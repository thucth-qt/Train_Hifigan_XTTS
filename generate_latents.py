import os
from tqdm import tqdm

import numpy as np
import torchaudio
import torch
from torch.utils.data import DataLoader
from pydub import AudioSegment

from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainerConfig, XttsAudioConfig
from TTS.tts.models.xtts import load_audio

from models.gpt_decode import GPTDecode
from datasets.dataset_xtts import GPTXTTSDataset

class GPTDecoder:
    def __init__(self, config, config_dataset):
        self.config = config
        self.config_dataset = config_dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_samples, _ = load_tts_samples(
            config_dataset
        )
        self.tokenizer = VoiceBpeTokenizer(config.model_args.tokenizer_file)
        self.dataset = GPTXTTSDataset(config, self.train_samples, self.tokenizer, config.audio.sample_rate, is_eval=True)
        self.loader = DataLoader(self.dataset, collate_fn=self.dataset.collate_fn, batch_size=self.config.batch_size)
        self.model = GPTDecode.init_from_config(config).to(self.device)
    
    def load_audio_16k(self, files):
        audios = []
        for file in files:
            audio = load_audio(file, self.config.audio.sample_rate).to(self.device)
            audio = audio[:, : self.config.audio.sample_rate * 30]

            audio_16k = torchaudio.functional.resample(audio, self.config.audio.sample_rate, 16000).squeeze(0)
            audios.append(audio_16k)

        max_len   = max([_.size(0) for _ in audios])
        audio_padded  = torch.zeros(len(audios), max_len)
        for i in range(len(audios)):
            audio_padded[i, : audios[i].size(0)] = audios[i]

        return audio_padded

    def generate(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "gpt_latents"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "speaker_embeddings"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "synthesis"), exist_ok=True)

        for id, batch in enumerate(tqdm(self.loader)):
            batch["text_lengths"] = batch["text_lengths"].to(self.device)
            batch["wav_lengths"] = batch["wav_lengths"].to(self.device)
            batch["cond_idxs"] = batch["cond_idxs"].to(self.device)
            batch["wav"] = batch["wav"].to(self.device)

            batch = self.model.format_batch_on_device(batch)

            cond_mels = batch["cond_mels"].to(self.device)
            text_inputs = batch["text_inputs"].to(self.device)
            text_lengths = batch["text_lengths"].to(self.device)
            audio_codes = batch["audio_codes"].to(self.device)
            wav_lengths = batch["wav_lengths"].to(self.device)
            cond_idxs = batch["cond_idxs"].to(self.device)
            cond_lens = batch["cond_lens"]
            code_lengths = torch.ceil(wav_lengths / self.model.xtts.gpt.code_stride_len).long()

            audio_16k = self.load_audio_16k(batch["filenames"]).to(self.device)
            speaker_embedding = self.model.xtts.hifigan_decoder.speaker_encoder.forward(audio_16k, l2_norm=True).unsqueeze(-1)

            latents = self.model.generate(
                text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens
            )

            wav = []
            for i in range(self.config.batch_size):
                wav.append(self.model.xtts.hifigan_decoder(latents[i][: code_lengths[i]].unsqueeze(0), g=speaker_embedding[i]).detach().cpu().squeeze())

            for i in range(self.config.batch_size):
                file_name = batch["filenames"][i].split("/")[-1]

                raw_audio = AudioSegment.from_file(batch["filenames"][i])
                raw_audio = raw_audio.set_frame_rate(self.config.audio.output_sample_rate)
                raw_audio.export(os.path.join(output_dir, "wavs", file_name), format="wav")
                torchaudio.save(os.path.join(output_dir, "synthesis", file_name), torch.tensor(wav[i]).unsqueeze(0), self.config.audio.output_sample_rate)

                with open(os.path.join(output_dir, "gpt_latents", file_name.replace(".wav", ".npy")), "wb") as f:
                    np.save(f, latents[i][: code_lengths[i]].detach().squeeze(0).transpose(0, 1).cpu())
                
                with open(os.path.join(output_dir, "speaker_embeddings", file_name.replace(".wav", ".npy")), "wb") as f:
                    np.save(f, speaker_embedding[i].detach().squeeze(0).squeeze(1).cpu())

if __name__ == "__main__":
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    DVAE_CHECKPOINT = "/data/weights/viXTTS/dvae.pth"
    MEL_NORM_FILE = "/data/weights/viXTTS/mel_stats.pth"
    TOKENIZER_FILE = "/data/weights/viXTTS/vocab.json"
    XTTS_CHECKPOINT = "/data/weights/viXTTS/model.pth"

    model_args = GPTArgs(
        enable_redaction=False,
        kv_cache=True,
        
        min_conditioning_length=66150,
        max_conditioning_length=132300,
        max_wav_length=255995,
        max_text_length=200,
        
        num_chars=255,
        gpt_layers=30,
        gpt_n_model_channels=1024,
        gpt_n_heads=16,
        
        gpt_max_audio_tokens=605,
        gpt_max_text_tokens=402,
        gpt_max_prompt_tokens=70,
        gpt_number_text_tokens=7544,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_code_stride_len=1024,
        
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
        
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        
        input_sample_rate=22050,
        output_sample_rate=24000,
        output_hop_length=256,
        decoder_input_dim=1024,
        d_vector_dim=512,
        cond_d_vector_in_each_upsampling_layer=True,
        duration_const=102400
    )

    
    config = GPTTrainerConfig(
        audio=audio_config,
        model_args=model_args,
        batch_size = 3,
        num_loader_workers=12,
    )

    dataset_vivoice = BaseDatasetConfig(
    formatter="vivoice",
    dataset_name="vivoice",
    path="/data/thucth/voice/viVoice_restructured/",
    meta_file_train="/data/thucth/voice/viVoice_restructured/metadata.csv",
    language="vi",
)
    
    dataset_config = [dataset_vivoice]

    gpt_decode = GPTDecoder(config, dataset_config)
    gpt_decode.generate(output_dir="/data/thucth/voice/viVoice_latents")
