#!/usr/bin/env python3
"""
convert_voice_seedvc.py — 二段接力法 (Seed-VC) 零樣本音色轉換與特徵混血腳本

功能：
1. 自動裁切並清洗參考音檔：
   - Ishikawa-sample.mp3 (日文女聲：去除前 8 秒空白、過長停頓與 27.5s 跳針處)
   - moi-sample.mp3 (台灣中文女聲：去除前 3.2 秒空白)
2. 支援單一音檔或整個目錄 (例如 ManimGX-Edu/<Project>/voice/*.mp3) 批次轉換
3. 支援 CAMPPlus 音色特徵向量混血 (timbre_blend) 與音高半音微調 (pitch_shift)
"""

import os
import sys
import glob
import argparse
import subprocess
import numpy as np
import torch
import torchaudio
import librosa

SEED_VC_DIR = "/content/seed-vc"
if os.path.exists(SEED_VC_DIR) and SEED_VC_DIR not in sys.path:
    sys.path.insert(0, SEED_VC_DIR)

from inference import load_models, adjust_f0_semitones, crossfade
from modules.bigvgan import bigvgan

# 修補 BigVGAN._from_pretrained 以相容新版 huggingface_hub
_orig_bigvgan_from_pretrained = bigvgan.BigVGAN._from_pretrained


@classmethod
def _patched_bigvgan_from_pretrained(
    cls,
    *,
    model_id,
    revision,
    cache_dir,
    force_download,
    proxies=None,
    resume_download=False,
    local_files_only,
    token,
    map_location="cpu",
    strict=False,
    use_cuda_kernel=False,
    **kwargs,
):
    return _orig_bigvgan_from_pretrained.__func__(
        cls,
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        resume_download=resume_download,
        local_files_only=local_files_only,
        token=token,
        map_location=map_location,
        strict=strict,
        use_cuda_kernel=use_cuda_kernel,
    )


bigvgan.BigVGAN._from_pretrained = _patched_bigvgan_from_pretrained


class SeedVCModelConfig:
    def __init__(self, f0_condition=True, fp16=True, checkpoint=None, config=None):
        self.f0_condition = f0_condition
        self.fp16 = fp16
        self.checkpoint = checkpoint
        self.config = config


def prepare_clean_references(ishikawa_mp3, moi_mp3, out_dir):
    """裁切出最適合做聲線特徵提取的乾淨段落"""
    os.makedirs(out_dir, exist_ok=True)
    ishikawa_clean = os.path.join(out_dir, "00_ishikawa_ref_clean.wav")
    ishikawa_natural = os.path.join(out_dir, "00_ishikawa_ref_natural_12s.wav")
    moi_full = os.path.join(out_dir, "00_moi_clean_full.wav")
    moi_10s = os.path.join(out_dir, "00_moi_clean_10s.wav")

    if ishikawa_mp3 and os.path.exists(ishikawa_mp3):
        # 1. 自然呼吸版 (7.8s ~ 19.6s)
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", ishikawa_mp3,
                "-ss", "7.8", "-to", "19.6",
                "-ar", "44100", "-ac", "1", ishikawa_natural,
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # 2. 高純度濃縮版 (去除前8秒空白、長停頓、並避開 27.5s 跳針片段)
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", ishikawa_mp3,
                "-filter_complex",
                "[0:a]atrim=start=7.9:end=9.45,asetpts=PTS-STARTPTS[a1];"
                "[0:a]atrim=start=10.25:end=12.2,asetpts=PTS-STARTPTS[a2];"
                "[0:a]atrim=start=14.1:end=16.4,asetpts=PTS-STARTPTS[a3];"
                "[0:a]atrim=start=17.45:end=19.4,asetpts=PTS-STARTPTS[a4];"
                "[0:a]atrim=start=21.35:end=22.6,asetpts=PTS-STARTPTS[a5];"
                "[0:a]atrim=start=23.55:end=25.5,asetpts=PTS-STARTPTS[a6];"
                "[0:a]atrim=start=30.95:end=35.1,asetpts=PTS-STARTPTS[a7];"
                "[a1][a2][a3][a4][a5][a6][a7]concat=n=7:v=0:a=1[out]",
                "-map", "[out]", "-ar", "44100", "-ac", "1", ishikawa_clean,
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    if moi_mp3 and os.path.exists(moi_mp3):
        # 去除前 3.2 秒空白
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", moi_mp3,
                "-ss", "3.2", "-to", "20.7",
                "-ar", "44100", "-ac", "1", moi_full,
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", moi_mp3,
                "-ss", "3.2", "-to", "13.3",
                "-ar", "44100", "-ac", "1", moi_10s,
            ],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    return ishikawa_clean, moi_full, moi_10s


@torch.no_grad()
def convert_single_audio(
    models_tuple,
    source_path,
    target_path,
    out_path,
    f0_condition=True,
    auto_f0_adjust=True,
    pitch_shift=0,
    diffusion_steps=30,
    inference_cfg_rate=0.7,
    timbre_blend=1.0,
    blend_target_path=None,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = models_tuple
    sr = 22050 if not f0_condition else 44100
    hop_length = 256 if not f0_condition else 512
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    source_audio = librosa.load(source_path, sr=sr)[0]
    ref_audio = librosa.load(target_path, sr=sr)[0]

    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device)

    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    S_alt = semantic_fn(converted_waves_16k)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    mel = mel_fn(source_audio.to(device).float())
    mel2 = mel_fn(ref_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2))]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(
        ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    # 雙聲線特徵插值融合 (例如 Ishikawa + moi 混血聲線)
    if blend_target_path is not None and timbre_blend < 1.0:
        ref_b = librosa.load(blend_target_path, sr=sr)[0]
        ref_b = torch.tensor(ref_b[: sr * 25]).unsqueeze(0).float().to(device)
        waves_b_16k = torchaudio.functional.resample(ref_b, sr, 16000)
        feat_b = torchaudio.compliance.kaldi.fbank(
            waves_b_16k, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat_b = feat_b - feat_b.mean(dim=0, keepdim=True)
        style_b = campplus_model(feat_b.unsqueeze(0))
        orig_norm = torch.norm(style2, dim=-1, keepdim=True)
        style2 = timbre_blend * style2 + (1.0 - timbre_blend) * style_b
        style2 = style2 / torch.norm(style2, dim=-1, keepdim=True) * orig_norm

    if f0_condition:
        F0_ori = f0_fn(ori_waves_16k[0], thred=0.03)
        F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)
        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]

        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
        voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
        median_log_f0_ori = torch.median(voiced_log_f0_ori)
        median_log_f0_alt = torch.median(voiced_log_f0_alt)

        shifted_log_f0_alt = log_f0_alt.clone()
        if auto_f0_adjust:
            shifted_log_f0_alt[F0_alt > 1] = (
                log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
            )
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if pitch_shift != 0:
            shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(
                shifted_f0_alt[F0_alt > 1], pitch_shift
            )
    else:
        F0_ori = None
        shifted_f0_alt = None

    cond, _, _, _, _ = model.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
    )
    prompt_condition, _, _, _, _ = model.length_regulator(
        S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
    )

    max_source_window = max_context_window - mel2.size(2)
    processed_frames = 0
    generated_wave_chunks = []

    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2,
                style2,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, mel2.size(-1) :]
        vc_wave = vocoder_fn(vc_target.float()).squeeze()[None, :]
        if processed_frames == 0:
            if is_last_chunk:
                generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                break
            generated_wave_chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            output_wave = crossfade(
                previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len
            )
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            output_wave = crossfade(
                previous_chunk.cpu().numpy(),
                vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                overlap_wave_len,
            )
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len

    vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torchaudio.save(out_path, vc_wave.cpu(), sr)
    print(f"✓ 已轉換並儲存: {out_path} (sr={sr}Hz)")


def main():
    parser = argparse.ArgumentParser(description="Seed-VC 台灣中文基底 + 日系聲優音色轉換工具")
    parser.add_argument("--source", type=str, required=True, help="來源台灣口音音檔或資料夾路徑")
    parser.add_argument("--target", type=str, required=True, help="目標聲優參考音檔 (如 00_ishikawa_ref_clean.wav)")
    parser.add_argument("--blend-target", type=str, default=None, help="第二參考音檔 (如 00_moi_clean_10s.wav，用於混血聲線)")
    parser.add_argument("--timbre-blend", type=float, default=1.0, help="目標音色比例 (0.0~1.0，預設 1.0 為純目標音色，0.65 為 65%% 主目標 + 35%% 第二目標)")
    parser.add_argument("--output", type=str, required=True, help="輸出音檔或資料夾路徑")
    parser.add_argument("--f0-condition", action="store_true", default=True, help="使用 44.1kHz F0 條件模型 (預設開啟)")
    parser.add_argument("--auto-f0-adjust", action="store_true", default=True, help="自動對齊目標聲優的音高中位數")
    parser.add_argument("--pitch-shift", type=int, default=0, help="額外半音偏移 (semi-tones)")
    parser.add_argument("--steps", type=int, default=30, help="Diffusion 步數 (預設 30)")
    args = parser.parse_args()

    os.chdir(SEED_VC_DIR)
    models_tuple = load_models(SeedVCModelConfig(f0_condition=args.f0_condition))

    if os.path.isdir(args.source):
        os.makedirs(args.output, exist_ok=True)
        files = sorted(
            glob.glob(os.path.join(args.source, "*.wav"))
            + glob.glob(os.path.join(args.source, "*.mp3"))
        )
        for f in files:
            base = os.path.splitext(os.path.basename(f))[0]
            out_file = os.path.join(args.output, f"{base}_vc.wav")
            convert_single_audio(
                models_tuple,
                source_path=f,
                target_path=args.target,
                out_path=out_file,
                f0_condition=args.f0_condition,
                auto_f0_adjust=args.auto_f0_adjust,
                pitch_shift=args.pitch_shift,
                diffusion_steps=args.steps,
                timbre_blend=args.timbre_blend,
                blend_target_path=args.blend_target,
            )
    else:
        convert_single_audio(
            models_tuple,
            source_path=args.source,
            target_path=args.target,
            out_path=args.output,
            f0_condition=args.f0_condition,
            auto_f0_adjust=args.auto_f0_adjust,
            pitch_shift=args.pitch_shift,
            diffusion_steps=args.steps,
            timbre_blend=args.timbre_blend,
            blend_target_path=args.blend_target,
        )


if __name__ == "__main__":
    main()
