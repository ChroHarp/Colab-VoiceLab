import os
import sys
import wave
import subprocess
import importlib.util

import torch
import torchaudio
from safetensors.torch import save_file

# ComfyUI executes all nodes inside torch.inference_mode()
torch.set_grad_enabled(False)

COMFY_DIR = "/content/ComfyUI"
sys.path.insert(0, COMFY_DIR)
sys.path.insert(0, os.path.join(COMFY_DIR, "custom_nodes"))

# Load ComfyUI-Drama3 package via importlib
drama3_root = os.path.join(COMFY_DIR, "custom_nodes", "ComfyUI-Drama3")
spec = importlib.util.spec_from_file_location("ComfyUI_Drama3", os.path.join(drama3_root, "__init__.py"), submodule_search_locations=[drama3_root])
pkg = importlib.util.module_from_spec(spec)
sys.modules["ComfyUI_Drama3"] = pkg
spec.loader.exec_module(pkg)

nodes_spec = importlib.util.spec_from_file_location("ComfyUI_Drama3.nodes", os.path.join(drama3_root, "nodes.py"))
nodes = importlib.util.module_from_spec(nodes_spec)
sys.modules["ComfyUI_Drama3.nodes"] = nodes
nodes_spec.loader.exec_module(nodes)

OUT_DIR = "/content/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def load_audio_comfy(path):
    waveform, sr = torchaudio.load(path)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": sr}

def save_wav(audio_dict, out_path):
    waveform = audio_dict["waveform"].squeeze().cpu()
    sr = audio_dict["sample_rate"]
    pcm = (waveform.clamp(-1, 1) * 32767).round().to(torch.int16).numpy().tobytes()
    with wave.open(out_path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm)
    print(f"✓ 已儲存音檔: {out_path}")

print("=== [1/5] 準備參考音檔 (VoiceSample 與 Edge TTS 台灣雲哲男聲) ===")
USER_SAMPLE_WAV = "/tmp/user_sample_10s.wav"
subprocess.run(["ffmpeg", "-y", "-i", "/content/voice_sample.mp3", "-t", "10", "-ar", "44100", "-ac", "1", USER_SAMPLE_WAV], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

EDGE_REF_WAV = os.path.join(OUT_DIR, "00_ref_edge_yunjhe_tw.wav")
subprocess.run([
    "ffmpeg", "-y",
    "-i", "/content/edge_clip1.mp3",
    "-i", "/content/edge_clip2.mp3",
    "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]",
    "-map", "[out]", "-ar", "44100", "-ac", "1", EDGE_REF_WAV
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

EDGE_REF_TEXT = "先考慮一個邊長為 a 的正方體，體積為 a 三方。將共同厚度乘上底面積總和，體積即可寫成 a 減 b，乘上 a 平方加 a b 加 b 平方。"

print("=== [2/5] 使用 Qwen3-ASR 辨識 VoiceSample 音檔並建立雙聲線混血參考音檔 ===")
asr_model = nodes.Qwen3ASRLoader.execute("qwen3_asr_1.7b_bf16.safetensors")[0]
user_audio = load_audio_comfy(USER_SAMPLE_WAV)
asr_out = nodes.Qwen3ASRTranscribe.execute(asr_model, user_audio, language="auto")
user_transcript = asr_out[0].strip()
user_lang = asr_out[1]
print(f"🎤 VoiceSample 偵測語言: {user_lang}")
print(f"📝 VoiceSample 轉錄逐字稿: {user_transcript}")

# Create blended audio: 6s of VoiceSample + Edge clip 2 (~6s of Taiwan YunJhe math speech)
USER_6S_WAV = "/tmp/user_6s.wav"
subprocess.run(["ffmpeg", "-y", "-i", USER_SAMPLE_WAV, "-t", "6", "-ar", "44100", "-ac", "1", USER_6S_WAV], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
user_6s_transcript = nodes.Qwen3ASRTranscribe.execute(asr_model, load_audio_comfy(USER_6S_WAV), language="auto")[0].strip()

BLEND_WAV = os.path.join(OUT_DIR, "00_ref_blended_sample_and_tw.wav")
subprocess.run([
    "ffmpeg", "-y",
    "-i", USER_6S_WAV,
    "-i", "/content/edge_clip2.mp3",
    "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]",
    "-map", "[out]", "-ar", "44100", "-ac", "1", BLEND_WAV
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

blend_transcript = f"{user_6s_transcript} 將共同厚度乘上底面積總和，體積即可寫成 a 減 b，乘上 a 平方加 a b 加 b 平方。"
print(f"📝 雙語混血參考逐字稿: {blend_transcript}")

del asr_model
torch.cuda.empty_cache()

print("=== [3/5] 使用 Drama 3 Codec 編碼三種聲音特徵 (並儲存為 .safetensors) ===")
codec = nodes.Drama3CodecLoader.execute("drama3_codec.safetensors")[0]

voice_user = nodes.Drama3EncodeVoice.execute(codec, user_audio, user_transcript)[0]
voice_edge = nodes.Drama3EncodeVoice.execute(codec, load_audio_comfy(EDGE_REF_WAV), EDGE_REF_TEXT)[0]
voice_blend = nodes.Drama3EncodeVoice.execute(codec, load_audio_comfy(BLEND_WAV), blend_transcript)[0]

save_file({"codes": voice_user["codes"].to(torch.int16).contiguous()}, os.path.join(OUT_DIR, "voice_profile_sample.safetensors"), metadata={"transcript": voice_user["transcript"]})
save_file({"codes": voice_edge["codes"].to(torch.int16).contiguous()}, os.path.join(OUT_DIR, "voice_profile_edge_tw.safetensors"), metadata={"transcript": voice_edge["transcript"]})
save_file({"codes": voice_blend["codes"].to(torch.int16).contiguous()}, os.path.join(OUT_DIR, "voice_profile_blended_tw.safetensors"), metadata={"transcript": voice_blend["transcript"]})
print("✓ 三組 .safetensors 專屬音色特徵檔已儲存！")

print("=== [4/5] 載入 Drama 3 (35B int8 Experts-in-RAM) 合成兩句數學解說台詞 ===")
drama3_model = nodes.Drama3Loader.execute("drama3_int8experts_bf16.safetensors", experts="in RAM", expert_cache_gb=0.0)[0]

# 台詞 1: 當 a = b 時，(a+b)/2 = \sqrt{a*b}
# 台詞 2: 最美的公式：e^{\pi i} = -1
LINE_1 = "當 a 等於 b 時，二分之 a 加 b，等於根號 a 乘以 b。 [language: zh]"
LINE_2 = "最美的公式：e 的 pi i 次方，等於負一。 [language: zh]"

profiles = [
    ("01_drama3_from_voicesample", voice_user),
    ("02_drama3_from_edge_yunjhe", voice_edge),
    ("03_drama3_blended_sample_and_yunjhe", voice_blend),
]

for prefix, v_profile in profiles:
    print(f"🎙️ 正在合成 [{prefix}] - 台詞 1 (算幾不等式等號成立條件)...")
    codes_1 = nodes.Drama3Generate.execute(
        model=drama3_model,
        text=LINE_1,
        voices={"voice0": v_profile},
        seed=42,
        temperature=1.0,
        top_p=0.9,
        top_k=0,
        max_frames=350,
        split_long_text=False,
        speakers="one take per line"
    )[0]
    audio_1 = nodes.Drama3Decode.execute(codec, codes_1, shorten_long_pauses=True)[0]
    save_wav(audio_1, os.path.join(OUT_DIR, f"{prefix}_line1_amgm.wav"))

    print(f"🎙️ 正在合成 [{prefix}] - 台詞 2 (歐拉公式)...")
    codes_2 = nodes.Drama3Generate.execute(
        model=drama3_model,
        text=LINE_2,
        voices={"voice0": v_profile},
        seed=42,
        temperature=1.0,
        top_p=0.9,
        top_k=0,
        max_frames=350,
        split_long_text=False,
        speakers="one take per line"
    )[0]
    audio_2 = nodes.Drama3Decode.execute(codec, codes_2, shorten_long_pauses=True)[0]
    save_wav(audio_2, os.path.join(OUT_DIR, f"{prefix}_line2_euler.wav"))

print("=== [5/5] 打包所有生成音檔與音色特徵檔 ===")
subprocess.run(["tar", "-czf", "/content/drama3_samples.tar.gz", "-C", OUT_DIR, "."], check=True)
print("🎉 全部合成並打包完成: /content/drama3_samples.tar.gz")
