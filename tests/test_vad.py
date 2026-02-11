import torch
import librosa
import time
from silero_vad import load_silero_vad, get_speech_timestamps

# 1. 加载模型
# onnx=True 通常推理速度更快
model = load_silero_vad(onnx=True)

# 2. 使用 librosa 代替 torchaudio 加载音频
# Librosa 会自动重采样到 16000Hz (Silero 模型的要求) 并转换为 float32 格式
# 请替换为您实际的文件路径
wav_numpy, sr = librosa.load('data/natsuyoshiyuko_s1.mp3', sr=16000)

# 3. 将 numpy 数组转换为 PyTorch 张量
wav = torch.tensor(wav_numpy)

# --- 性能指标计算开始 ---

# 计算音频总时长 (秒) = 样本总数 / 采样率
audio_duration = len(wav) / 16000

print(f"音频时长: {audio_duration:.2f} 秒")

# 记录开始时间
start_time = time.time()

# 4. 获取语音时间戳 (推理过程)
speech_timestamps = get_speech_timestamps(
    wav,
    model,
    return_seconds=True  # 返回秒为单位的时间戳 (默认是样本数)
)

# 记录结束时间
end_time = time.time()

# 计算处理耗时
processing_time = end_time - start_time

# 计算 RTF (实时率)
# RTF = 处理耗时 / 音频时长
# RTF < 1 表示处理速度快于实时 (越小越快)
rtf = processing_time / audio_duration

# --- 性能指标计算结束 ---

print("--------------------------------------------------")
print(f"处理耗时: {processing_time:.4f} 秒")
print(f"RTF (实时率): {rtf:.4f}")
print(f"检测到的语音段数量: {len(speech_timestamps)}")
print("--------------------------------------------------")
print("语音时间戳详情:", speech_timestamps)