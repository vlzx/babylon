import shutil
import numpy as np
import torch
import librosa
import time
import os
from collections import deque
from scipy.io.wavfile import write as wav_write
from silero_vad import load_silero_vad, VADIterator, get_speech_timestamps


# --- 1. 配置参数 ---
SAMPLING_RATE = 16000
CHUNK_SIZE = 512
VAD_THRESHOLD = 0.5
VAD_PADDING_MS = 500  # 前后缓冲

OUTPUT_DIR = "data/vad_segments_merged"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. 加载 ---
model = load_silero_vad(onnx=True)
wav_numpy, sr = librosa.load('data/nekoyashiki.mp3', sr=SAMPLING_RATE)

vad_iterator = VADIterator(model, threshold=VAD_THRESHOLD, sampling_rate=SAMPLING_RATE, min_silence_duration_ms=VAD_PADDING_MS, speech_pad_ms=VAD_PADDING_MS)

audio_duration = len(wav_numpy)/SAMPLING_RATE
print(f"音频时长: {audio_duration:.2f}s")
print("--------------------------------------------------")

def save_segment(chunks, idx):
    if not chunks: return
    full = np.concatenate(chunks)
    path = os.path.join(OUTPUT_DIR, f"seg_{idx:03d}.wav")
    wav_write(path, SAMPLING_RATE, full)
    print(f"  -> [保存文件] {path} (时长: {len(full)/SAMPLING_RATE:.2f}s)")

# --- 4. 循环 ---
count = 0
count_short_audio = 0
count_long_audio = 0
current_start = None # 用于暂存 start 时间戳

print("开始处理...")
start = time.time()

for i in range(0, len(wav_numpy), CHUNK_SIZE):
    chunk = wav_numpy[i : i + CHUNK_SIZE]
    if len(chunk) < CHUNK_SIZE: break
    
    speech_dict = vad_iterator(chunk)
    
    if speech_dict:
        # 1. 只有 start：记录开始点
        if 'start' in speech_dict:
            current_start = speech_dict['start']
            # print(f"检测到起点: {current_start}")

        # 2. 只有 end：获取结束点 -> 切片 -> 保存
        if 'end' in speech_dict:
            current_end = speech_dict['end']
            
            # 确保有过 start (防止只有 end 的异常情况)
            if current_start is not None:
                # 核心逻辑：直接根据时间戳从原始 numpy 数组切片
                # 注意：speech_dict返回的是采样点索引，直接用于切片即可
                segment = wav_numpy[current_start : current_end]
                duration_s = (current_end - current_start) / SAMPLING_RATE
                
                # 调用你的保存函数 (你的函数期望由 chunk 组成的 list，所以这里包一层 [])
                save_segment([segment], count)
                count += 1
                if duration_s > 28:
                    count_long_audio += 1
                if duration_s < 1:
                    count_short_audio += 1
                
                # 重置 start，准备下一段
                current_start = None 

processing_time = time.time() - start
print(f'Total segments saved: {count}')
print(f'Total segments > 28s: {count_long_audio}')
print(f'Total segments < 1s: {count_short_audio}')

# 计算 RTF (实时率)
# RTF = 处理耗时 / 音频时长
# RTF < 1 表示处理速度快于实时 (越小越快)
rtf = processing_time / audio_duration

# --- 性能指标计算结束 ---

print("--------------------------------------------------")
print(f"处理耗时: {processing_time:.4f} 秒")
print(f"音频时长: {audio_duration:.2f}s")
print(f"RTF (实时率): {rtf:.4f}")

vad_iterator.reset_states()