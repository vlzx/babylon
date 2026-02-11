import time
from faster_whisper import WhisperModel

model_path = "local_models/faster-whisper-large-v3-turbo-ct2"
print(f"Model: {model_path}")
audio_path = "data/natsuyoshiyuko_s1.mp3"

# 1. 加载模型
model = WhisperModel(model_path, device="cuda")

# 2. 开始计时
start_time = time.time()

# 执行转录
# 注意：segments 是一个生成器，真正的计算发生在遍历 segments 时
segments, info = model.transcribe(audio_path, language="ja")

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# 3. 遍历结果（此时模型在真正工作）
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

# 4. 结束计时
end_time = time.time()

### 结果计算与输出

# ASR 过程总耗时
asr_duration = end_time - start_time
# 音频总时长（从 info 对象中获取）
audio_duration = info.duration
# 计算 RTF
rtf = asr_duration / audio_duration

print("-" * 30)
print(f"音频时长 (Audio Duration): {audio_duration:.2f}s")
print(f"转录耗时 (ASR Duration):   {asr_duration:.2f}s")
print(f"实时率 (RTF):            {rtf:.4f}")
time.sleep(5)
