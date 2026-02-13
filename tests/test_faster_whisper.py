import os
import time
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions
import soundfile as sf
from natsort import natsorted


model_path = "local_models/faster-whisper-large-v3-turbo-ct2"
# model_path = "local_models/faster-whisper-tiny"
print(f"Model: {model_path}")
# wav_numpy, sr = librosa.load('data/natsuyoshiyuko_s1.mp3', sr=16000)
audio_dir = "data/"

# 1. 加载模型
model = WhisperModel(model_path, device="cuda")

files = os.listdir(audio_dir)
sorted_files = natsorted(files)

for filename in sorted_files:
    if not filename.endswith('.wav'):
        continue
    print(f'\n{filename}')
    wav_numpy, samplerate = sf.read(os.path.join(audio_dir, filename), dtype='float32')

    print(f"采样率: {samplerate} Hz")
    print(f"数据形状: {wav_numpy.shape}") # (samples, channels)
    print(f"数据类型: {wav_numpy.dtype}")

    f = open(os.path.join('data/transcript', filename.split('.')[0] + '.txt'), 'w+')

    # 2. 开始计时
    start_time = time.time()

    # 执行转录
    # 注意：segments 是一个生成器，真正的计算发生在遍历 segments 时
    segments, info = model.transcribe(wav_numpy,
                                      language="ja",
                                      word_timestamps=True,
                                      #   initial_prompt="",
                                      vad_filter=True,
                                      vad_parameters=VadOptions(
                                        threshold=0.05,
                                        min_silence_duration_ms=4000,
                                        speech_pad_ms=2000
                                    )
    )

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    # 3. 遍历结果（此时模型在真正工作）
    prev_end = 0.0
    for segment in segments:
        res = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
        print(res)
        # print(segment.words)
        if segment.start != prev_end:
            msg = f'Δ +{segment.start - prev_end:.2f}s'
            print(msg)
            f.write(msg + '\n')
        prev_end = segment.end
        f.write(res + '\n')

    # 4. 结束计时
    end_time = time.time()

    f.close()

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
