import os
import sys
import time
from faster_whisper import WhisperModel
import soundfile as sf

sys.path.append(os.getcwd())
from utils.chrono import format_hms


asr_model_path = "local_models/faster-whisper-large-v3-turbo-ct2"
vad_model_path = "local_models/faster-whisper-tiny"

asr_model = WhisperModel(asr_model_path, device="cuda")
vad_model = WhisperModel(vad_model_path, device="cuda")

wav_numpy, sample_rate = sf.read('data/nekoyashiki_utawaku.wav', dtype='float32')
wav_numpy = wav_numpy[:int(sample_rate*600)]
print(f"采样率: {sample_rate} Hz")
print(f"数据形状: {wav_numpy.shape}") # (samples, channels)
print(f"数据类型: {wav_numpy.dtype}")

f = open(os.path.join('data/transcript', 'nekoyashiki_utawaku.txt'), 'w+')

start_idx = 0
end_idx = 0
for i in range(0, len(wav_numpy), int(sample_rate * 1)):
    end_idx += int(sample_rate * 1)
    buffer = wav_numpy[start_idx:end_idx].copy()
    timestamp = f'audio->[{start_idx/sample_rate:.2f}s:{end_idx/sample_rate:.2f}s]'
    print(timestamp)
    f.write(timestamp + '\n')
    
    # 开始计时
    start_time = time.time()

    # 注意：segments 是一个生成器，真正的计算发生在遍历 segments 时
    detect_segments, info = vad_model.transcribe(buffer,
                                        language="ja",
                                        without_timestamps=True,
                                        condition_on_previous_text=False,
    )

    detect_results = list(detect_segments)
    # print(f'is_speech: {len(detect_results)}')
    if len(detect_results) == 0:
        start_idx += int(sample_rate * 1)
        print(f'[VAD] no_speech_probs: []')
        print(f'耗时: {time.time() - start_time:.2f}s')
        continue
    else:
        for segment in detect_results:
            res = f"[VAD] [{format_hms(segment.start)} -> {format_hms(segment.end)}] {segment.text}"
            # print(res)
            metrics = f'[VAD] temperature:{segment.temperature}  avg_logprob:{segment.avg_logprob:.2f}  compression_ratio:{segment.compression_ratio:.2f}  no_speech_prob:{segment.no_speech_prob}'
            # print(metrics)
        print(f'[VAD] no_speech_probs: {[segment.no_speech_prob for segment in detect_results]}')
        if all([segment.no_speech_prob > 0.8 for segment in detect_results]):
            start_idx += int(sample_rate * 1)
            print(f'耗时: {time.time() - start_time:.2f}s')
            continue

    segments, info = asr_model.transcribe(buffer,
                                        language="ja",
                                        word_timestamps=True,
                                        condition_on_previous_text=False,
    )
    
    prev_end = 0.0
    for segment in segments:
        res = f"[ASR] [{format_hms(segment.start)} -> {format_hms(segment.end)}] {segment.text}"
        print(res)
        metrics = f'[ASR] temperature:{segment.temperature}  avg_logprob:{segment.avg_logprob:.2f}  compression_ratio:{segment.compression_ratio:.2f}'
        print(metrics)
        if segment.start != prev_end:
            msg = f'Δ +{segment.start - prev_end:.2f}s'
            print(msg)
            f.write(msg + '\n')
        prev_end = segment.end
        f.write(res + '\n')
        # f.write(metrics + '\n')

    # 结束计时
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

f.close()
