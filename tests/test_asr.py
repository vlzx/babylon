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
max_sentence_sec = 15.0

wav_numpy, sample_rate = sf.read('data/nekoyashiki_utawaku.wav', dtype='float32')
wav_numpy = wav_numpy[:int(sample_rate*600)]
print(f"采样率: {sample_rate} Hz")
print(f"数据形状: {wav_numpy.shape}") # (samples, channels)
print(f"数据类型: {wav_numpy.dtype}")

f = open(os.path.join('data/transcript', 'nekoyashiki_utawaku.txt'), 'w+')

script_start_time = time.time()
total_audio_duration = len(wav_numpy) / sample_rate

last_merged_text = ""
same_merged_count = 0

start_idx = 0  # 当前缓冲区起始采样点
end_idx = 0  # 当前缓冲区结束采样点
for i in range(0, len(wav_numpy), int(sample_rate * 1)):
    end_idx += int(sample_rate * 1)  # 每轮将缓冲区结束点向后推进 1 秒
    buffer = wav_numpy[start_idx:end_idx].copy()  # 截取当前待识别窗口 [start_idx, end_idx)
    timestamp = f'audio->[{format_hms(start_idx / sample_rate)}:{format_hms(end_idx / sample_rate)}]'  # 输出当前窗口时间区间
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
        last_merged_text = ""
        same_merged_count = 0
        start_idx += int(sample_rate * 1)  # 无语音时起点同样前进 1 秒
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
            last_merged_text = ""
            same_merged_count = 0
            start_idx += int(sample_rate * 1)  # 高 no_speech 概率时起点前进 1 秒
            print(f'耗时: {time.time() - start_time:.2f}s')
            continue

    segments, info = asr_model.transcribe(buffer,
                                        language="ja",
                                        word_timestamps=True,
                                        condition_on_previous_text=False,
                                        initial_prompt=last_merged_text
    )
    
    prev_end = 0.0
    next_start_idx = start_idx  # 下一轮起始点候选，默认不截断
    merged_text_parts = []
    for segment in segments:
        merged_text_parts.append(segment.text)
        if segment.start != prev_end:
            msg = f'Δ +{segment.start - prev_end:.2f}s'
            print(msg)
            f.write(msg + '\n')
            # 在 ASR 分段出现断点时，以断点两侧时间戳中点作为下一轮音频起点
            cut_sec = (segment.start + prev_end) / 2
            candidate_start_idx = start_idx + int(cut_sec * sample_rate)  # 断点中点对应的候选起始采样点
            next_start_idx = max(next_start_idx, candidate_start_idx)  # 更新下一轮起始点候选（取更靠后位置）

        segment_duration = segment.end - segment.start
        if segment_duration > max_sentence_sec:
            msg = f'long segment {format_hms(segment_duration)} > {format_hms(max_sentence_sec)}, cut by segment.end={format_hms(segment.end)}'
            print(msg)
            f.write(msg + '\n')
            candidate_start_idx = start_idx + int(segment.end * sample_rate)  # 长句按 segment.end 计算候选起始点
            next_start_idx = max(next_start_idx, candidate_start_idx)  # 更新下一轮起始点候选（取更靠后位置）
        prev_end = segment.end

    merged_text = "".join(merged_text_parts).strip()
    asr_msg = f"[ASR] {merged_text}"
    print(asr_msg)
    f.write(asr_msg + '\n')

    if merged_text:
        if merged_text == last_merged_text:
            same_merged_count += 1
        else:
            last_merged_text = merged_text
            same_merged_count = 1
    else:
        last_merged_text = ""
        same_merged_count = 0

    if same_merged_count >= 3 and prev_end > 0:
        stable_msg = f'stable merged_text x{same_merged_count}, cut by sentence end={format_hms(prev_end)}'
        print(stable_msg)
        f.write(stable_msg + '\n')
        candidate_start_idx = end_idx  # 连续稳定文本时直接把候选起始点推进到当前窗口末端
        next_start_idx = max(next_start_idx, candidate_start_idx)  # 更新下一轮起始点候选（取更靠后位置）

    if next_start_idx > start_idx:  # 仅当候选起始点向前推进时才执行截断
        min_window_samples = int(sample_rate * 1)
        max_start_idx = end_idx - min_window_samples  # 为保证至少 1 秒窗口，允许的最大起始点
        next_start_idx = min(next_start_idx, max_start_idx)  # 限制起始点不能超过最大允许值
        next_start_idx = max(next_start_idx, start_idx)  # 限制起始点不能回退到当前起点之前
        cut_msg = f'cut start_idx: {format_hms(start_idx / sample_rate)} -> {format_hms(next_start_idx / sample_rate)}'  # 输出本轮截断结果
        print(cut_msg)
        f.write(cut_msg + '\n')
        start_idx = next_start_idx  # 提交下一轮缓冲区起始采样点

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

script_end_time = time.time()
script_duration = script_end_time - script_start_time
script_rtf = script_duration / total_audio_duration if total_audio_duration > 0 else 0.0

print("=" * 30)
print(f"脚本总音频时长 (Total Audio Duration): {format_hms(total_audio_duration)}")
print(f"脚本总耗时 (Script Duration):        {format_hms(script_duration)}")
print(f"脚本整体实时率 (Overall RTF):        {script_rtf:.4f}")
f.write("=" * 30 + '\n')
f.write(f"脚本总音频时长 (Total Audio Duration): {format_hms(total_audio_duration)}\n")
f.write(f"脚本总耗时 (Script Duration):        {format_hms(script_duration)}\n")
f.write(f"脚本整体实时率 (Overall RTF):        {script_rtf:.4f}\n")

f.close()
