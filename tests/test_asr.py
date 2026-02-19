import os
import sys
import time
from faster_whisper import WhisperModel
import soundfile as sf
from rich import print

sys.path.append(os.getcwd())
from utils.chrono import format_hms
from utils.string import get_added_part


hallucination_blacklist = [
    "ご視聴ありがとうございました",
    "チャンネル登録をお願いいたします"
]

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

last_valid_text = ""
last_merged_text = ""
same_merged_count = 0
initial_prompt = ""
is_speech = False

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

    # ----- VAD -----
    if not is_speech:
        # 注意：segments 是一个生成器，真正的计算发生在遍历 segments 时
        detect_segments, info = vad_model.transcribe(buffer,
                                            language="ja",
                                            without_timestamps=True,
                                            condition_on_previous_text=False,
        )

        detect_results = list(detect_segments)
        # 场景A1 静音
        if len(detect_results) == 0:
            last_merged_text = ""
            same_merged_count = 0
            start_idx = end_idx  # 无语音时跳过音频
            initial_prompt = ""
            is_speech = False
            vad_msg = f'[VAD] no_speech_probs: []'
            print(vad_msg)
            f.write(vad_msg + '\n')
            print(f'耗时: {time.time() - start_time:.2f}s')
            continue
        else:
            for segment in detect_results:
                res = f"[VAD] [{format_hms(segment.start)} -> {format_hms(segment.end)}] {segment.text}"
                # print(res)
                metrics = f'[VAD] temperature:{segment.temperature}  avg_logprob:{segment.avg_logprob:.2f}  compression_ratio:{segment.compression_ratio:.2f}  no_speech_prob:{segment.no_speech_prob}'
                # print(metrics)
            vad_msg = f'[VAD] no_speech_probs: {[segment.no_speech_prob for segment in detect_results]}'
            print(vad_msg)
            f.write(vad_msg + '\n')
            # 场景A2 非人声（no_speech_prob > 0.8）
            if all([segment.no_speech_prob > 0.8 for segment in detect_results]):
                last_merged_text = ""
                same_merged_count = 0
                start_idx = end_idx  # 高 no_speech 概率时跳过音频
                initial_prompt = ""
                is_speech = False
                print(f'耗时: {time.time() - start_time:.2f}s')
                continue
        # 不符合场景A1A2，判定为人声
        is_speech = True

    # ----- ASR -----
    segments, info = asr_model.transcribe(buffer,
                                        language="ja",
                                        word_timestamps=True,
                                        condition_on_previous_text=False,
                                        initial_prompt=initial_prompt
    )
    
    prev_end = 0.0
    next_start_idx = start_idx  # 下一轮起始点候选，默认不截断
    merged_text_parts = []
    for segment in segments:
        merged_text_parts.append(segment.text)
        # 场景B 音频分段
        if segment.start != prev_end:
            msg = f'Δ +{segment.start - prev_end:.2f}s'
            print(msg)
            f.write(msg + '\n')
            # 在 ASR 分段出现断点时，以断点两侧时间戳中点作为下一轮音频起点
            cut_sec = (segment.start + prev_end) / 2
            candidate_start_idx = start_idx + int(cut_sec * sample_rate)  # 断点中点对应的候选起始采样点
            next_start_idx = max(next_start_idx, candidate_start_idx)  # 更新下一轮起始点候选（取更靠后位置）

        segment_duration = segment.end - segment.start
        
        # 场景C 音频超长截断
        if segment_duration > max_sentence_sec:
            msg = f'long segment {format_hms(segment_duration)} > {format_hms(max_sentence_sec)}, cut by segment.end={format_hms(segment.end)}'
            print(msg)
            f.write(msg + '\n')
            candidate_start_idx = end_idx  # 长句直接截断
            next_start_idx = max(next_start_idx, candidate_start_idx)  # 更新下一轮起始点候选（取更靠后位置）
        prev_end = segment.end

    merged_text = "".join(merged_text_parts).strip()
    delta_text = get_added_part(last_valid_text, merged_text)
    delta_msg = f"[DELTA] {delta_text}"
    asr_msg = f"[ASR] {merged_text}"
    prompt_msg = f"[LAST] {last_valid_text}\n[INIT_PROMPT] {initial_prompt}"
    print(delta_msg)
    f.write(delta_msg + '\n')
    print(prompt_msg)
    f.write(prompt_msg + '\n')
    print(asr_msg)
    f.write(asr_msg + '\n')
    
    # 场景D 增量音频内识别出完整的黑名单文本，判定为幻觉，移除相关文本
    for item in hallucination_blacklist:
        if item in delta_text:
            merged_text = merged_text.replace(item, '')
            print(f'[REVISED] {merged_text}')
            f.write(f'[REVISED] {merged_text}\n')
    # 增量音频全部是幻觉，跳过
    if not merged_text:
        next_start_idx = end_idx

    if merged_text:
        last_valid_text = merged_text
        if merged_text == last_merged_text:
            same_merged_count += 1
        else:
            last_merged_text = merged_text
            same_merged_count = 1
    else:
        last_merged_text = ""
        same_merged_count = 0

    # 场景E 连续识别结果相同，说明后续音频非人声，截断
    if same_merged_count >= 3 and prev_end > 0:
        stable_msg = f'stable merged_text x{same_merged_count}, cut by sentence end={format_hms(prev_end)}'
        print(stable_msg)
        f.write(stable_msg + '\n')
        candidate_start_idx = end_idx  # 连续稳定文本时直接把候选起始点推进到当前窗口末端
        next_start_idx = max(next_start_idx, candidate_start_idx)  # 更新下一轮起始点候选（取更靠后位置）

    # 音频截断及其后处理
    if next_start_idx > start_idx:  # 仅当候选起始点向前推进时才执行截断
        # min_window_samples = int(sample_rate * 1)
        # max_start_idx = end_idx - min_window_samples  # 为保证至少 1 秒窗口，允许的最大起始点
        # next_start_idx = min(next_start_idx, max_start_idx)  # 限制起始点不能超过最大允许值
        next_start_idx = max(next_start_idx, start_idx)  # 限制起始点不能回退到当前起点之前
        cut_msg = f'cut start_idx: {format_hms(start_idx / sample_rate)} -> {format_hms(next_start_idx / sample_rate)}'  # 输出本轮截断结果
        print(cut_msg)
        f.write(cut_msg + '\n')
        start_idx = next_start_idx  # 提交下一轮缓冲区起始采样点
        initial_prompt = last_merged_text[-20:]
        # 新音频需要VAD
        if next_start_idx == end_idx:
            is_speech = False

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
