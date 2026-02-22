import os
import sys
import time
import threading

from faster_whisper import WhisperModel
from rich import print
import soundfile as sf
import numpy as np

sys.path.append(os.getcwd())
from utils.audio import stream_wav_realtime
from utils.chrono import format_hms
from utils.string import get_changed_part


hallucination_blacklist = [
    "ご視聴ありがとうございました",
    "チャンネル登録をお願いいたします",
]

asr_model_path = "local_models/faster-whisper-large-v3-turbo-ct2"
vad_model_path = "local_models/faster-whisper-tiny"

asr_model = WhisperModel(asr_model_path, device="cuda")
vad_model = WhisperModel(vad_model_path, device="cuda")
max_sentence_sec = 15.0

wav_path = "data/nekoyashiki_utawaku.wav"
audio_info = sf.info(wav_path)
sample_rate = audio_info.samplerate
one_second_samples = int(sample_rate * 1.0)
total_audio_duration = audio_info.duration

print(f"采样率: {sample_rate} Hz")
print(f"总时长: {total_audio_duration:.2f}s")
print(f"每轮累计时长: 1.00s ({one_second_samples} samples)")

script_start_time = time.time()

last_valid_text = ""
last_merged_text = ""
same_merged_count = 0
initial_prompt = ""
is_speech = False

start_idx = 0
end_idx = 0
audio_cache = None
shared = {
    "ready_chunks": [],
    "tail_samples": 0,
    "reader_done": False,
    "reader_error": None,
}
condition = threading.Condition()


def stream_reader_worker():
    pending_audio = None
    try:
        for audio_chunk in stream_wav_realtime(wav_path, frame_duration_ms=20, dtype="float32"):
            if pending_audio is None:
                pending_audio = audio_chunk.copy()
            else:
                pending_audio = np.concatenate([pending_audio, audio_chunk], axis=0)

            while len(pending_audio) >= one_second_samples:
                one_sec_chunk = pending_audio[:one_second_samples].copy()
                pending_audio = pending_audio[one_second_samples:]
                with condition:
                    shared["ready_chunks"].append(one_sec_chunk)
                    condition.notify()
    except Exception as exc:
        with condition:
            shared["reader_error"] = exc
    finally:
        with condition:
            shared["tail_samples"] = 0 if pending_audio is None else len(pending_audio)
            shared["reader_done"] = True
            condition.notify_all()


reader_thread = threading.Thread(target=stream_reader_worker, name="wav-stream-reader", daemon=True)
reader_thread.start()

f = open(os.path.join("data/transcript", "nekoyashiki_utawaku_stream.txt"), "w+", encoding="utf-8")
while True:
    with condition:
        while (
            len(shared["ready_chunks"]) == 0
            and shared["reader_error"] is None
            and not shared["reader_done"]
        ):
            condition.wait()

        if shared["reader_error"] is not None:
            raise RuntimeError("stream_wav_realtime reader failed") from shared["reader_error"]

        if len(shared["ready_chunks"]) == 0 and shared["reader_done"]:
            break

        one_sec_chunk = shared["ready_chunks"].pop(0)

    if audio_cache is None:
        audio_cache = one_sec_chunk
    else:
        audio_cache = np.concatenate([audio_cache, one_sec_chunk], axis=0)

    end_idx += len(one_sec_chunk)
    buffer = audio_cache[start_idx:end_idx].copy()
    timestamp = f"audio->[{format_hms(start_idx / sample_rate)}:{format_hms(end_idx / sample_rate)}]"
    print(timestamp)
    f.write(timestamp + "\n")

    start_time = time.time()

    # ----- VAD -----
    if not is_speech:
        detect_segments, _ = vad_model.transcribe(
            buffer,
            language="ja",
            without_timestamps=True,
            condition_on_previous_text=False,
        )

        detect_results = list(detect_segments)
        if len(detect_results) == 0:
            last_merged_text = ""
            same_merged_count = 0
            start_idx = end_idx
            initial_prompt = ""
            is_speech = False
            vad_msg = "[VAD] no_speech_probs: []"
            print(vad_msg)
            f.write(vad_msg + "\n")
            print(f"耗时: {time.time() - start_time:.2f}s")
            continue
        vad_msg = f"[VAD] no_speech_probs: {[segment.no_speech_prob for segment in detect_results]}"
        print(vad_msg)
        f.write(vad_msg + "\n")
        if all(segment.no_speech_prob > 0.8 for segment in detect_results):
            last_merged_text = ""
            same_merged_count = 0
            start_idx = end_idx
            initial_prompt = ""
            is_speech = False
            print(f"耗时: {time.time() - start_time:.2f}s")
            continue
        is_speech = True

    # ----- ASR -----
    segments, info = asr_model.transcribe(
        buffer,
        language="ja",
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt,
    )

    prev_end = 0.0
    next_start_idx = start_idx
    merged_text_parts = []
    for segment in segments:
        merged_text_parts.append(segment.text)
        if segment.start != prev_end:
            msg = f"Δ +{segment.start - prev_end:.2f}s"
            print(msg)
            f.write(msg + "\n")
            cut_sec = (segment.start + prev_end) / 2
            candidate_start_idx = start_idx + int(cut_sec * sample_rate)
            next_start_idx = max(next_start_idx, candidate_start_idx)

        segment_duration = segment.end - segment.start
        if segment_duration > max_sentence_sec:
            msg = (
                f"long segment {format_hms(segment_duration)} > {format_hms(max_sentence_sec)}, "
                f"cut by segment.end={format_hms(segment.end)}"
            )
            print(msg)
            f.write(msg + "\n")
            candidate_start_idx = end_idx
            next_start_idx = max(next_start_idx, candidate_start_idx)
        prev_end = segment.end

    merged_text = "".join(merged_text_parts).strip()
    delta_text = get_changed_part(
        last_valid_text.replace(' ', '').replace('\n', ''),
        merged_text.replace(' ', '').replace('\n', ''),
    )
    asr_msg = f"[LAST] {last_valid_text}\n[INIT_PROMPT] {initial_prompt}\n[ASR] {merged_text}\n[DELTA] {delta_text}"
    print(asr_msg)
    f.write(asr_msg + "\n")

    for item in hallucination_blacklist:
        if item in delta_text:
            merged_text = merged_text.replace(item, "")
            print(f"[REVISED] {merged_text}")
            f.write(f"[REVISED] {merged_text}\n")

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

    if same_merged_count >= 3 and prev_end > 0:
        stable_msg = f"stable merged_text x{same_merged_count}, cut by sentence end={format_hms(prev_end)}"
        print(stable_msg)
        f.write(stable_msg + "\n")
        candidate_start_idx = end_idx
        next_start_idx = max(next_start_idx, candidate_start_idx)

    if next_start_idx > start_idx:
        next_start_idx = max(next_start_idx, start_idx)
        cut_msg = (
            f"cut start_idx: {format_hms(start_idx / sample_rate)} -> "
            f"{format_hms(next_start_idx / sample_rate)}"
        )
        print(cut_msg)
        f.write(cut_msg + "\n")
        start_idx = next_start_idx
        initial_prompt = last_merged_text[-20:]
        if next_start_idx == end_idx:
            is_speech = False

    end_time = time.time()
    asr_duration = end_time - start_time
    audio_duration = info.duration
    rtf = asr_duration / audio_duration if audio_duration > 0 else 0.0

    print("-" * 30)
    print(f"音频时长 (Audio Duration): {audio_duration:.2f}s")
    print(f"转录耗时 (ASR Duration):   {asr_duration:.2f}s")
    print(f"实时率 (RTF):            {rtf:.4f}")

with condition:
    remainder = shared["tail_samples"]

if remainder > 0:
    remainder_msg = f"[TAIL] 剩余不足1秒音频未触发识别: {remainder / sample_rate:.2f}s"
    print(remainder_msg)
    f.write(remainder_msg + "\n")

script_end_time = time.time()
script_duration = script_end_time - script_start_time
script_rtf = script_duration / total_audio_duration if total_audio_duration > 0 else 0.0

print("=" * 30)
print(f"脚本总音频时长 (Total Audio Duration): {format_hms(total_audio_duration)}")
print(f"脚本总耗时 (Script Duration):        {format_hms(script_duration)}")
print(f"脚本整体实时率 (Overall RTF):        {script_rtf:.4f}")
f.write("=" * 30 + "\n")
f.write(f"脚本总音频时长 (Total Audio Duration): {format_hms(total_audio_duration)}\n")
f.write(f"脚本总耗时 (Script Duration):        {format_hms(script_duration)}\n")
f.write(f"脚本整体实时率 (Overall RTF):        {script_rtf:.4f}\n")

f.close()
reader_thread.join()
