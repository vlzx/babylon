import os
import sys
import time

import numpy as np
import soundfile as sf
from rich.live import Live
from rich.text import Text

sys.path.append(os.getcwd())
from asr_engine import ASREngine
from utils.audio import stream_wav_realtime
from utils.chrono import format_hms
from utils.logging import setup_logger


logger = setup_logger(level="WARNING")

hallucination_blacklist = [
    "ご視聴ありがとうございました",
    "チャンネル登録をお願いいたします",
]

asr_model_path = "local_models/faster-whisper-large-v3-turbo-ct2"
vad_model_path = "local_models/faster-whisper-tiny"
max_sentence_sec = 20.0

wav_path = "data/nekoyashiki_utawaku_test.wav"
audio_info = sf.info(wav_path)
sample_rate = audio_info.samplerate
one_second_samples = int(sample_rate * 1.0)
total_audio_duration = audio_info.duration

logger.info(f"采样率: {sample_rate} Hz")
logger.info(f"总时长: {total_audio_duration:.2f}s")
logger.info(f"每轮累计时长: 1.00s ({one_second_samples} samples)")

engine = ASREngine(
    asr_model_path=asr_model_path,
    vad_model_path=vad_model_path,
    device="cuda",
    sample_rate=sample_rate,
    min_process_sec=1.0,
    max_sentence_sec=max_sentence_sec,
    hallucination_blacklist=hallucination_blacklist,
)


def handle_result(result: dict, live: Live, output_file) -> None:
    if result["status"] == "error":
        raise RuntimeError(f"ASREngine failed: {result.get('error', 'unknown error')}")

    window_start = result["time"]["window_start_sec"]
    window_end = result["time"]["window_end_sec"]
    timestamp = f"[{format_hms(window_start)} -> {format_hms(window_end)}]"

    merged_text = result["text"]["merged"].replace("\n", "")
    delta_text = result["text"]["delta"]
    last_valid = result["text"]["last_valid"]
    prompt = result["text"]["prompt"]

    if result["status"] != "buffering":
        output_file.write(timestamp + "\n")

    if result["vad"]["ran"]:
        vad_msg = f"[VAD] no_speech_probs: {result['vad']['no_speech_probs']}"
        logger.debug(vad_msg)
        output_file.write(vad_msg + "\n")

    if result["status"] in {"partial", "committed"}:
        asr_msg = f"[LAST] {last_valid}\n[INIT_PROMPT] {prompt}\n[ASR] {result['text']['merged']}\n[DELTA] {delta_text}"
        logger.info(asr_msg)
        output_file.write(asr_msg + "\n")

        reasons = result["debug"].get("reasons", [])
        if reasons:
            output_file.write(f"[REASONS] {reasons}\n")

        cut_from_sec = result["time"]["cut_from_sec"]
        cut_to_sec = result["time"]["cut_to_sec"]
        if cut_from_sec is not None and cut_to_sec is not None:
            cut_msg = f"cut start_idx: {format_hms(cut_from_sec)} -> {format_hms(cut_to_sec)}"
            logger.debug(cut_msg)
            output_file.write(cut_msg + "\n")
            live.console.print(f"[white] {timestamp} {merged_text}[/]")

        metrics = result["metrics"]
        logger.debug("-" * 30)
        logger.debug(f"音频时长 (Audio Duration): {metrics['audio_duration_sec']:.2f}s")
        logger.debug(f"转录耗时 (ASR Duration):   {metrics['asr_duration_sec']:.2f}s")
        logger.debug(f"实时率 (RTF):            {metrics['rtf']:.4f}")

    live.update(
        Text().assemble(
            (f"{timestamp} {merged_text}", "bold cyan underline"),
            ("▋", "blink bold white"),
        )
    )


script_start_time = time.time()
transcript_path = os.path.join("data/transcript", "nekoyashiki_utawaku_stream.txt")

with open(transcript_path, "w+", encoding="utf-8") as output_file:
    waiting_text = Text("Listening...", style="dim italic blue")
    pending_audio = np.empty((0,), dtype=np.float32)

    with Live(waiting_text, refresh_per_second=10) as live:
        for audio_chunk in stream_wav_realtime(wav_path, frame_duration_ms=20, dtype="float32"):
            pending_audio = np.concatenate([pending_audio, audio_chunk], axis=0)

            while len(pending_audio) >= one_second_samples:
                one_sec_chunk = pending_audio[:one_second_samples].copy()
                pending_audio = pending_audio[one_second_samples:]
                result = engine.push_chunk(one_sec_chunk)
                handle_result(result, live, output_file)

        if len(pending_audio) > 0:
            remainder_result = engine.push_chunk(pending_audio.copy())
            handle_result(remainder_result, live, output_file)

    finalize_result = engine.finalize()
    if finalize_result["tail_samples"] > 0:
        tail_duration = finalize_result["tail_duration_sec"]
        remainder_msg = f"[TAIL] 剩余不足1秒音频未触发识别: {tail_duration:.2f}s"
        logger.debug(remainder_msg)
        output_file.write(remainder_msg + "\n")

    script_end_time = time.time()
    script_duration = script_end_time - script_start_time
    script_rtf = script_duration / total_audio_duration if total_audio_duration > 0 else 0.0

    logger.info("=" * 30)
    logger.info(f"脚本总音频时长 (Total Audio Duration): {format_hms(total_audio_duration)}")
    logger.info(f"脚本总耗时 (Script Duration):        {format_hms(script_duration)}")
    logger.info(f"脚本整体实时率 (Overall RTF):        {script_rtf:.4f}")

    output_file.write("=" * 30 + "\n")
    output_file.write(f"脚本总音频时长 (Total Audio Duration): {format_hms(total_audio_duration)}\n")
    output_file.write(f"脚本总耗时 (Script Duration):        {format_hms(script_duration)}\n")
    output_file.write(f"脚本整体实时率 (Overall RTF):        {script_rtf:.4f}\n")
