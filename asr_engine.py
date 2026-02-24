import time
from typing import Any

import numpy as np
from faster_whisper import WhisperModel

from utils.string import get_changed_part


class ASREngine:
    def __init__(
        self,
        asr_model: WhisperModel | None = None,
        vad_model: WhisperModel | None = None,
        *,
        asr_model_path: str | None = None,
        vad_model_path: str | None = None,
        device: str = "cuda",
        sample_rate: int = 16000,
        min_process_sec: float = 1.0,
        max_sentence_sec: float = 20.0,
        vad_no_speech_threshold: float = 0.8,
        stable_repeat_threshold: int = 3,
        min_cut_sec: float = 1.0,
        prompt_tail_chars: int = 20,
        hallucination_blacklist: list[str] | None = None,
    ) -> None:
        if asr_model is None:
            if not asr_model_path:
                raise ValueError("asr_model or asr_model_path is required")
            asr_model = WhisperModel(asr_model_path, device=device)
        if vad_model is None:
            if not vad_model_path:
                raise ValueError("vad_model or vad_model_path is required")
            vad_model = WhisperModel(vad_model_path, device=device)

        self.asr_model = asr_model
        self.vad_model = vad_model
        self.sample_rate = sample_rate
        self.one_second_samples = int(sample_rate * min_process_sec)
        self.max_sentence_sec = max_sentence_sec
        self.vad_no_speech_threshold = vad_no_speech_threshold
        self.stable_repeat_threshold = stable_repeat_threshold
        self.min_cut_sec = min_cut_sec
        self.prompt_tail_chars = prompt_tail_chars
        self.hallucination_blacklist = hallucination_blacklist or []

        self.reset()

    def reset(self) -> None:
        self.audio_cache = np.empty((0,), dtype=np.float32)
        self.start_idx = 0
        self.end_idx = 0

        self.last_valid_text = ""
        self.last_merged_text = ""
        self.same_merged_count = 0
        self.initial_prompt = ""
        self.is_speech = False

    def push_chunk(self, chunk: np.ndarray) -> dict[str, Any]:
        try:
            return self._push_chunk(chunk)
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "time": {
                    "window_start_sec": self.start_idx / self.sample_rate,
                    "window_end_sec": self.end_idx / self.sample_rate,
                    "cut_from_sec": None,
                    "cut_to_sec": None,
                },
                "text": {
                    "merged": "",
                    "delta": "",
                    "last_valid": self.last_valid_text,
                    "prompt": self.initial_prompt,
                },
                "vad": {"ran": False, "no_speech_probs": []},
                "metrics": {
                    "asr_duration_sec": 0.0,
                    "audio_duration_sec": 0.0,
                    "rtf": 0.0,
                },
                "debug": {"reasons": ["error"]},
            }

    def _push_chunk(self, chunk: np.ndarray) -> dict[str, Any]:
        if chunk is None or len(chunk) == 0:
            return self._build_result(
                status="buffering",
                merged_text="",
                delta_text="",
                no_speech_probs=[],
                reasons=["empty_chunk"],
                asr_duration_sec=0.0,
                audio_duration_sec=0.0,
                cut_from_sec=None,
                cut_to_sec=None,
                window_start_idx=self.start_idx,
                window_end_idx=self.end_idx,
                vad_ran=False,
            )

        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self.audio_cache = np.concatenate([self.audio_cache, chunk], axis=0)
        self.end_idx += len(chunk)
        window_start_idx = self.start_idx
        window_end_idx = self.end_idx

        active_samples = self.end_idx - self.start_idx
        if active_samples < self.one_second_samples:
            return self._build_result(
                status="buffering",
                merged_text="",
                delta_text="",
                no_speech_probs=[],
                reasons=["insufficient_window"],
                asr_duration_sec=0.0,
                audio_duration_sec=active_samples / self.sample_rate,
                cut_from_sec=None,
                cut_to_sec=None,
                window_start_idx=window_start_idx,
                window_end_idx=window_end_idx,
                vad_ran=False,
            )

        buffer = self.audio_cache[self.start_idx:self.end_idx].copy()

        no_speech_probs: list[float] = []
        reasons: list[str] = []
        vad_ran = False

        if not self.is_speech:
            vad_ran = True
            detect_segments, _ = self.vad_model.transcribe(
                buffer,
                language="ja",
                without_timestamps=True,
                condition_on_previous_text=False,
            )
            detect_results = list(detect_segments)
            no_speech_probs = [segment.no_speech_prob for segment in detect_results]

            if len(detect_results) == 0 or all(
                prob > self.vad_no_speech_threshold for prob in no_speech_probs
            ):
                self.last_merged_text = ""
                self.same_merged_count = 0
                self.start_idx = self.end_idx
                self.initial_prompt = ""
                self.is_speech = False
                reasons.append("vad_no_speech")
                return self._build_result(
                    status="no_speech",
                    merged_text="",
                    delta_text="",
                    no_speech_probs=no_speech_probs,
                    reasons=reasons,
                    asr_duration_sec=0.0,
                    audio_duration_sec=len(buffer) / self.sample_rate,
                    cut_from_sec=None,
                    cut_to_sec=self.start_idx / self.sample_rate,
                    window_start_idx=window_start_idx,
                    window_end_idx=window_end_idx,
                    vad_ran=vad_ran,
                )
            self.is_speech = True

        start_time = time.time()
        asr_segments, info = self.asr_model.transcribe(
            buffer,
            language="ja",
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
        )

        prev_end = 0.0
        next_start_idx = self.start_idx
        merged_text_parts: list[str] = []
        for segment in list(asr_segments):
            merged_text_parts.append(segment.text)
            if segment.start != prev_end:
                cut_sec = (segment.start + prev_end) / 2
                candidate_start_idx = self.start_idx + int(cut_sec * self.sample_rate)
                next_start_idx = max(next_start_idx, candidate_start_idx)
                reasons.append("segment_gap")

            segment_duration = segment.end - segment.start
            if segment_duration > self.max_sentence_sec:
                next_start_idx = max(next_start_idx, self.end_idx)
                reasons.append("long_segment")
            prev_end = segment.end

        merged_text = "".join(merged_text_parts).strip()
        delta_text = get_changed_part(
            self.last_valid_text.replace(" ", "").replace("\n", ""),
            merged_text.replace(" ", "").replace("\n", ""),
        )

        for item in self.hallucination_blacklist:
            if item in delta_text:
                merged_text = merged_text.replace(item, "")
                delta_text = delta_text.replace(item, "")
                reasons.append("hallucination_removed")

        if not merged_text:
            next_start_idx = self.end_idx
            reasons.append("empty_after_filter")

        if merged_text:
            self.last_valid_text = merged_text
            if merged_text == self.last_merged_text:
                self.same_merged_count += 1
            else:
                self.last_merged_text = merged_text
                self.same_merged_count = 1
        else:
            self.last_merged_text = ""
            self.same_merged_count = 0

        if self.same_merged_count >= self.stable_repeat_threshold and prev_end > 0:
            next_start_idx = max(next_start_idx, self.end_idx)
            reasons.append(f"stable_x{self.same_merged_count}")

        old_start_idx = self.start_idx
        cut_from_sec = None
        cut_to_sec = None
        status = "partial"
        if (next_start_idx - self.start_idx) / self.sample_rate > self.min_cut_sec:
            self.start_idx = max(next_start_idx, self.start_idx)
            self.initial_prompt = self.last_merged_text[-self.prompt_tail_chars :]
            if self.start_idx == self.end_idx:
                self.is_speech = False
            cut_from_sec = old_start_idx / self.sample_rate
            cut_to_sec = self.start_idx / self.sample_rate
            status = "committed"

        asr_duration_sec = time.time() - start_time
        audio_duration_sec = info.duration if info.duration > 0 else 0.0

        return self._build_result(
            status=status,
            merged_text=merged_text,
            delta_text=delta_text,
            no_speech_probs=no_speech_probs,
            reasons=reasons,
            asr_duration_sec=asr_duration_sec,
            audio_duration_sec=audio_duration_sec,
            cut_from_sec=cut_from_sec,
            cut_to_sec=cut_to_sec,
            window_start_idx=window_start_idx,
            window_end_idx=window_end_idx,
            vad_ran=vad_ran,
        )

    def finalize(self) -> dict[str, Any]:
        active_samples = self.end_idx - self.start_idx
        tail_samples = active_samples if active_samples < self.one_second_samples else 0
        return {
            "status": "tail",
            "tail_samples": tail_samples,
            "tail_duration_sec": tail_samples / self.sample_rate,
            "time": {
                "window_start_sec": self.start_idx / self.sample_rate,
                "window_end_sec": self.end_idx / self.sample_rate,
                "cut_from_sec": None,
                "cut_to_sec": None,
            },
            "text": {
                "merged": "",
                "delta": "",
                "last_valid": self.last_valid_text,
                "prompt": self.initial_prompt,
            },
            "vad": {"ran": False, "no_speech_probs": []},
            "metrics": {
                "asr_duration_sec": 0.0,
                "audio_duration_sec": 0.0,
                "rtf": 0.0,
            },
            "debug": {"reasons": ["finalize"]},
        }

    def _build_result(
        self,
        *,
        status: str,
        merged_text: str,
        delta_text: str,
        no_speech_probs: list[float],
        reasons: list[str],
        asr_duration_sec: float,
        audio_duration_sec: float,
        cut_from_sec: float | None,
        cut_to_sec: float | None,
        window_start_idx: int,
        window_end_idx: int,
        vad_ran: bool,
    ) -> dict[str, Any]:
        rtf = asr_duration_sec / audio_duration_sec if audio_duration_sec > 0 else 0.0
        return {
            "status": status,
            "time": {
                "window_start_sec": window_start_idx / self.sample_rate,
                "window_end_sec": window_end_idx / self.sample_rate,
                "cut_from_sec": cut_from_sec,
                "cut_to_sec": cut_to_sec,
            },
            "text": {
                "merged": merged_text,
                "delta": delta_text,
                "last_valid": self.last_valid_text,
                "prompt": self.initial_prompt,
            },
            "vad": {
                "ran": vad_ran,
                "no_speech_probs": no_speech_probs,
            },
            "metrics": {
                "asr_duration_sec": asr_duration_sec,
                "audio_duration_sec": audio_duration_sec,
                "rtf": rtf,
            },
            "buffer": {
                "cache_samples": len(self.audio_cache),
                "active_samples": self.end_idx - self.start_idx,
                "tail_samples": max(0, self.end_idx - self.start_idx),
            },
            "debug": {
                "reasons": reasons,
                "same_merged_count": self.same_merged_count,
                "is_speech": self.is_speech,
            },
        }
