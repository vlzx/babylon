import argparse
from pathlib import Path

import librosa
import soundfile as sf


def mp3_to_wav_16k_16bit_mono(input_path: str, output_path: str) -> None:
    # librosa.load with mono=True will down-mix to mono and resample to target sr.
    audio, _ = librosa.load(input_path, sr=16000, mono=True)
    sf.write(output_path, audio, 16000, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert mp3 to wav (16kHz, 16-bit, mono)."
    )
    parser.add_argument("input_mp3", help="Input mp3 file path")
    parser.add_argument(
        "-o",
        "--output",
        dest="output_wav",
        help="Output wav file path (default: same name as input with .wav)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_mp3)
    output_path = Path(args.output_wav) if args.output_wav else input_path.with_suffix(".wav")

    mp3_to_wav_16k_16bit_mono(str(input_path), str(output_path))
    print(f"Converted: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()

