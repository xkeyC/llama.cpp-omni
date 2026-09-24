#!/usr/bin/env python3
"""Builds a token2wav prompt bundle (voice clone) from a reference recording.

Writes the three files ``Token2Mel::load_prompt_bundle_dir`` reads:

    spk_f32.bin             CAM++ speaker embedding, 192 x f32
    prompt_tokens_i32.bin   S3 speech tokens (25 Hz) + 3 silence tokens, i32
    prompt_mel_btc_f32.bin  80-bin log mel (24 kHz, 50 Hz), T x 80 f32

It reproduces ``stepaudio2.Token2wav._prepare_prompt`` / ``set_stream_cache``
with numpy + onnxruntime only (no torch), using the two ONNX models shipped in
the MiniCPM-o 4.5 repo under ``assets/token2wav/``: ``campplus.onnx`` and
``speech_tokenizer_v2_25hz.onnx``.

    python make_voice_bundle.py ref.wav out_dir --models DIR
"""

import argparse
import sys
from pathlib import Path

import numpy as np

SILENCE_TOKEN = 4218
PRE_LOOKAHEAD = 3
MAX_SECONDS = 30.0  # the speech tokenizer takes at most 30 s


def load(path: Path, sr: int) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def whisper_log_mel(audio: np.ndarray) -> np.ndarray:
    """s3tokenizer.log_mel_spectrogram: 128 bins, 16 kHz, 10 ms hop."""
    import librosa

    stft = librosa.stft(audio, n_fft=400, hop_length=160, window="hann", center=True, pad_mode="reflect")
    power = np.abs(stft[:, :-1]) ** 2
    filters = librosa.filters.mel(sr=16000, n_fft=400, n_mels=128)
    log_spec = np.log10(np.maximum(filters @ power, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)  # [128, T]


def speech_tokens(audio: np.ndarray, model: Path) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    feats = whisper_log_mel(audio)[None]  # [1, 128, T]
    inputs = sess.get_inputs()
    out = sess.run(
        None,
        {inputs[0].name: feats, inputs[1].name: np.array([feats.shape[2]], dtype=np.int32)},
    )[0]
    return np.asarray(out).reshape(-1).astype(np.int32)


def speaker_embedding(audio: np.ndarray, model: Path) -> np.ndarray:
    """CAM++ on Kaldi fbank (80 bins, no dither), mean-normalised."""
    import kaldi_native_fbank as knf
    import onnxruntime as ort

    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(16000, audio.tolist())
    fbank.input_finished()
    feat = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)]).astype(np.float32)
    feat -= feat.mean(axis=0, keepdims=True)
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    emb = sess.run(None, {sess.get_inputs()[0].name: feat[None]})[0]
    return np.asarray(emb, np.float32).reshape(-1)


def prompt_mel(audio_24k: np.ndarray) -> np.ndarray:
    """CosyVoice2 mel: n_fft 1920, hop 480, 80 bins up to 8 kHz, log."""
    import librosa

    n_fft, hop = 1920, 480
    pad = (n_fft - hop) // 2
    y = np.pad(audio_24k, (pad, pad), mode="reflect")
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft, window="hann", center=False)
    mag = np.sqrt(np.abs(stft) ** 2 + 1e-9)
    basis = librosa.filters.mel(sr=24000, n_fft=n_fft, n_mels=80, fmin=0, fmax=8000)
    return np.log(np.maximum(basis @ mag, 1e-5)).T.astype(np.float32)  # [T, 80]


def build(wav: Path, out: Path, models: Path) -> None:
    audio16 = load(wav, 16000)[: int(MAX_SECONDS * 16000)]
    audio24 = load(wav, 24000)[: int(MAX_SECONDS * 24000)]
    if len(audio16) < 16000:
        raise ValueError("reference audio shorter than 1 s")

    tokens = speech_tokens(audio16, models / "speech_tokenizer_v2_25hz.onnx")
    spk = speaker_embedding(audio16, models / "campplus.onnx")
    mel = prompt_mel(audio24)
    # Mel frames are twice the token rate; pad (replicating the last frame)
    # or cut to exactly that, as set_stream_cache does.
    n_mel = len(tokens) * 2
    if len(mel) < n_mel:
        mel = np.concatenate([mel, np.repeat(mel[-1:], n_mel - len(mel), axis=0)])
    mel = mel[:n_mel]
    tokens = np.concatenate([tokens, np.full(PRE_LOOKAHEAD, SILENCE_TOKEN, np.int32)])
    if spk.shape != (192,):
        raise ValueError(f"unexpected speaker embedding shape {spk.shape}")

    out.mkdir(parents=True, exist_ok=True)
    # Written to temp names then renamed, so a reader never sees half a bundle.
    for name, arr in (
        ("spk_f32.bin", spk),
        ("prompt_tokens_i32.bin", tokens),
        ("prompt_mel_btc_f32.bin", np.ascontiguousarray(mel)),
    ):
        tmp = out / (name + ".tmp")
        tmp.write_bytes(arr.tobytes())
        tmp.replace(out / name)
    print(f"bundle: {len(tokens)} tokens, {len(mel)} mel frames, {len(audio16) / 16000:.1f}s -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("wav", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument(
        "--models",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
        help="directory holding campplus.onnx and speech_tokenizer_v2_25hz.onnx",
    )
    args = ap.parse_args()
    try:
        build(args.wav, args.out, args.models)
    except Exception as exc:  # noqa: BLE001 - reported to the calling server
        print(f"make_voice_bundle failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
