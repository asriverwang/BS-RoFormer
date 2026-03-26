"""Single-device vocal separator supporting BSRoFormer and MelRoFormer backbones.

Usage:
    python separator.py \
        --backbone bs-roformer \
        --model_ckpt_path mss_ckpt/bs_roformer.pt \
        --input_audio_folder test/ \
        --output_audio_folder result/ \
        --batch_size 8 \
        --hop_perc 0.5
"""
import argparse
import io
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

SAMPLE_RATE = 44100
CHUNK_LEN_S = 8
SEGMENT_SAMPLES = int(SAMPLE_RATE * CHUNK_LEN_S)

def find_audio_files(folder):
    paths = []
    for ext in ("mp3", "wav", "flac", "m4a", "ogg"):
        paths.extend(sorted(Path(folder).rglob(f"*.{ext}")))
    return paths


def load_audio(path, sr=SAMPLE_RATE):
    """Load audio as (channels, samples) at target sample rate via ffmpeg."""
    cmd = f'ffmpeg -i "{path}" -ac 2 -ar {sr} -f wav -'
    p = subprocess.Popen(
        cmd, shell=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    out, _ = p.communicate()
    audio, _ = sf.read(io.BytesIO(out))
    return audio.T  # (samples, channels) → (channels, samples)


def load_model(backbone, pt_path, device):
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=True)
    if backbone == "bs-roformer":
        from bsroformer import BSRoFormer
        model = BSRoFormer(**checkpoint["config"])
    elif backbone == "mel-roformer":
        from melroformer import MelRoFormer
        model = MelRoFormer(**checkpoint["config"])
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model.to(device)


class Separator:
    def __init__(self, model, segment_samples, hop_samples, batch_size, device):
        self.model = model.eval()
        self.segment_samples = segment_samples
        self.hop_samples = hop_samples
        self.batch_size = batch_size
        self.device = device
        self.overlap = segment_samples - hop_samples

    def separate(self, audio):
        """audio: (channels, samples) -> (channels, samples)."""
        orig_len = audio.shape[-1]
        segments = self._segment(audio)
        separated = self._forward_batched(segments)
        result = self._reassemble(separated)
        return result[:, :orig_len]

    def _segment(self, audio):
        """Pad audio and split into overlapping segments."""
        channels, samples = audio.shape
        buf = self.overlap // 2

        n_hops = int(np.ceil(samples / self.hop_samples))
        padded_len = n_hops * self.hop_samples + 2 * buf
        padded = np.zeros((channels, padded_len))
        padded[:, buf : buf + samples] = audio

        segments = []
        for start in range(0, padded_len - self.segment_samples + 1, self.hop_samples):
            segments.append(padded[:, start : start + self.segment_samples])
        return np.array(segments)

    def _reassemble(self, segments):
        """Reassemble segments using overlap-add with averaging."""
        n_segs, n_ch, seg_len = segments.shape
        buf = self.overlap // 2

        if n_segs == 1:
            return segments[0][:, buf : seg_len - buf] if buf > 0 else segments[0]

        total_len = (n_segs - 1) * self.hop_samples + seg_len
        out = np.zeros((n_ch, total_len))
        cnt = np.zeros_like(out)

        for i, seg in enumerate(segments):
            start = i * self.hop_samples
            out[:, start : start + seg_len] += seg
            cnt[:, start : start + seg_len] += 1

        out /= cnt
        return out[:, buf : total_len - buf] if buf > 0 else out

    @torch.no_grad()
    def _forward_batched(self, segments):
        results = []
        total_chunks = len(segments)
        use_amp = self.device.type == "cuda"
        with tqdm(total=total_chunks, desc="  chunks", unit="chunk", leave=False) as pbar:
            for i in range(0, total_chunks, self.batch_size):
                batch = torch.from_numpy(
                    segments[i : i + self.batch_size].copy()
                ).float().to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    out = self.model(batch)
                results.append(out.cpu().numpy())
                pbar.update(min(self.batch_size, total_chunks - i))
        return np.concatenate(results, axis=0)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Backbone: {args.backbone}")

    hop_samples = int(SEGMENT_SAMPLES * args.hop_perc)
    model = load_model(args.backbone, args.model_ckpt_path, device)
    if args.compile:
        print("Compiling transformer stack (first batch will be slow)...")
        model.compile_for_inference()
    separator = Separator(model, SEGMENT_SAMPLES, hop_samples, args.batch_size, device)

    target_dir = os.path.join(args.output_audio_folder, "target")
    residu_dir = os.path.join(args.output_audio_folder, "residual")
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(residu_dir, exist_ok=True)

    audio_paths = find_audio_files(args.input_audio_folder)
    total_songs = len(audio_paths)
    print(f"Found {total_songs} audio files")

    for song_idx, audio_path in enumerate(audio_paths, 1):
        print(f"Processing {song_idx}/{total_songs}: {audio_path.name}")
        try:
            audio = load_audio(audio_path)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        target = separator.separate(audio)
        residual = np.clip(audio - np.clip(target, -1.0, 1.0), -1.0, 1.0)

        stem = audio_path.stem.replace("_mixture", "")
        target_fn = os.path.join(target_dir, stem)
        residu_fn = os.path.join(residu_dir, stem)

        sf.write(f"{target_fn}.wav", target.T, SAMPLE_RATE)
        sf.write(f"{residu_fn}.wav", residual.T, SAMPLE_RATE)

        if args.output_mp3:
            for fn in (target_fn, residu_fn):
                subprocess.run(
                    ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                     "-i", f"{fn}.wav", "-b:a", "320k", f"{fn}.mp3"],
                    check=True,
                )
                os.remove(f"{fn}.wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BSRoFormer / MelRoFormer vocal separator")
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["bs-roformer", "mel-roformer"],
                        help="Model backbone to use")
    parser.add_argument("--model_ckpt_path", type=str, required=True, help=".pt checkpoint file")
    parser.add_argument("--input_audio_folder", type=str, default="test/")
    parser.add_argument("--output_audio_folder", type=str, default="test_result/")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_mp3", action="store_true")
    parser.add_argument("--hop_perc", type=float, default=0.5)
    parser.add_argument("--compile", action="store_true",
                        help="Compile transformer stack with torch.compile (PyTorch 2+ only)")
    main(parser.parse_args())
