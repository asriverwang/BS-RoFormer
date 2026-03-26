# BS-RoFormer & Mel-RoFormer — Music Source Separation

PyTorch implementations of two transformer-based music source separation models:

- **Band-Split RoPE Transformer (BS-RoFormer)**
  > *"Music Source Separation with Band-split RoPE Transformer"*
  > https://arxiv.org/abs/2309.02612

- **Mel-band RoFormer (MelRoFormer)**
  > *"Mel-band RoFormer for Music Source Separation"*
  > https://arxiv.org/abs/2310.01809

Both models take a stereo mixture waveform and output one or more separated source waveforms (e.g. vocals, drums, bass).

---

## How it works

Both architectures share the same high-level pipeline:

```
Waveform -> STFT -> Band-Split -> [Time Attn <-> Band Attn] × depth -> Mask Estimation -> iSTFT -> Waveform
```

The key difference is **how the frequency axis is split into bands**.

### BS-RoFormer — linear subspectrum bands

The frequency axis is divided into a fixed set of **subspectra** (coarse groups), each further split into equal-width subbands. The default splits 1024 frequency bins across 6 subspectra with 62 total bands:

| Subspectrum bins | Bands |
|------------------|-------|
| 0 – 47           | 24    |
| 48 – 95          | 12    |
| 96 – 191         | 8     |
| 192 – 383        | 8     |
| 384 – 767        | 8     |
| 768 – 1023       | 2     |

### MelRoFormer — mel-scale bands

Instead of hand-crafted subspectra, `librosa`'s mel filterbank is used to derive band boundaries automatically. Each mel band maps to a contiguous range of FFT bins, giving perceptually motivated frequency resolution (finer at low frequencies, coarser at high).

### Shared transformer core (`roformer.py`)

Both models use the same `RoFormerBlock`: a depth-stacked pair of pre-norm transformers that alternate attention axes:

1. **Time transformer** — each band attends across time frames (`(b·k, t, d)`)
2. **Band transformer** — each time frame attends across bands (`(b·t, k, d)`)

Both transformers use **Rotary Position Embeddings (RoPE)** via `rotary_embedding_torch`, with separate embedding instances for the time and band axes.

Normalization is **RMSNorm** throughout; activations use **GELU** in feed-forward layers and **GLU** in mask estimators.

### Complex masking

Each model estimates a complex-valued mask per band, scatters it into the full frequency grid, then applies it multiplicatively to the STFT of the mixture:

```
separated = (real * mask_real - imag * mask_imag) + j(real * mask_imag + imag * mask_real)
```

The separated spectrogram is converted back to audio with iSTFT.

---

## File overview

| File | Description |
|------|-------------|
| `roformer.py` | Shared building blocks: `RMSNorm`, `FeedForward`, `MultiHeadSelfAttention` (with RoPE), `Transformer`, `RoFormerBlock` |
| `bsroformer.py` | `BSRoFormer` model with linear subband splitting (`BandSplit`, `MaskEstimation`) |
| `melroformer.py` | `MelRoFormer` model with mel-scale band splitting (`MelBandSplit`, `MelBandMask`) |
| `separator.py` | Inference script: overlapping-segment separation, batched forward pass, overlap-add reassembly, FFmpeg audio I/O |

---

## Requirements

```
torch
torchaudio
rotary-embedding-torch
librosa          # MelRoFormer only
numpy
soundfile        # separator.py
tqdm             # separator.py
ffmpeg           # separator.py (system binary)
```

---

## Running inference

```bash
python separator.py \
    --backbone bs-roformer \
    --model_ckpt_path ckpt/bs_roformer.pt \
    --input_audio_folder test/ \
    --output_audio_folder result/ \
    --batch_size 8 \
    --hop_perc 0.5
```

**Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--backbone` | required | `bs-roformer` or `mel-roformer` |
| `--model_ckpt_path` | required | Path to `.pt` checkpoint |
| `--input_audio_folder` | `test/` | Folder with input audio (mp3/wav/flac/m4a/ogg, searched recursively) |
| `--output_audio_folder` | `test_result/` | Output folder; writes `target/` and `residual/` subdirs |
| `--batch_size` | `8` | Segments processed per forward pass |
| `--hop_perc` | `0.5` | Overlap between segments as a fraction of segment length |
| `--output_mp3` | off | Encode output as 320 kbps MP3 (requires ffmpeg) |
| `--compile` | off | Enable `torch.compile` on the transformer stack (PyTorch 2+) |

The separator processes audio in **8-second overlapping chunks** at 44.1 kHz stereo, using overlap-add to reconstruct a seamless output. The residual (mixture minus separated target) is also saved.

**Checkpoint format** — a `.pt` file with two keys:

```python
{
    "config": { ... },        # kwargs passed to BSRoFormer / MelRoFormer.__init__
    "state_dict": { ... },    # model weights
}
```

---

## Instantiating models directly

```python
from bsroformer import BSRoFormer

model = BSRoFormer(
    input_channels=2,
    output_channels=2,
    depth=12,
    num_feature=384,
    window_size=2048,
    hop_size=441,
)

import torch
mix = torch.randn(1, 2, 44100)   # (batch, channels, samples)
separated = model(mix)            # same shape as mix
```

```python
from melroformer import MelRoFormer

model = MelRoFormer(
    input_channels=2,
    output_channels=2,
    depth=12,
    num_feature=384,
    window_size=2048,
    hop_size=441,
    mel_bands=64,
)
import torch
# Length must be a multiple of hop_size; iSTFT returns floor(L/hop)*hop samples
mix = torch.randn(1, 2, 44100)   # 44100 = 100 * hop_size(441)
separated = model(mix)           # shape: (1, 2, 44100)
```

---

## Running the built-in tests

Each model file includes self-tests that run on CPU or GPU:

```bash
python bsroformer.py
python melroformer.py
```

Tests cover shape correctness, scatter locality of the mask estimator, full forward pass, and gradient flow.

---

## References

```bibtex
@inproceedings{lu2024music,
  title={Music Source Separation with Band-Split RoPE Transformer},
  author={Lu, Wei-Tsung and Wang, Ju-Chiang and Kong, Qiuqiang and Hung, Yun-Ning},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={481--485},
  year={2024},
  organization={IEEE}
}

@article{wang2023mel,
  title   = {Mel-band RoFormer for Music Source Separation},
  author  = {Wang, Ju-Chiang and Lu, Wei-Tsung and Won, Minz},
  journal = {arXiv preprint arXiv:2310.01809},
  year    = {2023}
}
```
