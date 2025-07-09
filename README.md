# 🧍‍♂️ Human Pose as Compositional Tokens

This repository implements a two-stage human pose estimation framework inspired by PCT (Pose as Compositional Tokens). The approach models human pose as a set of discrete tokens and trains a vector quantized autoencoder (VQ-VAE) in Stage I, followed by an image-based pose token classifier in Stage II.

🚧 **This project is currently under training and development.** The tokenizer (Stage I) is implemented and undergoing experiments. Stage II setup has begun.

## 📁 Directory Structure

```
/
├── api/                 # APIs for integrating models (e.g., inference)
│   └── models.py
├── model/               # Core architecture definitions
│   ├── encoder.py       # Encoder, VectorQuantizer, Decoder
│   └── modules.py       # Supporting layers (e.g., MLP Mixer)
├── process/
│   └── data.py          # MPII Dataset wrapper and preprocessing
├── train/
│   ├── stage1.py        # Stage I training script (tokenizer)
│   └── stage2.py        # Stage II training script (classifier)
```

## 🧩 Stage I — Tokenizer Training

✅ **Implemented**

This stage trains a vector-quantized encoder-decoder architecture to model human pose as a set of compositional tokens.

### 🔧 Architecture

- **Encoder**: CompositionalEncoder maps 2D keypoints (16 joints) into latent vectors.
- **Quantizer**: VectorQuantizer discretizes embeddings into a fixed codebook using VQ-VAE logic.
- **Decoder**: Reconstructs the original joint locations from quantized embeddings.

### 📦 Data

- **Dataset**: MPII Human Pose Dataset
- **Format**: .csv keypoints file + images
- **Class**: MPIIDataset from `process/data.py`
- **Transform**: Resizes input images to 256×256 and applies joint masking.

### 🧠 Training Logic (`train/stage1.py`)

- **Loss**: Combines Smooth L1 Loss on visible masked joints with VQ commitment loss.
- **Masking**: Randomly masks 30% of visible joints during training.
- **Optimizer**: AdamW with cosine learning rate schedule and warm-up.
- **Checkpoints**: Saved every 2 epochs under `weights/`.

```python
g_e = encoder(keypoints_masked)
v_q, vq_loss, _ = vq(g_e)
g_h = decoder(v_q)
loss = L1(g_h, keypoints) + β * vq_loss
```

## 🧠 Stage II — Classifier (WIP)

🔍 **Status**

- **File**: `train/stage2.py`
- **Goal**: Use a Swin Transformer backbone to predict token indices from input images.
- **Backbone output is processed by a classification head to estimate each token.**
- **Not fully implemented yet** — training logic and inference scripts are pending.

## 📚 Key Files

| File | Description |
|------|-------------|
| `model/encoder.py` | Encoder, decoder, and vector quantizer modules |
| `process/data.py` | MPII dataset class with preprocessing logic |
| `train/stage1.py` | Full training loop for Stage I tokenizer |
| `api/models.py` | Utility models or wrappers (placeholder) |

## 🛠️ Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision, tqdm, pandas, Pillow

## 🚀 Training Instructions (Stage I)

```bash
python train/stage1.py
```

**Default configs:**
- Batch size: 512
- Epochs: 50
- Learning rate: 0.01
- Image size: 256×256
- Codebook size: 2 × hidden dim

## 🔮 Roadmap

- [x] Implement tokenizer and training loop
- [ ] Integrate MPII preprocessing and joint masking
- [ ] Build Stage II classifier using Swin Transformer
- [ ] Implement inference API
- [ ] Evaluate on MPII/COCO metrics

## 📌 Notes

- Checkpoints are stored under `weights/`
- Codebook size and token count (M) are customizable via encoder params
- Your `stage1.py` is resume-aware and will load `../ckpt/6.pt` if available

Let me know if you'd like this formatted into a proper README.md file or want help extending documentation as Stage II progresses!
