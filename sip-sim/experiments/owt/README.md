# 125M Social Field on OpenWebText

Train a 125M GPT with stigmergic (social field) attention on OpenWebText,
comparing against published GPT-2 Small (124M) benchmarks.

## Architecture

- GPT-2 Small scale: d=768, 12 heads, 12 layers, context 1024
- Baseline: 123.6M params
- Social field: 130.7M params (+5.8% for field deposit/read/modulation)
- Chunk-parallel attention with C=128, additive key modulation, evap=0.05

Model code is shared with `../tinystories/model.py`.

## Setup

```bash
# Dependencies (same as tinystories)
pip install torch tiktoken datasets

# Local test (uses TinyStories, tiny model — verifies everything works)
python train.py --test
```

## Training on Lightning AI (A100)

```bash
# 1. Upload code
lightning cp -r learn/owt/ "lit://amartey/compact-ml-suite-project/studios/model-training-devbox/owt/"
lightning cp learn/tinystories/model.py "lit://amartey/compact-ml-suite-project/studios/model-training-devbox/tinystories/model.py"

# 2. Install deps + tokenise data on the studio (~30 min for OWT)
s.run("cd ~/owt && pip install tiktoken datasets")
s.run("cd ~/owt && python -c 'from data import OpenWebTextDataset; OpenWebTextDataset(\"train\")'")

# 3. Launch training (~12h on A100)
s.run_and_detach("cd ~/owt && nohup python train.py --name owt_social --use-field true > train.log 2>&1 &")

# 4. Monitor
s.run("tail -5 ~/owt/train.log")

# 5. Download checkpoint
s.download_file("owt/checkpoints/owt_social_latest.pt", "checkpoints/owt_social_latest.pt")
```

## Key training details

- **bf16 mixed precision** on CUDA (automatic, disable with `--no-amp`)
- **Gradient checkpointing** on CUDA (saves ~40% memory)
- **Cosine LR** with 2000-step warmup, decays to 10% of peak (not zero)
- **Weight decay** only on 2D params (no decay on biases, norms, embeddings)
- **Effective batch size**: 8 micro-batch * 8 grad_accum = 64
- **torch.compile** for throughput on A100

## Estimated cost

| Stage | Hours | Cost |
|-------|-------|------|
| Pretrain 125M (1 epoch OWT) | ~12h | ~$24 |
| Data tokenisation (one-time) | ~0.5h | ~$1 |
| Buffer | ~3h | ~$6 |
| **Total** | **~15h** | **~$31** |

## Evaluation

Compare against published GPT-2 Small numbers:

| Metric | GPT-2 Small (124M) | Social field 125M |
|--------|-------------------|-------------------|
| WikiText-103 PPL | ~29.4 | ? |
| OpenWebText val PPL | ~18.3 | ? |
| LAMBADA acc | ~45.9% | ? |
| HellaSwag acc | ~31.1% | ? |

## Local inference

```bash
# After downloading checkpoint
python ../tinystories/generate.py --checkpoint checkpoints/owt_social_latest.pt --interactive
```

## Files

- `train.py` — training script (bf16, grad ckpt, 125M defaults)
- `data.py` — OpenWebText dataset loader (train/0.5% val split)
- `../tinystories/model.py` — shared model (GPT + ChunkParallelAttention)
