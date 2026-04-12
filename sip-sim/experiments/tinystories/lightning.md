# Lightning AI Cheat Sheet

## Setup (one-time)

```bash
# Install CLI
uv tool install lightning-sdk

# Login (opens browser, saves to ~/.lightning/credentials.json)
lightning login
```

## Key concepts

- **Teamspace**: project container. Ours: `amartey/compact-ml-suite-project`
- **Studio**: a cloud dev machine (CPU or GPU). Persistent filesystem.
- **Job**: a fire-and-forget batch run on a machine type.

## List studios

```bash
lightning list studios --teamspace amartey/compact-ml-suite-project
```

The CLI truncates names. To get full names, use the Python SDK:

```python
from lightning_sdk import Teamspace
ts = Teamspace(name='compact-ml-suite-project', user='amartey')
for s in ts.studios:
    print(s)
```

## Upload files to a studio

```bash
# Single file
lightning cp local.py "lit://amartey/compact-ml-suite-project/studios/STUDIO_NAME/remote/path.py"

# Directory (recursive)
lightning cp -r ./local_dir/ "lit://amartey/compact-ml-suite-project/studios/STUDIO_NAME/remote_dir/"
```

**Gotcha**: `lightning cp` sometimes doesn't overwrite existing files. Use the SDK as fallback:

```python
from lightning_sdk import Studio
s = Studio(name='model-training-devbox', teamspace='compact-ml-suite-project', user='amartey')
s.upload_file("local.py", "remote/path.py")
```

Or write directly via base64:
```python
import base64
with open("local.py") as f:
    encoded = base64.b64encode(f.read().encode()).decode()
s.run(f"echo '{encoded}' | base64 -d > ~/remote/path.py")
```

## Run commands on a studio (Python SDK)

```python
from lightning_sdk import Studio

s = Studio(name='model-training-devbox', teamspace='compact-ml-suite-project', user='amartey')
print(s.status)   # 'Running', 'Stopped', etc.
print(s.machine)  # 'A100', 'CPU', etc.

# Run command (blocking, returns output)
output = s.run("nvidia-smi")
print(output)

# Run in background (non-blocking)
s.run_and_detach("nohup python train.py > train.log 2>&1 &")

# Download files
s.download_file("remote/path.pt", "local/path.pt")
```

## Start/stop a studio

```python
s.start(machine='A100')  # or 'T4', 'L4', 'CPU', etc.
s.stop()
```

## Run a standalone job (no studio needed)

```bash
lightning run job \
    --name "my-training-job" \
    --machine A100 \
    --teamspace amartey/compact-ml-suite-project \
    --image "nvcr.io/nvidia/pytorch:24.01-py3" \
    --command "pip install tiktoken datasets && python train.py"
```

## Available GPU machines

| Machine     | GPU          | VRAM  |
|-------------|-------------|-------|
| T4          | Tesla T4    | 16GB  |
| L4          | L4          | 24GB  |
| L40S        | L40S        | 48GB  |
| A100        | A100 SXM4   | 80GB  |
| H100        | H100        | 80GB  |
| H200        | H200        | 141GB |

Multi-GPU: append `_X_2`, `_X_4`, `_X_8` (e.g., `A100_X_4`).

## Our training workflow

```bash
# 1. Upload code
lightning cp -r learn/tinystories/ "lit://amartey/compact-ml-suite-project/studios/model-training-devbox/tinystories/"

# 2. SSH-like: run setup
s.run("cd ~/tinystories && pip install tiktoken datasets")
s.run("cd ~/tinystories && python -c 'from data import TinyStoriesDataset; TinyStoriesDataset(\"train\")'")

# 3. Launch training (detached)
s.run_and_detach("cd ~/tinystories && nohup python train.py --name baseline --use-field false > baseline.log 2>&1 &")
s.run_and_detach("cd ~/tinystories && nohup python train.py --name social --use-field true > social.log 2>&1 &")

# 4. Monitor
s.run("tail -5 ~/tinystories/baseline.log")
s.run("nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader")

# 5. Download checkpoints
s.download_file("tinystories/checkpoints/baseline_latest.pt", "checkpoints/baseline_latest.pt")

# 6. Local inference (Mac M4)
python generate.py --checkpoint checkpoints/baseline_latest.pt --interactive
```

## Gotchas

1. **File upload caching**: `lightning cp` may silently skip if file exists. Use SDK `upload_file` or base64 trick.
2. **torch.compile on A100**: first step is slow (30-60s compilation). After that, ~65K tok/s.
3. **Running multiple models**: A100 has 80GB. Two 30M models fit (35GB each). Three models need smaller batch sizes + gradient accumulation.
4. **Studio filesystem persists** when stopped. GPU is released, but data stays. Restart with `s.start(machine='A100')`.
5. **Environment variable**: `LIGHTENING_AI_API_KEY` in .env (note: Lightning SDK uses `~/.lightning/credentials.json` after `lightning login`).
