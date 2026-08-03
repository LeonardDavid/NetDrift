# Lamarr Cluster Setup Guide for MatQuant

This guide walks through deploying and running MatQuant on the Lamarr Slurm cluster (DGX A100 nodes, managed by the CS department at TU Dortmund).

---

## Storage Overview

The cluster has two storage tiers. Choosing the right one for each file type is important for both performance and data safety.

### Persistent Storage: `/home/{user}` (CephFS)
- Shared across **all** nodes and the gateway; survives job termination and node failure
- Backed by redundant Ceph storage (several hundred TB total)
- Slower I/O than local SSD

**Store here:**
- Framework source code and configs (`/home/{user}/matquant/`)
- Permanent dataset copies (`/home/{user}/datasets/`)
- Trained model checkpoints and final results (`/home/{user}/results/`)
- Archived W&B offline runs (`/home/{user}/wandb-archive/`)
- Python virtual environment (`/home/{user}/.venv/matquant/`)

### Non-Persistent Storage: `/raid/{user}` (local SSD per node)
- Local to **one** node only; very fast I/O (14–28 TB depending on node)
- Data is **not** purged after every job, but will eventually be deleted after extended inactivity or node failure
- Only accessible from jobs running on the same node

**Store here:**
- Active training datasets (copied from `/home` before each run)
- W&B offline cache during runs
- Ray Tune temp files during tuning

### Recommended directory layout

```
/home/{user}/
├── matquant/           ← git repo (code + configs)
├── datasets/           ← permanent dataset storage
│   ├── cifar-10-batches-py/
│   ├── cifar-100-python/
│   └── imagenette/
├── results/            ← permanent trained model checkpoints
│   └── {model_savename}/models/{stem}/model.pt
├── wandb-archive/      ← archived W&B offline runs (after syncing)
└── .venv/matquant/     ← Python virtual environment

/raid/{user}/
├── datasets/           ← fast copy for active training
├── wandb/              ← W&B cache during runs
└── ray_tmp/            ← Ray Tune temp dir during tuning
```

---

## Step-by-Step Setup

### Phase 1: Connect to the cluster

```bash
# SSH to the gateway from your local machine
ssh {USER}@gwkilab.cs.tu-dortmund.de

# Start a tmux session — required so your container keeps running if you disconnect
tmux new -s matquant
```

### Phase 2: Start a Slurm job

**First-time job creation (includes `--container-image`):**

```bash
srun --mem=64GB \
     --export ALL \
     -c 16 \
     --gres=gpu:1 \
     --container-name=matquant-dev \
     --job-name="matquant-dev" \
     -p GPU1 \
     --container-image=nvcr.io/ml2r/interactive_pytorch:23.12-py3 \
     --mail-user=leonard.bereholschi@cs.tu-dortmund.de \
     --mail-type=ALL \
     --pty /bin/bash
```

**Subsequent restarts (omit `--container-image` to reuse your existing container):**

```bash
srun --mem=64GB \
     --export ALL \
     -c 16 \
     --gres=gpu:1 \
     --container-name=matquant-dev1 \
     --job-name="matquant-dev1" \
     -p GPU1 \
     --mail-user=leonard.bereholschi@cs.tu-dortmund.de \
     --mail-type=ALL \
     --pty /bin/bash
```

> **Wall-time limits** (GPU1 partition): default 7 days, maximum 14 days.
> For longer runs use multiple stages or checkpoint/resume.

### Phase 3: Set up Python environment (first time only)

The PyTorch container has PyTorch pre-installed. Initialize a venv that inherits it:

```bash
# Inside the container
/bin/python -m venv --system-site-packages ~/.venv/matquant
source ~/.venv/matquant/bin/activate

# Clone the repo to persistent storage
cd /home/bereholschi
git clone https://github.com/LeonardDavid/matquant.git matquant
cd matquant

# Install remaining dependencies (torch is already in the venv via --system-site-packages)
pip install -r requirements.txt

# Persist venv activation across sessions
echo 'source ~/.venv/matquant/bin/activate' >> ~/.bashrc
```

### Phase 3b: Set up passwordless GitHub access (first time only)

The container does not have the GitHub CLI (`gh`). Use an SSH deploy key instead — it lives in `/home/{user}/.ssh/` (persistent) so this only needs to be done once.

```bash
# Inside the job, generate a new SSH key:
ssh-keygen -t ed25519 -C "lamarr-cluster" -f ~/.ssh/id_ed25519 -N ""

# Print the public key:
cat ~/.ssh/id_ed25519.pub
```

Go to **GitHub → your repo → Settings → Deploy keys → Add deploy key**, paste the public key, name it "lamarr-cluster", and check **Allow write access** if you need to push.

```bash
# Switch the remote from HTTPS to SSH:
git remote set-url origin git@github.com:LeonardDavid/matquant.git

# Test authentication:
ssh -T git@github.com
# Expected: "Hi LeonardDavid! You've successfully authenticated..."

# Now pull/push works without credentials:
git pull
```

### Phase 4: Set up storage layout (first time only)

```bash
# Persistent directories
mkdir -p /home/bereholschi/datasets
mkdir -p /home/bereholschi/results
mkdir -p /home/bereholschi/wandb-archive

# Fast local directories (scratch)
mkdir -p /raid/bereholschi/datasets
mkdir -p /raid/bereholschi/wandb
mkdir -p /raid/bereholschi/ray_tmp
```

**Transfer existing datasets to persistent storage:**

From inside the cluster (if your data is reachable via the filesystem mount):
```bash
rsync -av /path/to/current/data/ /home/bereholschi/datasets/
```

Or from your local machine:
```bash
rsync -av ./data/ {USER}@gwkilab.cs.tu-dortmund.de:/home/{USER}/datasets/
```

### Phase 5: Run training with the `--cluster` flag

The `--cluster` flag automatically overrides all storage paths:
- `dataset.data_dir` → `/raid/{user}/datasets` (fast local SSD)
- `training.model_dir` → `/home/{user}/results/{model_savename}` (persistent)
- W&B cache → `/raid/{user}/wandb` (fast local SSD)

> **Important**: Copy datasets to fast local storage before each training run for best I/O performance.

```bash
source ~/.venv/matquant/bin/activate

# Copy datasets to fast local storage
rsync -a /home/bereholschi/datasets/ /raid/bereholschi/datasets/

# Train
python main.py --config configs/resnet20_cifar10/resnet20_cifar10.yaml \
               --mode train --wandb batched --cluster

# Train then test
python main.py --config configs/resnet20_cifar10/resnet20_cifar10.yaml \
               --mode train_test --wandb batched --cluster

# Test a specific checkpoint (set testing.model_path in the YAML first)
python main.py --config configs/resnet20_cifar10/resnet20_cifar10.yaml \
               --mode test --wandb disabled --cluster

# Ray Tune hyperparameter search
python main.py --config configs/resnet20_cifar10/resnet20_cifar10_ray.yaml \
               --mode tune --cluster
```

**After training — sync W&B offline runs to cloud and archive:**

```bash
wandb sync /raid/bereholschi/wandb/offline-run-*
cp -r /raid/bereholschi/wandb/offline-run-* /home/bereholschi/wandb-archive/
```

**After training — checkpoint locations:**

Checkpoints are saved to `/home/{user}/results/{model_savename}/models/{stem}/model.pt`.
To use a cluster-trained checkpoint in a subsequent test or continued training run, set in your YAML:

```yaml
testing:
  model_path: /home/{user}/results/{model_savename}/models/{stem}/model.pt

training:
  stages:
    - load_from: /home/{user}/results/{prev_model}/models/{stem}/model.pt
```

### Phase 6: Running multiple configs in parallel

You can run multiple training jobs in parallel inside a **single Slurm job** using tmux panes — no new job needed. The job allocates a fixed resource pool and all panes share it.

```bash
Ctrl+b %    # split pane vertically
Ctrl+b "    # split pane horizontally
```

**New panes open on `gwkilab`, not inside the job.** tmux panes inherit the shell that launched the session (the gateway), so you need to SSH into the job from each new pane.

**In each new pane (on gwkilab), SSH into the job:**

```bash
# Find your job's node and port:
squeue -u $USER                          # get JOBID and NODELIST (e.g. ml2ran01)
scontrol show job ${JOBID} | grep Port   # get the SSH port (e.g. 26000)

# SSH in:
# get this info right after starting the slurm job
ssh ${NODE} -p ${PORT}                   # e.g. ssh ml2ran01s0 -p 24000

# Then activate venv and run:
source ~/.venv/matquant/bin/activate
```

In each pane, activate the venv and launch a different config.

**GPU allocation is the key constraint.** If you requested `--gres=gpu:1`, all panes share that one GPU — running two training jobs simultaneously will likely OOM. Request as many GPUs as you want parallel runs:

```bash
# Job with 2 GPUs (start from gwkilab, inside tmux, before entering the job)
srun --mem=128GB --export ALL -c 32 --gres=gpu:2 \
     --container-name=matquant-multi \
     --job-name="matquant-multi" \
     -p GPU2 \
     --mail-user=leonard.bereholschi@cs.tu-dortmund.de \
     --mail-type=ALL \
     --pty /bin/bash
```

Then pin each process to a specific GPU with `CUDA_VISIBLE_DEVICES`:

```bash
# Pane 1 — GPU 0
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/resnet20_cifar10/resnet20_cifar10_new212-ray2.yaml \
    --mode train --wandb batched --cluster

# Pane 2 — GPU 1
CUDA_VISIBLE_DEVICES=1 python main.py --config configs/resnet34_imagenet/resnet34_imagenet_new212-ray2.yaml \
    --mode train --wandb batched --cluster
```

Without `CUDA_VISIBLE_DEVICES`, both processes default to GPU 0 and conflict.

| Scenario | Action |
|----------|--------|
| Multiple panes, 1 run at a time | Reuse same job, no changes needed |
| N configs truly in parallel | Request N GPUs (`--gres=gpu:N`, `-p GPUN`), use `CUDA_VISIBLE_DEVICES=i` per pane |
| Forgot to request enough GPUs | `scancel ${JOBID}` and start a new job with more |

### Phase 7: Detach and monitor

```bash
# Detach from tmux (job keeps running in background)
Ctrl+b, then d

# Check job status
squeue -u $USER

# Re-attach
tmux attach -t matquant

# If the gateway crashes, use a backup gateway and re-attach via sattach:
ssh {USER}@gwkilab1.cs.tu-dortmund.de
tmux new -s recovery
squeue -u $USER          # note your job ID, e.g. 12345
sattach 12345.0
```

---

## Verification

Run these after setup to confirm everything works:

```bash
# Confirm GPU is visible
nvidia-smi

# Confirm PyTorch sees CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Smoke test (no training, just loads model and runs one eval pass)
python main.py --config configs/resnet20_cifar10/resnet20_cifar10.yaml \
               --mode test --wandb disabled --cluster
```

---

## Cluster path summary

| What | Cluster path | Persistence |
|------|-------------|-------------|
| Source code & configs | `/home/{user}/matquant/` | Permanent |
| Datasets (permanent copy) | `/home/{user}/datasets/` | Permanent |
| Datasets (active training) | `/raid/{user}/datasets/` | Node-local, may be deleted |
| Trained model checkpoints | `/home/{user}/results/` | Permanent |
| W&B offline cache | `/raid/{user}/wandb/` | Node-local, sync after run |
| W&B archive (synced runs) | `/home/{user}/wandb-archive/` | Permanent |
| Ray Tune temp files | `/raid/{user}/ray_tmp/` | Node-local, disposable |
| Python virtual environment | `/home/{user}/.venv/matquant/` | Permanent |

---

## Slurm quick reference

```bash
sinfo                    # cluster and queue status
squeue -u $USER          # your running jobs
scancel ${JOBID}         # cancel a job
sacct --format="CPUTime,MaxRSS"  # resource usage of finished jobs
sinfo -o %G              # list available GPU types
```
