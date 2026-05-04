# NetDrift

Presentation [Paper on Springer Nature](https://doi.org/10.1007/978-3-031-78377-7_16)

Models and other large files can be found [here](https://huggingface.co/leonarddavid/NetDrift-models/tree/main)

The NetDrift framework enables profiling Binary/Quantized Neural Networks (BNNs/QNNs) on unreliable Racetrack Memories (RTMs), facilitating the investigation into the impact of misalignment faults on model accuracy.

It enables controlled error injection in selected BNN/QNN layers with varying fault rates, simulates the impact of accumulated misalignments in weight tensors of multiple architectures (VGG3 for MNIST/FashionMNIST, VGG7 for CIFAR-10/100, ResNet18/34/50, MobileNetV2, ViT-B/16 — both ImageNet and CIFAR variants), and supports multiple quantization schemes (binary now, ternary/INT2/INT4/INT8/mixed-precision in upcoming phases) and pluggable storage layouts and fault models.

The framework allows tuning reliability for performance and vice versa, providing an estimate of the number of inference iterations required for a model to drop below a certain accuracy threshold under no, partial, or full layer protection.

---

## Status: framework overhaul in progress

A clean-room refactor under [`code/python/netdrift/`](code/python/netdrift/) replaces the legacy `code/python/{Models,QuantizedNN,Utils,Traintest_Utils}.py` + `run.py` + `run.sh` + `flags.conf` workflow with a typed, configurable, testable Python package. The legacy files remain in the repo for now and are slated for removal in Phase 4.

**Phase 1 (foundation) — complete.** What's shipped:

- **Decoupled abstractions.** `QuantScheme` (`netdrift.quant`), `FaultModel` (`netdrift.faults`), `WeightLayout` (`netdrift.storage`, scaffolded), `MitigationStep` (`netdrift.faults.mitigations`). Layers delegate to these via thin (~150 lines total) `QuantizedConv2d`/`QuantizedLinear` classes — replacing the ~600-line legacy forward passes.
- **Model registry + in-place quantization.** `replace_with_quantized(model, scheme)` walks any `nn.Module` tree and swaps `nn.Conv2d`/`nn.Linear` for their quantized cousins **without changing attribute paths**, preserving state-dict keys for HuggingFace checkpoint compatibility. 19 architectures registered: VGG3 (MNIST/FMNIST), VGG7 (CIFAR-10/100), ResNet{18,34,50}, MobileNetV2, ViT-B/16, each in ImageNet and CIFAR variants.
- **CheckpointAdapter** with three modes: `strict` (BNN→BNN exact match), `fp32_warmstart` (FP32 torchvision/HF checkpoint → quantized model with bias drops, key remaps, scale init), and `scheme_transfer` (BNN → different quantization scheme). Loading produces an `AdapterReport` listing every dropped/renamed/initialized key.
- **YAML config system.** A single `ExperimentConfig` dataclass tree drives everything. Configs support `defaults: [...]` includes for shared fragments and CLI `--override key.subkey=value` overrides. Replaces `run.sh` + `flags.conf`.
- **Single CLI entry point.** `python netdrift_run.py --config configs/<name>.yaml [--override key=value ...]`.
- **Eight datasets**: MNIST, FashionMNIST, KMNIST, SVHN, CIFAR-10, CIFAR-100, Imagenette, ImageNet-1k.
- **Fixed two latent CUDA RNG races** in the legacy racetrack kernels (see "Bugfixes" below). The new kernels are deterministic at a fixed seed.
- **37 GPU + CPU tests pass** — covering layer replacement, state-dict preservation, checkpoint adapter modes, RTM fault injection (zero-error identity, fault generation, accumulation, mitigation integration, kernel determinism), config loading and overrides, and mitigation-step semantics.

**Upcoming phases:**

- Phase 2 — structured metrics (JSONL + W&B sinks, run-dir layout) + fault-aware training (STE injection, knowledge distillation, sensitivity regularization).
- Phase 3 — `WeightLayout` strategies (Gray, ECC, replicated, importance-sorted) + multi-bit quantization (ternary, INT2/4/8, mixed-precision per channel).
- Phase 4 — skyrmion fault model + final test backfill, type hints, docs, and removal of the legacy code paths.

See [`/Users/leonard/.claude/plans/go-through-the-current-proud-storm.md`](https://github.com/) (local) for the full plan.

---

## Prerequisites

### Anaconda3 (Linux Ubuntu / WSL)

```bash
curl -O https://repo.anaconda.com/archive/Anaconda3-<INSTALLER_VERSION>-Linux-x86_64.sh
bash Anaconda3-<INSTALLER_VERSION>-Linux-x86_64.sh
source ~/anaconda3/bin/activate
conda init
source ~/.bashrc
```

### Conda environment

From the pre-made environment file:
```bash
conda env create -f environment.yml
conda activate netdrift
```

Or manually:
```bash
conda create -n netdrift python=3.10
conda activate netdrift
conda install nvidia::cuda-toolkit          # or: sudo apt install nvidia-cuda-toolkit
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
pip install matplotlib scipy pyyaml pytest
```

### Numba CUDA bindings (required for the new package)

The new `netdrift` package uses Numba CUDA for the RTM simulation kernels and requires the NVIDIA Python bindings to coexist cleanly with PyTorch's bundled CUDA runtime. Install once:

```bash
pip install "cuda-python>=12,<13"
```

The package sets `NUMBA_CUDA_USE_NVIDIA_BINDING=1` automatically on import (see `netdrift/__init__.py`), so you don't have to. **This is required**: without it, Numba probes the conda toolkit (typically a different CUDA minor version than PyTorch's bundled libs) and segfaults on the first kernel launch when both are loaded.

### Compile the legacy CUDA kernels (for the binary scheme + legacy paths)

```bash
bash ./code/cuda/install_kernels.sh
```

The new `BinaryScheme` uses these kernels when CUDA is available and falls back to a pure-PyTorch implementation otherwise.

---

## Run an experiment (new flow)

```bash
python netdrift_run.py --config configs/vgg7_cifar10_rtm.yaml
```

Or with overrides:

```bash
python netdrift_run.py --config configs/vgg7_cifar10_rtm.yaml \
    --override fault.rt_error=0.05 \
    --override training.loops=10 \
    --override fault.protection.policy=custom \
    --override fault.protection.layers=[1,2]
```

Each run writes its config snapshot, summary, and (Phase 2) metrics into `runs/<experiment_name>/<timestamp>/`.

### Example config

```yaml
# configs/vgg7_cifar10_rtm.yaml
experiment:
  name: vgg7_cifar10_rtm_baseline
  output_dir: runs/

model:
  name: vgg7_cifar10
  checkpoint: hf://leonarddavid/NetDrift-models/vgg7_cifar10.pth
  checkpoint_mode: strict          # strict | fp32_warmstart | scheme_transfer

data:
  name: cifar10                    # mnist | fmnist | kmnist | svhn | cifar10 | cifar100 | imagenette | imagenet
  batch_size: 256

quant:
  scheme: binary                   # binary (Phase 3+: ternary | int_uniform | mixed_precision)
  scale_init: max_abs              # max_abs | quantile_999 | quantile_<x>

storage:
  layout: row                      # row | col | mix (Phase 3+: interleaved | gray | importance_sorted | ecc | replicated)
  rt_size: 64
  kernel_mapping: row              # row | col | clw | acw

fault:
  model: rtm_misalignment
  rt_error: [0.001, 0.01, 0.05, 0.1]
  global_bitflip_budget: 0.0
  local_bitflip_budget: 0.0
  mitigations: []                  # default: empty (one step at a time recommended)
  protection:
    policy: all                    # all | custom | indiv
    layers: [1, 2]                 # for policy=custom: 1-based unprotected indices

training:
  mode: test                       # test | train
  fault_aware: none                # none (Phase 2: ste_inject | kd | regularization)
  loops: 1

metrics:
  online: [bitflips, misalign_faults, affected_units]
  offline: []
  sinks:
    - type: stdout
    - type: jsonl

gpu_num: 0
```

### Configuration cheatsheet

| Section       | Key                       | Meaning                                                               |
|---------------|---------------------------|-----------------------------------------------------------------------|
| `experiment`  | `name`, `output_dir`, `seed` | Run identification + RNG seed                                      |
| `model`       | `name`                    | Registry key — `python -c "from netdrift.models import list_models; print(list_models())"` |
|               | `checkpoint`, `checkpoint_mode` | Load a checkpoint with `strict` / `fp32_warmstart` / `scheme_transfer` |
|               | `skip_first_quant`, `skip_last_quant` | Leave the first / last layer at full precision (BNN convention) |
| `data`        | `name`, `batch_size`, ... | Dataset selection + DataLoader settings                              |
| `quant`       | `scheme`, `scale_init`    | Quantization scheme; warm-start scale-init policy                    |
| `storage`     | `layout`, `rt_size`, `kernel_mapping` | RT layout + 3×3 kernel mapping (legacy ROW/COL/CLW/ACW)    |
| `fault`       | `model`, `rt_error`       | Fault model registry key + scalar or list (sweep)                    |
|               | `mitigations`             | Ordered list of mitigation steps (default empty; chain at your own risk) |
|               | `protection.policy`       | `all` (all unprotected) / `custom` (list `layers`) / `indiv` (one at a time) |
| `training`    | `mode`, `loops`, `epochs`, `lr`, `gamma`, `step_size` | Train or test, with hyperparameters         |
| `metrics`     | `online`                  | Per-call stats: `bitflips`, `misalign_faults`, `affected_units`      |
|               | `sinks`                   | List of sinks: `stdout`, `jsonl`, (Phase 2) `wandb`                  |
| —             | `gpu_num`                 | GPU device ID                                                         |

---

## Run the test suite

```bash
pytest tests/ -v
```

CPU-safe tests run anywhere; GPU tests (marked `@pytest.mark.cuda`) auto-skip on machines without a working CUDA setup. The conftest probes Numba+CUDA in a subprocess to detect environment issues before any test imports.

To enable the optional HuggingFace-checkpoint accuracy regression test:

```bash
NETDRIFT_HF_CKPT=/path/to/vgg7_cifar10.pt \
NETDRIFT_HF_MODEL=vgg7_cifar10 \
NETDRIFT_HF_DATASET=cifar10 \
NETDRIFT_HF_ACC=89.42 \
pytest tests/test_hf_checkpoint.py -v
```

---

## Bugfixes vs. the legacy implementation

While porting the racetrack kernels into the new package, two latent CUDA RNG races were identified and fixed in [`netdrift/faults/kernels/rtm_numba.py`](code/python/netdrift/faults/kernels/rtm_numba.py):

1. **Direction-draw race in `calc_index_offset_kernel`.** The legacy passed `rand` (a `float32` in `[0, 1)`) as the *index* argument to the second `xoroshiro128p_uniform_float32` call. Numba truncates floats to int, collapsing every thread's direction draw onto `rng_states[0]` — a write-write race across all racetracks producing non-deterministic output. Fixed by reusing the per-thread state index `i*shape[1] + j` for both draws.

2. **Out-of-bounds-read race in `simulate_racetrack_kernel`.** The legacy used `q_out_index = j*rt_size + k` as the rng index — which collides across rows of racetracks at the same column, so multiple threads race on the same RNG state. Fixed by using each thread's own state index for all OOB draws.

These are intentional behavioral changes vs. the legacy: aggregate statistics (mean misalignment count, mean bitflip count) remain very close, but results are now reproducible at a fixed seed. Empirically verified: two consecutive `racetrack_sim(..., seed=12345)` calls in the legacy produce different offsets on row 0; the refactored kernels produce byte-identical output across runs.

---

## Legacy interface (still available, slated for removal in Phase 4)

The original `run.sh` / `run_all.sh` / `flags.conf` workflow continues to work unchanged. Its files remain at the repo root and `code/python/{Models,QuantizedNN,Utils,Traintest_Utils}.py`. The legacy CUDA package was renamed `code/python/cuda/` → `code/python/legacy_cuda/` to avoid shadowing the new `cuda-python` distribution; a single import in `QuantizedNN.py` was updated accordingly. No other behavior changes.

### Legacy `flags.conf`

Calculation Flags:
- `CALC_RESULTS: True/False` (**Leave True**. Calculate Results for each inference iteration across PERRORS misalignment fault rates)
- `CALC_BITFLIPS: True/False` (Calculate Bitflips per layer for each inference iteration)
- `CALC_MISALIGN_FAULTS: True/False` (Calculate Misalignment Faults per layer)
- `CALC_AFFECTED_RTS: True/False` (Calculate Affected Racetracks per layer)

Execution Flags:
- `EXEC_ENDLEN: True/False` (execute Blockhypothesis using Endlen optimization)
- `EXEC_ODD2EVEN_DEC/INC: True/False`
- `EXEC_EVEN2ODD_DEC/INC: True/False`
- `EXEC_BIN_REVERT_MID/EDGES: True/False`
- `EXEC_EVERY_NRUN: uint`

Read Flags & Parameters (at most 1 READ flag in `flags.conf` can be `True`):
- `READ_ENDLEN: True/False`, `FOLDER_ENDLEN: string`
- `READ_TEST: True/False`, `FOLDER_TEST: string`

Print Flags & Parameters: `PRNT_LAYER_NAME`, `PRNT_INPUT_FILE_INFO`, `PRNT_QWEIGHTS_BEFORE/AFTER`, `PRNT_QWEIGHTS_AFTER_NRUN`, `PRNT_IND_OFF_BEFORE/AFTER`, `PRNT_IND_OFF_AFTER_NRUN`.

### Legacy launch arguments (`run.sh`)

- `-o, --operation` `TRAIN|TEST|TEST_AUTO`
- `-l, --loops` Number of inference iterations (use `--epochs` for TRAIN)
- `-m, --model` `MNIST|FMNIST|CIFAR|RESNET`
- `-p, --perrors` Array of fault rates (space-separated)
- `-ks, --kernel-size` `0|3|5|7`
- `-km, --kernel-mapping` `ROW|COL|CLW|ACW`
- `-rs, --rt-size` Typically 64
- `-rm, --rt-mapping` `ROW|COL|MIX`
- `-lc, --layer-config` `ALL|CUSTOM|INDIV`
- `-ls, --layers` Layer IDs to leave unprotected
- `-g, --gpu-id` `0|1`
- Training: `-e, --epochs`, `-bs, --batch-size`, `-lr, --learning-rate`, `-ss, --step-size`
- Optional: `-mp, --model-path`, `-gb, --global-budget`, `-lb, --local-budget`

### Legacy examples

```bash
# Test MNIST, 1% fault rate, 2 iterations, default custom protection
bash ./run.sh -o TEST -l 2 --model MNIST --perrors 0.01 -ks 0 -km ROW -rs 64 -rm ROW --layer-config CUSTOM --gpu 0

# Test FMNIST, 10% rate, layers 1+3 unprotected
bash ./run.sh -o TEST -l 1 --model FMNIST --perrors 0.1 -ks 3 -km CLW -rs 64 -rm COL --layer-config CUSTOM --layers 1 3 --gpu 0

# Test CIFAR with budget enforcement
bash ./run.sh -o TEST -l 10 --model CIFAR --perrors 0.1 0.01 -ks 3 -km COL -rs 64 -rm COL --layer-config ALL --gpu 0 -gb 0.15 -lb 0.3

# Train FMNIST with fault injection
bash ./run.sh -o TRAIN --model FMNIST --perrors 0.0001 -ks 3 -km ROW -rs 64 -rm ROW --layer-config CUSTOM --layers 1 2 --gpu 0 --epochs 10 -bs 256 -lr 0.001 -ss 25
```

---

## Troubleshooting

- **`libc10.so: cannot open shared object file`** when running `python -m netdrift.runner.run`. The package isn't on `PYTHONPATH` and Python falls back to a misleading import path. Use `python netdrift_run.py ...` (preferred) or set `PYTHONPATH=code/python`.
- **Segfault on the first kernel launch.** Numba CUDA picked a different toolkit than PyTorch's bundled libs. Install `cuda-python>=12,<13` and confirm `NUMBA_CUDA_USE_NVIDIA_BINDING=1` is set (the new package sets this on import; if you bypass the package import, set it manually).
- **`ImportError: cannot import name 'cuda' from 'cuda'`**. The legacy `code/python/cuda/` directory shadows the `cuda-python` package. The repo has been renamed to `legacy_cuda/`; if you re-add a `cuda/` directory anywhere on `PYTHONPATH`, expect this error.
- **CUDA Memory errors.** Lower the test batch size in your config (`data.test_batch_size`).
- **Windows `bash ./code/cuda/install_kernels.sh` fails.** Convert line endings from CRLF to LF.

---

## Contact

Maintainer: [leonard.bereholschi@tu-dortmund.de](mailto:leonard.bereholschi@tu-dortmund.de)

## Acknowledgements

Special thanks to [Mikail Yayla](https://github.com/myay) for providing the original SPICE-Torch framework as a base.
