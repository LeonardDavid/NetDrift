# W1A{N} training — 1-bit weights + N-bit activations

**Date:** 2026-05-26
**Scope:** Training pipeline only. Inference under RTM faults is the next stage.

## Goal

Train models where weights are binarized to ±1 (existing `BinaryScheme`) and
activations are uniformly quantized to a configurable bit-width (e.g. 8/4/2).
Warm-start from a pretrained FP32 model. Hand off the resulting checkpoint to
the RTM simulation environment where 1-bit weights live on nanowires (with
misalignment faults) and quantized activations live on conventional RAM (no
faults).

For the headline target: provide a config file for VGG7 + CIFAR-10.

## Why uniform symmetric quantization on `[-1, 1]`

Every quantized activation in NetDrift sits immediately after `nn.Hardtanh()`,
which clamps to `[-1, 1]`. That gives us a pre-fixed input range — no learned
clipping (PACT) or LSQ-style step-size is needed for a first pass. Standard
DoReFa-style uniform quantization on `[-1, 1]` is the right baseline:

```
levels(N) = { -1 + 2k/(2^N - 1)  :  k = 0..2^N - 1 }
        N=1 → {-1, +1}             # reduces to BNN
        N=2 → {-1, -1/3, +1/3, +1}
        N=4 → 16 evenly-spaced levels on [-1, 1]
        N=8 → 256 evenly-spaced levels on [-1, 1]
```

Forward: clamp to `[-1, 1]` (already done by htanh, so a no-op), then snap to
nearest level. Backward: straight-through estimator (identity through the
quantizer). The existing `_STEQuantize` autograd Function already does this —
we just need a new `QuantScheme`.

Rationale for not doing more (YAGNI):

- **No PACT / learned clipping:** htanh provides a fixed clip. If N=2 accuracy
  is poor we revisit.
- **No per-channel activation scale:** activations are tensors; per-channel
  scaling on activation tensors is uncommon (PTQ, not QAT). Weights remain
  binary so no weight-scale either.
- **No first/last-layer skip:** the existing VGG7 BNN recipe doesn't skip
  these. Defaults stay `false`.

## Where things plug in

The codebase already has the right seams:

- `QuantScheme` ABC and `_STEQuantize` autograd Function are generic over
  activations vs. weights.
- `QuantizedActivation(scheme)` already exists and is wired into VGG7 as
  `qact1..qact7 = nn.Identity()` placeholders.
- `replace_with_quantized` swaps Conv/Linear in place; activations are not
  touched (they need a separate attach pass).
- `fp32_warmstart` checkpoint mode already loads FP32 weights into the BNN
  model and initialises per-channel scales (no-op here since binary needs
  none).

So the changes are small and local.

## Changes

### 1. New activation scheme

`code/python/netdrift/quant/uniform.py`:

```python
class IntUniformActScheme(QuantScheme):
    """Symmetric uniform quantization to N evenly-spaced levels on [-1, 1].

    Assumes input is already clamped to [-1, 1] (Hardtanh upstream).
    N=1 reduces to {-1, +1} (matches BinaryScheme on activations).
    """
    needs_per_channel_scale = False

    def __init__(self, bits: int):
        if bits < 1:
            raise ValueError(...)
        self.bits = bits
        self.n_levels = 2 ** bits
        self.step = 2.0 / (self.n_levels - 1) if self.n_levels > 1 else 2.0

    def quantize(self, x, per_channel_scale=None) -> QuantizedTensor:
        # clamp to be safe even if upstream doesn't clip (e.g. ResNet later)
        x_c = x.clamp(-1.0, 1.0)
        # snap to nearest of n_levels evenly spaced on [-1, 1]
        if self.n_levels == 2:                       # binary fast path: sign
            out = torch.where(x_c >= 0, ones, -ones)
        else:
            out = torch.round((x_c + 1.0) / self.step) * self.step - 1.0
        return QuantizedTensor(values=out, bits=self.bits,
                               levels=tuple(...), per_channel_scale=None)
```

Backward goes through `_STEQuantize` (identity). Outside `[-1, 1]` the htanh
upstream already zeros the gradient, so we don't need to mask in the STE.

### 2. VGG7 wires `QuantizedActivation` placeholders

Change `qactN = nn.Identity()` → `qactN = QuantizedActivation()` in VGG7 (and
VGG3 — same pattern, no harm).

**State-dict impact:** `QuantizedActivation` carries no parameters and no
buffers (the scheme is an attribute, not registered). So `state_dict()` keys
are unchanged. Existing BNN and FP checkpoints continue to load identically.

### 3. `attach_activation_scheme` helper

`code/python/netdrift/models/replace.py`: new function that walks
`named_modules()` and binds a scheme to every `QuantizedActivation`. Called by
the runner right after `replace_with_quantized`.

### 4. Config schema extension

`config/schema.py`, `QuantCfg`:

```python
@dataclass
class QuantCfg:
    scheme: str = "binary"                    # weight scheme
    bits: int = 1                             # weight bits
    bits_per_channel: Optional[list[int]] = None
    scale_init: str = "max_abs"
    activation_scheme: str = "none"           # NEW: "none" | "int_uniform"
    activation_bits: int = 1                  # NEW: 1, 2, 4, 8 (any N>=1)
```

Default `activation_scheme="none"` keeps existing configs (FP, plain BNN)
behaving identically.

### 5. Runner wiring

`runner/run.py`: `_build_scheme` becomes `_build_schemes` returning
`(weight_scheme, activation_scheme)`. Both get attached after
`replace_with_quantized`. `activation_scheme=None` ⇒ activations stay as
`nn.Identity`-equivalent (the `QuantizedActivation` with `scheme=None` is
already a no-op in forward).

### 6. Config files

Three near-identical files (different `activation_bits`):

- `configs/vgg7_cifar10_w1a8_train.yaml`
- `configs/vgg7_cifar10_w1a4_train.yaml`
- `configs/vgg7_cifar10_w1a2_train.yaml`

All warm-start from `models/fp/vgg7_cifar10/model_best.pt` (the artifact from
Stage 1). All write best checkpoints to `models/w1aN/vgg7_cifar10/`.

## Training recipe

Re-use the proven BNN recipe — same loss, same optimizer, same schedule. Only
the activation scheme changes.

- **Warm-start:** `checkpoint_mode: fp32_warmstart` from the FP VGG7.
- **Loss:** `BinaryHingeLoss(b=128)`.
- **Optimizer:** `Clippy` (Adam + `data.clamp(-1, 1)` post-step).
- **Schedule:** epochs=150, lr=1e-3, gamma=0.1, step_size=50 (decays at 50,
  100).
- **Augmentation:** whatever `build_datasets('cifar10', ...)` already returns
  (standard crop+flip).
- **Skip first/last weight quant:** `false` (matches existing BNN recipe).

Notes for N=2: if the loss stagnates, drop lr to 5e-4 or add 50 epochs. Don't
preemptively tune.

## Note for the downstream RTM-simulation stage

`QuantizedActivation.scheme` is a plain Python attribute, not part of
`state_dict` — same pattern as `BinaryScheme` on weights. When loading a
W1A_n_ checkpoint for inference under RTM faults, the caller must reattach
the activation scheme:

```python
load_checkpoint(model, "models/w1a4/.../model_best.pt", mode="strict")
attach_activation_scheme(model, IntUniformActScheme(bits=4))
```

The runner already does this when `quant.activation_scheme` is set in the
YAML, so the inference config just needs the same `quant:` block as training.

## Out of scope (not in this change)

- Inference under RTM faults — next stage, will reuse the produced `.pt`
  files unchanged.
- Per-channel activation scales / learned clipping / PACT / LSQ — only revisit
  if W1A2 underperforms.
- Activation quantization for ResNet / MobileNetV2 / ViT — those models don't
  use htanh, so the symmetric `[-1, 1]` assumption breaks. Future work.
- Activation-side fault injection — the RTM stage explicitly stores activations
  on conventional RAM (no faults). No fault model is attached to
  `QuantizedActivation`.

## Verification before declaring done

The user runs full training; this session can only verify the pipeline starts:

1. `python netdrift_run.py --config configs/vgg7_cifar10_w1a4_train.yaml`
   runs at least one batch and prints a decreasing loss.
2. The checkpoint adapter report shows no truly-missing keys (per-channel
   scales aren't introduced, so the warm-start should be clean).
3. The printed model shows `QuantizedActivation` with a bound
   `IntUniformActScheme` at every `qactN`.

Final accuracy targets are not claimed here — the user will report after a
full training run.
