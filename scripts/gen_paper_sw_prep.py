"""Emit + preflight the 6 COL-layout fault-aware training commands."""
import sys, shlex
sys.path.insert(0, "scripts")
sys.path.insert(0, "code/python")
from sweep_paper_runs import MODELS
from netdrift.config.loader import load, parse_overrides

SEED = 707
EPOCHS = 10
LAM = 0.05
# Training dose for the INJECTED arms. See note in the answer: fault_state_mode
# =fresh resets state every batch, so a batch sees ONE iteration's dose, while
# eval accumulates over 100 loops. 1e-4 puts each training batch at roughly the
# corruption level the eval sweep reaches at its hardest point (COL, 1e-6, 100
# loops). Inert for cat5 (faults detached).
RT_TRAIN_INJECT = 0.0001
RT_TRAIN_CLEAN = 0.000001

CATS = {
    "cat5_col":  {"fault_aware": "regularization", "lam": LAM,  "inject": "false", "rt": RT_TRAIN_CLEAN},
    "cat8_col":  {"fault_aware": "ste_inject",     "lam": 0.0,  "inject": "true",  "rt": RT_TRAIN_INJECT},
    "cat68_col": {"fault_aware": "regularization", "lam": LAM,  "inject": "true",  "rt": RT_TRAIN_INJECT},
}

def json_list(xs):
    return "[" + ",".join(str(int(x)) for x in xs) + "]"

def overrides(model, cat, spec):
    m = MODELS[model]
    root = m["ckpt_root"]
    return [
        f"experiment.name=paper-sw_{model}",
        "experiment.output_dir=runs/paper-runs/",
        f"experiment.seed={SEED}",
        f"model.checkpoint={root}/model_best.pt",
        "model.checkpoint_mode=strict",
        "storage.layout=col",
        "storage.base_layout=col",
        "storage.rt_size=64",
        "storage.kernel_mapping=row",
        f"fault.rt_error=[{spec['rt']!r}]",
        "fault.weight_encoder=null",
        "fault.mitigations=[]",
        "fault.edge_mode=saturate",
        "fault.protection.policy=custom",
        f"fault.protection.layers={json_list(m['unprotected_custom'])}",
        "training.mode=train",
        f"training.fault_aware={spec['fault_aware']}",
        f"training.reg.lambda={spec['lam']}",
        "training.reg.beta=4.0",
        f"training.reg.inject_faults={spec['inject']}",
        "training.fault_state_mode=fresh",
        "training.fault_aware_criterion=hinge",
        "training.fault_aware_hinge_b=128.0",
        "training.lr=0.001",
        f"training.epochs={EPOCHS}",
        "training.step_size=5",
        "training.gamma=0.1",
        f"training.save_dir={root}/{cat}/",
    ]

order = [(m, c) for m in ("vgg7_cifar10", "resnet18_imagenette") for c in CATS]
for gpu, (model, cat) in enumerate(order):
    spec = CATS[cat]
    ov = overrides(model, cat, spec) + [f"gpu_num={gpu}"]
    argv = ["--config", MODELS[model]["config"]]
    for o in ov:
        argv += ["--override", o]
    argv += ["--metrics", "none",
             "--wandb-project", "netdrift-paper-runs-sw",
             "--wandb-category", f"train_{cat}",
             "--wandb-subcategory", f"train_{cat}_seed{SEED}"]

    # PREFLIGHT: resolve through the real loader and assert the flags landed.
    cfg = load(MODELS[model]["config"], overrides=parse_overrides(ov))
    assert cfg.training.mode == "train"
    assert cfg.storage.layout == "col", cfg.storage.layout
    assert cfg.training.lr == 0.001, cfg.training.lr
    assert cfg.fault.protection.policy == "custom"
    assert cfg.fault.weight_encoder is None
    assert cfg.training.save_dir == f"{MODELS[model]['ckpt_root']}/{cat}/"
    # the flags that decide WHICH mechanism runs
    inject = bool(cfg.training.reg.inject_faults) or cfg.training.fault_aware == "ste_inject"
    use_reg = cfg.training.fault_aware == "regularization" and cfg.training.reg.lambda_ > 0
    exp = {"cat5_col": (False, True), "cat8_col": (True, False), "cat68_col": (True, True)}[cat]
    assert (inject, use_reg) == exp, f"{cat}: got {(inject, use_reg)} want {exp}"

    print(f"# --- {model} / {cat}  (gpu {gpu})   inject={inject} run_length_penalty={use_reg}")
    print("python netdrift_run.py \\\n    " +
          " \\\n    ".join(" ".join(shlex.quote(t) for t in argv[i:i+2])
                           for i in range(0, len(argv), 2)))
    print()
print("ALL 6 PREFLIGHTED OK", file=sys.stderr)
