"""Diagnose the MagFace collapse: measure the embedding-norm distribution.

MagFace ties the per-sample angular margin to ``||f||`` via the linear
schedule ``m(a)`` over ``[l_a, u_a]`` (defaults 10..110). If our
ResNet-50 embeddings fall *outside* that range the margin saturates at
a constant (l_m or u_m) from epoch 1 — degenerating training. The
sanity run showed exactly that (val_loss frozen at ln(7), acc ~= chance).

This probe loads the real backbone + a few real batches, extracts the
pre-head embedding (``extract_features``), and reports the norm
distribution against the MagFace [l_a, u_a] window. fp32, num_workers=0
(matched with the factor-3 protocol). Read-only — trains nothing.
"""

from __future__ import annotations

import argparse

import torch

from face_bias.config import load_config
from face_bias.data.dataset import setup_dataset
from face_bias.models.resnet import LResNet50E_IR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments_factor3/exp_f3_magface_s42.yaml")
    ap.add_argument("--batches", type=int, default=4, help="train batches to sample")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    l_a = cfg["model"].get("magface_l_a", 10.0)
    u_a = cfg["model"].get("magface_u_a", 110.0)

    dataloaders, _, num_classes = setup_dataset(cfg)
    model = LResNet50E_IR(
        num_classes=num_classes,
        dropout=cfg["model"]["dropout"],
        head="magface",
        pretrained=cfg["model"].get("pretrained", True),
    ).to(device)

    def collect(train_mode: bool) -> torch.Tensor:
        # MagFace computes a=||features|| in TRAIN mode (BN batch stats +
        # dropout). eval mode (BN running/ImageNet stats, dropout off) is a
        # different distribution — measure both to see which regime matters.
        model.train(train_mode)
        ns: list[torch.Tensor] = []
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloaders["train"]):
                feats = model.extract_features(images.to(device))
                ns.append(torch.norm(feats, dim=1).cpu())
                if i + 1 >= args.batches:
                    break
        return torch.cat(ns)

    a_train = collect(train_mode=True)
    a = collect(train_mode=False)  # eval-mode, kept for the headline report

    qt = torch.quantile(a_train, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0]))
    print(f"\n[TRAIN-mode norm — what MagFace's m(a) actually sees]  n={a_train.numel()}")
    print(f"  min={qt[0]:.2f} p5={qt[1]:.2f} median={qt[2]:.2f} "
          f"p95={qt[3]:.2f} max={qt[4]:.2f}  mean={a_train.mean():.2f}")
    it = ((a_train >= l_a) & (a_train <= u_a)).float().mean().item()
    print(f"  in MagFace window [{l_a},{u_a}] = {it:.1%}  "
          f"above_u_a={(a_train > u_a).float().mean().item():.1%}")
    qs = torch.quantile(a, torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]))
    in_window = ((a >= l_a) & (a <= u_a)).float().mean().item()
    below = (a < l_a).float().mean().item()
    above = (a > u_a).float().mean().item()

    print(f"\nEmbedding-norm probe — n={a.numel()} samples, fp32, eval mode")
    print(f"MagFace window [l_a={l_a}, u_a={u_a}]")
    print(f"  min={qs[0]:.3f}  p1={qs[1]:.3f}  p5={qs[2]:.3f}  "
          f"median={qs[3]:.3f}  p95={qs[4]:.3f}  p99={qs[5]:.3f}  max={qs[6]:.3f}")
    print(f"  mean={a.mean():.3f}  std={a.std():.3f}")
    print(f"  in-window={in_window:.1%}  below_l_a={below:.1%}  above_u_a={above:.1%}")

    if in_window < 0.5:
        side = "BELOW l_a" if below > above else "ABOVE u_a"
        sat = "l_m (min margin)" if below > above else "u_m (max margin)"
        print(f"\nVERDICT: norms mostly {side} -> margin saturates at {sat} "
              f"from epoch 1 -> collapse confirmed. Recalibrate [l_a,u_a] to the "
              f"measured ~[p5,p95] range (the MagFace-paper prescription).")
    else:
        print("\nVERDICT: norms largely in-window; collapse is NOT a range "
              "mismatch — investigate scale s / margin magnitude instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
