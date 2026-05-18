"""Instrument the REAL training loop to watch the MagFace collapse happen.

Same backbone / data / loss / optimiser as the factor-3 config, but we
log every N steps what actually degenerates:
  - train_loss
  - mean ||features|| (the `a` MagFace's margin sees, TRAIN mode)
  - eval-logit spread: std across the 7 classes of inference_logits on a
    fixed val batch (-> 0 == uniform == the ln(7) symptom)
  - val_acc on that fixed batch
  - mean pairwise cosine between the 7 classifier-weight rows
    (-> 1 == weights collapsed to one direction)

ArcFace is run as the control under identical conditions: it must NOT
collapse. Whatever metric diverges between the two IS the mechanism.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from face_bias.config import load_config
from face_bias.data.dataset import setup_dataset
from face_bias.models.losses import ArcFaceLoss, MagFaceLoss
from face_bias.models.resnet import LResNet50E_IR


def weight_collapse(head) -> float:
    w = F.normalize(head.weight, dim=1)
    sim = w @ w.t()
    n = w.size(0)
    return (sim.sum() - n) / (n * (n - 1))  # mean off-diagonal cosine


def run(head_name, cfg, device, steps, log_every):
    torch.manual_seed(42)
    dl, _, ncls = setup_dataset(cfg)
    mc = cfg["model"]
    model = LResNet50E_IR(
        num_classes=ncls, dropout=mc["dropout"],
        head=head_name, pretrained=True,
        magface_l_a=mc.get("magface_l_a", 10.0),
        magface_u_a=mc.get("magface_u_a", 110.0),
        magface_l_m=mc.get("magface_l_m", 0.45),
        magface_u_m=mc.get("magface_u_m", 0.8),
        magface_lambda_g=mc.get("magface_lambda_g", 0.0),
    ).to(device)
    loss_fn = MagFaceLoss() if head_name == "magface" else ArcFaceLoss()
    lam = getattr(model.head, "lambda_g", 0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])

    # one fixed val batch for a stable eval probe
    vimg, vlab = next(iter(dl["val"]))
    vimg, vlab = vimg.to(device), vlab.to(device)

    print(f"\n=== {head_name} ===")
    print(f"{'step':>5} {'loss':>8} {'||f||':>7} {'eval_logit_std':>14} "
          f"{'val_acc':>8} {'W_collapse':>11}")
    it = iter(dl["train"])
    for step in range(steps + 1):
        if step % log_every == 0:
            model.eval()
            with torch.no_grad():
                fe = model.extract_features(vimg)
                ev = model.head.inference_logits(fe)
                std = ev.std(dim=1).mean().item()
                acc = (ev.argmax(1) == vlab).float().mean().item()
                fn = torch.norm(fe, dim=1).mean().item()
                wc = weight_collapse(model.head).item()
            print(f"{step:>5} {cur_loss:>8.3f} {fn:>7.2f} {std:>14.4f} "
                  f"{acc:>8.3f} {wc:>11.4f}"
                  if step else
                  f"{step:>5} {'-':>8} {fn:>7.2f} {std:>14.4f} "
                  f"{acc:>8.3f} {wc:>11.4f}")
            model.train()
        try:
            img, lab = next(it)
        except StopIteration:
            it = iter(dl["train"])
            img, lab = next(it)
        img, lab = img.to(device), lab.to(device)
        opt.zero_grad()
        out = model(img, lab)
        loss = loss_fn(out, lab)
        if lam > 0.0:  # canonical MagFace: mirror trainer._magface_reg
            loss = loss + lam * model.head.last_g_reg
        loss.backward()
        opt.step()
        cur_loss = loss.item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments_factor3/exp_f3_magface_s42.yaml")
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run("magface", cfg, dev, args.steps, args.log_every)
    run("arcface", cfg, dev, args.steps, args.log_every)
    print("\nRead: the metric that diverges magface-vs-arcface is the mechanism.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
