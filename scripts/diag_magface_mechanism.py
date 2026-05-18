"""Controlled ablation: isolate WHY MagFace collapses while ArcFace converges.

Synthetic, deterministic, CPU, seconds. Separable 7-class data + a small
trainable backbone + each head variant. We watch, per epoch:
  - train loss (does it go down?)
  - val acc via inference_logits (does the EVAL path learn?)
  - mean target-class cosine (is the representation discriminative?)
  - mean ||features|| (does the embedding norm collapse?)

Variants:
  arcface              : reference (constant margin) — must converge
  mag_orig             : MagFace as first written (NO guard, a NOT detached)
  mag_guard            : + monotonicity guard only (== current magface.py)
  mag_guard_detach     : + guard AND a.detach() in the margin schedule
  mag_lambda_g          : mag_orig + magnitude regulariser g(a) active (paper)

If mag_guard still collapses but mag_guard_detach / mag_lambda_g converge,
the mechanism is the norm-gradient escape with lambda_g=0 — not the guard.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

C, D, PER = 7, 128, 90          # classes, embed dim, samples/class
S = 30.0
L_A, U_A, L_M, U_M = 10.0, 110.0, 0.45, 0.8
EPOCHS, LR = 60, 5e-3


def make_data():
    centroids = F.normalize(torch.randn(C, D), dim=1)
    xs, ys = [], []
    for c in range(C):
        x = centroids[c] + 0.6 * torch.randn(PER, D)   # separable but noisy
        xs.append(x)
        ys.append(torch.full((PER,), c, dtype=torch.long))
    X, Y = torch.cat(xs), torch.cat(ys)
    perm = torch.randperm(len(Y))
    X, Y = X[perm], Y[perm]
    n = int(0.8 * len(Y))
    return X[:n], Y[:n], X[n:], Y[n:]


Xtr, Ytr, Xva, Yva = make_data()


def backbone():
    return nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))


def m_of_a(a):
    slope = (U_M - L_M) / (U_A - L_A)
    return slope * (a - L_A) + L_M


def mag_logits(feats, label, *, use_guard, detach_a, lambda_g):
    """Parametrisable MagFace head logic (weight is module-held)."""
    w = mag_logits.weight
    cosine = F.linear(F.normalize(feats), F.normalize(w)).clamp(-1 + 1e-7, 1 - 1e-7)
    if label is None:
        return cosine * S, torch.tensor(0.0)
    a = torch.norm(feats, dim=1, keepdim=True).clamp(L_A, U_A)
    if detach_a:
        a = a.detach()
    m_a = m_of_a(a)
    cos_m, sin_m = torch.cos(m_a), torch.sin(m_a)
    sine = torch.sqrt((1.0 - cosine ** 2).clamp_min(1e-7))
    phi = cosine * cos_m - sine * sin_m
    if use_guard:
        phi = torch.where(cosine > -cos_m, phi, cosine - sin_m * m_a)
    one_hot = torch.zeros_like(cosine)
    one_hot.scatter_(1, label.view(-1, 1).long(), 1)
    out = (one_hot * phi + (1.0 - one_hot) * cosine) * S
    g_reg = ((1.0 / (U_A ** 2)) * a + 1.0 / a).mean() if lambda_g > 0 else torch.tensor(0.0)
    return out, lambda_g * g_reg


def arc_logits(feats, label):
    w = arc_logits.weight
    cosine = F.linear(F.normalize(feats), F.normalize(w))
    if label is None:
        return cosine * S
    cosine = cosine.clamp(-1, 1)
    cos_m, sin_m = math.cos(0.5), math.sin(0.5)
    th, mm = math.cos(math.pi - 0.5), math.sin(math.pi - 0.5) * 0.5
    sine = torch.sqrt(1.0 - cosine ** 2 + 1e-7)
    phi = cosine * cos_m - sine * sin_m
    phi = torch.where(cosine > th, phi, cosine - mm)
    one_hot = torch.zeros_like(cosine)
    one_hot.scatter_(1, label.view(-1, 1).long(), 1)
    return (one_hot * phi + (1.0 - one_hot) * cosine) * S


def run(name, head_fn):
    torch.manual_seed(0)
    bb = backbone()
    weight = nn.Parameter(F.normalize(torch.randn(C, D), dim=1).clone())
    if name == "arcface":
        arc_logits.weight = weight
    else:
        mag_logits.weight = weight
    opt = torch.optim.Adam(list(bb.parameters()) + [weight], lr=LR)
    ce = nn.CrossEntropyLoss()

    last = {}
    for ep in range(EPOCHS):
        bb.train()
        feats = bb(Xtr)
        if name == "arcface":
            logits, reg = arc_logits(feats, Ytr), torch.tensor(0.0)
        else:
            logits, reg = head_fn(feats, Ytr)
        loss = ce(logits, Ytr) + reg
        opt.zero_grad()
        loss.backward()
        opt.step()

        bb.eval()
        with torch.no_grad():
            fv = bb(Xva)
            ev = arc_logits(fv, None) if name == "arcface" else head_fn(fv, None)[0]
            acc = (ev.argmax(1) == Yva).float().mean().item()
            cos = F.linear(F.normalize(fv), F.normalize(weight))
            tgt_cos = cos.gather(1, Yva.view(-1, 1)).mean().item()
            fnorm = torch.norm(fv, dim=1).mean().item()
        last = dict(loss=loss.item(), acc=acc, tgt_cos=tgt_cos, fnorm=fnorm)
    return last


VARIANTS = {
    "arcface":          None,
    "mag_orig":         lambda f, y: mag_logits(f, y, use_guard=False, detach_a=False, lambda_g=0.0),
    "mag_guard":        lambda f, y: mag_logits(f, y, use_guard=True,  detach_a=False, lambda_g=0.0),
    "mag_guard_detach": lambda f, y: mag_logits(f, y, use_guard=True,  detach_a=True,  lambda_g=0.0),
    "mag_lambda_g":     lambda f, y: mag_logits(f, y, use_guard=True,  detach_a=False, lambda_g=35.0),
}

print(f"\n{'variant':<18} {'loss':>8} {'val_acc':>8} {'tgt_cos':>9} {'||f||':>8}  verdict")
print("-" * 70)
chance = 1.0 / C
for nm, fn in VARIANTS.items():
    r = run(nm, fn)
    ok = r["acc"] > 2 * chance and r["tgt_cos"] > 0.15
    verdict = "CONVERGE" if ok else "COLLAPSE"
    print(f"{nm:<18} {r['loss']:>8.3f} {r['acc']:>8.3f} "
          f"{r['tgt_cos']:>9.3f} {r['fnorm']:>8.2f}  {verdict}")
print(f"\n(chance acc = {chance:.3f}; COLLAPSE = eval path did not learn / "
      f"representation degenerate)")
