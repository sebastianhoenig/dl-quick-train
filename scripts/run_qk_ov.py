#!/usr/bin/env python3
"""P0.2: QK and OV weight-matrix analysis for the Layer-1 retrieval heads.

For each Layer-1 head we report:
  - top eigenvalues of the symmetric part of QK
  - full SVD of effective OV
  - projection energy of the top-k QK eigenvectors onto the address / payload /
    random basis (the address/payload bases are built from Layer-0 OV circuits).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from dl_quick_train.analysis.subspaces import address_basis, payload_basis
from dl_quick_train.analysis.weights import (
    projection_energy,
    qk_circuit,
    ov_circuit,
    random_basis_like,
    svd,
    sym_eig,
)
from minimal_sae_train import build_model, load_weights


DEFAULT_ROLES = "experiments/results/verification/head_roles.json"
DEFAULT_OUT = "experiments/results/qk_ov"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", type=str, default=DEFAULT_ROLES)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--rank", type=int, default=32, help="subspace rank")
    ap.add_argument("--topk", type=int, default=8, help="top-k QK eigvecs to score")
    args = ap.parse_args()

    with open(args.roles) as f:
        roles = json.load(f)
    address_head = int(roles["address_head"])
    payload_head = int(roles["payload_head"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    load_weights(model, device)

    addr = address_basis(model, address_head, rank=args.rank, layer=0)
    pay = payload_basis(model, payload_head, rank=args.rank, layer=0)
    rand = random_basis_like(addr, seed=0)

    os.makedirs(args.out, exist_ok=True)

    summary = {
        "address_head_L0": address_head,
        "payload_head_L0": payload_head,
        "rank": args.rank,
        "topk": args.topk,
        "L1_heads": {},
    }

    for h in range(model.cfg.n_heads):
        qk = qk_circuit(model, layer=1, head=h)  # torch.Tensor
        ov = ov_circuit(model, layer=1, head=h)
        eigvals, eigvecs = sym_eig(qk)
        U, S, Vh = svd(ov)

        top_eigvecs = eigvecs[:, : args.topk]  # [d_model, topk]

        energy_addr = projection_energy(top_eigvecs, addr)
        energy_pay = projection_energy(top_eigvecs, pay)
        energy_rand = projection_energy(top_eigvecs, rand)

        # Identity-on-payload check: cosine of payload basis pre vs post OV.
        OV = ov.numpy()
        transported = OV.T @ pay  # OV input = row-vector residual; for v in payload basis,
        # output = v @ OV (since OV is [d_model, d_model] applied as x @ OV in TL).
        transported = transported / (np.linalg.norm(transported, axis=0, keepdims=True) + 1e-9)
        pay_norm = pay / (np.linalg.norm(pay, axis=0, keepdims=True) + 1e-9)
        payload_identity_cos = (transported * pay_norm).sum(axis=0)

        np.savez(
            os.path.join(args.out, f"qk_L1_h{h}.npz"),
            eigvals=eigvals,
            eigvecs=eigvecs,
            top_eigvec_energy_address=energy_addr,
            top_eigvec_energy_payload=energy_pay,
            top_eigvec_energy_random=energy_rand,
        )
        np.savez(
            os.path.join(args.out, f"ov_L1_h{h}.npz"),
            U=U,
            S=S,
            Vh=Vh,
            payload_identity_cos=payload_identity_cos,
        )

        head_summary = {
            "qk_top_eigvals": eigvals[: args.topk].tolist(),
            "ov_top_singular": S[: args.topk].tolist(),
            "top_eigvec_energy_address_mean": float(energy_addr.mean()),
            "top_eigvec_energy_payload_mean": float(energy_pay.mean()),
            "top_eigvec_energy_random_mean": float(energy_rand.mean()),
            "payload_identity_cos_mean": float(payload_identity_cos.mean()),
        }
        summary["L1_heads"][f"h{h}"] = head_summary
        print(f"L1 H{h}: top-{args.topk} QK eigvec projection onto")
        print(f"   address basis : {energy_addr.mean():.3f}")
        print(f"   payload basis : {energy_pay.mean():.3f}")
        print(f"   random  basis : {energy_rand.mean():.3f}")
        print(f"   payload identity cos (mean): {payload_identity_cos.mean():.3f}")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
