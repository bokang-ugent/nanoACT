"""Train ACT on a pinned ALOHA demonstration set.

    python -m nanoact.train --steps 100000 --seed 1000 --out outputs/nanoact_seed1000
    python -m nanoact.train --dataset transfer_cube_human --steps 100000 --seed 1000 ...

The learning rate is constant, as in LeRobot's ACT recipe.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from nanoact.datasets import add_argument, get

from .data import Batches
from .model import ACT, ACTConfig, compute_loss


def git_commit() -> str:
    """Commit that produced this run.

    A source export can provide a nanoact/COMMIT file when Git is unavailable.
    A wheel installed inside another repository must not inherit its HEAD.
    Source hashes in the checkpoint also identify uncommitted code.
    """
    here = Path(__file__).resolve().parent
    if (here.parent / ".git").exists():
        try:
            out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=here.parent)
            if out.returncode == 0:
                return out.stdout.strip()
        except OSError:
            pass
    stamp = here / "COMMIT"
    return stamp.read_text().strip() if stamp.exists() else "unknown"


def save(path: Path, model: ACT, cfg: ACTConfig, sidecar: dict, meta: dict) -> None:
    """Weights plus everything needed to reproduce and to run inference without the dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": vars(cfg), "stats": sidecar["stats"],
                "data_repo_id": sidecar["repo_id"], "data_revision": sidecar["revision"],
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
                **meta}, path)


def main() -> None:
    p = argparse.ArgumentParser()
    add_argument(p)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", type=Path, default=Path("outputs/nanoact"))
    p.add_argument("--save-every", type=int, default=25_000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mp4", action="store_true", help="decode video during training instead of using cached images")
    p.add_argument("--no-vae", action="store_true", help="the CVAE ablation: encoder bypassed, latent pinned to zero, no KL term")
    p.add_argument("--noise-latent", action="store_true",
                   help="no encoder, N(0,I) into the latent token at training, zeros at "
                        "inference")
    p.add_argument("--grad-trace", action="store_true",
                   help="write per-step pre-clip gradient norms by group and the applied clip "
                        "scale to grad_trace.jsonl in --out")
    p.add_argument("--clip-scope", choices=("global", "trunk"), default="global",
                   help="global = the recipe's single clip over all params; trunk = encoder and "
                        "trunk clipped by their own norms, severing the shared-clip coupling")
    p.add_argument("--free-bits", type=float, default=ACTConfig.free_bits,
                   help="per-dimension KL floor in nats; 0 = the standard objective")
    p.add_argument("--kl-weight", type=float, default=ACTConfig.kl_weight,
                   help="weight on the KL penalty (default: 10)")
    p.add_argument("--encoder-sees-image", action="store_true",
                   help="give the CVAE encoder the decoder's backbone feature map, detached and "
                        "pooled to 12 tokens; off by default")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    cfg = ACTConfig(use_vae=not (args.no_vae or args.noise_latent),
                    noise_latent=args.noise_latent,
                    kl_weight=args.kl_weight, free_bits=args.free_bits,
                    encoder_sees_image=args.encoder_sees_image)
    batches = Batches(get(args.dataset), cfg.chunk_size, cfg.batch_size, args.seed,
                      args.device, use_mp4=args.mp4)

    model = ACT(cfg).to(args.device)
    model.backbone.load_imagenet_weights()
    model.train()

    # Two groups so the backbone LR can be tuned separately, as in the original.
    backbone = {id(q) for q in model.backbone.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {"params": [q for q in model.parameters() if id(q) not in backbone], "lr": cfg.lr},
            {"params": list(model.backbone.parameters()), "lr": cfg.lr_backbone},
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # The global clip is the only per-step channel coupling encoder gradients to trunk
    # updates (per-param AdamW, fp32, frozen BN, no weight sharing) — so it is the one
    # thing worth tracing, and the one coupling --clip-scope trunk can sever. Norms must
    # be read before clip_grad_norm_, which scales grads in place.
    enc_params = [q for n, q in model.named_parameters() if n.startswith("vae_encoder")]
    rest_params = [q for n, q in model.named_parameters() if not n.startswith("vae_encoder")]
    bb_params = list(model.backbone.parameters())
    trunk_params = [q for q in rest_params if id(q) not in backbone]
    trace = None
    if args.grad_trace:
        args.out.mkdir(parents=True, exist_ok=True)
        trace = (args.out / "grad_trace.jsonl").open("w")

    def group_norm(params: list) -> float:
        grads = [q.grad for q in params if q.grad is not None]
        if not grads:
            return 0.0
        return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads])).item()

    stream, start = batches.stream(), time.time()
    for step in range(1, args.steps + 1):
        loss, l1, kl = compute_loss(model, *next(stream))
        if not loss.isfinite():
            raise SystemExit(f"non-finite loss at step {step} (l1 {l1}, kl {kl}) — aborting")
        loss.backward()
        if trace is not None:
            pre = {"enc": group_norm(enc_params), "bb": group_norm(bb_params),
                   "trunk": group_norm(trunk_params)}
        if args.clip_scope == "trunk":
            total = torch.nn.utils.clip_grad_norm_(rest_params, cfg.grad_clip_norm)
            if enc_params:
                torch.nn.utils.clip_grad_norm_(enc_params, cfg.grad_clip_norm)
        else:
            total = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        if trace is not None:
            t = total.item()
            scale = 1.0 if t <= cfg.grad_clip_norm else cfg.grad_clip_norm / t
            trace.write(json.dumps({"step": step, "l1": round(float(l1), 6),
                                    "kl": round(float(kl), 6),
                                    **{k: round(v, 4) for k, v in pre.items()},
                                    "total": round(t, 4), "scale": round(scale, 6)}) + "\n")
            if step % args.log_every == 0:
                trace.flush()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - start
            print(f"step {step}/{args.steps}  loss {loss.item():.4f}  l1 {l1:.4f}  kl {kl:.4f}  "
                  f"{elapsed:.0f}s  ({step / elapsed:.1f} it/s)", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            meta = {"seed": args.seed, "steps": step, "wall_seconds": round(time.time() - start, 1),
                    "git_commit": git_commit(), "mp4_fallback": args.mp4}
            save(args.out / f"step_{step:06d}.pt", model, cfg, batches.sidecar, meta)
            print(f"saved {args.out / f'step_{step:06d}.pt'}  ({meta['wall_seconds']}s)", flush=True)

    print(json.dumps({"dataset": args.dataset, "steps": args.steps, "seed": args.seed,
                      "wall_seconds": round(time.time() - start, 1)}))


if __name__ == "__main__":
    main()
