"""Attack a batch of images with GAQA.

This is the only entry point of the repository: give it a victim checkpoint and
a number of images, and it writes ``results.csv`` (one row per image) plus
``summary.json`` (the ASR / MQ / AQ / NED / MCSD metrics of the paper).

Examples
--------
    python attack.py --model victim_cifar10.pt --num-images 5
    python attack.py --model victim_cifar10.pt --data cifar100 --eps 8/255 \
                     --iters 1000 --population 100 --output results/demo
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from dataset import CIFAR, load_cifar
from fitness import mcsd, normalised_distance
from gaqa import ABLATIONS, GAQA
from victim import Victim

DATASETS = CIFAR


def parse_eps(text: str) -> float:
    """Accept either ``8/255`` or a decimal."""
    if "/" in text:
        numerator, denominator = text.split("/")
        return float(numerator) / float(denominator)
    return float(text)


def load_images(name: str, root: str, split: str, start: int, count: int):
    images, labels = load_cifar(name, root, split, download=True)
    stop = min(start + count, len(images))
    for index in range(start, stop):
        yield index, images[index].astype(np.float32) / 255.0, int(labels[index])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GAQA black-box AE generation")
    parser.add_argument("--data", default="cifar10", choices=sorted(DATASETS),
                        help="benchmark dataset (default: cifar10)")
    parser.add_argument("--data-root", default="data", help="where the dataset is cached")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--start", type=int, default=0, help="index of the first image")
    parser.add_argument("--num-images", type=int, default=10, help="how many images to attack")
    parser.add_argument("--model", default="victim_cifar10.pt", help="victim checkpoint")
    parser.add_argument("--arch", default="resnet18", help="torchvision architecture of the victim")
    parser.add_argument("--eps", default="8/255", help="perturbation bound, e.g. 8/255")
    parser.add_argument("--iters", type=int, default=1000, help="iteration budget N (Table I)")
    parser.add_argument("--population", type=int, default=100, help="population size Q (Table I)")
    parser.add_argument("--rho", type=int, default=10, help="perturbation-varied children (Table I)")
    parser.add_argument("--nu", type=int, default=300, help="extra iterations after the first AE")
    parser.add_argument("--theta", type=int, default=1000, help="PSO-Jump threshold Theta")
    parser.add_argument("--psi", type=float, default=0.1,
                        help="minimal attack effectiveness psi (Eq. 6)")
    parser.add_argument("--selection-pi", type=int, default=8, help="selection exponent pi (Eq. 8)")
    parser.add_argument("--pso-iters", type=int, default=20, help="PSO-Jump steps U (Table I)")
    parser.add_argument("--pso-particles", type=int, default=10, help="PSO-Jump particles W")
    parser.add_argument("--c1", type=float, default=0.5, help="PSO coefficient C1 (Eq. 17)")
    parser.add_argument("--c2", type=float, default=0.5, help="PSO coefficient C2 (Eq. 17)")
    parser.add_argument("--keep-misclassified", action="store_true",
                        help="also attack images the victim already gets wrong")
    parser.add_argument("--ablate", nargs="*", default=[], choices=ABLATIONS,
                        help="switch off individual operators (Table II)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/run", help="output directory")
    parser.add_argument("--quiet", action="store_true", help="no per-iteration progress")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eps = parse_eps(args.eps)

    if args.data not in DATASETS:
        raise SystemExit(f"unsupported dataset {args.data}")
    num_classes = DATASETS[args.data]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    victim = Victim(checkpoint=args.model, arch=args.arch, num_classes=num_classes,
                    device=args.device)
    print(f"victim      : {args.arch} <- {args.model} ({num_classes} classes, {victim.device})")
    print(f"dataset     : {args.data}/{args.split}, images {args.start}..{args.start + args.num_images - 1}")
    print(f"budget      : eps={eps:.6f} N={args.iters} Q={args.population} rho={args.rho} "
          f"Theta={args.theta} nu={args.nu}")
    if args.ablate:
        print(f"ablations   : {', '.join(args.ablate)}")

    rows = []
    attacked = 0
    started = time.time()

    for index, image, label in load_images(args.data, args.data_root, args.split,
                                           args.start, args.num_images):
        queries_before = victim.queries
        clean_prediction = victim.predict(image)
        if clean_prediction != label and not args.keep_misclassified:
            print(f"[{index:5d}] skipped: the victim already predicts {clean_prediction} "
                  f"instead of {label}")
            continue

        attacked += 1
        runner = GAQA(victim, image, label, eps=eps, population=args.population,
                      rho=args.rho, theta=args.theta, iters=args.iters, nu=args.nu,
                      psi=args.psi, pso_iters=args.pso_iters,
                      pso_particles=args.pso_particles, c1=args.c1, c2=args.c2,
                      selection_pi=args.selection_pi, seed=args.seed + index,
                      ablate=tuple(args.ablate), verbose=not args.quiet)
        if not args.quiet:
            print(f"[{index:5d}] label={label} clean_pred={clean_prediction} attacking...")
        clock = time.time()
        result = runner.run()
        elapsed = time.time() - clock

        delta = result.delta[None]
        row = {
            "index": index,
            "label": label,
            "clean_prediction": clean_prediction,
            "adversarial_prediction": result.prediction,
            "success": int(result.success),
            "queries": victim.queries - queries_before,
            "queries_to_success": result.queries_to_success,
            "iterations": result.iterations,
            "success_iteration": result.success_iteration if result.success_iteration else "",
            "fitness": round(result.fitness, 6),
            "ned": round(float(normalised_distance(delta, eps)[0]), 6),
            "mcsd": round(float(mcsd(delta)[0]), 6),
            "seconds": round(elapsed, 1),
        }
        rows.append(row)
        print(f"[{index:5d}] {'SUCCESS' if result.success else 'failed ':<7} "
              f"queries={row['queries_to_success']:6d}/{row['queries']:6d} iters={row['iterations']:5d} "
              f"ned={row['ned']:.4f} mcsd={row['mcsd']:.4f} ({elapsed:.0f}s)")

    if not rows:
        print("no image was attacked")
        return 1

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    successes = [r for r in rows if r["success"]]
    # MQ/AQ of the paper count the queries spent until the first effective AE;
    # the evasiveness refinement that follows is reported separately.
    queries = np.array([r["queries_to_success"] for r in rows])
    total_queries = np.array([r["queries"] for r in rows])
    summary = {
        "dataset": args.data,
        "split": args.split,
        "arch": args.arch,
        "checkpoint": args.model,
        "ablation": args.ablate,
        "images_attacked": attacked,
        "successes": len(successes),
        "asr": round(len(successes) / attacked, 4),
        "median_queries": float(np.median(queries)),
        "average_queries": round(float(queries.mean()), 1),
        "median_queries_total": float(np.median(total_queries)),
        "average_queries_total": round(float(total_queries.mean()), 1),
        "ned": round(float(np.mean([r["ned"] for r in rows])), 6),
        "mcsd": round(float(np.mean([r["mcsd"] for r in rows])), 6),
        "eps": eps,
        "hyperparameters": {
            "iters": args.iters, "population": args.population, "rho": args.rho,
            "theta": args.theta, "nu": args.nu, "psi": args.psi,
            "pso_iters": args.pso_iters, "pso_particles": args.pso_particles,
            "c1": args.c1, "c2": args.c2, "selection_pi": args.selection_pi,
        },
        "wall_clock_seconds": round(time.time() - started, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"attacked   : {attacked}")
    print(f"ASR        : {summary['asr']:.2%}")
    print(f"MQ / AQ    : {summary['median_queries']:.0f} / {summary['average_queries']:.1f} "
          f"(first AE; {summary['median_queries_total']:.0f} / "
          f"{summary['average_queries_total']:.1f} including the evasiveness phase)")
    print(f"NED / MCSD : {summary['ned']:.4f} / {summary['mcsd']:.4f}")
    print(f"written    : {out_dir / 'results.csv'}, {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
