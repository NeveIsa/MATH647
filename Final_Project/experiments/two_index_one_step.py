#!/usr/bin/env python3
"""
Experiment 3 (Dandi et al.): two-index teacher, *one* giant full-batch step on W.

Aligned with the single-step batch-size story (Thms.~4–5): with $r>1$ and $n \\sim \\Theta(d)$,
feature overlap should not fully specialize across *both* unknown directions in one update.
We report max-row alignment to each orthonormal teacher separately.

Training helpers live in one_step_vs_rf (paper init, η = c p √(n/d), ridge readout).
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

import one_step_vs_rf as d


def hermite2(t: np.ndarray) -> np.ndarray:
    return t * t - 1.0


def orthonormal_teachers(d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    w1 = rng.standard_normal(d)
    w1 /= np.linalg.norm(w1)
    w2 = rng.standard_normal(d)
    w2 -= w1 * float(np.dot(w2, w1))
    w2 /= np.linalg.norm(w2)
    return w1, w2


def max_row_cosine(W: np.ndarray, w: np.ndarray) -> float:
    norms = np.maximum(np.linalg.norm(W, axis=1), 1e-12)
    return float(np.max(np.abs(W @ w) / norms))


def save_figure(
    *,
    results: dict,
    fz_text: str,
    title_suffix: str,
    out_path: Path,
    noise_labels: list[tuple[str, str]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [r"$w_1^\star$", r"$w_2^\star$"]
    rows = [("frozen", "frozen $W$"), ("step", "one step on $W$")]
    base_cols = [("iid", "Eq.(10) i.i.d. init"), ("sym", "Eq.(10)+(11) symmetric init")]
    cols = []
    for mode_key, mode_title in base_cols:
        for noise_key, noise_title in noise_labels:
            cols.append((mode_key, mode_title, noise_key, noise_title))
    fig, axes = plt.subplots(2, len(cols), figsize=(5 * len(cols), 7))

    for r_idx, (row_key, row_title) in enumerate(rows):
        for c_idx, (col_key, col_title, noise_key, noise_title) in enumerate(cols):
            ax = axes[r_idx, c_idx]
            bucket = results[col_key][noise_key]
            vals = bucket["frozen"] if row_key == "frozen" else bucket["step"]
            m1 = statistics.mean(vals["a1"])
            m2 = statistics.mean(vals["a2"])
            e1 = statistics.pstdev(vals["a1"])
            e2 = statistics.pstdev(vals["a2"])
            ax.bar([0, 1], [m1, m2], width=0.55, yerr=[e1, e2], capsize=4, color=["#4C72B0", "#55A868"])
            ax.set_xticks([0, 1])
            ax.set_xticklabels(labels)
            ax.set_ylim(0.0, max(0.45, max(m1, m2, 0.1) * 1.35))
            ax.set_ylabel(r"$\max_i |\cos(w_i, w^\star)|$")
            ax.set_title(f"{row_title} | {col_title} | {noise_title}", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            ax.text(
                0.5,
                0.79,
                fz_text,
                transform=ax.transAxes,
                ha="center",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7", "pad": 2},
            )
            asym = abs(m1 - m2)
            ax.text(
                0.5,
                0.92,
                rf"$|a_1 - a_2| = {asym:.3f}$",
                transform=ax.transAxes,
                ha="center",
                fontsize=9,
            )

    fig.suptitle(f"Two-index teacher, single giant step — init comparison\n{title_suffix}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Two-index teacher, one GD step vs frozen W.")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n", type=int, default=None, help="default: 8*d")
    ap.add_argument("--n-test", type=int, default=4096)
    ap.add_argument("--p", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--paper-lr-c", type=float, default=5.0)
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--noise-std", type=float, default=0.2)
    ap.add_argument(
        "--compare-noise",
        action="store_true",
        help="run both clean and noisy labels in one run and show both in same plot",
    )
    ap.add_argument(
        "--noise-grid",
        type=str,
        default="",
        help="comma-separated sigma values for noise sweep, e.g. 0,0.05,0.1,0.2,0.4",
    )
    ap.add_argument(
        "--symmetric-target",
        action="store_true",
        help="y = He_2(t1)+He_2(t2); default is weighted (stronger along w1*)",
    )
    ap.add_argument(
        "--plot-dir",
        type=str,
        default="",
        help="PNG path directory (default: experiments/figures alongside script); '-' disables",
    )
    args = ap.parse_args()

    n = args.n if args.n is not None else 8 * args.d
    if args.plot_dir == "-":
        plot_dir = None
    elif args.plot_dir:
        plot_dir = Path(args.plot_dir).resolve()
    else:
        plot_dir = Path(__file__).resolve().parent / "figures"

    eta = d.learning_rate_paper(args.p, n, args.d, args.paper_lr_c)

    wt = (
        "f*(z) = (<w1*, z>)^2 + (<w2*, z>)^2 - 2"
        if args.symmetric_target
        else "f*(z) = (<w1*, z>)^2 + 0.35(<w2*, z>)^2 - 1.35"
    )
    wt_plot = (
        r"$f^\star(z)=(\langle w_1^\star,z\rangle)^2+(\langle w_2^\star,z\rangle)^2-2$"
        if args.symmetric_target
        else r"$f^\star(z)=(\langle w_1^\star,z\rangle)^2+0.35(\langle w_2^\star,z\rangle)^2-1.35$"
    )
    print(
        f"Two-index experiment: d={args.d}, n={n}, p={args.p}, n_test={args.n_test}, "
        f"η={args.paper_lr_c}·p·√(n/d)={eta:.3f}, ridge={args.ridge}, seeds={args.seeds}"
    )
    print(f"Target: {wt}, with orthonormal w1*, w2*.")
    if args.compare_noise:
        print(f"Noise comparison enabled: epsilon ~ N(0, {args.noise_std}^2)")
    if args.noise_grid:
        print(f"Noise sweep requested: {args.noise_grid}")

    if args.noise_grid:
        vals = [float(x.strip()) for x in args.noise_grid.split(",") if x.strip()]
        noise_keys = []
        for s in vals:
            key = f"sigma_{str(s).replace('.', 'p')}"
            if s == 0.0:
                noise_keys.append((key, r"clean labels ($\sigma=0$)", 0.0))
            else:
                noise_keys.append((key, rf"noisy labels ($\sigma={s}$)", s))
    else:
        noise_keys = [("clean", "clean labels", 0.0)]
        if args.compare_noise:
            noise_keys.append(("noisy", rf"noisy labels ($\sigma={args.noise_std}$)", args.noise_std))
    results = {
        "iid": {
            nk: {"frozen": {"a1": [], "a2": [], "mse": []}, "step": {"a1": [], "a2": [], "mse": []}}
            for nk, _, _ in noise_keys
        },
        "sym": {
            nk: {"frozen": {"a1": [], "a2": [], "mse": []}, "step": {"a1": [], "a2": [], "mse": []}}
            for nk, _, _ in noise_keys
        },
    }

    for seed in range(args.seeds):
        rng = np.random.default_rng(20_000 + seed)
        w1, w2 = orthonormal_teachers(args.d, rng)
        Z_tr = rng.standard_normal((n, args.d))
        Z_te = rng.standard_normal((args.n_test, args.d))
        t1_tr, t1_te = Z_tr @ w1, Z_te @ w1
        t2_tr, t2_te = Z_tr @ w2, Z_te @ w2

        if args.symmetric_target:
            y_tr = hermite2(t1_tr) + hermite2(t2_tr)
            y_te = hermite2(t1_te) + hermite2(t2_te)
        else:
            y_tr = hermite2(t1_tr) + 0.35 * hermite2(t2_tr)
            y_te = hermite2(t1_te) + 0.35 * hermite2(t2_te)

        modes = [("iid", d.init_weights_paper), ("sym", d.init_weights_paper_symmetric)]

        for mode_key, init_fn in modes:
            for noise_key, _noise_title, sigma in noise_keys:
                if sigma > 0:
                    eps_tr = sigma * rng.standard_normal(y_tr.shape)
                    eps_te = sigma * rng.standard_normal(y_te.shape)
                    ytr_use = y_tr + eps_tr
                    yte_use = y_te + eps_te
                else:
                    ytr_use = y_tr
                    yte_use = y_te
                snap = rng.bit_generator.state

                def run_once(do_step: bool) -> tuple[float, float, float]:
                    rng.bit_generator.state = snap
                    W0, a0 = init_fn(args.p, args.d, rng)
                    W = W0
                    if do_step:
                        W = d.one_step_W(W0, a0, Z_tr, ytr_use, eta, normalize_grad=False)
                    af = d.fit_second_layer_ridge(W, Z_tr, ytr_use, ridge=args.ridge)
                    pred = d.forward(W, af, Z_te)
                    mse = float(np.mean((pred - yte_use) ** 2))
                    return mse, max_row_cosine(W, w1), max_row_cosine(W, w2)

                mf, f1, f2 = run_once(False)
                ms, s1, s2 = run_once(True)
                results[mode_key][noise_key]["frozen"]["mse"].append(mf)
                results[mode_key][noise_key]["frozen"]["a1"].append(f1)
                results[mode_key][noise_key]["frozen"]["a2"].append(f2)
                results[mode_key][noise_key]["step"]["mse"].append(ms)
                results[mode_key][noise_key]["step"]["a1"].append(s1)
                results[mode_key][noise_key]["step"]["a2"].append(s2)

    def line(name: str, bucket: dict, mode_title: str) -> None:
        print(f"\n=== {name} [{mode_title}] ===")
        print(
            f"  test MSE = {statistics.mean(bucket['mse']):.4f} ± {statistics.pstdev(bucket['mse']):.4f}"
        )
        print(
            f"  max |cos|, w1* = {statistics.mean(bucket['a1']):.4f} ± {statistics.pstdev(bucket['a1']):.4f}"
        )
        print(
            f"  max |cos|, w2* = {statistics.mean(bucket['a2']):.4f} ± {statistics.pstdev(bucket['a2']):.4f}"
        )
        asym = [
            abs(bucket["a1"][i] - bucket["a2"][i]) for i in range(len(bucket["a1"]))
        ]
        print(
            f"  |align(w1*) - align(w2*)| = {statistics.mean(asym):.4f} ± {statistics.pstdev(asym):.4f}"
        )

    for noise_key, noise_title, _sigma in noise_keys:
        for mode_key, mode_title in [("iid", "Eq.(10) i.i.d."), ("sym", "Eq.(10)+(11) symmetric")]:
            line("frozen W", results[mode_key][noise_key]["frozen"], f"{mode_title} | {noise_title}")
            line("one step on W", results[mode_key][noise_key]["step"], f"{mode_title} | {noise_title}")
            d_as_f = statistics.mean(
                [
                    abs(results[mode_key][noise_key]["frozen"]["a1"][i] - results[mode_key][noise_key]["frozen"]["a2"][i])
                    for i in range(args.seeds)
                ]
            )
            d_as_s = statistics.mean(
                [
                    abs(results[mode_key][noise_key]["step"]["a1"][i] - results[mode_key][noise_key]["step"]["a2"][i])
                    for i in range(args.seeds)
                ]
            )
            print(f"  [{mode_title} | {noise_title}] mean |a1-a2|: frozen {d_as_f:.4f} → one step {d_as_s:.4f}")

    if plot_dir is not None:
        out = plot_dir / "two_index_one_step_alignments.png"
        try:
            save_figure(
                results=results,
                fz_text=wt_plot,
                title_suffix=(
                    rf"$d={args.d},\ n={n},\ p={args.p},\ \eta={args.paper_lr_c}p\sqrt{{n/d}}$"
                ),
                out_path=out,
                noise_labels=[(k, t) for k, t, _ in noise_keys],
            )
            print(f"\nSaved plot: {out}")
        except ImportError:
            print("Matplotlib missing; skip figure.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
