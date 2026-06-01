#!/usr/bin/env python3
"""
Fig.~3-style toy experiment for Dandi et al. (JMLR 2024):
one full-batch gradient step on W vs frozen W, with the second layer fit by ridge
regression (proxy for training a to convergence).

Case A: probabilists' Hermite H_2(t) = t^2 - 1  on t = <w*, z>
Case B: H_4(t) = t^4 - 6 t^2 + 3  (higher Hermite degree / harder early alignment)

NumPy for the experiment; Matplotlib only if you save figures (--plot-dir).

Learning rate (Dandi et al., Alg.~1 / Appendix~A): theory and experiments use
``η = O(p √(n/d))``; Fig.~3 uses ``η = 5 p √(n/d)`` with second-layer ridge ``λ = 10⁻⁶``.
Eq.~(10) init: ``w_i^0`` uniform on the sphere, ``√p · a_i^0 ~ Unif([-1,1])``.
Use ``--paper`` to match that scaling (default ``--lr`` / unit-norm grad is only a heuristic).
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np


def hermite2(t: np.ndarray) -> np.ndarray:
    return t * t - 1.0


def hermite4(t: np.ndarray) -> np.ndarray:
    return t**4 - 6.0 * t**2 + 3.0


def sample_teacher(d: int, rng: np.random.Generator) -> np.ndarray:
    w = rng.standard_normal(d)
    return w / np.linalg.norm(w)


def init_weights_heuristic(p: int, d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian rows scaled like 1/sqrt(d); a ~ 0.5 * N(0, I) (older toy default)."""
    W = rng.standard_normal((p, d)) / np.sqrt(d)
    a = rng.standard_normal(p) * 0.5
    return W, a


def init_weights_paper(p: int, d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Eq.~(10): w_i uniform on S^{d-1}; √p · a_i ~ Unif([-1, 1])."""
    W = rng.standard_normal((p, d))
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    a = rng.uniform(-1.0, 1.0, size=p) / np.sqrt(p)
    return W, a


def init_weights_paper_symmetric(
    p: int, d: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Eq.~(10) + Eq.~(11): pair neurons with opposite second-layer signs."""
    if p % 2 != 0:
        raise ValueError("Symmetric initialization requires even p.")
    h = p // 2
    W_half = rng.standard_normal((h, d))
    W_half = W_half / np.linalg.norm(W_half, axis=1, keepdims=True)
    a_half = rng.uniform(-1.0, 1.0, size=h) / np.sqrt(p)
    W = np.concatenate([W_half, W_half], axis=0)
    a = np.concatenate([a_half, -a_half], axis=0)
    return W, a


def learning_rate_paper(p: int, n: int, d: int, c: float) -> float:
    """Appendix~A / Fig.~3: η = c · p · √(n/d) with c ≈ 5 in the Fig.~3 caption."""
    return float(c * p * np.sqrt(n / d))


def forward(W: np.ndarray, a: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """f(Z) = (1/sqrt(p)) sum_i a_i relu(<w_i, Z>) — shape (n,)."""
    p = W.shape[0]
    phi = np.maximum(0.0, Z @ W.T)
    return (phi @ a) / np.sqrt(p)


def grad_W_mse(W: np.ndarray, a: np.ndarray, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of mean squared error w.r.t. W for fixed a; W has shape (p, d)."""
    n, d = Z.shape
    p = W.shape[0]
    pre = Z @ W.T
    phi = np.maximum(0.0, pre)
    f = (phi @ a) / np.sqrt(p)
    r = f - y
    dphi = (2.0 / n) * r[:, np.newaxis] * (a[np.newaxis, :] / np.sqrt(p))
    mask = (pre > 0).astype(np.float64)
    dpre = dphi * mask
    # d/dW: row j gets sum_i dpre[i,j] * Z[i, :]
    return dpre.T @ Z


def one_step_W(
    W: np.ndarray,
    a: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    lr: float,
    normalize_grad: bool,
) -> np.ndarray:
    g = grad_W_mse(W, a, Z, y)
    if normalize_grad:
        nrm = np.linalg.norm(g)
        if nrm > 0:
            g = g * (1.0 / nrm)
    return W - lr * g


def fit_second_layer_ridge(W: np.ndarray, Z: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    p = W.shape[0]
    phi = np.maximum(0.0, Z @ W.T)
    X = phi / np.sqrt(p)
    h = X.T @ X + ridge * np.eye(p)
    return np.linalg.solve(h, X.T @ y)


def max_row_alignment(W: np.ndarray, w_star: np.ndarray) -> float:
    norms = np.maximum(np.linalg.norm(W, axis=1), 1e-12)
    return float(np.max(np.abs(W @ w_star) / norms))


def save_figures(
    *,
    results: dict,
    out_dir: Path,
    title_suffix: str,
    noise_labels: list[tuple[str, str]],
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("A", r"quadratic target", r"$f^\star(z)=(\langle w^\star,z\rangle)^2-1$"),
        (
            "B",
            r"quartic target",
            r"$f^\star(z)=(\langle w^\star,z\rangle)^4-6(\langle w^\star,z\rangle)^2+3$",
        ),
    ]
    init_modes = [("iid", "Eq.(10) i.i.d. init")]
    first_noise = noise_labels[0][0]
    if len(results["sym"][first_noise]["A"]["frozen"]) > 0:
        init_modes.append(("sym", "Eq.(10)+(11) symmetric init"))
    plot_cols: list[tuple[str, str, str, str]] = []
    for mode_key, mode_title in init_modes:
        for noise_key, noise_title in noise_labels:
            plot_cols.append((mode_key, mode_title, noise_key, noise_title))
    labels_m = ["frozen $W$", "one step on $W$"]

    fig_m, axes_m = plt.subplots(2, len(plot_cols), figsize=(5 * len(plot_cols), 7))
    for r_idx, (case_key, case_title, fz_text) in enumerate(cases):
        for c_idx, (mode_key, mode_title, noise_key, noise_title) in enumerate(plot_cols):
            ax_m = axes_m[r_idx, c_idx]
            bucket = results[mode_key][noise_key][case_key]
            mf = statistics.mean(bucket["frozen"])
            ms = statistics.mean(bucket["step"])
            ef = statistics.pstdev(bucket["frozen"])
            es = statistics.pstdev(bucket["step"])
            ax_m.bar(
                [0, 1],
                [mf, ms],
                width=0.55,
                yerr=[ef, es],
                capsize=4,
                color=["#4C72B0", "#DD8452"],
            )
            ax_m.set_xticks([0, 1])
            ax_m.set_xticklabels(labels_m, rotation=15, ha="right")
            ax_m.set_ylabel("test MSE")
            ax_m.set_title(f"{case_title} | {mode_title} | {noise_title}", fontsize=9)
            ax_m.text(
                0.5,
                0.80,
                fz_text,
                transform=ax_m.transAxes,
                ha="center",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7", "pad": 2},
            )
            ax_m.grid(axis="y", alpha=0.3)
            imp = (mf - ms) / mf * 100 if mf > 0 else 0.0
            ax_m.text(
                0.5,
                0.92,
                rf"$\Delta$ MSE ${imp:+.1f}\%$",
                transform=ax_m.transAxes,
                ha="center",
                fontsize=9,
            )
    fig_m.suptitle(f"Test MSE comparison by initialization\n{title_suffix}", fontsize=11)
    fig_m.tight_layout()
    path_m = out_dir / "one_step_vs_rf_mse.png"
    fig_m.savefig(path_m, dpi=150)
    plt.close(fig_m)

    fig_a, axes_a = plt.subplots(2, len(plot_cols), figsize=(5 * len(plot_cols), 7))
    for r_idx, (case_key, case_title, fz_text) in enumerate(cases):
        for c_idx, (mode_key, mode_title, noise_key, noise_title) in enumerate(plot_cols):
            ax_a = axes_a[r_idx, c_idx]
            bucket = results[mode_key][noise_key][case_key]
            af = statistics.mean(bucket["align_f"])
            a_s = statistics.mean(bucket["align_s"])
            ef = statistics.pstdev(bucket["align_f"])
            es = statistics.pstdev(bucket["align_s"])
            ax_a.bar(
                [0, 1],
                [af, a_s],
                width=0.55,
                yerr=[ef, es],
                capsize=4,
                color=["#4C72B0", "#DD8452"],
            )
            ax_a.set_xticks([0, 1])
            ax_a.set_xticklabels(labels_m, rotation=15, ha="right")
            ax_a.set_ylabel(r"max$_i\, |\cos(w_i, w^\star)|$")
            ax_a.set_title(f"{case_title} | {mode_title} | {noise_title}", fontsize=9)
            ax_a.text(
                0.5,
                0.80,
                fz_text,
                transform=ax_a.transAxes,
                ha="center",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7", "pad": 2},
            )
            ax_a.grid(axis="y", alpha=0.3)
            ax_a.text(
                0.5,
                0.92,
                rf"$\Delta\cos$ ${a_s - af:+.3f}$",
                transform=ax_a.transAxes,
                ha="center",
                fontsize=9,
            )
    fig_a.suptitle(f"Teacher alignment comparison by initialization\n{title_suffix}", fontsize=11)
    fig_a.tight_layout()
    path_a = out_dir / "one_step_vs_rf_alignment.png"
    fig_a.savefig(path_a, dpi=150)
    plt.close(fig_a)

    return path_m, path_a


def main() -> int:
    ap = argparse.ArgumentParser(description="One GD step on W vs RF baseline (ridge readout).")
    ap.add_argument("--d", type=int, default=128, help="input dimension")
    ap.add_argument("--n", type=int, default=None, help="train batch size (default: 8*d)")
    ap.add_argument("--n-test", type=int, default=4096, help="test sample size")
    ap.add_argument("--p", type=int, default=512, help="hidden width")
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (mean ± std)")
    ap.add_argument(
        "--paper",
        action="store_true",
        help="paper scaling: Eq.~(10) init, η = --paper-lr-c · p · √(n/d), raw batch gradient (no unit norm)",
    )
    ap.add_argument(
        "--paper-lr-c",
        type=float,
        default=5.0,
        help="constant c in η = c · p · √(n/d) (Fig.~3 uses c = 5)",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1.0,
        help="step size when not --paper (after optional unit normalization unless --raw-grad)",
    )
    ap.add_argument(
        "--raw-grad",
        action="store_true",
        help="use unnormalized MSE gradient (required for meaningful --lr when not --paper)",
    )
    ap.add_argument(
        "--ridge",
        type=float,
        default=1e-3,
        help="ridge λ for readout a (Fig.~3 uses 1e-6; Appendix A often uses λ = 1)",
    )
    ap.add_argument(
        "--noise-std",
        type=float,
        default=0.2,
        help="std of additive Gaussian label noise (used only with --compare-noise)",
    )
    ap.add_argument(
        "--compare-noise",
        action="store_true",
        help="run both clean and noisy labels in one run and show both in same plots",
    )
    ap.add_argument(
        "--noise-grid",
        type=str,
        default="",
        help="comma-separated sigma values for noise sweep, e.g. 0,0.05,0.1,0.2,0.4",
    )
    ap.add_argument(
        "--plot-dir",
        type=str,
        default="",
        help="if set (default: experiments/figures next to this script), save PNG plots here; "
        "use --plot-dir - to disable",
    )
    args = ap.parse_args()

    n = args.n if args.n is not None else 8 * args.d

    if args.paper:
        lr_used = learning_rate_paper(args.p, n, args.d, args.paper_lr_c)
        normalize_grad = False
        init_fn = init_weights_paper
    else:
        lr_used = args.lr
        normalize_grad = not args.raw_grad
        init_fn = init_weights_heuristic

    if args.plot_dir == "-":
        plot_dir: Path | None = None
    elif args.plot_dir:
        plot_dir = Path(args.plot_dir).resolve()
    else:
        plot_dir = Path(__file__).resolve().parent / "figures"

    if args.paper:
        print(
            f"Setup (paper scaling): d={args.d}, n_train={n}, n_test={args.n_test}, p={args.p}, "
            f"η = {args.paper_lr_c}·p·√(n/d) = {lr_used:.4f}, raw grad, Eq.(10) init, ridge={args.ridge}, seeds={args.seeds}"
        )
    else:
        print(
            f"Setup (toy defaults): d={args.d}, n_train={n}, n_test={args.n_test}, p={args.p}, "
            f"lr={args.lr}, unit_norm_grad={normalize_grad}, ridge={args.ridge}, seeds={args.seeds}"
        )
    print("Targets:")
    print("  Case A: f*(z) = (<w*, z>)^2 - 1")
    print("  Case B: f*(z) = (<w*, z>)^4 - 6(<w*, z>)^2 + 3")
    if args.compare_noise:
        print(f"Noise comparison enabled: epsilon ~ N(0, {args.noise_std}^2)")
    if args.noise_grid:
        print(f"Noise sweep requested: {args.noise_grid}")

    def new_bucket() -> dict:
        return {"frozen": [], "step": [], "align_f": [], "align_s": []}

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
        "iid": {nk: {"A": new_bucket(), "B": new_bucket()} for nk, _, _ in noise_keys},
        "sym": {nk: {"A": new_bucket(), "B": new_bucket()} for nk, _, _ in noise_keys},
    }

    for seed in range(args.seeds):
        rng = np.random.default_rng(10_000 + seed)

        w_star = sample_teacher(args.d, rng)
        Z_train = rng.standard_normal((n, args.d))
        Z_test = rng.standard_normal((args.n_test, args.d))
        t_train = Z_train @ w_star
        t_test = Z_test @ w_star

        cases = [
            ("A", hermite2(t_train), hermite2(t_test)),
            ("B", hermite4(t_train), hermite4(t_test)),
        ]
        modes = [("iid", init_fn)]
        if args.paper:
            modes.append(("sym", init_weights_paper_symmetric))

        for case_key, y_train_clean, y_test_clean in cases:
            snap_case = rng.bit_generator.state
            for mode_key, mode_init_fn in modes:
                for noise_key, _noise_title, sigma in noise_keys:
                    rng.bit_generator.state = snap_case
                    if sigma > 0:
                        eps_tr = sigma * rng.standard_normal(y_train_clean.shape)
                        eps_te = sigma * rng.standard_normal(y_test_clean.shape)
                        y_train = y_train_clean + eps_tr
                        y_test = y_test_clean + eps_te
                    else:
                        y_train = y_train_clean
                        y_test = y_test_clean
                    snap_mode = rng.bit_generator.state

                    def run_once(frozen: bool) -> tuple[float, float]:
                        rng.bit_generator.state = snap_mode
                        W, a = mode_init_fn(args.p, args.d, rng)
                        if not frozen:
                            W = one_step_W(W, a, Z_train, y_train, lr_used, normalize_grad=normalize_grad)
                        a_fit = fit_second_layer_ridge(W, Z_train, y_train, ridge=args.ridge)
                        pred = forward(W, a_fit, Z_test)
                        mse = float(np.mean((pred - y_test) ** 2))
                        align = max_row_alignment(W, w_star)
                        return mse, align

                    mf, af = run_once(True)
                    ms, a_s = run_once(False)
                    bucket = results[mode_key][noise_key][case_key]
                    bucket["frozen"].append(mf)
                    bucket["step"].append(ms)
                    bucket["align_f"].append(af)
                    bucket["align_s"].append(a_s)

    def summarize(name: str, bucket: dict, mode_title: str) -> None:
        print(f"\n=== {name} [{mode_title}] (mean ± std over seeds) ===")
        print(
            f"  frozen W        test MSE = {statistics.mean(bucket['frozen']):.6f} ± "
            f"{statistics.pstdev(bucket['frozen']):.6f}   "
            f"max |cos(row, w*)| = {statistics.mean(bucket['align_f']):.4f}"
        )
        print(
            f"  one step on W   test MSE = {statistics.mean(bucket['step']):.6f} ± "
            f"{statistics.pstdev(bucket['step']):.6f}   "
            f"max |cos(row, w*)| = {statistics.mean(bucket['align_s']):.4f}"
        )
        mf = statistics.mean(bucket["frozen"])
        ms = statistics.mean(bucket["step"])
        if mf > 0:
            imp = (mf - ms) / mf * 100
            print(f"  relative MSE change (frozen → one step): {imp:+.1f}%")

    for noise_key, noise_title, _sigma in noise_keys:
        summarize("Case A (H_2)", results["iid"][noise_key]["A"], f"Eq.(10) i.i.d. | {noise_title}")
        summarize("Case B (H_4)", results["iid"][noise_key]["B"], f"Eq.(10) i.i.d. | {noise_title}")
        if args.paper:
            summarize(
                "Case A (H_2)",
                results["sym"][noise_key]["A"],
                f"Eq.(10)+(11) symmetric | {noise_title}",
            )
            summarize(
                "Case B (H_4)",
                results["sym"][noise_key]["B"],
                f"Eq.(10)+(11) symmetric | {noise_title}",
            )

    if plot_dir is not None:
        if args.paper:
            suffix = (
                rf"$d={args.d}$, $n={n}$, $p={args.p}$, "
                rf"$\eta={args.paper_lr_c}p\sqrt{{n/d}}$, seeds={args.seeds}"
            )
        else:
            suffix = f"$d={args.d}$, $n={n}$, $p={args.p}$, seeds={args.seeds}"
        try:
            p_m, p_a = save_figures(
                results=results,
                out_dir=plot_dir,
                title_suffix=suffix,
                noise_labels=[(k, t) for k, t, _ in noise_keys],
            )
            print(f"\nSaved plots:\n  {p_m}\n  {p_a}")
        except ImportError:
            print("\nMatplotlib not installed; skipping figures (`pip install matplotlib`).", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
