#import "@preview/clear-iclr:0.7.0": iclr2025
#import "clear-iclr/logo.typ": LaTeX, LaTeXe

/**
 * EE 592 submission draft — critical exposition of:
 * Dandi et al., JMLR 2024 / arXiv:2305.18270.
 * Template assets live in `clear-iclr/`; this file is your source.
 */
#let authors = (
  (
    names: ([Sampad B Mohanty],),
    affilation: [Computational Methods for Inverse Problems],
  ),
)

#show: iclr2025.with(
  title: [Critical exposition: How two-layer networks learn in one \
    (giant) gradient step],
  authors: authors,
  keywords: (
    "feature learning",
    "multi-index models",
    "two-layer neural networks",
    "high-dimensional asymptotics",
    "gradient descent",
  ),
  abstract: [
    This report studies *How Two-Layer Neural Networks Learn, One (Giant) Step at a Time*
    @dandi2024two (JMLR, 2024). The paper analyzes early training of a two-layer network on
    Gaussian inputs with multi-index labels, where all low-dimensional structure is encoded in
    a link function over unknown teacher projections. The core question is when a small number
    of large-batch gradient steps on first-layer weights yields nontrivial alignment with the
    teacher subspace $V^(star)$, thus departing from the initialization (lazy) kernel.

    Following @dandi2024two, I separate three ingredients---batch size $n$, number of gradient steps,
    and target structure (Hermite / leap / staircase)---and stress that they are not
    interchangeable. In one giant full-batch step, batch scaling $n =
    Theta(d)$ supports alignment but only to a single teacher direction unless $n$ reaches
    $Theta(d^2)$ scale for multi-direction specialization; still harder directions may demand
    $n = Theta(d^ell)$ with leap index $ell$. With multiple steps and fresh batches of size $n =
    Theta(d)$, targets satisfying a staircase coupling can progressively recruit several
    directions over time, while obstructed pieces retain $Theta(d^2)$ bottlenecks. Finally,
    @dandi2024two link these geometric facts to approximation and generalization relative to the
    initialization kernel---including settings where one gradient step helps or fails versus
    frozen features.

    Critical discussion of assumptions, limitations, and preliminary EE592 experiments is in
    @sec:critical; the technical overview
    begins in @sec:setting.
  ],
  bibliography: bibliography("references.bib"),
  appendix: [
    = Appendix

    == Reproducibility notes for preliminary experiments

    Scripts:
    - `experiments/one_step_vs_rf.py`
    - `experiments/two_index_one_step.py`

    Representative commands used for the figures included in this report:
    - `python experiments/one_step_vs_rf.py --paper --ridge 1e-6 --seeds 4 --noise-grid 0,0.05,0.1,0.2,0.4`
    - `python experiments/two_index_one_step.py --seeds 4 --noise-grid 0,0.05,0.1,0.2,0.4`

    Outputs:
    - `experiments/figures/one_step_vs_rf_mse.png`
    - `experiments/figures/one_step_vs_rf_alignment.png`
    - `experiments/figures/two_index_one_step_alignments.png`
  ],
  // `none` = show real author block (not the ICLR "Anonymous authors" placeholder used when
  // `accepted: false`) and omit the ICLR preprint header; see clear-iclr `make-authors`.
  accepted: none,
)

#let url(uri) = link(uri, raw(uri))

#align(center, block(spacing: 0.5em, width: 100%)[
  #set text(size: 9pt)
  *AI disclosure:* Generative AI tools were used for drafting and ideation; the author edited the
  text and takes responsibility for all content.
])
#v(0.35em)

= Setting and contributions <sec:setting>

Unless noted, asymptotic statements match @dandi2024two (their §1--2; large-$d$ scaling).

== Notation at a glance

#set text(size: 8.8pt)
#table(
  columns: (2fr, 6fr),
  stroke: 0.35pt,
  inset: 2pt,
  align: left,
  [*Symbol*], [*Meaning*],
  [$d$], [Input dimension ($z in RR^d$).],
  [$r$], [Number of teacher directions.],
  [$n$], [Batch size / samples per update stage.],
  [$p$], [Network width (hidden units).],
  [$W in RR^(p times d)$], [First-layer matrix; row $w_i$ is neuron-$i$ direction.],
  [$a in RR^p$], [Second-layer vector.],
  [$W^star$], [Teacher matrix (rows $w_j^star$).],
  [$f^(star), g^(star)$], [Target: $f^(star)(z)=g^(star)(W^(star) z)$.],
  [$V^(star)$], [Teacher span $op("span"){w_1^star, dots, w_r^star}$.],
  [$eta$], [Learning rate for $W$ updates.],
  [$ell$], [Leap index (first Hermite order with signal).],
  [$Theta(·)$], [Asymptotic scaling notation.]
)
#set text(size: 10pt)

== Problem setup <sec:setup>

*Data and target.* Covariates $z prop cal(N)(0, I_d)$ in $RR^d$. Labels follow a multi-index
representation: orthogonal teachers $w_1^(star), dots, w_r^(star)$ and a link $g^(star) : RR^r -> RR$
with
$
  y = f^(star)(z) := g^(star)( chevron.l w_1^(star), z chevron.r, dots.h, chevron.l w_r^(star), z chevron.r).
$
Isotropic inputs put *all* exploitable structure in $f^(star)$. That modeling choice is deliberate:
classical kernels depend on isotropic geometry of $z$ but do not automatically exploit hidden
low-dimensional structure in $y$, so improvements over kernel behavior here are evidence of
*representation learning* rather than an artifact of input clustering @dandi2024two.

*Two-layer predictor.* First- and second-layer parameters $W in RR^(p times d)$, $a in RR^p$, and
activation $sigma$,
$
  hat(f)(z; W, a)
  = 1/sqrt(p) sum_(i=1)^p a_i thin sigma( chevron.l w_i, z chevron.r).
$

*Training regime analyzed.* @dandi2024two focus on the *early* phase: one or more full-batch
gradient steps on $W$ under squared loss, with batch size $n$ coupled to $d$ (typically $n =
Theta(d)$ versus $Theta(d^2)$ comparisons) and learning-rate scaling tied to width $p$ as in their
theorems. For multi-step analyses they use *fresh i.i.d. batches* each iteration and (like related
work) treat second-layer training separately---idealizations that simplify the Hermite and
concentration analysis @dandi2024two.

*Operational notion of learning.* Beating the kernel at initialization means growing overlap between
rows of $W$ and $V^(star) := op("span"){w_1^(star), dots, w_r^(star)}$. Whether overlap arises, and in how
many orthogonal directions, is driven jointly by $(n, "steps")$ and the Hermite/*leap*/*staircase*
structure of $g^(star)$ @dandi2024two.

== Main contributions <sec:contributions>

=== A single organizing tension

@dandi2024two argue that *specialization*---encoding distinct teacher directions in distinct hidden
units---is not a monotone consequence of ``more data.'' Their comparison is between one full-batch
update built from $n$ samples (each sample contributes an additive term to the gradient; those terms
are independent and can be farmed out to parallel workers, then averaged) and *classical one-pass SGD*,
where updates happen one after another so later steps depend on earlier ones. Thus needing a large $n$
for one full batch is not the same as needing many *serial* gradient steps: *sample counts* and
*wall-clock complexity* (elapsed real time for the computation to finish, as opposed to an abstract
count of serial operations or steps) need not track one another; what matters for their theory is which
directions appear in the Hermite expansion of $g^star$ at an order detectable by one batch step at that $(n, "step")$
pair @dandi2024two (their leap index and staircase definitions formalize this).

=== One full-batch gradient step on $W$

Their Theorems 4--5 and Fig.~1 summarize a batch-size hierarchy for one step: alignment
already requires $n$ at least on $Theta(d)$ order with $eta$ coupled to $p$, yet *at most one*
direction in $V^(star)$ can emerge at that scale when $r>1$. Acquiring several directions
*simultaneously* in one update pushes the batch to $Theta(d^2)$. Even then, some directions can
force $n = Theta(d^ell)$ where $ell$ is the leap index (Definition 3): the first Hermite degree at
which certain components of $g^(star)$ appear @dandi2024two.

=== Multiple steps: staircase structure and obstructions

Theorem 7 and Fig.~2 describe a different regime: with repeated steps, *fresh* batches of size $n =
Theta(d)$ suffice to learn *multiple* directions over time *provided* new directions are coupled to
previous ones through the staircase property (made precise in their §3.4). Intuitively, later
directions become statistically predictable once earlier ones align---so the network can climb a
sequence of ``linearized'' subproblems. Directions lacking this coupling (and with degenerate low
Hermite mass) remain stuck behind the same $Theta(d^2)$ barrier that already obstructs single-step
multi-index learning @dandi2024two.

=== Approximation, generalization, and comparison to random features

Section 2.3 connects learned subspaces to what degree-$k$ kernels could approximate at initialization
(benchmarking against random-features scaling @mei2022randomfeatures, as discussed in @dandi2024two).
Fig.~3-style experiments show when
one gradient step on $W$ lowers test error relative to random features and when it
does not: the deciding issue is whether the nonlinear signal has components along directions the first step can
align with @dandi2024two.

== Comparison to the reference paper <sec:compare>

=== Ba *et al.* @ba2022gradient as neighbor work

@ba2022gradient study one gradient step on the first layer of a two-layer network in a proportional
high-dimensional limit: isotropic Gaussian inputs, single-index teacher, additive noise on $y$, and a
sharp comparison between small versus large learning-rate regimes for representation improvement over
the initial kernel @ba2022gradient. @dandi2024two retain the same ``early feature learning'' spirit but
change the target class, the noise model, and the algorithmic questions they prioritize (batch scaling
and multiple steps); we spell out overlaps and splits below.

=== Where the assumptions agree

- *Inputs.* Both works take features to be standard Gaussian (isotropic), so all nontrivial structure is
  encoded in the label map rather than in anisotropic covariates @ba2022gradient @dandi2024two.
- *Architecture and scaling.* Shallow two-layer networks with the usual $1/sqrt(p)$ width normalization
  appear in both analyses @ba2022gradient @dandi2024two.
- *Training protocol (family resemblance).* Both separate learning of the first layer from fitting the
  second layer and use *fresh* batches for different phases (Ba *et al.* use two independent batches for
  the two layers in their one-step setup; @dandi2024two use a fresh batch at each gradient step when
  analyzing multi-step training). In either case, decoupling layers avoids coupled finite-width pathologies
  and matches how this line of work models early full-batch GD on the first layer @dandi2024two.
- *Hermite viewpoint.* Each paper expands the target and loss geometry using Hermite analysis on Gaussian
  inputs; leap-index language enters when discussing which Hermite orders drive alignment @ba2022gradient
  @dandi2024two.

=== Where they differ (and what ``leap index'' means in each)

In @ba2022gradient the teacher is *single-index*: $y = f^star(x) + epsilon$, with a scalar projection of
$x$ driving $f^star$. There the target has no constant Hermite component and already depends linearly on the
teacher direction, so the leap index is $ell = 1$ in @dandi2024two's terminology: the first Hermite order at
which that direction contributes is linear @ba2022gradient.

@dandi2024two instead treat *multi-index* teachers $y = g^star(W^star z)$ with $r$ orthonormal directions
and *noiseless* labels. Their leap index $ell$ is *not* fixed to $1$: it indexes how deep one must go in
the Hermite expansion along a direction before the target's dependence on that direction ``turns on.''
When $ell = 1$, a toy linear functional $w_1^T z$ has leap one (the minimal case); when $ell > 1$, @dandi2024two show that even enormous single-batch updates may
still fail to learn that direction until $n$ reaches $Theta(d^ell)$. Thus Ba *et al.*'s baseline
single-index story aligns with the *simplest* leap-$1$ picture, whereas @dandi2024two also treats cases
with $ell > 1$ and with several interacting directions through
$g^star$ @dandi2024two.

*Summary.* Similar isotropic-Gaussian + shallow-net setup; dissimilar *target complexity* (single- vs
multi-index), *noise* (present vs absent), and *role of $ell$* (Ba *et al.* emphasize a leap-$1$
single-index tractable case; @dandi2024two treat general $ell$ and multi-step/staircase structure).

=== Map back to @dandi2024two (Sections / figures)

Equations (1)--(2) in @dandi2024two match our setup block above. Their abstract's three bullets correspond
to: §2.1 / Thms.~4--5 / Fig.~1 (single-step batch hierarchy and leap complexity); §2.2 / Thm.~7 / Fig.~2
(multi-step staircase learning at $n = Theta(d)$); §2.3 / Fig.~3 (approximation and generalization versus
initialization kernels, including comparison to random-features scaling @mei2022randomfeatures). Section
@sec:contributions tracks that order intentionally @dandi2024two.

*Reading across papers.* The progression from @ba2022gradient to @dandi2024two is not ``replace Ba's
theorem'' but ``change the task class and ask more detailed resource questions'': how large must a *single* batch
be to unlock multiple directions in one update, how does leap $ell$ obstruct that, and when do *several*
large-batch steps at $n = Theta(d)$ recover staircase targets that a single step cannot @dandi2024two.

= Critical discussion and limitations <sec:critical>

This section summarizes scope conditions and limitations of @dandi2024two and relates them to nearby deep-learning
theory.

== Gaussian, noiseless, teacher-only structure

The analysis assumes isotropic Gaussian inputs and *noiseless* labels from a realized multi-index teacher
@dandi2024two. This supports Hermite/Stein/Gaussian-equivalence arguments, but excludes correlated designs, heavy
tails, covariate shift, and label noise. @ba2022gradient use additive label noise in their single-index setting,
so the two papers compare different statistical regimes. In practice, part of the observed error may be dominated
by noise rather than by representation mismatch.

In related statistical and signal-processing settings, controlled noise can act as regularization (noise injection
as Tikhonov smoothing) or improve detectability in nonlinear systems (stochastic resonance) @bishop1995noise
@gammaitoni1998stochastic. Modern deep-learning analyses also study noise scale, batch size, and generalization
in SGD-based training @keskar2017largebatch @mandt2017sgdbayes @smith2018bayesian.

*Can moderate noise ever improve learning dynamics by regularizing alignment, rather than only increasing
estimation error?*

== Protocol: fresh batches, two stages, symmetrized init

@dandi2024two study giant-batch GD on $W$ with fresh samples across steps, then re-estimate the second layer on new
data. This two-stage protocol is useful analytically, but differs from common pipelines that reuse minibatches,
update $(W, a)$ jointly, and use momentum or adaptive methods. The analysis also uses an auxiliary symmetrized
initialization that sets the initial predictor to zero and simplifies the first update. As a result, the reported
rates should be interpreted for this training protocol, not for arbitrary optimization settings.

*What changes in the early-time feature-learning picture when initialization is not symmetrized?*

*What changes if $(W, a)$ are trained simultaneously with separated time scales (larger step size for the first
layer, smaller step size for the second layer), a setup closer to practical training than strict two-stage
optimization, and can the paper's conditioning/concentration framework be adapted to analyze this smoothed
two-step regime?*

*Do adaptive optimizers such as Adam induce comparable layerwise time-scale separation in practice (or the opposite,
if backpropagated gradients attenuate in earlier layers), and can training be accelerated by schedule designs
inspired by this paper that avoid treating all layer parameters identically?*

*Can the early-time feature-learning phenomena analyzed here motivate alternatives to standard backpropagation (for
example, forward-only local objectives such as the forward-forward algorithm @hinton2022forwardforward), especially
if one seeks biologically plausible update mechanisms?*

*Generalized-depth alignment question (exact setup):* let the teacher be two-layer,
$
  y = h^(star)( U^(star) g^(star)(V^(star) z)),
$
with $V^(star) in RR^(r_1 times d)$ (inner teacher directions), $U^(star) in RR^(r_2 times r_1)$ (outer teacher
directions), inner nonlinearity $g^(star): RR^(r_1) -> RR^(r_1)$, and outer map $h^(star): RR^(r_2) -> RR$.
Let the student be three-layer,
$
  hat(y) = a^T sigma_2(W_2 sigma_1(W_1 z)),
$
with $W_1 in RR^(p_1 times d)$ and $W_2 in RR^(p_2 times p_1)$. Under what conditions do we get staged alignment
$op("Row")(W_1^t) -> op("Row")(V^(star))$ and $op("Row")(W_2^t) -> op("Row")(U^(star))$ during early training?
Which geometric conditions (principal-angle / range-nullspace separation between teacher and student subspaces)
and which complexity descriptors of $g^(star), h^(star)$ (e.g., separate leap indices, Hermite tensor ranks,
derivative bounds) are sufficient for this behavior?

== Preliminary experiments addressing open questions

This subsection is an empirical check of two questions posed above. It does *not* add new proofs
beyond @dandi2024two.

=== Motivation and hypotheses

*Scope.* The simulations below do not prove new results. The single-step directional bottleneck and the role of
batch scaling and leap structure are proved in @dandi2024two; we use simulations only to probe robustness under
protocol variants (noise and initialization) that the paper does not treat in full generality.

=== What is simulated (step-by-step)

*Student model.* A two-layer network as in @dandi2024two:
$
  hat(f)(z; W, a) = 1/sqrt(p) sum_(i=1)^p a_i thin sigma(chevron.l w_i, z chevron.r),
$
with ReLU $sigma$, width $p$, and rows $w_i$ of $W in RR^(p times d)$.

*Data.* Draw covariates $z prop cal(N)(0, I_d)$. Teacher directions $w_j^star$ are sampled once per random seed;
targets $f^star(z)$ are explicit polynomials in inner products $chevron.l w_j^star, z chevron.r$
(single-index cases A/B in the first figures; two-index weighted quadratic in the third figure).

*Training labels.* Either clean $y = f^star(z)$ or noisy $y = f^star(z) + epsilon$ with
$epsilon prop cal(N)(0, sigma^2)$; we sweep $sigma in {0, 0.05, 0.1, 0.2, 0.4}$.

*Initialization.* Two protocols from @dandi2024two: Eq.~(10) i.i.d. draws for $(W_0, a_0)$, versus
Eq.~(10)+(11) symmetrized pairing (duplicate hidden directions with opposite output weights).

*Stage 1---first layer only (two ablations).*
- *Frozen-$W$ (random features baseline):* keep $W = W_0$; do *not* update the first layer.
- *One-step-$W$:* take *one* full-batch gradient step on $W$ only, using squared loss on the *training* batch
  and *fixed* initial second layer $a_0$ (Alg.~1 style). Learning rate follows the paper-style scaling
  $eta prop p sqrt(n/d)$.

*Stage 2---readout.* Refit $a$ by ridge regression on the *same* training batch with $W$ fixed at its Stage~1 value.
This matches ``train $a$ after an early-time move of $W$,'' not joint training of $(W, a)$.

*Evaluation.* Report performance on a fresh Gaussian test set (same $f^star$, clean labels for test error).
*Alignment metric:* $max_i |cos angle(w_i, w_j^star)|$ over hidden rows---a concrete cosine overlap with teacher
directions, comparable in spirit to measuring whether $op("Row")(W)$ aligns with teacher subspaces as in @dandi2024two
§2.1 / Fig.~1, but not identical to any single quantity in their proofs.

*Figure layout (single-index plots).* Case A vs Case B form the two rows; columns cycle initialization $times$ $sigma$.
Within each subplot, two bars compare frozen-$W$ vs one-step-$W$. The *Results* subsection states what each figure plots.
Hyperparameters $(d, n, p, eta, lambda)$ are held fixed across $sigma$ (no retuning per noise level).

*Typical numerics (unless overridden when generating figures).* Reference scripts use $d=128$, training batch
$n=8d$, width $p=512$, test sample size $4096$, learning rate $eta = 5 dot p sqrt(n/d)$, and ridge readout
$lambda$ on the order of $10^(-3)$ to $10^(-6)$ depending on the run.

*Motivation of these experiments.*
- *Noise-motivation test:* the question in §2 asks whether moderate label noise changes alignment between student
  rows and teacher directions beyond what follows from higher MSE alone, inspired by noise-as-regularizer
  and stochastic-resonance discussions @bishop1995noise @gammaitoni1998stochastic.
- *Initialization-motivation test:* @dandi2024two rely on symmetrized initialization for analysis. We test
  whether the observed early-time behavior is sensitive to Eq.~(11), or whether the qualitative alignment
  pattern survives without that symmetry constraint.

=== Results and interpretation

The figures below summarize runs from the Stage~1--2 pipeline in the preceding subsection *What is simulated
(step-by-step)*: sample $(W_0, a_0)$, optionally take one gradient step on $W$ with fixed $a_0$ under squared loss on
noisy training labels, ridge-fit $a$, then evaluate on i.i.d. test draws from $cal(N)(0, I_d)$ with clean targets
$f^star(z)$ for MSE. Commands and paths to regenerate the plots are listed in the Appendix.
@tab:prelim-figures maps each panel to objects and results in @dandi2024two (Eq., Alg., Thm.) versus what we measure on
finite draws; the bullets below spell out subplot grids.

#figure(
  {
    set text(size: 8.3pt)
    table(
      columns: (1.05fr, 1.2fr, 1.15fr, 1.15fr),
      stroke: 0.4pt,
      inset: 5pt,
      align: (left, left, left, left),
      table.header(
        [*Panel*],
        [*Question (narrow)*],
        [*Where the protocol is anchored in @dandi2024two*],
        [*Outcomes \& caveats (finite Monte Carlo trials)*],
      ),
      [Single-index test MSE @fig:ee592-mse],
      [
        Does one GD step on $W$ lower *clean* test MSE vs frozen-$W$, across Case A/B targets, init (Eq.~(10) vs
        Eq.~(10)+(11)), and training noise $sigma$?
      ],
      [
        Same model class and idealized training as @dandi2024two: predictor as in their Eq.~(1)--(2); two phases as in
        Alg.~1 (here one optional gradient step on $W$ with $a = a_0$, then refit $a$); learning rate chosen in the same
        scaling family as their Fig.~3 experiments ($eta prop p sqrt(n/d)$). Initializations follow Eq.~(10) and,
        when used, Eq.~(10)+(11). *Extension:* Gaussian inputs and teacher structure match the paper's setting; additive
        label noise on training data is an *extrapolation*---their theorems assume noiseless $y = g^star(W^star z)$
        @dandi2024two.
      ],
      [
        Case A: one-step mean below frozen at every $sigma$ in the plot; smallest Case A error at $sigma = 0$. Case B:
        smaller margins and stronger init dependence; one-step can exceed frozen under Eq.~(11). No turn-key evidence that
        raising $sigma$ *reduces* test MSE (exploratory sweep, not tuned $lambda$ per $sigma$).
      ],
      [Single-index alignment],
      [
        *Same Monte Carlo trials as @fig:ee592-mse.* Does the same step increase $max_i |cos angle(w_i, w^star)|$ vs
        frozen-$W$? (A second summary statistic, not a separate experiment.)
      ],
      [
        Identical trials as the preceding row. The cosine measures overlap between hidden rows and $w^star$, i.e.
        row-level overlap with the teacher direction---the same geometric object Thms.~4--5 / Fig.~1 discuss via alignment
        with (subspaces spanned by) teacher directions at finite batch step @dandi2024two; we report a concrete scalar
        diagnostic, not their asymptotic theorem conclusion.
      ],
      [
        In all panels of the saved figure, one-step alignment mean $>$ frozen mean at each tested $sigma$. Training noise
        hurts test MSE (first panel) but does not reverse this cosine ordering in these panels.
      ],
      [Two-index alignment],
      [
        With $r = 2$, does one step yield unequal alignment to $w_1^star$ vs $w_2^star$ compared with frozen-$W$? This is
        motivated by Thms.~4--5 @dandi2024two: at batch scale $n = Theta(d)$, a *single* giant step cannot fully specialize
        across multiple generic directions (their Fig.~1 illustrates the batch-threshold picture).
      ],
      [
        Alg.~1-style training and Eq.~(10)--(11) inits as above @dandi2024two; explicit polynomial $f^star$ with two
        orthonormal directions (an illustrative multi-index teacher, not one of the paper's formal instantiations).
        Same $(sigma, "init")$ comparison logic as row 1; figure layout differs from @fig:ee592-mse.
      ],
      [
        In these plots $|m_1 - m_2|$ often increases after one step vs frozen under clean and noisy training labels; seed
        dependence remains. Interpreting strictly: Thms.~4--5 concern large-$d$ limits and noiseless labels @dandi2024two,
        so this panel is only a finite-width *illustration* of ``asymmetric'' alignment across directions after one step,
        not a numerical verification of their statements.
      ],
    )
  },
  caption: [
    Companion to the plots below. Rows 1--2 are one *experiment* with two summaries per replicate; row 3 uses a
    different teacher. A *seed* fixes the pseudorandom generator state for one replicate; changing it redraws
    $(W_0, a_0)$, teacher directions, training batches, and noise---each seed is one independent Monte Carlo trial. Bars
    average over several seeds; error bars indicate spread across those trials. Protocol objects (Eq.~(1)--(2), Alg.~1
    phases, Eq.~(10)--(11), $eta prop p sqrt(n/d)$) follow @dandi2024two; Hermite-style Case A/B targets and label-noise
    sweeps are exploratory. ``Alignment'' means $max_i |cos angle(w_i, w_j^star)|$. No formal testing;
    $(d,n,p,eta,lambda)$ fixed across $sigma$.
  ],
) <tab:prelim-figures>
#v(0.35em)

*What each figure displays.*
- *Single-index, test MSE.* Teacher uses one unknown direction $w^star$. Each subplot shows mean test MSE (error bars:
  variability across random seeds). *Rows:* Case A (quadratic / Hermite-$H_2$-style target) and Case B (quartic /
  $H_4$-style target). *Columns:* initialization---Eq.~(10) i.i.d., and Eq.~(10)+(11) symmetric when included---crossed
  with training-noise level $sigma in {0, 0.05, 0.1, 0.2, 0.4}$. *Bars:* frozen-$W$ (left) vs one-step-$W$ (right).
  ``$Delta$ MSE'' printed on each panel is percent change from the frozen mean to the one-step mean.
- *Single-index, alignment.* Same experimental grid as the test-MSE figure for the same seeds and hyperparameters; only
  the plotted statistic changes to $max_i |cos angle(w_i, w^star)|$ from $W$ after Stage~2 (the same $W$ used for
  prediction).
- *Two-index teacher, alignment.* Same Stage~1--2 machinery, but $f^star$ depends on two orthonormal directions
  $(w_1^star, w_2^star)$ (formula in the figure caption). *Rows:* frozen-$W$ vs one-step-$W$. *Columns:*
  initialization $times$ $sigma$. Each subplot has two bars (alignment to $w_1^star$ and to $w_2^star$). The value
  $|a_1-a_2|$ printed on the plot is the absolute difference between those two reported means.

#figure(
  image("../experiments/figures/one_step_vs_rf_mse.png", width: 100%),
  caption: [
    Single-index teacher: mean test MSE (± variability across seeds). Rows: Case A / Case B. Columns: Eq.~(10) vs
    Eq.~(10)+(11) and training-label noise $sigma$. Bars: frozen-$W$, one-step-$W$. ``$Delta$ MSE'' is percent change from
    frozen mean to one-step mean.
  ],
) <fig:ee592-mse>
#v(0.2em)

*Figure interpretation (MSE).*
- Case A (quadratic): one-step improves over frozen-$W$ for all tested $sigma$, but best absolute error is at $sigma=0$.
- Case B (quartic): gains are weaker and more initialization-sensitive; under Eq.~(11), one-step can worsen MSE.
- Overall: no evidence here that moderate noise improves test error.

#figure(
  image("../experiments/figures/one_step_vs_rf_alignment.png", width: 100%),
  caption: [
    Same seeds and layout as @fig:ee592-mse. Vertical axis: $max_i |cos angle(w_i, w^star)|$ after ridge readout (same
    $W$ as in @fig:ee592-mse).
  ],
)
#v(0.2em)

*Figure interpretation (alignment).* Each bar plots $max_i |cos angle(w_i, w^star)|$ (maximum cosine similarity between
a hidden row and the teacher direction $w^star$), i.e. the usual ``best neuron'' alignment score; this is not test MSE.
For each subplot we compare two procedures at the same $(sigma, "init")$: *frozen-$W$* leaves the first layer at its
initialization; *one-step-$W$* takes one full-batch gradient step on $W$ before fitting $a$. In every panel the one-step
bar is strictly larger than the frozen bar: the alignment score after one step exceeds the score with no step, at each
tested noise level. So label noise worsens test error in the previous figure, but it does not reverse this ordering on
the cosine statistic shown here.

#figure(
  image("../experiments/figures/two_index_one_step_alignments.png", width: 100%),
  caption: [
    Two-index teacher,
    $f^star(z) = (chevron.l w_1^star, z chevron.r)^2 + 0.35(chevron.l w_2^star, z chevron.r)^2 - 1.35$ with orthonormal
    $(w_1^star, w_2^star)$. Rows: frozen-$W$ vs one-step-$W$. Columns: initialization $times$ training noise $sigma$. Two
    bars per panel: alignment to each teacher direction; $|a_1-a_2|$ is the absolute gap between those two means.
  ],
)
#v(0.2em)

*Figure interpretation (two-index teacher).*
- Each panel has two bars: alignment to $w_1^star$ and alignment to $w_2^star$, using the same cosine statistic as above.
  The printed $|a_1-a_2|$ is the absolute difference between those two bar heights (not a second-layer weight difference).
- In the plotted means, the absolute difference between the two bar heights is often larger after one step on $W$ than
  for frozen-$W$; the same qualitative pattern appears with clean and with noisy training labels in these runs.
- Qualitatively, this matches the batch-size story in @dandi2024two (Thms.~4--5, Fig.~1): at $n = Theta(d)$ one giant step
  can align strongly with one teacher direction while leaving another weaker, rather than specializing evenly across both.

=== Quantitative summary and limits

*Preliminary takeaway (not a theorem).* In these finite-width runs, I do not observe a robust
``moderate-noise improves test error'' effect. At each tested $sigma$, one-step-$W$ still yields higher cosine alignment
than frozen-$W$ in the plots above, while test MSE generally rises as $sigma$ increases. Symmetrized initialization
changes the baseline predictor and can change measured one-step improvements; @dandi2024two likewise stress that
protocol choices affect such comparisons.

*Quantified examples (mean ± std over 4 seeds; conditional on fixed hyperparameters).* In the one-step
single-index experiment with Eq.~(10) i.i.d. init and $sigma=0$, Case A test MSE changes
$4.01 \+- 0.11 -> 1.91 \+- 0.12$ (frozen to one-step). At $sigma=0.4$, Case A changes
$4.25 \+- 0.14 -> 2.23 \+- 0.10$; this is not evidence of noise-induced improvement over the clean-label
baseline. In the two-index setting, alignment to $w_1^star$ and to $w_2^star$ after one step remain unequal under both
clean and noisy labels in these runs (quantified on the figure by $|a_1-a_2|$), while reported MSE depends on $sigma$.

*Inference caveat.* These are finite-width, finite-seed observations under fixed schedules; we do not
re-optimize hyperparameters per noise level, and we do not perform formal hypothesis testing. Therefore,
the noise-related statements should be interpreted as setup-conditional empirical observations.

== Target geometry: regularity, staircase, leap, obstructed targets

The main bounds use staircase/polynomial structure in $g^star$ and leap index $ell$ along each direction
@dandi2024two. Figure~2 includes cases where no teacher direction is recovered in finite depth, and some directions
require batch scaling $Theta(d^ell)$. These are algebraic constraints in Hermite space (which orders carry mass
along which directions). This perspective is different from landscape-based analyses of non-convex optimization and
does not directly cover deeper compositional architectures.

*Do recently proposed non-backprop / random-search post-training approaches (e.g., random perturbation experts in
@gan2026neuralthickets and hyperscale ES variants in @sarkar2026eggroll) admit a connection to staircase-style
sequential feature recruitment, or do they rely on a different geometric mechanism?*

*Related ``jumping staircase'' question:* if recovery of a new direction at Hermite order $k+1$ depends on how many
tensor factors are already shared with learned order-$k$ structure, does the theory imply an intermediate regime where
$n = Theta(d^2)$ (rather than $Theta(d)$) can unlock additional directions in one giant step when overlap is partial?
If such jumps exist, can they motivate layerwise learning-rate or batch-size schedules that differ across layers
instead of uniform adaptive updates?

== Concentration regime: $n$ of order $d$ and the Gaussian toolbox

Operator-norm concentration for empirical covariances already requires $n$ to scale with $d$ when directions are
unconstrained. This underlies the paper's batch-size thresholds before leap/staircase refinements. The proof
techniques (projection conditioning, Gaussian equivalence, Hermite calculus) are specialized to Gaussian
covariates, so extension to non-Gaussian designs requires new arguments.

== What is proved vs conjectured (Table 1) + scope in DL theory

Table~1 in @dandi2024two includes both proved results and entries labeled as educated guesses, so claims should be
distinguished accordingly. Relative to broader theory, the paper focuses on finite-time, finite-width directional
alignment in $V^(star)$, while NTK/lazy and mean-field lines emphasize different asymptotic objects. The structural
contribution is the interaction of batch size, iteration depth, and Hermite geometry of $g^star$; conclusions that
depend on Gaussian noiseless inputs, fresh batches, and decoupled training should be read as model-specific.

// Keep report within the course page limit (assignment: at most 8 pages).
