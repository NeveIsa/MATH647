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
    affilation: [Topics in Statistics: Random Matrix Theory in Machine Learning],
  ),
)

#show: iclr2025.with(
  title: [Exposition: How two-layer networks learn in one \
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
    This report engages with *How Two-Layer Neural Networks Learn, One (Giant) Step at a Time*
    @dandi2024two (JMLR, 2024). The reference studies early training of a two-layer network on
    standard Gaussian inputs when the label follows a *multi-index* model: all low-dimensional
    structure lies in a link function of projections onto unknown teacher directions. The central
    question is when a small number of large-batch gradient steps on the first-layer weights
    produces nontrivial alignment with the teacher subspace $V^(star)$---thereby leaving the
    random-feature (lazy) kernel induced by initialization @dandi2024two.

    Following @dandi2024two, we separate three ingredients---batch size $n$, number of gradient steps,
    and the target's structure (Hermite / leap / staircase)---and stress that they are not
    interchangeable. In one giant full-batch step, batch scaling $n =
    Theta(d)$ supports alignment but only to a single teacher direction unless $n$ reaches
    $Theta(d^2)$ scale for multi-direction specialization; still harder directions may demand
    $n = Theta(d^ell)$ with leap index $ell$. With multiple steps and fresh batches of size $n =
    Theta(d)$, targets satisfying a staircase coupling can progressively recruit several
    directions over time, while obstructed pieces retain $Theta(d^2)$ bottlenecks. Finally,
    @dandi2024two link these geometric facts to approximation and generalization relative to the
    initialization kernel---including experiments where one gradient step helps or fails versus
    frozen features.

    Critical discussion of assumptions and limitations is in @sec:critical; the technical overview
    begins in @sec:setting.
  ],
  bibliography: bibliography("references.bib"),
  appendix: [
    = Appendix

    Optional: longer calculations, extra figures, or extended experiments.
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
directions are #strong[algebraic]ally accessible at a given $(n, "step")$ pair @dandi2024two.

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

=== Approximation, generalization, and separation from random features

Section 2.3 connects learned subspaces to what degree-$k$ kernels could approximate at initialization
(benchmarking against random-features scaling @mei2022randomfeatures, as discussed in @dandi2024two).
Fig.~3-style experiments show when
one gradient step on $W$ materially lowers test error relative to random features and when it
cannot: the deciding issue is whether the nonlinear signal shares directions the first step can
actually align @dandi2024two.

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
  and matches how this literature idealizes early-time GD @dandi2024two.
- *Hermite viewpoint.* Each paper expands the target and loss geometry using Hermite analysis on Gaussian
  inputs; leap-index language enters when discussing which Hermite orders drive alignment @ba2022gradient
  @dandi2024two.

=== Where they differ (and what ``leap index'' means in each)

In @ba2022gradient the teacher is *single-index*: $y = f^star(x) + epsilon$, with a scalar projection of
$x$ driving $f^star$. The comparison table in my presentation slides pins their typical Hermite regime to *no constant
Hermite mass*, a nonzero first-order (linear) Hermite contribution, and---in that same schematic---
*leap index* equal to $1$. Informally, the first nontrivial coupling between target and a teacher direction
already appears at the *linear* level, which is the simplest regime for a one-step alignment effect in the
proportional limit @ba2022gradient.

@dandi2024two instead treat *multi-index* teachers $y = g^star(W^star z)$ with $r$ orthonormal directions
and *noiseless* labels. Their leap index $ell$ is *not* fixed to $1$: it indexes how deep one must go in
the Hermite expansion along a direction before the target's dependence on that direction ``turns on.''
When $ell = 1$, a toy linear functional $w_1^T z$ has leap one (as in the short ``leap index toy
example'' on my presentation slides); when $ell > 1$, @dandi2024two show that even enormous single-batch updates may
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

To partially address the open questions above (label noise and symmetrized initialization), I ran
paper-style one-step experiments with Gaussian inputs and explicit teacher functions $f^star(z)$, while
varying (i) initialization protocol and (ii) additive label-noise level.

*Protocol used in this report's experiments.* I compare Eq.~(10) i.i.d. initialization to
Eq.~(10)+(11) symmetrized initialization, and clean labels ($sigma = 0$) to noisy labels
$y = f^star(z) + epsilon$ with $epsilon prop cal(N)(0, sigma^2)$ on a sweep
$sigma in {0, 0.05, 0.1, 0.2, 0.4}$. In each condition, I compare frozen-$W$ to one giant
step on $W$, then ridge-fit the readout layer.

#figure(
  image("../experiments/figures/one_step_vs_rf_mse.png", width: 100%),
  caption: [
    Preliminary one-step experiment (single-index targets): test MSE comparison across
    Eq.~(10) i.i.d. vs Eq.~(10)+(11) symmetric initialization, and across noise levels
    $sigma in {0, 0.05, 0.1, 0.2, 0.4}$. Each panel compares frozen-$W$ and one-step-$W$.
  ],
)

#figure(
  image("../experiments/figures/one_step_vs_rf_alignment.png", width: 100%),
  caption: [
    Same sweep as above, but reporting max row-teacher alignment.
    One-step updates continue to increase alignment under moderate noise, though this does
    not automatically imply lower test error in every target/init condition.
  ],
)

#figure(
  image("../experiments/figures/two_index_one_step_alignments.png", width: 100%),
  caption: [
    Two-index preliminary experiment:
    $f^star(z) = (chevron.l w_1^star, z chevron.r)^2 + 0.35(chevron.l w_2^star, z chevron.r)^2 - 1.35$.
    Panels compare frozen-vs-one-step alignment, split by initialization and noise level.
    In this setup, one-step training still exhibits directional concentration, while added noise
    mainly shifts error levels rather than producing a clear stochastic-resonance gain.
  ],
)

*Preliminary takeaway (not a theorem).* In these finite-width runs, I do not observe a robust
``moderate-noise improves test error'' effect. The dominant pattern is that one-step updates still
increase alignment under noise, while test error typically worsens or remains similar as $sigma$ grows.
Symmetrized initialization changes the baseline predictor and can materially alter apparent one-step
gains, consistent with the paper's caution that protocol details matter for interpreting early-time
feature learning.

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

== RMT and concentration: $n$ of order $d$ and the Gaussian toolbox

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
