# Research notes: what the literature says, and what this engine does about it

This document is the reading behind the v0.3 additions. It surveys the
portfolio-construction literature the engine draws on, states plainly which
ideas were already implemented, which were added in this round, and which were
read and deliberately left out — with the reason in each case.

The organizing claim across all of it is the same one the engine was built
around: **an allocation is a function of estimates, and the estimates are
noisy.** Almost every method below is an answer to some version of "given that
your inputs are wrong, how wrong will your portfolio be?" They differ in where
they intervene — the covariance, the objective, the search, or the reporting.

---

## 1. Marcos López de Prado

### 1.1 Hierarchical Risk Parity (2016)

*Building Diversified Portfolios that Outperform Out-of-Sample*, JPM 42(4).

The observation: quadratic optimizers require an inverted covariance matrix,
and the condition number of that matrix grows with the correlation among
assets. So the more you need diversification, the less stable the optimizer
that provides it — "Markowitz's curse". HRP sidesteps inversion entirely by
building a correlation dendrogram, quasi-diagonalizing the matrix, and
splitting risk down the tree by inverse variance.

**Status: already implemented** (`HRPOptimizer`), with configurable linkage and
reported clusters.

### 1.2 Denoising and detoning (2020)

*Machine Learning for Asset Managers*, CUP, ch. 2.

Shrinkage estimators pull the whole matrix toward a target, attenuating signal
and noise alike. Random matrix theory offers a sharper instrument: eigenvalues
below the Marchenko-Pastur edge λ₊ are statistically indistinguishable from
those of a pure-noise matrix, so they can be replaced by their common average
while the eigenvectors carrying the factor structure are left untouched.
Detoning additionally removes the first eigenvector — the market — which
otherwise dominates every pairwise correlation and degrades any clustering
built on them.

**Status: added.** `optimization_engine.data.denoise` implements the KDE-fitted
Marchenko-Pastur cutoff, both the constant-residual and targeted-shrinkage
rules, and detoning. Available as `covariance_method="denoised"`, as
`covariance_matrix(..., denoise=True, detone=k)`, and as `EngineConfig.denoise`
/ `EngineConfig.detone`.

One finding worth recording, because it is the sort of thing a library should
tell you rather than let you assume. On the engine's own sample panel:

```
The correlation's condition number went 18.4 → 11.2;
the covariance's went 1.25e+04 → 1.23e+04
```

Denoising did what it claims — it improved the correlation's conditioning by
39% — and the covariance barely moved, because *this* panel's conditioning
comes from the spread of the volatilities (cash against EM equity), not from
correlation noise. The `DenoiseReport` says so in words. Denoising is not a
general-purpose fix for an ill-conditioned covariance, and quoting only the
correlation improvement would have been the flattering half of the truth.

### 1.3 Nested Clustered Optimization (2019)

*A Robust Estimator of the Efficient Frontier*, SSRN 3469961; also *MLAM* ch. 7.

Denoising removes noise-induced instability. NCO addresses the second kind:
instability induced by the *signal*, i.e. by the block structure of a
correlated universe. Rather than inverting one N×N matrix, NCO clusters, solves
inside each cluster, collapses each cluster to a synthetic asset, solves across
those, and multiplies the layers. Every inverted matrix is then either small or
nearly diagonal.

**Status: added** (`NCOOptimizer`). Both layers reuse the engine's existing
convex optimizers, so signs and the long-only setting hold throughout; the
mandate is applied to the combined result by projection, and the distance moved
is reported. The correlation matrix is detoned before clustering by default.
The diagnostics name the condition numbers being avoided:

```
A direct solve would invert a matrix with condition number 2.11e+03.
NCO instead inverts 2 cluster matrices (worst condition 334)
and one 2×2 matrix (condition 192).
```

### 1.4 Monte Carlo Optimization Selection

The experimental apparatus from the same paper: declare the sample estimates to
be ground truth, draw synthetic histories from that distribution, re-estimate
and re-solve on each, and measure how far each method's weights land from the
answer that truth implies.

**Status: added** (`monte_carlo_optimization_selection`). This is the single
most useful addition for the engine's stated purpose, because it turns "which
method should I use" from an argument into a measurement. On the sample panel,
20 simulations:

| Method | Weight RMSE | Worst single position |
| --- | --- | --- |
| `nco` | 0.01% | 0.01% |
| `herc` | 0.02% | 0.05% |
| `hrp` | 0.05% | 0.13% |
| `min_variance` | 0.59% | 1.51% |
| `mean_variance` | **14.79%** | **35.53%** |

The comparison is deliberately self-referential — each method is scored against
*its own* answer on the truth, not a common one — because the question is
estimation stability, not which objective is correct. A method that is wrong
but consistent scores well here, and should: it is telling you the method is
not where your uncertainty is coming from.

### 1.5 The Sharpe ratio, with Bailey

*The Sharpe Ratio Efficient Frontier* (2012), *The Deflated Sharpe Ratio*
(2014), *The Probability of Backtest Overfitting* (2017).

Three connected results. The probabilistic Sharpe ratio converts a point
estimate into a confidence statement, widening the standard error for negative
skew and fat tails. The deflated Sharpe ratio then corrects for *selection*:
the benchmark a strategy must beat is not zero, but the expected maximum across
however many configurations were tried,
`√V[SR] · ((1−γ)Z⁻¹[1−1/N] + γZ⁻¹[1−1/(N·e)])`. CSCV asks the complementary
question — across every balanced split of the sample, how often does the
in-sample winner land below the out-of-sample median?

**Status: PSR was already implemented; the rest added** in
`optimization_engine.analytics.selection`. The test suite validates the
correction on 50 strategies drawn from the *same zero-mean distribution*: the
best of them posts an annualized Sharpe of 1.24 and a probabilistic Sharpe of
99.3%, and the deflated Sharpe correctly reports 45.6%.

This matters for this engine specifically. It ships ten-plus methods, six
covariance estimators and a grid of constraints; running forty combinations and
reporting the best is easy and is exactly what the deflation exists to catch.
`optengine optimize --trials 40` makes the declaration explicit, and the
default of 1 is a claim the output states out loud.

---

## 2. Dany Cajas (Riskfolio-Lib)

Cajas's contribution is largely one of *coverage*: expressing a very wide class
of risk measures as disciplined convex programs so they can be optimized with
off-the-shelf conic solvers rather than bespoke code. The relevant papers are
*Entropic Portfolio Optimization* (2021), *OWA Portfolio Optimization: a
Disciplined Convex Programming Framework* (2021), *Kelly Portfolio
Optimization* (2021), *Convex Optimization of Portfolio Kurtosis* (2022),
*Portfolio Optimization of Relativistic Value at Risk* (2023), and the 2025
book *Advanced Portfolio Optimization*.

**Already implemented:** mean-CVaR (Rockafellar-Uryasev), semicovariance, and a
passthrough to riskfolio-lib's shrinkage when installed.

**Added:** mean-CDaR (§4 below), and downside risk measures (CVaR, conditional
drawdown) as the cluster-splitting criterion inside HERC — the pluggable-risk-
measure idea applied where it costs nothing to support.

**Read and deliberately deferred**, with reasons:

* **EVaR / RLVaR** (entropic and relativistic value at risk). Genuinely
  attractive: EVaR is the tightest coherent upper bound on VaR and is
  exponential-cone representable. The blocker is operational — the engine's
  solver fallback chain is `CLARABEL → ECOS → SCS → OSQP`, and only the first
  and third handle exponential cones. Adding EVaR without reworking that chain
  would produce a method that silently fails over to a solver that cannot
  express its own objective. Worth doing; worth doing properly.
* **OWA / range risk measures** (Gini mean difference, tail Gini, CVaR range).
  A framework rather than a method, and one riskfolio-lib already implements
  well. The right move is a routed passthrough, not a reimplementation.
* **Kelly / logarithmic mean-risk.** `cp.log` makes the program straightforward,
  but the objective changes what "expected return" means throughout the
  reporting, backtest and frontier layers. That is a coherent piece of work, not
  an optimizer file.
* **Portfolio kurtosis.** The SDP reformulations require the N²×N² co-kurtosis
  matrix; on a 13-asset panel that is fine and on anything realistic it is not.

---

## 3. Richard Grinold and Ronald Kahn

*Active Portfolio Management* (2nd ed., 1999); Grinold (1989) on the
fundamental law; Clarke, de Silva & Thorley (2002) on the transfer coefficient.

The engine optimized entirely in absolute space. Everything a manager with a
benchmark cares about — active weights, active risk, and whether the forecasts
justify either — sits in a different space, and the translation is not obvious.

**Added** in `optimization_engine.analytics.active`:

* **`IR ≈ TC · IC · √BR`** and its inverse, `implied_breadth`. Run backwards the
  law is a plausibility check: an information ratio of 1.0 built on an IC of
  0.03 requires 1,111 independent bets a year. A manager with 40 positions
  turned over quarterly does not have them.
* **The transfer coefficient**, in both the Grinold-Kahn definition (the
  correlation, in the risk metric, between the held active weights and the
  unconstrained optimum `Σ⁻¹α`) and the Clarke-de Silva-Thorley one (the
  cross-sectional correlation of `α_i/σ_i` against `Δw_i·σ_i`, which needs no
  matrix inverse and therefore survives a detoned covariance).

  This is the piece the engine is unusually well placed to compute, because it
  already knows exactly which constraints were applied and what the
  unconstrained answer would have been. A transfer coefficient of 0.35 says two
  thirds of the skill is being absorbed by the mandate — a statement about the
  *constraints*, pointing at a different remedy than "get better signals".
* **Grinold's alpha**, `α = IC · σ · z`. The most useful single formula in the
  book, and the one most often skipped. A score is not an expected return. With
  an IC of 0.05, a two-standard-deviation view on a 20%-volatility asset earns
  an alpha of 2% — not the 10% that gets typed into a spreadsheet, and the
  difference is precisely what stops mean-variance producing corner solutions.
* **Risk-aversion calibration.** `λ_A = IR / (2ψ*)` turns a tracking-error
  budget into the utility coefficient the optimizer wants. "Risk aversion = 2"
  means nothing on its own; "a 4% tracking-error budget at an IR of 0.5" is a
  sentence an investment committee can argue with.
* **Active risk decomposition** — the Euler decomposition of *tracking error*
  rather than of total risk. The two disagree exactly where it matters: a 40%
  position in the largest index constituent can be the biggest single source of
  absolute risk and contribute nothing to tracking error, because the benchmark
  holds it too.

**Deferred:** the structural (fundamental) factor risk model. The engine is
covariance-based; a real factor model is a different data pipeline — exposures,
factor returns, specific risk — rather than an additional function.

---

## 4. Other sources

### Attilio Meucci — the effective number of bets

*Managing Diversification* (2009); Meucci, Santangelo & Deguest (2015).

The engine already reported the effective number of positions and the effective
number of risk contributions. Both are computed asset by asset, and so neither
notices that ten European bank stocks are one bet. Meucci's answer is to rotate
into uncorrelated factors and take the exponential entropy of the resulting
variance shares.

**Added** in `optimization_engine.analytics.diversification`, with both the
minimum-torsion rotation (Meucci's own iterative algorithm) and PCA.

The two disagree, and the disagreement is the point. An equal-weight book on
the sample panel scores:

| Rotation | Effective bets | Largest single bet |
| --- | --- | --- |
| Minimum torsion | 9.78 of 13 | 15% |
| PCA | 1.56 of 13 | 89% |

Thirteen distinct, nameable positions; one dominant driver. Neither number is
"the" answer — PCA measures how much independent variation the book is exposed
to, minimum torsion measures how many distinct positions it takes — so
`compare_diversification()` reports both. Note in particular that minimum
torsion does *not* collapse to 1 for a perfectly equicorrelated panel held at
equal weights: symmetry leaves the rotation nothing to work with. That is a
real property of the method, documented in the module rather than papered over.

### Thierry Raffinot — Hierarchical Equal Risk Contribution

*Hierarchical Clustering-Based Asset Allocation* (2017); *The Hierarchical Equal
Risk Contribution Portfolio* (2018).

HRP quasi-diagonalizes the matrix into a list and bisects that list down the
middle. The ordering comes from the tree; the splits do not. HERC splits at the
dendrogram's own merge points and stops at the number of clusters the data
supports, allocating between siblings by equal risk contribution.

**Added** (`HERCOptimizer`), with a pluggable cluster risk measure (variance,
standard deviation, CVaR, conditional drawdown, or equal weight, the last
reproducing Raffinot's HCAA).

### Chekhlov, Uryasev & Zabarankin — conditional drawdown at risk

*Drawdown Measure in Portfolio Optimization* (2005).

Variance and CVaR are both order-independent: reorder the return history and
neither changes. Drawdown is the one risk measure a client actually experiences
that is not. CDaR — the average of the worst α of drawdowns along the path — is
coherent and, on an uncompounded equity curve, linear-programmable.

**Added** (`CDaROptimizer`). On the sample panel it cuts the maximum drawdown
from 15.8% (equal weight) to 3.6%.

Two limits are stated in the module docstring rather than buried: the effective
sample size is far smaller than the observation count suggests, because the
points of a drawdown path are highly dependent (the optimizer reports the number
of *distinct underwater episodes* for this reason); and optimizing a path
statistic on a single path is the most overfittable thing in this library. It
is the method most in need of the walk-forward and the deflated Sharpe, which
is a pleasant symmetry with §1.5.

### Already present before this round

Michaud resampling and the bootstrap frontier band; Jorion (1986) Bayes-Stein
shrinkage of the mean; Ledoit-Wolf and OAS shrinkage; Black-Litterman with
absolute and relative views and δ calibrated from a market Sharpe;
Rockafellar-Uryasev mean-CVaR; Choueifaty-Coignard maximum diversification;
equal risk contribution; Newey-West volatility; Cornish-Fisher VaR.

---

## 5. What is still missing

Ranked by the ratio of value to work, as a roadmap rather than a promise:

1. **A solver chain that can express exponential cones**, unlocking EVaR and
   RLVaR (Cajas). The blocker is infrastructural, not mathematical.
2. **Nonlinear shrinkage** (Ledoit & Wolf, 2017/2020). Dominates linear
   shrinkage almost everywhere; the QuEST inversion is the reason it is not
   already here.
3. **Entropy pooling** (Meucci, 2008) as a non-normal alternative to
   Black-Litterman, which would let views be expressed on any statistic rather
   than only on means.
4. **A structural factor risk model**, which would make the transfer
   coefficient and the active risk decomposition considerably more useful than
   they are against a sample covariance.
5. **The full ONC algorithm** (López de Prado, MLAM §4.4). The engine currently
   selects the cluster count by maximizing the silhouette t-statistic on the
   hierarchical tree — the same criterion, applied to the dendrogram the caller
   already has rather than to k-means restarts. Deterministic, and a
   simplification; the chosen `k` and every candidate's score are reported so
   the choice can be inspected rather than trusted.

---

## References

Bailey, D. and López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier".
*Journal of Risk* 15(2).

Bailey, D. and López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting
for Selection Bias, Backtest Overfitting and Non-Normality". *The Journal of
Portfolio Management* 40(5).
[SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)

Bailey, D., Borwein, J., López de Prado, M. and Zhu, Q. (2017). "The Probability
of Backtest Overfitting". *Journal of Computational Finance* 20(4).

Cajas, D. (2021). "Entropic Portfolio Optimization: A Disciplined Convex
Programming Framework".
[SSRN 3792520](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3792520)

Cajas, D. (2021). "OWA Portfolio Optimization: A Disciplined Convex Programming
Framework".
[SSRN 3988927](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3988927)

Cajas, D. (2021). "Kelly Portfolio Optimization: A Disciplined Convex
Programming Framework".
[SSRN 3833617](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3833617)

Cajas, D. (2022). "Convex Optimization of Portfolio Kurtosis".
[SSRN 4202967](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4202967)

Cajas, D. (2023). "Portfolio Optimization of Relativistic Value at Risk".
[SSRN 4378498](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4378498)

Cajas, D. (2025). *Advanced Portfolio Optimization: A Cutting-edge Quantitative
Approach*. Springer.
[Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib)

Chekhlov, A., Uryasev, S. and Zabarankin, M. (2005). "Drawdown Measure in
Portfolio Optimization". *International Journal of Theoretical and Applied
Finance* 8(1).

Clarke, R., de Silva, H. and Thorley, S. (2002). "Portfolio Constraints and the
Fundamental Law of Active Management". *Financial Analysts Journal* 58(5).

Grinold, R. (1989). "The Fundamental Law of Active Management". *The Journal of
Portfolio Management* 15(3).

Grinold, R. and Kahn, R. (1999). *Active Portfolio Management*, 2nd ed.
McGraw-Hill.

Laloux, L., Cizeau, P., Bouchaud, J-P. and Potters, M. (1999). "Noise Dressing of
Financial Correlation Matrices". *Physical Review Letters* 83(7).

López de Prado, M. (2016). "Building Diversified Portfolios that Outperform
Out-of-Sample". *The Journal of Portfolio Management* 42(4).

López de Prado, M. (2019). "A Robust Estimator of the Efficient Frontier".
[SSRN 3469961](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3469961)

López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge
University Press.

Meucci, A. (2009). "Managing Diversification". *Risk* 22(5).

Meucci, A., Santangelo, A. and Deguest, R. (2015). "Risk Budgeting and
Diversification Based on Optimized Uncorrelated Factors". *Risk* 28(11).
[SSRN 2276632](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2276632)

Raffinot, T. (2017). "Hierarchical Clustering-Based Asset Allocation". *The
Journal of Portfolio Management* 44(2).

Raffinot, T. (2018). "The Hierarchical Equal Risk Contribution Portfolio".
[SSRN 3237540](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3237540)

Rockafellar, R. T. and Uryasev, S. (2000). "Optimization of Conditional
Value-at-Risk". *Journal of Risk* 2(3).
