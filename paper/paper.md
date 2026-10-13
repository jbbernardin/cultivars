---
title: "cultivars: Research-grade time-series econometrics in Python, from reduced forms to structural identification"
tags:
  - Python
  - econometrics
  - time series
  - vector autoregression
  - structural identification
  - Bayesian inference
  - state space
  - macroeconomics
authors:
  - name: Nikhil Sunder
    orcid: 0000-0000-0000-0000 # TODO: real ORCID
    corresponding: true
    affiliation: 1
affiliations:
  - name: University of Miami, United States
    index: 1
    ror: 02dgjyy92 # TODO: verify ROR id
date: 11 October 2026 # TODO: set on submission
bibliography: paper.bib
---

# Summary

Economists, central-bank researchers and financial analysts study how
variables such as output, inflation, interest rates and asset returns move
together over time, and what happens to them after an unexpected event such
as a policy-rate change or an oil-supply disruption. Answering such questions
takes two steps: fitting a statistical model to the data, and then imposing an
economic assumption that turns the model's correlations into causal
statements. `cultivars` is a Python library that implements both steps for the
model families used in applied macroeconometrics and financial econometrics:
univariate autoregressions and volatility models; vector autoregressions
(VARs) in reduced, Bayesian, large-dimensional, nonlinear and regime-switching
forms; structural identification schemes; state-space filters and smoothers;
spectral methods; and the diagnostic tests and forecast-evaluation tools that
surround them. Every fitted model returns an immutable result object that
prints a uniform summary, states the assumptions its numbers depend on, and
declines to report quantities those assumptions cannot support. The package
depends only on NumPy and SciPy [@harris2020array; @virtanen2020scipy], is
fully typed, and is released under the MIT license.

# Statement of need

Since @sims1980macroeconomics, the VAR has been the central empirical tool of
macroeconomics. The identification literature built on it has moved well past
recursive orderings: long-run restrictions [@blanchard1989dynamic], sign and
narrative-sign restrictions [@uhlig2005effects; @antolin2018narrative],
exact bounds for set-identified models [@giacomini2021robust], external
instruments [@mertens2013dynamic; @stock2018identification], and
identification through heteroskedasticity [@rigobon2003identification] or
non-Gaussianity [@lanne2017identification]. In parallel, Bayesian shrinkage
[@litterman1986forecasting; @banbura2010large; @giannone2015prior] made VARs
with dozens of variables routine for forecasting, and time-varying-parameter
VARs with stochastic volatility [@primiceri2005time] became the standard
instrument for studying structural change. These methods form the core of
graduate macroeconometrics [@kilian2017structural] and of empirical work at
central banks.

Python users who need them face a fragmented choice. Reduced-form VARs, vector
error-correction models and recursive structural VARs are available; almost
everything after that exists only as R packages, MATLAB toolboxes, or
replication code attached to individual papers. A survey of public
repositories conducted for this project found [TODO: N, ~40] active Python
projects re-deriving generalized variance decompositions, sign-restriction
samplers, Minnesota priors or mixed-frequency filters in one-off scripts,
rarely tested against a reference implementation.

`cultivars` is written for three audiences: applied researchers who want the
modern identification and Bayesian VAR toolkit in the same language as their
data pipeline; instructors and students of graduate time-series courses, for
whom one consistent interface across model families lowers the cost of
comparing methods on the same data; and methodologists, who need tested
filters, samplers and identification utilities to build on rather than
reimplement.

# State of the field

The closest Python package is statsmodels [@seabold2010statsmodels], whose
time-series module provides ARIMA and SARIMAX, unobserved components, dynamic
factor models, univariate Markov switching, VAR, VECM and an A/B-model
structural VAR on a mature state-space foundation. arch [@sheppard_arch]
provides univariate GARCH-family models with a range of error distributions.
Outside Python, the reference implementations are R packages, including vars
[@pfaff2008vars], svars [@lange2021svars], BVAR [@kuschnig2021bvar] and tsDyn
[@tsdyn], and MATLAB toolboxes such as BEAR [@dieppe2016bear] and Dynare
[@adjemian2024dynare]. \autoref{tab:coverage} compares coverage across the 76
estimating classes in `cultivars` against statsmodels 0.15.0 and the wider
Python ecosystem.

| Model group            | Classes | statsmodels | Any Python package | No packaged implementation in any language |
| :--------------------- | ------: | ----------: | -----------------: | -----------------------------------------: |
| Univariate             |      26 |           8 |                 12 |                                          2 |
| VAR, reduced form      |      12 |           5 |                  5 |                                          3 |
| VAR, large-dimensional |      11 |           1 |                  1 |                                          0 |
| VAR, structural        |      13 |           2 |                  3 |                                          3 |
| VAR, nonlinear         |       7 |           0 |                  0 |                                          2 |
| VAR, Markov-switching  |       3 |           0 |                  0 |                                          1 |
| State space            |       4 |           1 |                  2 |                                          2 |
| **Total**              |  **76** |      **17** |             **23** |                                     **13** |

: Counterparts to `cultivars` estimating classes. \label{tab:coverage}

Where statsmodels or arch already implement a model well (ARIMA, unobserved
components, univariate Markov switching, the linear-Gaussian Kalman filter,
standalone GARCH), `cultivars` does not claim to improve on them; its versions
exist so that downstream tools such as identification, forecast comparison
and spectral analysis can consume every model through one result interface,
and the established implementations serve as numerical oracles in its test
suite. The contribution lies in the 53 classes with no maintained Python
counterpart, concentrated in structural identification, Bayesian and
large-dimensional VARs, nonlinear and regime-switching VARs, and
regime-switching state space.

We chose to build a separate package rather than contribute to statsmodels
for two reasons tied to design rather than preference. First, the
identification layer requires that structural schemes take a fitted reduced
form as input and that set-identified schemes return sets rather than points;
statsmodels' `SVAR` estimates and identifies within one object, and
separating the two would break its public interface. Second, Bayesian VARs
need purpose-built Gibbs samplers that exploit conditional conjugacy, which
falls outside statsmodels' frequentist scope, while general-purpose
probabilistic programming frameworks do not exploit that structure.
[TODO: link any upstream statsmodels issue or discussion, if one exists.]

# Software design

Three decisions shape the package.

**Estimation and identification are separate acts.** A reduced form reports
only what the data determine. Any assumption that converts it into causal
statements (an ordering, a sign pattern, an instrument, a volatility regime)
is a separate object constructed from the fitted result, and the structural
result restates that assumption in its own summary:

```python
var = VAR(Y, order=4, names=("ip", "cpi", "ffr")).fit()
svar = RecursiveSVAR(var, order=("ip", "cpi", "ffr")).identify()
sign = SignRestrictedSVAR(var, restrictions=R).identify(seed=0)
```

All identification schemes share one generic base class, so a new scheme
inherits impulse responses, variance decompositions and historical
decompositions, and the same reduced form can be identified several ways and
the answers compared. The cost is an extra call in the simplest recursive
case; the benefit is that every impulse response can be traced to the
assumption that produced it.

**Results are records, not handles.** Every `fit` returns a frozen dataclass
holding the estimates, the data needed to forecast or simulate, and a summary
whose notes state what the numbers assume. Consumers such as backtests,
spectral densities and model-confidence-set comparisons
[@hansen2011model] read the result's surface rather than the model's type, so
they work across families without special cases. Results also refuse what
they cannot support: a set-identified scheme returns the identified set, a
long-run restriction rejects a system with a unit root, a particle-filter
likelihood is labelled as an estimate, and a mixture model declines the
regime-count likelihood-ratio test whose asymptotics are non-standard.

**A thin public surface over a shared engine.** Public modules hold
specifications and result types; the numerical work, including recursions,
Kalman, extended, unscented, particle and Hamilton filters, smoothers,
posterior samplers and prior machinery, lives in a private engine of roughly
36,000 lines shared across model families. The same Gibbs components serve
the hierarchical, independent Normal-Wishart, factor stochastic-volatility and
time-varying-parameter VARs. We traded per-model speed tuning and accelerator
backends (JAX, Numba) for a NumPy/SciPy-only dependency footprint that is
easier to install, audit and reproduce; accelerated engines are planned
behind a versioned engine contract that leaves the public API unchanged.

Errors follow a three-way taxonomy (`DimensionError`, `SpecificationError`,
`NumericalError`) that tells the user whether to reshape the data, change the
specification, or inspect the numbers; each class also subclasses the
matching built-in exception so existing handlers keep working. Every
stochastic routine takes an explicit seed or generator, and every public
object carries a deterministic, runnable docstring example; the roughly 8,700
lines of examples run as a doctest suite in continuous integration. Breadth
and validation are deliberately decoupled: the semantic-versioning guarantee
covers a validated core of [TODO: 20] classes, and the remainder are marked
experimental until they reproduce a reference.

# Research impact statement

`cultivars` ships ten worked examples, executed during each documentation
build, that reproduce the empirical designs of published studies on public
FRED data: recursive monetary VARs [@christiano1999monetary], long-run
restrictions [@blanchard1989dynamic], sign, narrative and set identification
of oil shocks [@kilian2009not], identification by heteroskedasticity and
external instruments [@rigobon2003identification; @mertens2013dynamic],
dynamic Nelson-Siegel yield-curve forecasting [@diebold2006forecasting],
Markov-switching business-cycle dating [@hamilton1989new], cointegration of
the great ratios [@king1991stochastic], medium-scale Bayesian VAR forecasting
[@banbura2010large], GARCH and stochastic-volatility risk backtests, and an
inflation-forecasting comparison [@stock2007why].

Numerical agreement is checked against reference implementations through
pinned fixtures that run in continuous integration: statsmodels for VAR, VARX,
VECM, Markov-switching AR, unobserved components and the Kalman filter; the R
packages BVAR, bvarsv, tsDyn, svars and bsvarSIGNs for the Bayesian, threshold
and structural models; and published tables for the proxy SVAR
[@mertens2013dynamic] and regime-switching state space [@kim1999state].
[TODO: report results, e.g. "N of M core classes match their oracle to within
1e-6 for closed-form estimators; samplers agree in posterior means within
documented Monte Carlo tolerance."]

[TODO: concrete realized use. Examples of what qualifies: the course or
research project at the University of Miami Herbert Business School that uses
the package; your own working paper or thesis chapter that uses it; issues or
pull requests from external users; download counts; any talk or poster.
Aspirational adoption targets do not count and should not appear here.]

The package is distributed on PyPI with signed releases and archived on
Zenodo, and holds an OpenSSF Best Practices badge; contribution, security and
governance policies, a criteria-gated roadmap, and full API documentation are
public.

# AI usage disclosure

[TODO: JOSS requires all three of the following; vague language is grounds
for rejection.]

[1. Tools and versions: name each tool and model (e.g., "Claude Code with
Claude Opus 4.x", "GitHub Copilot", "ChatGPT GPT-5") and where it was used:
source code, tests, docstrings and documentation, and this paper.]

[2. Nature and scope: e.g., code generation from author-written
specifications, refactoring, docstring and doctest drafting, test
scaffolding, copy-editing of the manuscript.]

[3. Human oversight: e.g., "The author designed the architecture, the
model scope and the identification interface, reviewed and edited all
AI-assisted output, and validated estimators against the reference
implementations described above."]

# Acknowledgements

[TODO: funding statement. If none: "This work received no external
funding."] [TODO: thank anyone who reviewed code or tested early versions.]

# References
