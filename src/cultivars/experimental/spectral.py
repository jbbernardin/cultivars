# filepath: /src/cultivars/experimental/spectral.py
#
# Copyright (c) 2026 Nikhil Sunder
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
r"""Non-Stationary Spectral Decomposition Network, re-exported from ``ns-sdn``.

The NS-SDN (Sunder and Leon, 2026) is a neural state-space model of a
scalar series as a trend plus :math:`K` sinusoids whose amplitude,
instantaneous frequency, phase offset and participation evolve with a
latent state,

.. math::

   \hat y_t = B_t + \sum_{k=1}^{K} w_k\, g_{t,k}\, A_{t,k}\,
   \sin(\theta_{t,k} + \phi_{t,k}),
   \qquad
   \theta_{t,k} = \theta_{t-1,k} + \omega_{t,k}\,\Delta t_t,
   \qquad
   h_t = F_\psi(h_{t-1}, x_t, \Delta t_t),

trained as a one-step forecaster and read as a time-varying spectral
decomposition: the result carries the trend, the per-component
amplitude, frequency, phase and gate paths, and the spectral-state
coordinates (power-weighted centroid, dispersion, effective rank and
Wasserstein motion) that summarize where the power sits and how it
moves. It answers the question :mod:`cultivars.spectral` answers for
stationary systems -- at which frequencies, with what power -- for a
series whose frequencies drift.

The model lives in its own package, ``ns-sdn``, because it is estimated
by gradient descent in ``torch`` rather than on numpy and scipy, which
is the contract the rest of cultivars keeps. This module is the bridge:
every public name of ``ns_sdn`` is re-exported here under an ``NSSDN``
prefix, so a cultivars user reaches the model through one import path
and never confuses its records with the package's own. ``NSSDNResult``
is not a cultivars result and ``NSSDNSummaryTable`` is not
:class:`cultivars.summary.SummaryTable`: the two packages share the
model-fit-result design and the exception hierarchy by construction,
but their records are distinct types, and a consumer written against
cultivars protocols (identification, backtests, spectral views) does not
accept an ``NSSDNResult``. What does carry across is the shape of the
error handling: ``NSSDNDimensionError``, ``NSSDNSpecificationError`` and
``NSSDNNumericalError`` derive from ``ValueError`` and ``RuntimeError``
exactly as their cultivars namesakes do, so one ``except`` clause covers
both packages.

Held to the :mod:`cultivars.experimental` rules. The package is pinned
to one pre-release version in ``pyproject.toml``, the API may change in
any minor release, and the model is promoted into
:mod:`cultivars.spectral` once the replication fixture against the
published DGS10 run lands and the surface has held for two minor
releases. The defaults reproduce that run -- sixteen components, a
128-dimensional GRU state, the filtered alignment, min-max scaling and
a two-stage Adam schedule of 3,000 steps -- so every exploratory fit
should pass a shorter ``schedule`` of :class:`NSSDNStage` records.

Layout. :class:`NSSDN` is the specification and :class:`NSSDNResult`
the frozen record ``fit()`` returns, with ``decompose``, ``forecast``,
``predict`` and ``spectral_state``; :class:`NSSDNStage` and
:class:`NSSDNPenalties` are the schedule records the constructor takes;
:class:`NSSDNSpectralState` is the measure-valued reading returned by
``spectral_state()``; :class:`NSSDNSummaryTable` and
:class:`NSSDNInformationCriteria` are the reporting records;
:class:`NSSDNAlignment`, :class:`NSSDNObjective`, :class:`NSSDNScaling`
and :class:`NSSDNTransition` are the categorical option aliases; and
the four ``NSSDN...Error`` names are the error hierarchy. The
``ns_sdn.engine`` layers are private and are not re-exported.

References:
    Sunder, N., & Leon, M. (2026). Non-Stationary Spectral Decomposition
    Network: Adaptive Spectral Emission Heads and Frequency Modulation.
    *IEEE ORSS*.

    Sunder, N. (2026). Non-Stationary Spectral Decomposition Network for
    Econometric Time Series Forecasting. *FLAIRS-39*.

Example:
    A chirp with a step in amplitude and a linear trend, fitted with a
    short schedule. The decomposition has one row per observation; the
    one-step fit has one fewer:

    >>> import numpy as np
    >>> from cultivars.experimental.spectral import NSSDN, NSSDNStage
    >>> t = np.arange(300.0)
    >>> amplitude = 0.4 + 0.3 * (t > 150)
    >>> phase = np.cumsum(0.15 + 0.0004 * t)
    >>> y = 0.002 * t + amplitude * np.sin(phase)
    >>> res = NSSDN(y, k=3, state_dim=16, schedule=(NSSDNStage(1e-2, 150),), seed=0).fit()
    >>> res.emission.shape, res.amplitude.shape, res.fittedvalues.shape
    ((300,), (300, 3), (299,))
    >>> bool(np.allclose(res.trend + res.components.sum(axis=1), res.emission, atol=1e-5))
    True
    >>> res.forecast(5).shape
    (5,)
"""

from ns_sdn.exceptions import DimensionError as NSSDNDimensionError
from ns_sdn.exceptions import NSSDNError
from ns_sdn.exceptions import NumericalError as NSSDNNumericalError
from ns_sdn.exceptions import SpecificationError as NSSDNSpecificationError
from ns_sdn.spectral_decomposition import NSSDN, NSSDNResult
from ns_sdn.spectral_decomposition import Penalties as NSSDNPenalties
from ns_sdn.spectral_decomposition import Stage as NSSDNStage
from ns_sdn.spectral_state import SpectralState as NSSDNSpectralState
from ns_sdn.summary import InformationCriteria as NSSDNInformationCriteria
from ns_sdn.summary import SummaryTable as NSSDNSummaryTable
from ns_sdn.typing import Alignment as NSSDNAlignment
from ns_sdn.typing import Objective as NSSDNObjective
from ns_sdn.typing import Scaling as NSSDNScaling
from ns_sdn.typing import Transition as NSSDNTransition

__all__ = [
    "NSSDN",
    "NSSDNAlignment",
    "NSSDNDimensionError",
    "NSSDNError",
    "NSSDNInformationCriteria",
    "NSSDNNumericalError",
    "NSSDNObjective",
    "NSSDNPenalties",
    "NSSDNResult",
    "NSSDNScaling",
    "NSSDNSpecificationError",
    "NSSDNSpectralState",
    "NSSDNStage",
    "NSSDNSummaryTable",
    "NSSDNTransition",
]
