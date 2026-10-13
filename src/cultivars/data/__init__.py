# filepath: /src/cultivars/data/__init__.py
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
r"""Data access: the clients that bring series into the package, re-exported.

A reduced form is only as good as the series behind it, and the series
of applied macroeconometrics come from a small number of public
providers whose APIs are worth a client of their own. This package
re-exports those clients under names that say what kind of data they
carry rather than which service serves them, so that a workflow reads
``from cultivars.data import economic`` and the choice of provider is
an implementation detail that can change without the import changing.

Nothing here is estimated or transformed. The clients return
dataframes of observations with their own date index and vintage
handling, and the models in the rest of the package take arrays, so a
fetched frame is reduced to the columns and sample the question needs
before it is passed to a model. The :doc:`examples <auto_examples/index>`
do exactly that for every model family, through a shared helper that
fetches, aligns and labels a dictionary of series in one call.

The providers need credentials. The economic client reads its API key
from the ``FRED_API_KEY`` environment variable or from
:func:`~fedfred.set_api_key`, and raises a clear error when neither is
set; a free key is issued at
https://fred.stlouisfed.org/docs/api/api_key.html. No module in the
package touches the network unless a client is called, so importing
``cultivars.data`` is safe offline.

Layout. :mod:`~cultivars.data.economic` is `fedfred
<https://nikhilxsunder.github.io/fedfred/>`_, a client for the Federal
Reserve Bank of St. Louis FRED and ALFRED databases: series, releases,
sources, categories, tags, vintage dates and the GeoFRED maps, in
synchronous (:class:`~fedfred.FredAPI`) and asynchronous
(:class:`~fedfred.AsyncAPI`) forms, with pandas, polars and dask
output. A financial client for SEC EDGAR filings is planned under the
same package. Both are independent projects with their own
documentation, versioning and release cycle; this package pins a
version and re-exports it, and their APIs are documented at their own
sites rather than here.

Example:
    Fetch two FRED series as a frame, keep the common sample, and hand
    the array to a VAR. Needs a key and a network connection, so the
    example is not run as a test:

    >>> from cultivars.data import economic                       # doctest: +SKIP
    >>> from cultivars.multivariate.reduced_form.vector_autoregression import VAR
    >>> api = economic.FredAPI()                                   # doctest: +SKIP
    >>> frames = [
    ...     api.get_series_observations(code, observation_start="1990-01-01", units=unit)
    ...     for code, unit in (("INDPRO", "pch"), ("FEDFUNDS", "lin"))
    ... ]                                                          # doctest: +SKIP
    >>> data = frames[0].join(frames[1], lsuffix="_ip", rsuffix="_ffr").dropna()  # doctest: +SKIP
    >>> res = VAR(data.to_numpy(), order=4, names=("ip", "ffr")).fit()         # doctest: +SKIP
    >>> res.names                                                  # doctest: +SKIP
    ('ip', 'ffr')
"""

from __future__ import annotations

import fedfred as economic

# A financial client (edgar-sec) is planned: ``import edgar_sec as financial``.

__all__ = ["economic"]
