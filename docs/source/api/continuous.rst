===============
Continuous Data
===============

This module provides classes for analyzing relationships involving continuous
numerical data, as follows:

+--------------------------+-----------------------------+-----------------------------+-------+
| Class                    | Predictor (IV)              | Outcome (DV)                | Notes |
+==========================+=============================+=============================+=======+
| `CorrStats`              | * numeric predictor         | * numeric outcome           |       |
|                          | * implicit; separate Series | * `pd.Series` of values for |       |
|                          |   for each level            |   DV column                 |       |
+--------------------------+-----------------------------+-----------------------------+-------+
| `TwoSeriesStats`         | * binary predictor          | * numeric outcome           |       |
|                          | * implicit; separate DV     | * `pd.Series` of values for |       |
|                          |   Series for each group     |   DV column                 |       |
+--------------------------+-----------------------------+-----------------------------+-------+
| `TwoSampleStats`         | * binary predictor          | * numeric outcome           |       |
|                          | * Boolean grouping column   | * `pd.Series` of values for |       |
|                          |   for each IV level         |   DV column                 |       |
+--------------------------+-----------------------------+-----------------------------+-------+
| `MultiSeries1WayBGStats` | * categorical predictor     | * numeric outcome           |       |
|                          | * implicit; separate DV     | * `pd.Series` of values for |       |
|                          |   series for each level     |   DV column                 |       |
+--------------------------+-----------------------------+-----------------------------+-------+
| `MultiSample1WayBGStats` | * categorical predictor     | * numeric outcome           |       |
|                          | * categorical grouping      | * `pd.Series` of values for |       |
|                          |   column for each IV level  |   DV column                 |       |
+--------------------------+-----------------------------+-----------------------------+-------+
|                          |                             |                             |       |
|                          |                             |                             |       |
|                          |                             |                             |       |
+--------------------------+-----------------------------+-----------------------------+-------+

All classes support both parametric and non-parametric tests for the same IV/DV
structure. For multi-sample tests, an initial omnibus test is performed; if
significant, pairwise post-hoc tests with *P*-value adjustment are also
performed.

+--------------------------+------------------------------+-------------------------------+
| Class                    | Parametric Test              | Non-Parametric Test           |
+==========================+==============================+===============================+
| `CorrStats`              | Pearson's correlation        | Spearman's correlation        |
+--------------------------+------------------------------+-------------------------------+
| `TwoSeriesStats`         | Welch's *t*-test             | Mann-Whitney *U*-test         |
+--------------------------+------------------------------+-------------------------------+
| `TwoSampleStats`         | Welch's *t*-test             | Mann-Whitney *U*-test         |
+--------------------------+------------------------------+-------------------------------+
| `MultiSeries1WayBGStats` | * **Omnibus:** Welch's ANOVA | * **Omnibus:** Kruskal-Wallis |
|                          | * **Post hoc:** pairwise     | * **Post hoc:** pairwise      |
|                          |   Welch's *t*-tests          |   Mann-Whitney *U*-tests      |
+--------------------------+------------------------------+-------------------------------+
| `MultiSample1WayBGStats` | * **Omnibus:** Welch's ANOVA | * **Omnibus:** Kruskal-Wallis |
|                          | * **Post hoc:** pairwise     | * **Post hoc:** pairwise      |
|                          |   Welch's *t*-tests          |   Mann-Whitney *U*-tests      |
+--------------------------+------------------------------+-------------------------------+
|                          |                              |                               |
+--------------------------+------------------------------+-------------------------------+


Classes assume input series are appropriately typed (e.g., continuous for
numerical tests, boolean/categorical for grouping). Handles missing data by
dropping NaNs.

.. automodule:: unistat.continuous
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: ControlTestStats, parametric, alpha, data