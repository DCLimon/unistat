==========
Resampling
==========

Module for resampling statistics, including bootstrapping and permutations.

This module provides classes for bootstrapping and permutation tests on two
series or samples. It supports tests on means and medians for bootstrapping,
and additional t-tests and Mann-Whitney U for permutations. Results include
observed statistics, distributions, confidence intervals, and p-values.

BootResult
    Dataclass for bootstrap results.
PermResult
    Dataclass for permutation results.
TwoSeriesBootstrap
    Class for bootstrapping two series.
TwoSampleBootstrap
    Class for bootstrapping with a boolean grouping variable.
TwoSeriesPermutation
    Class for permutation tests on two series.
TwoSamplePermutation
    Class for permutation tests with a boolean grouping variable.

Note that while bootstrapped hypothesis tests are possible, they are much more
complicated to perform than permutation hypothesis tests. Bootstrapped
hypothesis tests and p-values are a WIP. While bootstrapped CIs are reliable
and preferred, currently (and probably generally), permutation tests should be
preferred for hypothesis tests. To avoid accidentally using an unreliable or
biased bootstrapped p-value in actual practice, displaying of bootstrapped
p-values is currently controlled by the ``boot_p_value`` Boolean parameter,
which currently defaults to ``False``; setting it to ``True`` will allow access
to the WIP bootstrapped p-value.

Assumes input series are continuous. Handles missing data by dropping NaNs.
Some features, like bootstrap p-values, are experimental.

.. automodule:: unistat.resampling
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: boot_ci_hi, boot_ci_lo, boot_ci_pct, boot_dist, boot_mean, boot_sem, obs_stat, p, perm_dist