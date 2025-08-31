# Future Changes

Future changes for new versions of `unistat`

## Complete `README.md`


## Improve code documentation

* Docstrings everywhere
* Type hints everywhere

## Implement integration tests


## Implement a univariate summary stats caller

* see `summ_stats()` & `groupby_summ_stats` in `ax-subclav` project
* Should basically allow a univariate or by-category Table 1 to be made easily


## Changes to `TwoSampleStats`

## Changes to `RegressionStats` base class

### Add a ProbitStats class

### Add a log-binomial class to estimate RR

## Greater categorical support for 3+ levels

* `MulticlassContingencyStats` class should include method for pairwise post-hoc testing
  * May be best implemented by repeatedly calling `BooleanContingencyStats`, and then manually implementing Holm-Bonferroni (or other) p-value correction
  * o/w, `ax-subclav` project has a jank post-hoc fxn ripped from somewhere online
    * GitHub may have it starred
* formal ANOVA omnibus testing class
  * should include a pairwise t-testing post-hoc method
* formal Kruskal-Wallis omnibus testing class
  * should include a pairwise M-W U-testing post-hoc method


## Integrate with Polars DataFrames

* Low-priority
* Will likely just be a `polars_df.to_pandas()` call to integrate well with SciPy & statsmodels


## Implement some basic plotting functions for common plots

* Univariate combined histogram & Q-Q figure
* Possibly a bivariate/categorical multi-axis hist/QQ figure??
  * low priority
* A colorful univariate logistic regression plot, if feasible
  * could be a method in `LogitStats` class, e.g. `.univariate_plot(X: str)`
  * Is there an similar plot that could be drawn in the case of a continuous Logit predictor?
  * Is a LinReg version possible too?
