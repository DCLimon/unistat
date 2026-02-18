# Future Changes

Future changes for new versions of `unistat`

-------------------------------------------------------------------------------


## Complete `README.md`

-------------------------------------------------------------------------------


## Improve code documentation

* Docstrings everywhere
* Type hints everywhere

-------------------------------------------------------------------------------


## Implement integration tests

-------------------------------------------------------------------------------


## Implement error handling

* Should probably implement an error for when calling a standardized 
  regression but all predictor columns are Boolean
  * Seems like this may no longer be an issue as of `v0.2.3`, but fully check 
    before removing this item

-------------------------------------------------------------------------------


## Implement a univariate summary stats caller

* see `summ_stats()` & `groupby_summ_stats` in `ax-subclav` project
* Should basically allow a univariate or by-category Table 1 to be made easily

-------------------------------------------------------------------------------


## Changes to `resampling.py`

### Bootstrapping classes

* Unbiased calculation of p-values using bootstrapped distribution under the 
  null hypothesis

### Permutation classes

* Support for regression models

-------------------------------------------------------------------------------


## Changes to `TwoSampleStats`

-------------------------------------------------------------------------------


## Changes to `RegressionStats` base class

See if there's some sort of modified VIF that gets used with Logit or in 
other cases. I vaguely remember this coming up when calculating some regression 
in R for the Retro HypoCa stats.

### Add feature selection support

* backward feature elimination
* augmented backward feature elimination
* stepwise feature selection
* +/- forward selection

### Consider beginning to deprecate `RegressionStats`?

* Maybe `FormulaRegression` is useful enough
* Maybe the API in-progress in `feat/keyword-regression` branch could take 
  over as an improved `RegressionStats` API
  * with `FormulaRegression` working, it is no longer imperative for a 
    keyword-based regression class to have complete API coverage

### Add a ProbitStats class

### Add a log-binomial class to estimate RR

Has had an experimental implementation, but is buggy.

#### Known bugs:

* **RR point estimate falls outside the bounds of displayed 95% CI**
  * Cause has not yet been isolated
  * Example below taken from `tests/test_regression.py`:
```
>>> print(
...     LogBinStats(
...         X=outcome_df[[
...             'ca_grams_per_unit_4h',  # pd.Float64
...             'age',  # pd.Int64
...             'iss',  # pd.Int64
...             'moi_pen',  # pd.Boolean
...             'sex_female',  # boolean-like float
...         ]],
...         y=outcome_df['mortality_24h'],
...         bool_col_names=['moi_pen', 'sex_female']
...     )
... )

ca_grams_per_unit_4h    1.776993
age                     3.100851
iss                     2.479153
moi_pen                 1.467021
sex_female              1.396250
Name: vif, dtype: float64
                 Results: Generalized linear model
====================================================================
Model:                 GLM               AIC:             nan       
Link Function:         Log               BIC:             -1350.2242
Dependent Variable:    mortality_24h     Log-Likelihood:  nan       
Date:                  2026-02-05 12:32  LL-Null:         -107.58   
No. Observations:      268               Deviance:        114.61    
Df Model:              5                 Pearson chi2:    6.32e+15  
Df Residuals:          262               Scale:           1.0000    
Method:                IRLS                                         
--------------------------------------------------------------------
                      Coef.  Std.Err.    z    P>|z|   [0.025  0.975]
--------------------------------------------------------------------
const                -2.7876   0.6004 -4.6433 0.0000 -3.9643 -1.6110
ca_grams_per_unit_4h -1.5291   0.5742 -2.6629 0.0077 -2.6546 -0.4036
age                   0.0255   0.0076  3.3741 0.0007  0.0107  0.0404
iss                   0.0151   0.0110  1.3781 0.1682 -0.0064  0.0366
moi_pen               0.1943   0.3181  0.6109 0.5413 -0.4291  0.8178
sex_female            0.0516   0.3365  0.1534 0.8781 -0.6079  0.7111
====================================================================
                            RR  95% CI lower  95% CI upper
const                 1.063502      0.018981      0.199697
ca_grams_per_unit_4h  1.242010      0.070329      0.667894
age                   2.789489      1.010758      1.041185
iss                   2.760014      0.993636      1.037305
moi_pen               3.368553      0.651071      2.265451
sex_female            2.866193      0.544505      2.036301
                   Results: Generalized linear model
========================================================================
Model:                 GLM                 AIC:               nan       
Link Function:         Log                 BIC:               -1350.2242
Dependent Variable:    mortality_24h       Log-Likelihood:    nan       
Date:                  2026-02-05 12:32    LL-Null:           -107.58   
No. Observations:      268                 Deviance:          114.61    
Df Model:              5                   Pearson chi2:      6.32e+15  
Df Residuals:          262                 Scale:             1.0000    
Method:                IRLS                                             
------------------------------------------------------------------------
                          Coef.  Std.Err.    z    P>|z|   [0.025  0.975]
------------------------------------------------------------------------
const                    -2.4143   0.2709 -8.9132 0.0000 -2.9451 -1.8834
ca_grams_per_unit_4h_std -0.9064   0.3404 -2.6629 0.0077 -1.5735 -0.2392
age_std                   0.4493   0.1332  3.3741 0.0007  0.1883  0.7103
iss_std                   0.1966   0.1427  1.3781 0.1682 -0.0830  0.4762
moi_pen                   0.1943   0.3181  0.6109 0.5413 -0.4291  0.8178
sex_female                0.0516   0.3365  0.1534 0.8781 -0.6079  0.7111
========================================================================
                                RR  95% CI lower  95% CI upper
const                     1.093555      0.052595      0.152077
ca_grams_per_unit_4h_std  1.497790      0.207323      0.787221
age_std                   4.793366      1.207214      2.034618
iss_std                   3.377892      0.920341      1.609949
moi_pen                   3.368553      0.651071      2.265451
sex_female                2.866193      0.544505      2.036301
```
* When passing a custom warning message to `ExperimentalWarning` instance, 
  both the custom & default messages are displayed in console
  * Example below taken from `tests/test_regression.py`:
```
>>> print(logbin_1float_2int_2bool())

C:\Users\dclim\PycharmProjects\unistat\src\unistat\regression.py:584: ExperimentalWarning: 
        
            This feature is experimental and has not been fully tested nor 
            optimized; calculations may be incorrect, and/or errors may occur. 
            In critical applications, output should be verified for accuracy 
            and/or LogitStats preferred instead.
         is still experimental, and may contain errors or
        fail to function as expected. In critical applications, output
        should be carefully checked, or an alternative used.
        
  warn_experimental('''
```
* Producing multiple `statsmodels` warnings:
  * `DomainWarning`:
```
>>> print(logbin_1float_2int_2bool())

C:\Users\dclim\PycharmProjects\unistat\.venv-tests\Lib\site-packages\statsmodels\genmod\generalized_linear_model.py:308: DomainWarning: The Log link function does not respect the domain of the Binomial family.
  warnings.warn((f"The {type(family.link).__name__} link function "
C:\Users\dclim\PycharmProjects\unistat\.venv-tests\Lib\site-packages\statsmodels\genmod\generalized_linear_model.py:308: DomainWarning: The Log link function does not respect the domain of the Binomial family.
  warnings.warn((f"The {type(family.link).__name__} link function "
```
* * `RuntimeWarning`:
```
>>> print(logbin_1float_2int_2bool())

C:\Users\dclim\PycharmProjects\unistat\.venv-tests\Lib\site-packages\statsmodels\genmod\families\family.py:1056: RuntimeWarning: invalid value encountered in log
  special.gammaln(n - y + 1) + y * np.log(mu / (1 - mu + 1e-20)) +
C:\Users\dclim\PycharmProjects\unistat\.venv-tests\Lib\site-packages\statsmodels\genmod\families\family.py:1057: RuntimeWarning: invalid value encountered in log
  n * np.log(1 - mu + 1e-20)) * var_weights
C:\Users\dclim\PycharmProjects\unistat\.venv-tests\Lib\site-packages\statsmodels\genmod\families\family.py:1056: RuntimeWarning: invalid value encountered in log
  special.gammaln(n - y + 1) + y * np.log(mu / (1 - mu + 1e-20)) +
C:\Users\dclim\PycharmProjects\unistat\.venv-tests\Lib\site-packages\statsmodels\genmod\families\family.py:1057: RuntimeWarning: invalid value encountered in log
  n * np.log(1 - mu + 1e-20)) * var_weights
```

-------------------------------------------------------------------------------


## Changes to `formula_regression.py`

### Implement `FormulaProbit` class

### Implement `FormulaLogBin` class

-------------------------------------------------------------------------------


## Changes to `RegressionStats` base class

-------------------------------------------------------------------------------


## Greater categorical support for 3+ levels

* `MulticlassContingencyStats` class should include method for pairwise 
  post-hoc testing
  * May be best implemented by repeatedly calling `BooleanContingencyStats`,
    and then manually implementing Holm-Bonferroni (or other) p-value correction
  * o/w, `ax-subclav` project has a jank post-hoc fxn ripped from somewhere 
    online
    * GitHub may have it starred
* formal ANOVA omnibus testing class
  * should include a pairwise t-testing post-hoc method
* formal Kruskal-Wallis omnibus testing class
  * should include a pairwise M-W U-testing post-hoc method

-------------------------------------------------------------------------------


## Integrate with Polars DataFrames

* Low-priority
* Will likely just be a `polars_df.to_pandas()` call to integrate well with 
  SciPy & statsmodels

-------------------------------------------------------------------------------


## Implement some basic plotting functions for common plots

* Univariate combined histogram & Q-Q figure
* Possibly a bivariate/categorical multi-axis hist/QQ figure??
  * low priority
* A colorful univariate logistic regression plot, if feasible
  * could be a method in `LogitStats` class, e.g. `.univariate_plot(X: str)`
  * Is there an similar plot that could be drawn in the case of a continuous 
    Logit predictor?
  * Is a LinReg version possible too?

-------------------------------------------------------------------------------
