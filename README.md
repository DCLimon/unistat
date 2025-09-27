# unistat
A unified statistics interface for simple running and displaying of statistics,
as commonly used in medicine and biostatistics. `unistat` is built on top of
SciPy and statsmodels, and aims to integrate coding and displaying results of
or each individual test, so that it's faster to run tests, and display results
for easy reporting.

## Features

* **Regression Analysis**:
  * Univariate and multivariable linear, logistic, and log-binomial regression
  * Integrated display of:
    * VIF
    * Regression coefficients (linear regression),
    * ORs (logit)
    * risk ratios (log-binomial regression)
  * Easy printing methods for quick copy-pasting of coefficients & CIs
* **Contingency Tables**:
  * Multiclass and 2x2 table support, with displays for absolute & percent counts
  * Integrated display of $\chi^2$ & odds ratios
  * For 2x2 tables, displays Fisher's exact test with indicator for which to use
    * Will likely replace include Boschloo's exact test in the near future
* **Correlation**:
  * Pearson and Spearman correlation for continuous variables.
* **2-Sample Tests**:
  * Parametric (t-test) and nonparametric (Mann-Whitney U) comparisons
* **Resampling methods**:
  * Bootstrap confidence intervals for means & medians
  * Permutation hypothesis tests for means, medians, t-tests, and Mann-Whitney U-tests

## Installation

Install StatsPy using pip (assuming it's published to PyPI):

```bash
pip install unistat
```

Ensure dependencies are installed:
* Python 3.10+
* pandas
* numpy
* scipy
* statsmodels

## Usage

Below are brief examples of functionality of some `unistat` modules.
For comprehensive documentation, see our 
[ReadTheDocs page](https://unistat.readthedocs.io/en/latest/).

### Boostrapped 95% CI for Difference of Means

```
import pandas as pd
from unistat.resampling import TwoSeriesBootstrap

# Observed data
test = pd.Series([1, 2, 3, 4, 5], name='test group')
control = pd.Series([2, 3, 4, 5, 6], name='control group')

# Bootstrap test for difference of means
bootstrap = TwoSeriesBootstrap(test, control, test_type='means', n_resamples=1000)
print(bootstrap.results)
```

Output:
```
          test Ha   control
n     5.000000  =  5.000000
mean  3.000000  =  4.000000
std   1.581139  =  1.581139
min   1.000000  =  2.000000
max   5.000000  =  6.000000
            Bootstrapped Hypothesis Test
                * Test Stat: Difference of Means
                * Test Group: test group
                * Control Group: control group
            
        Observed Test Stat: -1.0000
        Mean of Bootstrap Dist.: -0.9962
        SEM of Bootstrap Dist.: 0.8993
        BCa 95% CI: -2.8000 to 0.8000
        No hypothesis testing performed.
```

## Documentation

Detailed usage, including API references and examples, is available on
[ReadTheDocs](https://unistat.readthedocs.io/en/latest/).

## Experimental Features

* log-binomial regression (`LogBinStats`)
  * Buggy implementation, and more research into functionality required
* Bootstrapped p-values
  * Results given are not *wrong*, but there are more robust means to calculate a bootstrapped hypothesis test, particularly in cases of heteroskedastic groups
  * Bootstrapped CIs are reliable
  * Permuted p-values are reliable

Experimental features are prone to unannounced changes. Features may also
contain statistical errors or faux pas, buggy implementation, or be otherwise
unreliable. If these features are used, it is highly recommended that results
are verified using more reliable means (e.g. verify a bootstrapped p-value with
a permutation or asymptotic p-value to ensure the results are at least ballpark
correct), particularly if results are intended for publication or other critical
applications.

## Contributing

Contributions welcome. Please submit issues or pull requests on
[GitHub](https://github.com/DCLimon/unistat). Follow NumPy style guidelines, as
also followed by SciPy and statsmodels.

## License

`unistat` is licensed under the BSD 3-Clause license.

## Author

`unistat` is created and maintained by David C. Limón, MD.
