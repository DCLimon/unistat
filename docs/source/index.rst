.. unistat documentation master file, created by
   sphinx-quickstart on Wed Sep 17 03:14:30 2025.

unistat Documentation (Version 0.2.3)
=====================================

Welcome to the documentation for ``unistat``, a Python library to simplify
performing and reporting of medical, biostatistics, and social sciences
statistical analyses.

``unistat`` is built on top of common Python data analysis & statistics
libraries, including `pandas <https://pandas.pydata.org/>`__,
`SciPy <https://scipy.org/>`__, and
`statsmodels <https://www.statsmodels.org/stable/index.html>`__. This
library aims to implement best practices for publication-quality statistical
analysis, and to implement a simple, straightforward API to to run these
analyses, and view all data that is pertinent for reporting in the context
of academic publications.

Accordingly, unlike the statistics libraries on which ``unistat`` is built,
``unistat`` is relatively *opinionated*: whereas parent libraries tend to
offer optionality in statistical methodologies, this library often selects
approaches that are generally accepted best practices for academic
publication, or at least are justifiable in such a setting. As a corollary,
``unistat`` documentation aims to offer copious citations for its chosen
methods, so that a manuscript *Methods* section can appropriately justify
any methodological choices.

``unistat``'s *raison d'être* was to simplify statistics for medical (in
particular, surgical) clinical research; as such, to the extent that
accepted statistical methodologies in medical/surgical research differ from
other biostatistics or social sciences, the norms for clinical surgical
research will be prioritized. Nonetheless, where genuine methodological
optionality exists, some degree of choice is left to the user. At times,
this may not be easily accessible or obvious in the API, and in those cases,
users should access and choose non-default options only with informed
rationale for doing so.

This documentation covers ``unistat`` **version 0.2.3**, released
2025-10-17.

Getting Started
---------------

Install ``unistat`` with:

.. code-block:: bash

    pip install unistat

In future updates, more information will be available in the
:doc:`getting_started` guide.

Examples
--------

Coming in future updates to :doc:`examples`.

API Reference
-------------

Detailed documentation for ``unistat`` modules:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   examples
   api/contingency
   api/continuous
   api/exceptions
   api/formula_regression
   api/regression
   api/resampling

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`