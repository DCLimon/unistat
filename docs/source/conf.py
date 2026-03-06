# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import tomllib

# Project info ---------------------------------------------------------------
fallback_author = 'David Limon, MD'
fallback_version = '0.2.4'

# Add path to src/unistat/ so Sphinx can import the package for autodoc
sys.path.insert(0, os.path.abspath('../../src'))

# Get data from  pyproject.toml, with fallbacks
try:
    with open('../../pyproject.toml', 'rb') as f:
        data = tomllib.load(f)

    # Extract version from pyproject.toml
    release = data['project'].get('version', fallback_version)

    # Extract first listed author from pyproject.toml
    if 'authors' in data['project'] and data['project']['authors']:
        author = data['project']['authors'][0].get('name', fallback_author)
    elif 'author' in data['project']:
        author = data['project'].get('author', fallback_author)
    else:
        author = fallback_author
except (FileNotFoundError, KeyError, tomllib.TOMLDecodeError):
    release = fallback_version  # Fallback version
    author = fallback_author   # Fallback author name

# Project Metadata
project = 'unistat'
copyright = f'2026 {author}'  # Dynamically include author in copyright
version = release  # Short version (e.g., '0.2.0')
release_date = '2025-10-17'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',    # Auto-generate API documentation
    'sphinx.ext.napoleon',   # Support Google/NumPy docstrings
    'sphinx.ext.intersphinx',  # Link to external documentation
    'sphinx.ext.viewcode',   # Add links to source code
    'sphinxcontrib.bibtex',  # BibTeX citation support
]

# Autodoc settings
autodoc_default_options = {
    'members': True,         # Include all members
    'undoc-members': True,   # Include members without docstrings
    'show-inheritance': True,  # Show inheritance hierarchy
}
autoclass_content = 'class'
autodoc_member_order = 'bysource'

# Intersphinx configuration for external links
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# Napoleon settings for Google/NumPy docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# BibTeX configuration
bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'super'

# Add any custom static files (e.g., custom CSS)
html_static_path = ['_static']

# Add any custom templates
templates_path = ['_templates']

# List of patterns to ignore when looking for source files
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'  # Use Read the Docs theme
html_theme_options = {
    'collapse_navigation': False,
    # 'sticky_navigation': True,
    'navigation_depth': 4,
    # 'includehidden': True,
    # 'titles_only': False,
}

# Remove left sidebar from all pages
html_sidebars = {
  "**": []
}

# Optional: Add a logo or favicon if you have one
# html_logo = '_static/logo.png'
# html_favicon = '_static/favicon.ico'

# -- Options for internationalization ----------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-internationalization

language = 'en'

# -- Additional configuration ------------------------------------------------

# Ensure Python code is highlighted correctly
highlight_language = 'python3'

# -- Global reference labels -------------------------------------------------
rst_prolog = f"""
.. |release-date| replace:: {release_date}
.. _pandas-homepage: https://pandas.pydata.org/
.. _scipy-homepage: https://scipy.org/
.. _statsmodels-homepage: https://www.statsmodels.org/stable/index.html
"""
