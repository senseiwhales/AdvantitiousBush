# conf.py

import os
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'AdvantitiousBush'
author = 'senseiwhales'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings
extensions = []

# The suffix of source filenames
source_suffix = '.rst'

# The master toctree document
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'
