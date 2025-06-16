# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from importlib.metadata import version as get_version
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ZEN-garden'
copyright = '2025, Reliability and Risk Engineering lab, ETH Zurich'
author = 'Jacob Mannhardt, Alissa Ganter, Johannes Burger, Francesco de Marco, Lukas Kunz, Lukas Schmidt-Engelbertz, Nour Boulos, Christoph Funke, Giovanni Sansavini'
release = get_version("zen_garden")
language = "en"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.mathjax',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
#              "sphinx.ext.autosectionlabel",
              'sphinx_reredirects',
              'nbsphinx',
              'nbsphinx_link',
              'myst_parser',
              "sphinx.ext.imgconverter",  # for SVG conversion
             ]
# allow errors in the notebooks
nbsphinx_allow_errors = True

# Specify the special members to include in the documentation
autodoc_default_options = {
    'members': True,
    'special-members': '__init__',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude all jupyter notebooks
exclude_patterns = ['_build', 'dataset_examples', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints',
                    'files/tutorial/add_carrier.rst', 'files/tutorial/add_policy.rst', 
                    'files/tutorial/add_technology.rst', 'files/tutorial/add_transport.rst', 
                    'files/tutorial/handle_infeasibilities.rst', 'files/api_v2/**', 
                    'files/dataset_examples/**', 'files/developer_guide/testing.rst',
                    'files/welcome/use_cases.rst', 'files/references/release_notes.rst']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

# Theme-specific options to customize the look of a theme
# For a list of options available for each theme, see the documentation.
html_theme_options = {
    "repository_url": "https://github.com/ZEN-universe/ZEN-garden",
    "use_repository_button": True,
    "show_navbar_depth": 1,
    "show_toc_level": 2,
}

# The name for this set of Sphinx documents.  
html_title = "ZEN-garden"
html_short_title = "ZEN-garden"

# The name of an image file (relative to this directory)
html_logo = "files/figures/general/zen_garden_logo_text.png"

# html_favicon = "images/zen_garden_logo.svg"
html_favicon = "files/figures/general/zen_garden_logo_text.png"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
