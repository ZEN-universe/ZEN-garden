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
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ZEN-garden'
copyright = '2024, Jacob Mannhardt, Alissa Ganter, Johannes Burger, Francesco de Marco, Giovanni Sansavini'
author = 'Jacob Mannhardt, Alissa Ganter, Johannes Burger, Francesco de Marco, Giovanni Sansavini'
release = 'v1.2.0'
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
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              "sphinx.ext.autosectionlabel",
              'sphinx_reredirects',
              'nbsphinx',
              'nbsphinx_link',
              'myst_parser',
              "sphinx.ext.imgconverter",  # for SVG conversion
             ]

# Specify the special members to include in the documentation
autodoc_default_options = {
    'special-members': '__init__',
}
autodoc_flags = ['members']
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


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
html_logo = "images/zen_garden_logo_text.png"

html_favicon = "images/zen_garden_logo.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
