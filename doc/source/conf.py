# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from unittest.mock import MagicMock
project = 'STORM'
copyright = '2025, Zongxu Zhang'
author = 'Zongxu Zhang'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'jupyter_sphinx'
]

templates_path = ['_templates']
exclude_patterns = []

napoleon_google_docstring = True
napoleon_numpy_docstring = False
doctest_test_doctest_blocks = 'default'
jupyter_execute_notebooks = "off"
nbsphinx_execute = 'never'
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

sys.path.insert(0, os.path.abspath('../..'))
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()
MOCK_MODULES = ['openslide','pyvips','cv2', 'models']
sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
