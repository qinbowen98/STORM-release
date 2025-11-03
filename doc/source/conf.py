# Configuration file for the Sphinx documentation builder.

import os
import sys
from unittest.mock import MagicMock

# -- Path setup --------------------------------------------------------------
# ✅ 确保 storm 可以被找到
sys.path.insert(0, os.path.abspath('../..'))

# ✅ Mock 不可安装模块（必须在此位置）
MOCK_MODULES = ['spams', 'openslide', 'pyvips', 'cv2', 'models']
for mod in MOCK_MODULES:
    sys.modules[mod] = MagicMock()


# -- Project information -----------------------------------------------------
project = 'STORM'
author = 'Zongxu Zhang'
copyright = '2025'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'jupyter_sphinx'
]

templates_path = ['_templates']
exclude_patterns = []

jupyter_execute_notebooks = "auto"

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
