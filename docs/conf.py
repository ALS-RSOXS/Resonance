"""Configuration for the Sphinx documentation builder."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
src = root / "src"
sys.path.insert(0, str(src))

project = "resonance"
copyright = "2024, ALS-RSOXS"
author = "ALS-RSOXS"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

html_theme = "furo"
html_title = "resonance Documentation"
html_static_path = ["_static"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0066CC",
        "color-brand-content": "#0066CC",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4A9EFF",
        "color-brand-content": "#4A9EFF",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/ALS-RSOXS/auto-reflect",
    "source_branch": "main",
    "source_directory": "docs/",
}

autodoc_mock_imports = ["bcs"]
