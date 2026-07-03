# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'regrender'
copyright = '2025, yu-ting wei'
author = 'yu-ting wei'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx_copybutton']

exclude_patterns = []

# -- Options for autodoc ------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_class_signature = 'separated'
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'inherited-members': True,
    'show-inheritance': True,
}

# -- Options for autosummary ------------------------------------------------
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "navigation_depth": 2,
    "show_toc_level": 4,
    "github_url": "https://github.com/ytsimon2004/regrender",

    # ─── Linear reading aids ────────────────────────────────
    # show “Previous” / “Next” links at bottom of each page
    "show_prev_next": True,

    # ─── In-page navigation ─────────────────────────────────
    # adds a “Back to top” floating button once you scroll down
    "back_to_top_button": True,

    # ─── Sidebar behavior ───────────────────────────────────
    # keep the sidebar fully expanded by default
    "collapse_navigation": False,

    # include “hidden” links (search, version selector, etc.) in main sidebar
    "sidebar_includehidden": True,

    # which widgets to show in the right-hand sidebar
    "secondary_sidebar_items": [
        "page-toc",  # in-page TOC
        "edit-this-page",  # the button above
        "sourcelink",  # links to the .rst source
    ],

}

# -- Copy Button --------------------------------
copybutton_prompt_text = r'^\$ '
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True
