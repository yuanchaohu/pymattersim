# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyMatterSim'
copyright = '2024, Yuan-Chao Hu'
author = 'Yuan-Chao Hu'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser", # enable markdown support
]

myst_enable_extensions = [
    "dollarmath",  # Optional: support for LaTeX-style math
    "colon_fence", # Optional: ::: directive support
    "deflist",     # Optional: definition list support
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_theme = "sphinx_rtd_theme"
html_theme = "furo"
# html_theme = "sphinx_material"
html_static_path = ['_static']

if html_theme == "furo":
    html_css_files = [
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/fontawesome.min.css",
        "furo.css",
    ]
    html_theme_options = {
        "sidebar_hide_name": True,
    }

# sphinx_rtd_theme
"""
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "style_external_links": True,
}
"""

# furo
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#1e88e5",
        "color-brand-content": "#1e88e5",
    },
    "dark_css_variables": {
        "color-brand-primary": "#bb86fc",
        "color-brand-content": "#bb86fc",
    },
}


# material for sphinx
"""
html_theme_options = {
    "nav_title": "My Project",
    "color_primary": "blue",
    "color_accent": "light-blue",
    "repo_url": "https://github.com/username/repo",
    "repo_name": "GitHub",
    "globaltoc_depth": 2,
}
"""