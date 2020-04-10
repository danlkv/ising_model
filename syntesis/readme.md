# Syntesis

This folder is a forgery of novelity. 
Experimentation and dirty hacks for quick testing is a welcome guest here.
Gradual packaging is done in following stages:
    
    0. Go wild with experiments
    1. Absract out pieces of code: define variables and make a function
    2. Convertt to script and refactor
    3. Move/link to the package, write tests and documentation.

## Linking notebooks

use jupytext to link notebooks with scripts,
https://jupytext.readthedocs.io/en/latest/using-server.html#paired-notebooks

    jupytext --set-formats ipynb,scripts//py --sync Preparations.ipynb


