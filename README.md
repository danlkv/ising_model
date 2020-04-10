# Work framework

The framework is to streamline the proocess of development experimentation and research (`DER`)

Output of `DER` is `packaged code`, `data` which often represented as plots and `figures`
and `research outcomes` which are usually go in form of article/notes/jupyter notebooks

# DER stages

States

    1. Syntesis of some code, in form of notebooks, usually in easy-to-tinker scratchpad fashion.
    2. Gradually abstracting and packaging pieces of notebooks into .py files using `jupytext`
    3. Plotting theoretical plots from experimentation.
    4. Packaging the solution src for later use by copying or linking files from syntesis
    5. Generating data by running the simulation/recourse heavy computation job
    6. Analysing the data and plotting figures in jupyter notebooks

Transitions

    i -> i+1
    1 -> 3
    3 -> 1
    4 -> 1, 2

After packaging, a testing stage may be added
