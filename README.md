prompred
=============================

Machine learning models in promoter prediction of microorganisms

## Installation 

To use this package, simply clone the project into a local folder

```shell
$ git clone https://github.com/Kleurenprinter/prompred.git
```

## Code Example

The main module *prompred.py* is located in src/

The functionality of prompred can be directly accessed through the terminal. For more information on all implemented functions, simply type:

```shell
$ python prompred.py -h
```

The following script executes a gridsearch for the parameters alpha and gamma for a ridge regression using a third degree polynomial kernel, with features extracted over the [-7 12] and [-7 12] regions of the promoters (with respect to the 35- and 10-box) 

```shell
$ python prompred.py GS -d ../data/external/mut_rand_mod_lib.csv -s -7 12 -7 12 -m ridge -k poly -g 3 --parL alpha gamma --parR 10 10  
``` 

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── log.txt			   <- Log file holding a record all executed scripts
    │   └── result_logger  <- folder holding logs of the results of finished scripts
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── prompred.py    <- Main module, accessible through terminal
    │   ├── log_utils.py   <- functions to create log files
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
