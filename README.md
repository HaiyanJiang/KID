
# The KID algorithm

The main KID algorithm is defined in `utils_lda.py`, whose core components are wrapped in our new developed `ridger` package.
- The `ridger` package provides the main components of the OLS problem, including the ridge regression and ridgeless regression.
- To obtain the random feature maps, we use the `pyrfm` package, a library for random feature maps in Python, which can be downloaded from https://neonnnnn.github.io/pyrfm/.


## The dependencies of the KID algorithm.
All dependencies are included in `environment.yml`. To install, run
```
conda env create -f environment.yml
```
(Make sure you have installed `Anaconda` before running.)
Then, activate the installed environment by
```
conda activate t1.7
```

## Installation of `ridger` and `pyrfm`

1. The `ridger` provides the main components of the OLS problem.
2. We use the [pyrfm](https://neonnnnn.github.io/pyrfm/) package to get the random feature transformation in high dimensions.


### 1. Installation of `ridger`
Download the source codes, and use pip to install offline
```
pip install ridger-0.0.1.tar.gz
```


### 2. Installation of `pyrfm`
Please follow the instructions from https://neonnnnn.github.io/pyrfm/.

2.1. Download the source codes by:
```
git clone https://github.com/neonnnnn/pyrfm.git
```
or download as a ZIP from GitHub.

2.2. Install the dependencies::
```
cd pyrfm
pip install -r requirements.txt
```
2.3. Finally, build and install pyrfm by:
```
python setup.py install
```


## How to use the package?

### Overview
We provide code and models for our experiments on dataset such as `ARCX`, `ARCZ`, `Colon`, `CNS`, `PGD`, `LSVT`, `Leukemia`.

## Data preparation
By default, we put the datasets in `./data/Colon/` and save trained models in `./data/Colon/Results/` (taking `Colon` for example). You may also use any other directories you like by setting the `--doc_root` argument to `/your/data/path/`, and the `--transformation` argument to the different kernel transformations when running all experiments below.
Alternatively, you can try different activation function by changing the `--actfun` argument.


### Three steps
There are some util functions:
- `utils_lda.py`
- `cv_clf.py`
- `lda_rf_sgd.py`
- `args.py`

1. Change the parameters in `run_rbf.sh` and `run_rft.sh`, and use the following command line to run the simulation
```
# tmux new -s rbf
conda activate t1.7
sh run_rbf.sh  # sh run_rff.sh

# or use the following command line running in the background with no hang up
conda activate t1.7
nohup sh run_rbf.sh >> output_rbf.log 2>&1 &
# jobs -l
```
2. The `lda_rf_summary.py` summarizes the results of the algorithm.
3. The `result_assemble.py` assembles all the results.

### Tuning parameters

- For RBF kernel, set `gamma=1.0` for features transformation.
- For RandomFourier kernel, set `gamma='auto'`, which is equivalent to `gamma=1/X.shape[1]`.
- For RandomKernel, just the default.
