[![Build](https://github.com/gabalz/cvxreg/actions/workflows/python-package.yml/badge.svg)](https://github.com/gabalz/cvxreg/actions/workflows/python-package.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This Python library implements convex regression algorithms of various papers.

----------------------------------------------------------------------------------------------------
# ALGORITHMS

**Convex Nonparametric Least-Squares (CNLS)** \
`algorithm/cnls/cnls.py`

> Convex Optimization, Section 6.5.5, \
> *Stephen Boyd, Lieven Vandenberghe,* \
> Cambridge University Press, 2004
([book](https://web.stanford.edu/~boyd/cvxbook/)).

**Least-Squares Partition Algorithm (LSPA)** \
`algorithm/lspa/lspa.py`

> Convex Piecewise-Linear Fitting, \
> *Alessandro Magnani, Stephen P. Boyd,* \
> Optimization and Engineering, vol.10, 2009
([paper](https://web.stanford.edu/~boyd/papers/pdf/cvx_pwl_fit.pdf)).

**Convex Adaptive Partitioning (CAP), and FastCAP** \
`algorithm/cap/cap.py`

> Multivariate Convex Regression with Adaptive Partitioning, \
> *Lauren A. Hannah, David B. Dunson,* \
> JMLR, vol.14, 2013
([paper](https://www.jmlr.org/papers/v14/hannah13a.html),
[MATLAB code](https://github.com/laurenahannah/convex-function))

**Partitioning Convex Nonparametric Least-Squares (PCNLS) with uniformly random Voronoi partition** \
`algorithm/pcnls/pcnls_voronoi.py`

> Near-Optimal Max-Affine Estimators for Convex Regression, \
> *Gabor Balazs, Andras Gyorgy, Csaba Szepesvari,* \
> AISTATS, 2015
([paper](https://proceedings.mlr.press/v38/balazs15.html),
[MATLAB code](https://proceedings.mlr.press/v38/balazs15-supp.zip)).

**Adaptive Max-Affine Partitioning (AMAP)** \
`algorithm/amap/amap.py`

> Convex Regression: Theory, Practice, and Applications, Section 6.2.3, \
> *Gabor Balazs,* \
> PhD Thesis, University of Alberta, 2016
([thesis](https://doi.org/10.7939/R3T43J98B),
[MATLAB code](https://gabalz.github.io/code/macsp2016-src.zip)).

**Adaptively Partitioning Convex Nonparametric Least-Squares (APCNLS)** \
`algorithm/apcnls/apcnls.py`

> Adaptively Partitioning Max-Affine Estimators for Convex Regression, \
> *Gabor Balazs,* \
> AISTATS, 2022
([paper](https://proceedings.mlr.press/v151/balazs22a.html)).

----------------------------------------------------------------------------------------------------
# PYTHON ENVIRONMENT

The installation of a minimal virtual environment to show the requirements of running the code.

```bash
python3 -m venv .../pyenv  # creating empty virtual environment
source .../pyenv/bin/activate  # activating the virtual environment

pip install --upgrade pip
pip install --upgrade setuptools

pip install numpy scipy osqp clarabel numba

# For "external algorithms" (in algorithms/external):
pip install scikit-learn scikit-fda xgboost

# Jupyter notebook (Optional):
pip install joblib pandas
pip install widgetsnbextension jupyter matplotlib seaborn papermill
```

---------------------------------------------------------------------------------------------------
# UNIT TESTING

For examples, see the doctests in the files mentioned in the ALGORITHMS section above.

All the doctests can be run by using the nose package:
```bash
source .../pyenv/bin/activate  # if not done yet
pip install pytest
cd .../cvxreg  # go to the root directory of this project
PYTHONPATH=. pytest --doctest-modules common/ optim/ algorithm/
```

---------------------------------------------------------------------------------------------------
# EXPERIMENTS

There is a Jupyter notebook `ipynb/synthetic.ipynb`
which provides basic experimenting on synthetic regression problems.

---------------------------------------------------------------------------------------------------
# DEVELOPMENT SETUP

git clone https://github.com/gabalz/cvxreg.git
cd cvxreg
git config core.hooksPath .githooks

