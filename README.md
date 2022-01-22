
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

> Convex piecewise-linear fitting, \
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

> Near-optimal max-affine estimators for convex regression, \
> *Gabor Balazs, Andras Gyorgy, Csaba Szepesvari,* \
> AISTATS, 2015
([paper](http://jmlr.org/proceedings/papers/v38/balazs15.html),
[MATLAB code](http://proceedings.mlr.press/v38/balazs15-supp.zip)).

**Adaptive Max-Affine Partitioning (AMAP)** \
`algorithm/amap/amap.py`

> Convex regression: theory, practice, and applications, Section 6.2.3, \
> *Gabor Balazs,* \
> PhD Thesis, University of Alberta, 2016
([thesis](https://era.library.ualberta.ca/files/c7d278t254/Balazs_Gabor_201609_PhD.pdf),
[MATLAB code](https://gabalz.github.io/code/macsp2016-src.zip)).

----------------------------------------------------------------------------------------------------
# PYTHON ENVIRONMENT

The installation of a minimal virtual environment to show the requirements of running the code.

```bash
python -m venv .../pyenv  # creating empty virtual environment
source .../pyenv/bin/activate  # activating the virtual environment

pip install --upgrade pip
pip install --upgrade setuptools

pip install nose
pip install numpy
pip install scipy
pip install osqp

pip install widgetsnbextension
pip install jupyter
pip install joblib
pip install matplotlib
pip install pandas
```

---------------------------------------------------------------------------------------------------
# UNIT TESTING

For examples, see the doctests in the files mentioned in the ALGORITHMS section above.

All the doctests can be run by using the nose package:
```bash
source .../pyenv/bin/activate  # if not done yet
cd .../cvxreg  # go to the root directory of this project
PYTHONPATH=. nosetests --with-doctests --doctest-test
```

---------------------------------------------------------------------------------------------------
# EXPERIMENTS

There is a Jupyter notebook `ipynb/cvxreg_synt.ipynb`
which provides basic experimenting on synthetic convex regression problems.

