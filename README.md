
This Python library implements convex regression algorithms of various papers.

----------------------------------------------------------------------------------------------------
# ALGORITHMS

**Convex Nonparametric Least-Squares (CNLS)** \
`y2004_cnls/cnls.py`

> Convex Optimization, Section 6.5.5, \
> *Stephen Boyd, Lieven Vandenberghe,* \
> Cambridge University Press, 2004
([book](https://web.stanford.edu/~boyd/cvxbook/)).

**Least-Squares Partition Algorithm (LSPA)** \
`y2009_lspa/lspa.py`

> Convex piecewise-linear fitting, \
> *Alessandro Magnani, Stephen P. Boyd,* \
> Optimization and Engineering, vol.10, 2009
([paper](https://web.stanford.edu/~boyd/papers/pdf/cvx_pwl_fit.pdf)).

**Convex Adaptive Partitioning (CAP), and FastCAP** \
`y2013_cap/cap.py`

> Multivariate Convex Regression with Adaptive Partitioning, \
> *Lauren A. Hannah, David B. Dunson,* \
> JMLR, vol.14, 2013
([paper](https://www.jmlr.org/papers/v14/hannah13a.html),
[MATLAB code](https://github.com/laurenahannah/convex-function))

**Partitioning Convex Nonparametric Least-Squares (PCNLS) with uniformly random Voronoi partition** \
`y2015_pcnls/pcnls_voronoi.py`

> Near-optimal max-affine estimators for convex regression, \
> *Gabor Balazs, Andras Gyorgy, Csaba Szepesvari,* \
> AISTATS, 2015
([paper](http://jmlr.org/proceedings/papers/v38/balazs15.html),
[MATLAB code](http://proceedings.mlr.press/v38/balazs15-supp.zip)).

**Adaptive Max-Affine Partitioning (AMAP)** \
`y2016_amap/amap.py`

> Convex regression: theory, practice, and applications, Section 6.2.3, \
> *Gabor Balazs,* \
> PhD Thesis, University of Alberta, 2016
([thesis](https://era.library.ualberta.ca/files/c7d278t254/Balazs_Gabor_201609_PhD.pdf),
[MATLAB code](https://gabalz.github.io/code/macsp2016-src.zip)).

----------------------------------------------------------------------------------------------------
# PYTHON ENVIRONMENT

The installation of a minimal virtual environment to show the requirements of running the code. The library has been tested with Python 3.5.2 and the shown package versions, but it should work with newer Python and packages as well.

```bash
python -m venv .../pyenv # creating empty virtual environment
source .../pyenv/bin/activate # activating the virtual environment

pip install --upgrade pip  # == 20.3.3
pip install --upgrade setuptools  # == 41.2.0

pip install nose  # == 1.3.7
pip install numpy  # == 1.17.2
pip install bottleneck  # == 1.2.1 (optional, but recommended)
pip install scipy  # == 1.3.1
pip install osqp  # == 0.6.1

pip install widgetsnbextension  # == 3.5.0
pip install jupyter  # == 1.0.0
pip install joblib  # == 0.13.2
pip install matplotlib  # == 3.0.3
pip install pandas  # == 0.24.2
```

---------------------------------------------------------------------------------------------------
# UNIT TESTING

For examples, see the doctests in the files mentioned in the ALGORITHMS section above.

All the doctests can be run by using the nose package:
```bash
source .../pyenv/bin/activate  # if not done yet
cd .../cvxreg  # go to the root directory of this project
export PYTHONPATH=.
nosetests --with-doctests --doctest-test
```

---------------------------------------------------------------------------------------------------
# EXPERIMENTS

There is a notebook which provides basic experimenting on synthetic convex regression problems.

To run the Jupyter notebook:
```bash
source .../pyenv/bin/activate # if not done yet
cd .../cvxreg  # go to the root directory of this project
jupyter-notebook  # other options: --ip <ip|hostname> --port <port> --no-browser
```
Then open the notebook `ipynb/cvxreg.ipynb`, review its description, and enjoy...
