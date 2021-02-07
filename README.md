
This Python library implements convex regression algorithms of various papers.

----------------------------------------------------------------------------------------------------
# ALGORITHMS

`y2004_cnls/cnls.py`

    **Convex Nonparametric Least-Squares (CNLS)**

    Convex Optimization, Section 6.5.5,
    Stephen Boyd, Lieven Vandenberghe,
    Cambridge University Press, 2004.
    [book](https://web.stanford.edu/~boyd/cvxbook/)

`y2009_lspa/lspa.py`

    **Least-Squares Partition Algorithm (LSPA)**

    Convex piecewise-linear fitting,
    Alessandro Magnani, Stephen P. Boyd,
    Optimization and Engineering, vol.10, 2009.
    [paper](https://web.stanford.edu/~boyd/papers/pdf/cvx_pwl_fit.pdf)

`y2013_cap/cap.py`

    **Convex Adaptive Partitioning (CAP), and FastCAP**

    Multivariate Convex Regression with Adaptive Partitioning,
    Lauren A. Hannah, David B. Dunson,
    JMLR, vol.14, 2013.
    [paper](https://www.jmlr.org/papers/v14/hannah13a.html)
    [MATLAB code](https://github.com/laurenahannah/convex-function)

`y2015_pcnls/pcnls_voronoi.py`

    **Partitioning Convex Nonparametric Least-Squares (PCNLS) with uniformly random Voronoi partition**

    Near-optimal max-affine estimators for convex regression,
    Gabor Balazs, Andras Gyorgy, Csaba Szepesvari,
    AISTATS, 2015.
    [paper](http://jmlr.org/proceedings/papers/v38/balazs15.html)
    [MATLAB code](http://proceedings.mlr.press/v38/balazs15-supp.zip)

`y2016_amap/amap.py`

    **Adaptive Max-Affine Partitioning (AMAP)**

    Convex regression: theory, practice, and applications, Section 6.2.3,
    Gabor Balazs,
    PhD Thesis, University of Alberta, 2016.
    [thesis](https://era.library.ualberta.ca/files/c7d278t254/Balazs_Gabor_201609_PhD.pdf)
    [MATLAB code](https://gabalz.github.io/code/macsp2016-src.zip)

----------------------------------------------------------------------------------------------------
# INSTALLATION

```bash
python -m venv .../pyenv # creating virtual empty environment
source .../pyenv/bin/activate # activating the virtual environment

pip install --upgrade pip # == 20.3.3
pip install --upgrade setuptools # == 41.2.0

pip install nose # == 1.3.7
pip install numpy # == 1.17.2
pip install bottleneck # == 1.2.1 (optional, but recommended)
pip install scipy # == 1.3.1
pip install osqp # == 0.6.1

pip install widgetsnbextension # == 3.5.0
pip install jupyter # == 1.0.0
pip install joblib # == 0.13.2
pip install matplotlib # == 3.0.3
pip install pandas # == 0.24.2
```

---------------------------------------------------------------------------------------------------
# UNIT TESTING

For examples, see the doctests in the above mentioned files.

All the doctests can be run by using the nose package:
```bash
source .../pyenv/bin/activate # if not done yet
cd .../cvxreg # go to the root directory of this project
nosetests --with-doctests --doctest-test
```

---------------------------------------------------------------------------------------------------
# STARTING JUPYTER NOTEBOOK

There is a notebook which provides basic experimenting on synthetic convex regression problems.

To run the Jupyter notebook:
```bash
source .../pyenv/bin/activate # if not done yet
cd .../cvxreg # go to the root directory of this project
jupyter-notebook # then, copy the showed link to your browser and open: ipynb/cvxreg.ipynb
```
