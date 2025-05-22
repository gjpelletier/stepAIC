from setuptools import setup
import sys
sys.path.insert(0, ".")
from stepAIC import __version__

setup(
    name='stepAIC',
    version=__version__,
    author='Greg Pelletier',
    py_modules=['stepAIC'], 
    install_requires=['numpy','pandas','statsmodels','scikit-learn','tabulate','matplotlib'],
)