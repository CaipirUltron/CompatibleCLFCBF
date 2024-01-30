from setuptools import setup, find_packages

setup(name='compatible_clf_cbf',
      version='1.0.0',
      description='Safety-critical control based on CLF and CBFs',
      url='https://github.com/CaipirUltron/CompatibleCLFCBF',
      author='Matheus Reis',
      license='GPLv3',
      packages=find_packages(include=['compatible_clf_cbf.*']),
      zip_safe=False,
      install_requires=[ 'numpy', 'scipy', 'sympy', 'matplotlib', 'qpsolvers', 'picos', 'cvxpy', 'shapely' ])