from setuptools import setup, find_namespace_packages

setup(name='compatible_clf_cbf',
      version='0.1',
      description='Packages for QP control based on compatible CLF/CBFs',
      url='https://github.com/CaipirUltron/CompatibleCLFCBF',
      author='Matheus Reis',
      license='GPLv3',
      packages=find_namespace_packages(include=['compatible_clf_cbf.*']),
      zip_safe=False)