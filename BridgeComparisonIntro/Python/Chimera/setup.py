from setuptools import setup

exec(compile(open('chimera/version.py').read(),
             'chimera/version.py', 'exec'))


setup(name='chimera',
      version=__version__,
      description='Chimera implements bridged inverse probability weighting estimator of Zivich et al. (2022) arXiv',
      packages=['chimera'],
      include_package_data=True,
      author='Paul Zivich'
      )
