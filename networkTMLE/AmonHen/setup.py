from setuptools import setup

exec(compile(open('amonhen/version.py').read(),
             'amonhen/version.py', 'exec'))


setup(name='amonhen',
      version=__version__,
      description='AmonHen is implementations of stochastic TMLE',
      keywords='TMLE',
      packages=['amonhen'],
      include_package_data=True,
      author='Paul Zivich'
      )
