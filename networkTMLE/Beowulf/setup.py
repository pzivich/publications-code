from setuptools import setup

exec(compile(open('beowulf/version.py').read(),
             'beowulf/version.py', 'exec'))


setup(name='beowulf',
      version=__version__,
      description='Beowulf is the various data generating mechanisms',
      keywords='DGM',
      packages=['beowulf'],
      include_package_data=True,
      author='Paul Zivich'
      )
