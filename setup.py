from setuptools import setup, find_packages
from os import path


from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='pyGSM',  # Required
    version='0.1',  # Required
    description='Reaction path searching',  # Optional
    url='https://github.com/ZimmermanGroup/pyGSM',
    author='Cody Aldaz',
    author_email='codyaldaz@gmail.com',

    long_description_content_type='text/markdown',  # Optional (see note above)
    long_description=long_description,  # Optional

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
    install_requires=[
        'numpy>=1.11',
        'networkx',
        'scipy>=1.1.0',
        'six',
        'matplotlib',
        #'openbabel>=2.4.1',
        ],

    entry_points={'console_scripts': [
        'gsm=pygsm.wrappers.main:main',
            ]},

    )

