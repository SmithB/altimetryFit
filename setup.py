import os
from setuptools import setup, find_packages

# get long_description from README.md
with open("README.md", "r") as fh:
    long_description = fh.read()

# list of all scripts to be included with package
scripts = [os.path.join('scripts',f) for f in os.listdir('scripts')]

scripts = [os.path.join('scripts',f) for f in os.listdir('scripts') if not (f[0]=='.' or f[-1]=='~' or os.path.isdir(os.path.join('scripts', f)))] +\
    [os.path.join('altimetryFit', f) for f in ['fit_OIB_aug.py', 'fit_altimetry.py'] +\
     [os.path.join('register_DEMs/', f) for f in ['register_WV_DEM_with_IS2.py', 'register_WV_DEM_with_CS2.py' ]]
    ]
print(scripts)
setup(
    name='altimetryFit',
    version='1.0.0.0',
    description='scripts to fit altimetry data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='',
    author='Ben Smith', 
    author_email='besmith@uw.edu',
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='IceBridge',
    packages=find_packages(),
    scripts=scripts,
)
