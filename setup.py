from setuptools import setup, find_packages

with open('README.md', 'r') as readme_file:
    README = readme_file.read()

with open('HISTORY.md', 'r') as history_file:
    HISTORY = history_file.read()


# Most basic version
install_requires = ['numpy', 'pandas', 'joblib']

_extras_light = [
    'mega.py',
    'dask[complete]',
    'matplotlib',
    'scipy',
    'sklearn'
]

_extras_datavis = [
    'sqlalchemy',
    'plotly',
    'dash_daq',
    'dash-bootstrap-components',
    'dash',
]

_extras_all = [
    'torch',
    *_extras_light,
    *_extras_datavis
]


setup_args = dict(
    name="pyadlml",
    version="0.0.8.0alpha",
    url="https://github.com/tcsvn/pyadlml",
    author="Christian Meier",
    description="Sklearn flavored library containing numerous Activity of Daily Livings datasets, preprocessing methods, visualizations and models.",
    long_description_content_type="text/markdown",
    long_description=README + '\n\n' + HISTORY,
    author_email="account@meier-lossburg.de",
    license="MIT",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=dict(
        complete=_extras_all,
        datavis=_extras_datavis,
        light=_extras_light,
    ),
    keywords=['Activity of Daily Living'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
)


if __name__ == '__main__':
    setup(**setup_args)
