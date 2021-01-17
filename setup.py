from setuptools import setup, find_packages

with open('README.md', 'r') as readme_file:
  README = readme_file.read()

with open('HISTORY.md', 'r') as history_file:
  HISTORY = history_file.read()

setup_args = dict(
  name = "pyadlml",
  version = "0.0.6.9.2-alpha",
  url = "https://github.com/tcsvn/pyadlml",
  author = "Christian Meier",
  description = "Sklearn like library supporting numerous Activity of Daily Livings datasets",
  long_description_content_type="text/markdown",
  long_description=README + '\n\n' + HISTORY,
  author_email = "account@meier-lossburg.de",
  license = "MIT",
  packages = find_packages(),
  keywords = ['Activity of Daily Living'],
  classifiers = [
    "Programming Language :: Python :: 3",
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Visualization"
  ],
)

install_requires = ['numpy', 'pandas', 'joblib', 'mega.py', 'dask[complete]', 
                    'matplotlib', 'scipy', 'sklearn'
]

if __name__ == '__main__':
  setup(**setup_args, install_requires=install_requires)
