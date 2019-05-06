from setuptools import setup, find_packages

setup(
  name = "hbhmm",
  version = "0.0.5",
  url = "https://gitlab.com/hassbrain/hassbrain_algorithm/hbhmm",
  author = "Christian Meier",
  author_email = "christian@meier-lossburg.de",
  license = "MIT",
  packages = find_packages(),
  install_requires = ['numpy']
)
