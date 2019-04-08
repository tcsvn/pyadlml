from setuptools import setup, find_packages

setup(
  name = "hassbrain_algorithm",
  version = "0.0.5",
  url = "https://gitlab.com/hassbrain/hassbrain_algorithm",
  author = "Christian Meier",
  author_email = "christian@meier-lossburg.de",
  license = "MIT",
  packages = find_packages(),
  install_requires = ['numpy']
)
