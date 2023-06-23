source ~/venvs/pyadlml/bin/activate

echo 'creating distribution'
python3 setup.py sdist bdist_wheel

echo 'uploading to pypi'
python3 -m twine upload --verbose dist/*

echo 'cleaning up...'
rm -rf dist/*
rm -rf build/*
rm -rf pyadlml.egg-info/*