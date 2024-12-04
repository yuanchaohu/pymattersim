#!/bin/bash
rm -rf dist/*
python setup.py sdist bdist_wheel
twine upload dist/*

# documentation update
# push changes to master
# cd docs/
# make html
# git subtree push --prefix docs/_build/html origin gh-pages