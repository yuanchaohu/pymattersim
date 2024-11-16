mkdir logs
python -m unittest discover tests/static -p '*_test.py' &> logs/test_static.log