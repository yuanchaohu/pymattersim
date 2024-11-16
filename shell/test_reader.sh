mkdir logs
python -m unittest discover tests/reader -p '*_test.py' &> logs/test_reader.log
