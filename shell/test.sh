python -m unittest discover tests/reader -p '*_test.py' &> logs/reader_test.log
python -m unittest discover tests/writer -p '*_test.py' &> logs/writer_test.log
python -m unittest discover tests/pbc -p '*_test.py' &> logs/pbc_test.log
