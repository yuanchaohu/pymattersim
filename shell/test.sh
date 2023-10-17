# python -m unittest reader.tests.lammps_reader_helper_test &> logs/reader_test.log
python -m unittest discover tests -p '*_test.py' &> logs/reader_test.log
