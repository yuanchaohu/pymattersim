mkdir logs
python -m unittest discover tests/dynamics -p '*_test.py' &> logs/test_dynamics.log