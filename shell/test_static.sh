#python -m unittest discover tests/static -p '*_test.py' &> logs/test_static.log
#python -m unittest discover tests/static -p '*tetrahedral_test.py' &> logs/test_static.log
python -m unittest discover tests/static -p 'packing_capability_test.py' &> logs/test_static.log