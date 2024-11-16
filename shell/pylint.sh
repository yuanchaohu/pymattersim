rm -f logs/lint_error.log
pylint PyMatterSim.reader >> logs/lint_error.log
pylint PyMatterSim.writer >> logs/lint_error.log
pylint PyMatterSim.utils.pbc >> logs/lint_error.log
pylint PyMatterSim.neighbors >> logs/lint_error.log
pylint PyMatterSim.static >> logs/lint_error.log