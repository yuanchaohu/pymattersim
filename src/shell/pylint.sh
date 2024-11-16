rm -f logs/lint_error.log
pylint reader >> logs/lint_error.log
pylint writer >> logs/lint_error.log
pylint utils.pbc >> logs/lint_error.log
pylint neighbors >> logs/lint_error.log
pylint static >> logs/lint_error.log