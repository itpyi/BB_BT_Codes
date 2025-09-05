.PHONY: typecheck mypy test check

typecheck:
	mypy --config-file mypy.ini .

mypy: typecheck

test:
	python -m unittest discover -s test -p "test_*.py" -v

check: typecheck test
