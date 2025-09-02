.PHONY: typecheck mypy test

typecheck:
	mypy --config-file mypy.ini .

mypy: typecheck

test:
	python -m unittest discover -s test -p "test_*.py" -v
