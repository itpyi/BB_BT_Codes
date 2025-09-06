.PHONY: typecheck mypy test check format style

typecheck:
	mypy --config-file mypy.ini .

mypy: typecheck

test:
	python -m unittest discover -s test -p "test_*.py" -v

style:
	black --check --diff .

format:
	black .

check: style typecheck test
