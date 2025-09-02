.PHONY: typecheck mypy

typecheck:
	mypy --config-file mypy.ini .

mypy: typecheck
