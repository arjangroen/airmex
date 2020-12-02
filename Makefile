.PHONY: precommit
precommit:
	pre-commit run --all

.PHONY: test
test:
	python -m pytest

.PHONY: ci
ci: test precommit
