install:
	@pip install poetry
	@poetry install
	@poetry run pre-commit install --install-hooks

test:
	@poetry run pre-commit run --all-files

run:
	poetry run python src/baysian_bandits_animation.py
