
lint:
	uv run isort src
	uv run black src -l 79
	uv run flake8 src
