
run:
	uv run gradio src/visualize_svd.py

lint:
	uv run isort src
	uv run black src -l 79
	uv run flake8 src
