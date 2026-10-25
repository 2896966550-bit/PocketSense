.PHONY: test smoke data-fetch

test:
	python -m pytest tests -q

smoke:
	python -m psense.pipeline.smoke

data-fetch:
	python -m psense.data.fetch
