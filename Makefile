.PHONY: install test lint demo ui doctor
install:
	pip install -e ".[hf,ui,dev]"
test:
	pytest
lint:
	ruff check src tests
doctor:
	nudge doctor --backend tiny
demo:
	nudge demo
ui:
	nudge ui
