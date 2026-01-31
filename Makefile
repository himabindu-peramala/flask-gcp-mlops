PROJ_ID=your-project-id
IMAGE_NAME=iris-app
VENV_PATH=venv/bin/

.PHONY: install train run test lint clean venv

venv:
	python3 -m venv venv

install:
	$(VENV_PATH)pip install -r requirements.txt

train:
	$(VENV_PATH)python src/train.py

run:
	$(VENV_PATH)python src/main.py

test:
	$(VENV_PATH)pytest tests/ -v

lint:
	# simpler lint for now
	@echo "Linting not strictly enforced yet."

clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	rm -rf venv
