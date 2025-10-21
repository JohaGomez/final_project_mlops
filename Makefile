.PHONY: install lint format train test run clean mlflow-ui

PY=python

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

lint:
	flake8 src

format:
	black src tests

train:
	$(PY) src/train.py --config configs/config.yaml

test:
	pytest -q

run: lint test train

clean:
	- rmdir /S /Q mlruns 2>NUL || true
	- rmdir /S /Q models 2>NUL || true
	- rmdir /S /Q data\\processed 2>NUL || true

mlflow-ui:
	mlflow ui --backend-store-uri "file:./mlruns" --port 5000
