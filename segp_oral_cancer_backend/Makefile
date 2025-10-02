PY=python
PIP=pip

init:
	$(PIP) install -r requirements.txt

train:
	$(PY) -m src.training.train_tvnt --config configs/tvnt.yaml

eval:
	$(PY) -m src.training.evaluate_tvnt --config configs/tvnt.yaml

api:
	uvicorn src.api.main:app --reload --port 8000
