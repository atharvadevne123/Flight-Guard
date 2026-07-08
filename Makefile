.PHONY: install test lint format run docker-up docker-down train diagram clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

lint:
	ruff check .

format:
	ruff format .

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

train:
	python -c "from app.features import generate_synthetic_data; from app.model import train_model; X, y = generate_synthetic_data(2000); _, m = train_model(X, y); print(m)"

diagram:
	python scripts/generate_diagram.py

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
