install:
	pip install -r requirements.txt

test:
	pytest tests

run-default:
	bash scripts/run_default.sh

reproduce:
	bash scripts/reproduce_all.sh
