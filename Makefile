# Makefile

PYTHON = /home/jose/venvs/finance/bin/python

.PHONY: run test refresh-universe cache-fund lint ops-check preopen-check

run:          ## Start all three pipeline processes
	$(PYTHON) -m src.main

test:         ## Run full pytest suite
	$(PYTHON) -m pytest tests/unit/ -v

refresh-universe: ## Run preselection to find top 40 moat stocks
	export PYTHONPATH=$$PYTHONPATH:. && $(PYTHON) scripts/refresh_universe.py

cache-fund:      ## Cache daily financial data for the 40 stocks
	export PYTHONPATH=$$PYTHONPATH:. && $(PYTHON) scripts/cache_fundamentals.py

lint:         ## Black + mypy type check
	black src/ tests/ && mypy src/

ops-check:    ## Validate Makefile targets reference existing files
	$(PYTHON) scripts/check_makefile_targets.py

preopen-check: ## One-command readiness checker for environment and data
	export PYTHONPATH=$$PYTHONPATH:. && $(PYTHON) scripts/preopen_check.py

help:         ## Show this help message
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
