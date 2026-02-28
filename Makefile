# Makefile

run:          ## Start all three pipeline processes
	python -m src.main

test:         ## Run full pytest suite with coverage report
	pytest tests/ --cov=src --cov-report=term-missing -v

audit:        ## Parse rejected_ticker logs and print weekly summary
	python scripts/audit_gatekeeper.py --week

halt:         ## Emergency manual halt (calls REST flatten, logs MANUAL_HALT)
	python scripts/emergency_halt.py

config-check: ## Validate config.yaml schema before deployment
	python scripts/validate_config.py

backtest:     ## Run walk-forward backtest with purged CV (last 2 years)
	python -m src.models.backtest --years 2

refresh-universe: ## Run preselection to find top 40 moat stocks
	export PYTHONPATH=$$PYTHONPATH:. && /home/jose/venvs/finance/bin/python scripts/refresh_universe.py

cache-fund:      ## Cache daily financial data for the 40 stocks
	export PYTHONPATH=$$PYTHONPATH:. && /home/jose/venvs/finance/bin/python scripts/cache_fundamentals.py

lint:         ## Black + mypy type check
	black src/ tests/ && mypy src/

deps-audit:   ## Scan dependencies for known CVEs
	pip-audit -r requirements.txt
