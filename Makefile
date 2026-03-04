# Makefile for systemd service management
SERVICE_NAME = trading-bot.service
SERVICE_FILE_PATH = /etc/systemd/system/$(SERVICE_NAME)
PASSWORD = Rocky2024
MAIN_LOG_FILE = ./logs/main.log


.PHONY: install start stop restart enable disable status log log-follow log-clear main-log main-log-follow

define log_action
	@mkdir -p $(dir $(LOG_FILE))
	@echo "[$(shell date '+%Y-%m-%d %H:%M:%S')] $(1)" | tee -a $(LOG_FILE)
endef

install: ## Install the systemd service
	$(call log_action,INFO  | Installing $(SERVICE_NAME) to $(SERVICE_FILE_PATH)...)
	@echo "$(PASSWORD)" | sudo -S cp $(SERVICE_NAME) $(SERVICE_FILE_PATH)
	@echo "$(PASSWORD)" | sudo -S systemctl daemon-reload
	$(call log_action,INFO  | Service installed successfully.)

enable: ## Enable the service to start on boot
	$(call log_action,INFO  | Enabling $(SERVICE_NAME)...)
	@echo "$(PASSWORD)" | sudo -S systemctl enable $(SERVICE_NAME)
	$(call log_action,INFO  | Service enabled.)

disable: ## Disable the service from starting on boot
	$(call log_action,INFO  | Disabling $(SERVICE_NAME)...)
	@echo "$(PASSWORD)" | sudo -S systemctl disable $(SERVICE_NAME)
	$(call log_action,INFO  | Service disabled.)

start: ## Start the service
	$(call log_action,INFO  | Starting $(SERVICE_NAME)...)
	@echo "$(PASSWORD)" | sudo -S systemctl start $(SERVICE_NAME)
	$(call log_action,INFO  | Service started.)

stop: ## Stop the service
	$(call log_action,INFO  | Stopping $(SERVICE_NAME)...)
	@echo "$(PASSWORD)" | sudo -S systemctl stop $(SERVICE_NAME)
	$(call log_action,INFO  | Service stopped.)

restart: ## Restart the service
	$(call log_action,INFO  | Restarting $(SERVICE_NAME)...)
	@echo "$(PASSWORD)" | sudo -S systemctl restart $(SERVICE_NAME)
	$(call log_action,INFO  | Service restarted.)

status: ## Check the status of the service
	@systemctl status $(SERVICE_NAME)

uninstall: ## Uninstall the systemd service
	$(call log_action,WARN  | Uninstalling $(SERVICE_NAME)...)
	@echo "$(PASSWORD)" | sudo -S systemctl stop $(SERVICE_NAME) || true
	@echo "$(PASSWORD)" | sudo -S systemctl disable $(SERVICE_NAME) || true
	@echo "$(PASSWORD)" | sudo -S rm $(SERVICE_FILE_PATH)
	@echo "$(PASSWORD)" | sudo -S systemctl daemon-reload
	$(call log_action,WARN  | Service uninstalled.)

log: ## View the service management log
	@echo "=== Service Management Log ($(LOG_FILE)) ==="
	@if [ -f $(LOG_FILE) ]; then cat $(LOG_FILE); else echo "No log file found at $(LOG_FILE)"; fi

log-follow: ## Tail the service management log (live)
	@echo "=== Tailing Service Management Log ($(LOG_FILE)) — Ctrl+C to exit ==="
	@if [ -f $(LOG_FILE) ]; then tail -f $(LOG_FILE); else echo "No log file found at $(LOG_FILE)"; fi

log-clear: ## Clear the service management log
	$(call log_action,INFO  | Log cleared.)
	@echo "" > $(LOG_FILE)
	@echo "Log cleared."

main-log: ## View the main application log
	@echo "=== Main Application Log ($(MAIN_LOG_FILE)) ==="
	@if [ -f $(MAIN_LOG_FILE) ]; then cat $(MAIN_LOG_FILE); else echo "No log file found at $(MAIN_LOG_FILE)"; fi

main-log-follow: ## Tail the main application log (live)
	@echo "=== Tailing Main Application Log ($(MAIN_LOG_FILE)) — Ctrl+C to exit ==="
	@if [ -f $(MAIN_LOG_FILE) ]; then tail -f $(MAIN_LOG_FILE); else echo "No log file found at $(MAIN_LOG_FILE)"; fi

log-journald: ## View journald logs for the service
	@echo "=== Journald Logs for $(SERVICE_NAME) ==="
	@journalctl -u $(SERVICE_NAME) --no-pager

log-journald-follow: ## Tail journald logs for the service (live)
	@echo "=== Tailing Journald Logs for $(SERVICE_NAME) — Ctrl+C to exit ==="
	@journalctl -u $(SERVICE_NAME) -f

help: ## Show this help message
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'