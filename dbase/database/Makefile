PYTHON ?= python
ENV ?= none
NAME ?= none
SOURCE_ENV ?= none
TARGET_ENV ?= none
BRANCH ?= none
CONFIRM ?=
WITH_DATA ?=
EXCLUDE ?=
ARGS ?=

DATABASE_MK_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

ifndef REPO_ROOT
REPO_ROOT := $(abspath $(DATABASE_MK_DIR)/../..)
endif
export PYTHONPATH := $(REPO_ROOT)$(if $(PYTHONPATH),:$(PYTHONPATH),)

DB_MGMT := $(PYTHON) -m dbase.database.db_management

CREATE_EXCLUDE :=
ifneq ($(strip $(EXCLUDE)),)
CREATE_EXCLUDE := $(foreach db,$(EXCLUDE),--exclude $(db))
endif

CONFIRM_FLAG := $(if $(filter 1 yes true,$(CONFIRM)),--confirm,)
WITH_DATA_FLAG := $(if $(filter 1 yes true,$(WITH_DATA)),--with-data,)
SYNC_BRANCH_FLAG := $(if $(filter none,$(BRANCH)),,--branch $(BRANCH))
CREATE_DB_BRANCH_FLAG := $(if $(filter none,$(BRANCH)),,--branch $(BRANCH))

.DEFAULT_GOAL := help
.PHONY: help test test-db-management list create-env create-db create-env-with-data delete diff sync sync-apply

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' $(DATABASE_MK_DIR)/Makefile | awk -F':.*## ' '{printf "  make %-22s %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables (defaults): ENV=$(ENV) NAME=$(NAME) SOURCE_ENV=$(SOURCE_ENV) TARGET_ENV=$(TARGET_ENV) BRANCH=$(BRANCH)"
	@echo "  CONFIRM=1 skip delete prompt; WITH_DATA=1 copy data on sync-apply (create-env-with-data always copies data)"
	@echo "  EXCLUDE='base1 base2' extra --exclude on create-env; ARGS extra CLI flags"

test: ## Run db_management unit tests (from repo root)
	cd $(REPO_ROOT) && $(PYTHON) -m pytest tests/test_db_management_create.py tests/test_db_management_create_db.py $(ARGS)

test-db-management: test ## Alias for test (repo root convenience)

list: ## List non-production environments
	$(DB_MGMT) list $(ARGS)

create-db: ## Create empty DB + registry (ENV, NAME required; BRANCH optional)
	@if [ "$(ENV)" = "none" ]; then echo "ENV is required (e.g. make create-db ENV=scratch NAME=portfolio_config)"; exit 1; fi
	@if [ "$(NAME)" = "none" ]; then echo "NAME is required (base_name e.g. portfolio_config)"; exit 1; fi
	$(DB_MGMT) create-db --env $(ENV) --name $(NAME) $(CREATE_DB_BRANCH_FLAG) $(ARGS)

create-env: ## Create env schema-only (ENV, BRANCH, SOURCE_ENV required; EXCLUDE optional)
	@if [ "$(ENV)" = "none" ]; then echo "ENV is required (e.g. make create-env ENV=mean-reversion)"; exit 1; fi
	@if [ "$(BRANCH)" = "none" ]; then echo "BRANCH is required"; exit 1; fi
	@if [ "$(SOURCE_ENV)" = "none" ]; then echo "SOURCE_ENV is required"; exit 1; fi
	$(DB_MGMT) create --env $(ENV) --branch $(BRANCH) --source-env $(SOURCE_ENV) --schema-only $(CREATE_EXCLUDE) $(ARGS)

create-env-with-data: ## Create env with data (ENV, BRANCH, SOURCE_ENV required; EXCLUDE optional)
	@if [ "$(ENV)" = "none" ]; then echo "ENV is required"; exit 1; fi
	@if [ "$(BRANCH)" = "none" ]; then echo "BRANCH is required"; exit 1; fi
	@if [ "$(SOURCE_ENV)" = "none" ]; then echo "SOURCE_ENV is required"; exit 1; fi
	$(DB_MGMT) create --env $(ENV) --branch $(BRANCH) --source-env $(SOURCE_ENV) --with-data $(CREATE_EXCLUDE) $(ARGS)

delete: ## Delete environment ENV (CONFIRM=1 to skip prompt)
	@if [ "$(ENV)" = "none" ]; then echo "ENV is required (e.g. make delete ENV=test-mean-reversion)"; exit 1; fi
	$(DB_MGMT) delete --delete-env $(ENV) $(CONFIRM_FLAG) $(ARGS)

diff: ## Diff SOURCE_ENV vs TARGET_ENV (what target is missing)
	@if [ "$(SOURCE_ENV)" = "none" ]; then echo "SOURCE_ENV is required"; exit 1; fi
	@if [ "$(TARGET_ENV)" = "none" ]; then echo "TARGET_ENV is required"; exit 1; fi
	$(DB_MGMT) diff --source-env $(SOURCE_ENV) --target-env $(TARGET_ENV) $(ARGS)

sync: ## Sync dry run (SOURCE_ENV, TARGET_ENV required)
	@if [ "$(SOURCE_ENV)" = "none" ]; then echo "SOURCE_ENV is required"; exit 1; fi
	@if [ "$(TARGET_ENV)" = "none" ]; then echo "TARGET_ENV is required"; exit 1; fi
	$(DB_MGMT) sync --source-env $(SOURCE_ENV) --target-env $(TARGET_ENV) $(SYNC_BRANCH_FLAG) $(ARGS)

sync-apply: ## Sync and apply (WITH_DATA=1 for --with-data; BRANCH optional for new DBs)
	@if [ "$(SOURCE_ENV)" = "none" ]; then echo "SOURCE_ENV is required"; exit 1; fi
	@if [ "$(TARGET_ENV)" = "none" ]; then echo "TARGET_ENV is required"; exit 1; fi
	$(DB_MGMT) sync --source-env $(SOURCE_ENV) --target-env $(TARGET_ENV) --apply $(WITH_DATA_FLAG) $(SYNC_BRANCH_FLAG) $(ARGS)
