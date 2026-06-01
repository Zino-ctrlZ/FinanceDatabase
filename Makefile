# FinanceDatabase — thin wrapper for dbase/database tooling
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
export REPO_ROOT

include dbase/database/Makefile
