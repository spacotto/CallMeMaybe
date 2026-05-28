# ============================================================
#  VARIABLES
# ============================================================

VENV    := .venv
UV	:= $(VENV)/bin/uv
PYTHON  := $(VENV)/bin/python3
MAIN    := main.py
TEST	:= tests/

# ------------------------------------------------------------
#  Ansi colors
# ------------------------------------------------------------

RESET	:= \033[0m
GRAY	:= \033[1;90m
RED	:= \033[1;91m
GREEN	:= \033[1;92m
YELLOW	:= \033[1;93m
BLUE	:= \033[1;94m
MAGENTA	:= \033[1;95m
CYAN	:= \033[1;96m
WHITE	:= \033[1;97m

# ------------------------------------------------------------
#  Additional commands
# ------------------------------------------------------------

ECHO	:= echo 
FIND	:= /bin/find
IGNORE	:= 2>/dev/null || true
MV	:= /bin/mv
RM	:= /bin/rm -rf
FLAKE	:= flake8 . --extend-exclude=.venv,llm_sdk,tests $(IGNORE)

# ============================================================
#  RULES
# ============================================================

.PHONY: install run debug clean lint lint-strict \
       	help activate clean-venv flag test

# ------------------------------------------------------------
#  Default target
# ------------------------------------------------------------

help:
	@$(ECHO) ""
	@$(ECHO) " $(BLUE)MANDATORY RULES$(RESET)"
	@$(ECHO) ""
	@$(ECHO) "     $(BLUE)install$(RESET)      Install dependencies in venv"
	@$(ECHO) "     $(BLUE)run$(RESET)          Run the main script with MAP=<map_path>"
	@$(ECHO) "     $(BLUE)debug$(RESET)        Run the main script with pdb"
	@$(ECHO) "     $(BLUE)clean$(RESET)        Remove temporary files and caches"
	@$(ECHO) "     $(BLUE)lint$(RESET)         Execute flake8 + mypy (standard flags)"
	@$(ECHO) "     $(BLUE)lint-strict$(RESET)  Execute flake8 + mypy --strict"
	@$(ECHO) ""
	@$(ECHO) " $(CYAN)BONUS RULES$(RESET)"
	@$(ECHO) ""
	@$(ECHO) "     $(CYAN)activate$(RESET)     Display the command to activate the venv"
	@$(ECHO) "     $(CYAN)clean-venv$(RESET)   Remove the venv"
	@$(ECHO) "     $(CYAN)test$(RESET)         Run pytest test set"
	@$(ECHO) ""

# ------------------------------------------------------------
#  install — create a virtual environment and install deps
# ------------------------------------------------------------

install:
	@if [ ! -d "$(VENV)" ]; then \
		$(ECHO) "$(YELLOW)>>> No venv detected$(RESET)"; \
		$(ECHO) "$(YELLOW)>>> Creating new venv...$(RESET)"; \
		python3 -m venv $(VENV); \
	fi
	@$(ECHO) "$(YELLOW)>>> Installing dependencies inside $(VENV)...$(RESET)" 
	@$(PIP) install --upgrade pip 
	@$(PIP) install -r requirements.txt
	@$(ECHO) "$(CYAN)>>> Installation complete.$(RESET)"

# ------------------------------------------------------------
#  run — execute the main script
# ------------------------------------------------------------

run:
	@$(ECHO) "$(YELLOW)>>> Running $(MAIN)...$(RESET)"
	$(UV) run $(PYTHON) -m src \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calls.json

# ------------------------------------------------------------
#  debug — launch the main script under pdb
# ------------------------------------------------------------

debug:
	@$(ECHO) "$(YELLOW)>>> Launching $(MAIN) under pdb...$(RESET)"
	@$(PYTHON) -m pdb $(MAIN) $(CONFIG)

# ------------------------------------------------------------
#  clean — remove byte-compiled files and tool caches
# ------------------------------------------------------------

clean:
	@$(ECHO) "$(YELLOW)>>> Cleaning __pycache__$(RESET)"
	@$(FIND) . -type d -name "__pycache__" -exec $(RM) {} + $(IGNORE)
	@$(ECHO) "$(YELLOW)>>> Cleaning *.pyc$(RESET)"
	@$(FIND) . -type f -name "*.pyc" -delete $(IGNORE)
	@$(ECHO) "$(YELLOW)>>> Cleaning *.pyo$(RESET)"
	@$(FIND) . -type f -name "*.pyo" -delete $(IGNORE)
	@$(ECHO) "$(YELLOW)>>> Cleaning .mypy_cache$(RESET)"
	@$(FIND) . -type d -name ".mypy_cache" -exec $(RM) {} + $(IGNORE)
	@$(ECHO) "$(YELLOW)>>> Cleaning .ruff_cache$(RESET)"
	@$(FIND) . -type d -name ".ruff_cache" -exec $(RM) {} + $(IGNORE)
	@$(ECHO) "$(YELLOW)>>> Cleaning .pytest_cache$(RESET)"
	@$(FIND) . -type d -name ".pytest_cache" -exec $(RM) {} + $(IGNORE)
	@$(ECHO) "$(YELLOW)>>> Cleaning *.egg-info$(RESET)"
	@$(FIND) . -type d -name "*.egg-info" -exec $(RM) {} + $(IGNORE)
	@$(ECHO) "$(YELLOW)>>> Cleaning .pytest_cache$(RESET)"
	@$(FIND) . -type d -name ".pytest_cache" -exec $(RM) {} + $(IGNORE)
	@$(ECHO) "$(CYAN)>>> Done.$(RESET)"

# ------------------------------------------------------------
#  lint — standard type-checking and style enforcement
# ------------------------------------------------------------

lint:
	@$(ECHO) "$(YELLOW)>>> Running flake8...$(RESET)"
	@$(FLAKE)
	@$(ECHO) "$(YELLOW)>>> Running mypy (standard)...$(RESET)"
	@mypy . \
	    --warn-return-any \
	    --warn-unused-ignores \
	    --ignore-missing-imports \
	    --disallow-untyped-defs \
	    --check-untyped-defs \
	    $(IGNORE)

# ------------------------------------------------------------
#  lint-strict — maximum mypy strictness (recommended)
# ------------------------------------------------------------

lint-strict:
	@$(ECHO) "$(YELLOW)>>> Running flake8...$(RESET)"
	@$(FLAKE)
	@$(ECHO) "$(YELLOW)>>> Running mypy (strict)...$(RESET)"
	@mypy . --strict $(IGNORE)

# ------------------------------------------------------------
#  activate — display the commmand to activate the venv
# ------------------------------------------------------------

activate:
	@$(ECHO) "$(YELLOW)>>> Run this command to activate the venv:$(RESET)"
	@$(ECHO) "source $(VENV)/bin/activate"

# ------------------------------------------------------------
#  clean-venv — remove virtual environment
# ------------------------------------------------------------

clean-venv:
	@$(ECHO) "$(YELLOW)>>> Cleaning venv$(RESET)"
	@$(RM) $(VENV) $(IGNORE)
	@$(ECHO) "$(CYAN)>>> Done.$(RESET)"

# ------------------------------------------------------------
#  test — test the project with pytest framework 
# ------------------------------------------------------------

test:
	@$(ECHO) "$(YELLOW)>>> Running pytest...$(RESET)"
	@$(ECHO) "$(YELLOW)>>> Testing with $(TEST) ...$(RESET)"
	@$(PYTHON) -m pytest -v -s $(TEST) 
