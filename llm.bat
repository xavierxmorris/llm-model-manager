@echo off
:: LLM Model Manager - Quick launcher
:: Usage: llm.bat <command> [model-id]
::
:: Commands:
::   list              List all available models
::   start <model-id>  Start a model server
::   stop              Stop the running server
::   switch <model-id> Stop current + start new model
::   status            Show what's running
::   test              Quick health + completions test
::   add               Interactive: add a new model

set SCRIPT_DIR=%~dp0
python "%SCRIPT_DIR%llm_manager.py" %*
