@echo off
rem Pipeline launcher. ".\run" for the menu, ".\run train" etc. to start one directly.
cd /d "%~dp0"
uv run python run.py %*
