#!/bin/sh

PYTHON=".venv/bin/python"

OUTPATH=out/testdump.csv
$PYTHON -m data_analysis dump_to_csv -o "$OUTPATH" -t 10