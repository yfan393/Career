#!/bin/sh

PYTHON=".venv/bin/python"

if [ -f out/testdb.sqlite ]; then
    rm out/testdb.sqlite
fi
export DB_CONNECTION_STRING=sqlite:///out/testdb.sqlite
$PYTHON db.py -T "../Training_final/" -H "../Holdout_final/" --test_run