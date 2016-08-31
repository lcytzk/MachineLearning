#!/usr/bin/env bash
cat $1 | ./getlabel.py > label.txt
./getTog.py $2 > vw_test.res
cat vw_test.res | ./vw_auc.py

