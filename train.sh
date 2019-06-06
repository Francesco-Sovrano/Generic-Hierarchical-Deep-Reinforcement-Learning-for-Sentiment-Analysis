#!/bin/bash

pkill -9 -f python

MY_PATH="`dirname \"$0\"`"
cd $MY_PATH
. .env2/bin/activate

# python3 ./A3C/build.py
python3 ./A3C/train.py