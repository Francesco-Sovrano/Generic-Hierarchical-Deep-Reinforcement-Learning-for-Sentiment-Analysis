#!/bin/bash

pkill -9 -f python

MY_PATH="`dirname \"$0\"`"
cd $MY_PATH
. .env2/bin/activate

# rm -r log
if [ ! -d "log" ]; then
  mkdir log
fi
cd ./log
if [ ! -d "performance" ]; then
  mkdir performance
fi
cd ..

python3 ./A3C/test.py