#!/usr/bin/bash

# script to store HBNL Qt PeakPicker results

source /usr/local/anaconda3/bin/activate dbI

#cd /usr/local/PeakPicker
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $DIR
cd $DIR
python3 ./storePicks.py $1 #/usr/local/PeakPicker/PeakPickerQ.py $1
