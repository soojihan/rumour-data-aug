#!/bin/sh
#nohup /oak1/data/sooji/anaconda3/envs/py3/bin/python3.6 -u /oak01/data/sooji/data_annotation/src/preprocessing.py > /oak01/data/sooji/data_annotation/src/preprocessing.log 2>&1 &
nohup ./anaconda3/envs/py3/bin/python3.6 -u /oak01/data/sooji/data_annotation/src/batch_annotation.py 14 > /oak01/data/sooji/data_annotation/train_augmentation/batch_annotation14.log 2>&1 &




