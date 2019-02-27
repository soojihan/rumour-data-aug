#!/bin/sh
#nohup /oak1/data/sooji/anaconda3/envs/py3/bin/python3.6 -u /oak01/data/sooji/data_annotation/src/preprocessing.py > /oak01/data/sooji/data_annotation/src/preprocessing.log 2>&1 &

#disable hdf5 file locking (e.g., for training ELMo model) which we do not have permission in iceberg
export HDF5_USE_FILE_LOCKING=FALSE

nohup ./anaconda3/envs/py3/bin/python3.6 -u /oak01/data/sooji/data_annotation/src/sem_sim_elmo.py 1 > /oak01/data/sooji/data_annotation/train_augmentation/sem_sim_elmo_1.log 2>&1 &




