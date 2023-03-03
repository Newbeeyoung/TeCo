import os
import shutil
import sys

BASE_DIR=sys.argv[1]

# BASE_DIR="/data/Dataset/kinetics/mini_kinetics-c/h265_crf_hdf5/27"

for classname in os.listdir(BASE_DIR):
    for filename in os.listdir(os.path.join(BASE_DIR,classname)):
        if filename[-5:]!='.hdf5':
            shutil.rmtree(os.path.join(BASE_DIR,classname,filename))
            print(classname,filename)