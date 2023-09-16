import numpy as np
import glob
import random
import tensorflow as tf
import os
import shutil

kfold = 5
seed = 0
train_folder = "/local_datasets/tpugraphs/npz/tile/xla_cv"

random.seed(seed)
np.random.seed(seed)
samples = list(glob.glob(train_folder+"/train/*"))
N = len(samples)
idxs = np.arange(N)
np.random.shuffle(idxs)
valid_size = int(N/kfold)
valid_idxs = np.split(idxs, [valid_size, 2*valid_size, 3*valid_size, 4*valid_size])

for kf in range(kfold):
    # Create train and valid folders
    train_fold, valid_fold = os.path.join(train_folder,"train"+str(kf)), os.path.join(train_folder,"valid"+str(kf))
    if tf.io.gfile.exists(train_fold):
        os.rmdir(train_fold)
    tf.io.gfile.makedirs(train_fold)
    if tf.io.gfile.exists(valid_fold):
        os.rmdir(valid_fold)
    tf.io.gfile.makedirs(valid_fold)

    # Copy samples to train or valid folder
    for i in range(N):
        if i in valid_idxs[kf]:
            shutil.copy(samples[i], valid_fold)
        else:
            shutil.copy(samples[i], train_fold)

print("Spliting is done!!!")



