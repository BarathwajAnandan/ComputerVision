import numpy as np
import os
import sys
import scipy.io
import shutil


output_folder = '../data/oxford-flowers17'
splits = scipy.io.loadmat('datasplits.mat')
maps = [('trn2','train'),('val2','val'),('tst2','test')]
labels = {'labels': (np.arange(1360,dtype=np.int) )//80 }
base_str = "image_{:04d}.jpg"


input_folder = 'jpg'


img_to_split = {}
img_to_label = {}

for split in splits.keys():
    if '__' in split: continue
    for mp in maps:
        if mp[0] in split:
            vec = np.squeeze(splits[split])
            for l in vec:
                img_to_split[l] = mp[1]

for idx, label in enumerate(np.squeeze(labels['labels'])):
    img_to_label[idx+1] = label

os.mkdir(output_folder)
for mp in maps:
    os.mkdir(os.path.join(output_folder,mp[1]))
for lbl in np.unique(list(img_to_label.values())):
    for mp in maps:
        os.mkdir(os.path.join(output_folder,mp[1],str(lbl)))
for i in img_to_label.keys():
    name = base_str.format(i)
    inp = os.path.join(input_folder,name)
    otp = os.path.join(output_folder,str(img_to_split[i]),str(img_to_label[i]),name)
    shutil.move(inp, otp)
