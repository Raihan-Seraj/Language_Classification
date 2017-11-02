
# coding: utf-8

# In[13]:

import csv
import re
import random
import pickle
import pdb


f = open('kaggle_set.csv', 'w')
print('Id,Text', file=f)

reader = csv.reader(open('dataset/train_set_x.csv', newline=''), delimiter=',')
next(reader,None) #skip header
for row in reader:
    lower = row[1].lower()
    clean_text = lower.replace(' ','')
    sample = list(clean_text)
    #print(len(sample))
    #pdb.set_trace()
    if len(sample) < 20:
        #print("entering the less condition")
        rand_sample = random.sample(sample, len(sample))
    else:
        rand_sample = [ sample[i] for i in sorted(random.sample(range(len(sample)), 20)) ]
    print(row[0] + ',' + ' '.join(rand_sample), file=f)
f.close()
