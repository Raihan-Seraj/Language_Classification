import copy
import numpy as np
import csv
import re
from collections import defaultdict

def read_file(file_name):
    data = []
    reader = csv.reader(open(file_name, newline=''), delimiter=',')
    next(reader,None) #skip header
    for row in reader:
            data.append(row[1])
    return data

data_x = read_file('dataset/train_set_x.csv')
data_y = read_file('dataset/train_set_y.csv')
test_x = read_file('dataset/test_set_x.csv')
train = [x for x in list(zip(data_y, data_x))]

def does_string_contain_chars(string, charx):
    strip = string
    t = 0
    for c in charx:
        t += 1
        if c not in strip:
            return False
        strip = strip.replace(c,'', 1)
    return True

chars_by_frequency = defaultdict(int)
total_chars_by_language = [0 for i in range(5)]
chars_by_language = [defaultdict(int) for i in range(5)]
alphabet = dict()
print("{0: Slovak, 1: French, 2: Spanish, 3: German, 4: Polish}")
raw_features = np.zeros((300,5))
indx = 0

for inst in train:
    for c in inst[1]:
        if c not in alphabet:
            alphabet[c] = indx
            indx += 1
        label = int(inst[0])
        chars_by_frequency[c] += 1
        raw_features[alphabet[c]][label] += 1
        chars_by_language[label][c] += 1
        total_chars_by_language[label] += 1


scaling_factors = [max(total_chars_by_language) / i for i in total_chars_by_language]

for label in chars_by_language:
    chars_by_language[label]

print(scaling_factors)
print(total_chars_by_language)
print("Finished histogram")

i = 0
results = [['Id', 'Category']]
output = open("test_set_y.py", "w")

for inst in test_x:
    if i > 99:
        break
    if inst.__len__() == 0:
        results.append([i, 2])
        i+=1
        continue

    chars = inst.split(' ')

    foo = defaultdict(int)

    for comment in train:
        if does_string_contain_chars(comment[1], chars):
            foo[comment[0]] += 1
    
    if(foo.__len__() == 0):
        print(foo)
        print(chars)

    max_val = 0
    max_langx = 0
    for pair in foo.items():
        if pair[1] > max_val:
            max_val = pair[1]
            max_langx = pair[0]

    out = [i, max_langx]
    print(str(i) + ',' + str(max_langx), file=output)

    #if i % 100 == 0:
        #print(out)
        #print(results.__len__())

    i += 1

print(i)
print("done")
