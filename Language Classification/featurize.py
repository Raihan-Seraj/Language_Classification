import copy
import numpy as np
import csv
import regex as re
from collections import defaultdict
from utils import read_file

data_x = read_file('dataset/train_set_x.csv')
data_y = read_file('dataset/train_set_y.csv')
train = [x for x in list(zip(data_y, data_x))]

chars_by_frequency = defaultdict(int)
total_chars_by_language = [0 for i in range(5)]
chars_by_language = [defaultdict(int) for i in range(5)]
alphabet = dict()
word_dictionary = defaultdict(int)
lang_word_dictionaries = [defaultdict(int) for _ in range(5)]
lang_totals = defaultdict(int)
dictionaries = [defaultdict(int) for _ in range(5)]
print("{0: Slovak, 1: French, 2: Spanish, 3: German, 4: Polish}")
dictionary = defaultdict(int)
raw_features = np.zeros((690, 5), dtype=np.int)
# bigrams = np.zeros((1000, 1000, 5), dtype=np.int)
indx = 0
i = 0

def find_ngrams(input_list, label):
    ngrams = list()
    for a, b, c in zip(input_list, input_list[1:], input_list[2:]):
        string = ''.join(sorted(a + b + c))
        try:
            string.encode('ascii')
        except UnicodeEncodeError:
            continue
        else:
            pass
        if len(string) > 0:
            ngrams.append(string)

    for a, b in zip(input_list, input_list[1:]):
        string = ''.join(sorted(a + b))
        try:
            string.encode('ascii')
        except UnicodeEncodeError:
            continue
        else:
            pass
        if len(string) > 4:
            ngrams.append(string)

    for a in input_list:
        string = ''.join(sorted(a))
        try:
            string.encode('ascii')
        except UnicodeEncodeError:
            continue
        else:
            pass
        if len(string) > 3:
            word_dictionary[string] += 1
            lang_word_dictionaries[label][string] += 1
            lang_totals[label] += 1

    return ngrams

for instance in train:
    if i % 1000 == 0:
        print(str(i / len(train)))
    label = int(instance[0])
    comment = instance[1].lower()
    words = re.split("[^\p{L}\p{M}]", comment)

    for ngram in find_ngrams(words, label):
        dictionary[ngram] += 1
        dictionaries[label][ngram] += 1

    symbols = comment.replace(' ', '')

    w = ''.join(sorted(re.sub("[^\p{L}\p{M}]","", symbols)))
    if 0 < len(w) < 20:
        dictionary[w] += 1
        dictionaries[label][w] += 1

    for c in symbols:
        if c not in alphabet:
            alphabet[c] = indx
            indx += 1

        chars_by_frequency[c] += 1
        raw_features[alphabet[c]][label] += 1
        if c.isalpha() and not ord(c) < 127:
            raw_features[alphabet[c]][label] += 2
        chars_by_language[label][c] += 1
        total_chars_by_language[label] += 1
    i += 1

print(len(word_dictionary))
np.savetxt('raw_features.csv', raw_features, delimiter=',')
np.save('dictionary.npy', dictionary)
np.save('dictionaries.npy', dictionaries)
np.save('word_dictionary.npy', word_dictionary)
np.save('lang_word_dictionaries.npy', lang_word_dictionaries)
np.save('lang_totals.npy', lang_totals)
np.save('alphabet.npy', alphabet)
# np.save('bigrams.npy', bigrams)
