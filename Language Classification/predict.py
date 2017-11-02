from joblib import Parallel, delayed
import random
import editdistance
import multiprocessing
import copy
import numpy as np
import math
import csv
import re
from collections import defaultdict
from utils import read_file_with_instance_no

np.set_printoptions(suppress=True, precision=9, threshold=np.nan)
dictionary = np.load('dictionary.npy')
dictionaries = np.load('dictionaries.npy')

word_dictionary = np.load('word_dictionary.npy').item()
lang_word_dictionaries = np.load('lang_word_dictionaries.npy')
lang_totals = np.load('lang_totals.npy').item()
anagram_indices = dict()
p_word_given_lang = np.zeros((len(word_dictionary), len(lang_totals)))
word_indxs = {k: v for v, k in enumerate(word_dictionary)}

for li, lang_dict in enumerate(lang_word_dictionaries):
    for word in lang_dict:
        p_word_given_lang[word_indxs[word]][li] = lang_word_dictionaries[li][word] / lang_totals[li]

average_anagrams_per_line = 0
anagrams = [None] * 118509

with open('python_filtered_anagrams.csv', 'rt', encoding='ascii') as csvfile:
    anagram_reader = csv.reader(csvfile, delimiter=',')
    for row in anagram_reader:
        anagrams[int(row[0])] = row[1:]
        average_anagrams_per_line += len(row)

i = 0
for dict in dictionaries:
    for w in list(dict.keys()):
        if dict[w] < 2:
            del dict[w]
        else:
            i += 1

alphabet = np.load('alphabet.npy').item()
bigrams = np.load('bigrams.npy')
raw_features = np.loadtxt('raw_features.csv', delimiter=',')
english_frequency = np.array(
    [7381, 24841, 14678, 6916, 14395, 19606, 4889, 6022, 18718, 2674, 12373, 15670, 20479, 8332, 731, 8130, 563, 9166,
     3931, 0, 341, 0, 3907, 10903, 5453, 10, 419, 5700, 2449, 862, 817, 721, 458, 858, 0, 22, 0, 0, 0, 11, 506, 363, 0,
     0, 0, 99, 11, 0, 0, 461, 37, 0, 0, 532, 0, 0, 374, 0, 414, 10, 47, 390, 404, 0, 0, 0, 0, 0, 682, 2398, 43, 33, 438,
     0, 0, 0, 132, 0, 0, 0, 3, 0, 15, 0, 0, 0, 11, 0, 77, 0, 0, 0, 86, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0,
     11, 0, 0, 0, 0, 0, 5, 0, 0, 3, 0, 77, 0, 0, 0, 264, 264, 19, 0, 14, 0, 0, 0, 0, 0, 0, 29, 10, 0, 22, 4, 0, 0, 0, 0,
     6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

rev_alphabet = {k: v for v, k in alphabet.items()}

raw_features = np.c_[raw_features, english_frequency.reshape(690, 1)]
raw_features[raw_features < 3] = 1
bigrams[bigrams < 3] = 1

languages = {0: 'Slovak', 1: 'French', 2: 'Spanis', 3: 'German', 4: 'Polish', 5: 'English'}
total_chars_by_language = np.sum(raw_features, axis=0)
print("\nTotal characters by language:")
for i in range(raw_features.shape[1]):
    print(languages[i] + ": " + str(total_chars_by_language[i]))

print("\nMost common characters by language:")
for lang in range(raw_features.shape[1]):
    print(languages[lang] + ": ", end='')
    for i in np.argsort(-raw_features[:, lang]):
        if raw_features[i][lang] > 1:
            print(rev_alphabet[i], end=' ')
    print()

scaling_factors = [max(total_chars_by_language) / i for i in total_chars_by_language]
scaled_features = scaling_factors * raw_features
p_lang_given_char = scaled_features / scaled_features.sum(axis=1)[:, None]
p_char_given_lang = raw_features / raw_features.sum(axis=0)
p_langs = total_chars_by_language / sum(total_chars_by_language)
bigrams_normed = bigrams / bigrams.sum(axis=0)
p_big = np.sum(bigrams, axis=0) / sum(np.sum(bigrams, axis=0))

print("\nLeast ambiguous characters by language:")
for lang in range(raw_features.shape[1]):
    print(languages[lang] + ": ", end='')
    for i in np.argsort(-np.multiply(p_lang_given_char[:, lang], p_char_given_lang[:, lang])):
        if i in rev_alphabet:
            print(rev_alphabet[i], end=' ')
    print()

print("\nMost common phrases by language:")
for di, dx in enumerate(dictionaries):
    print(languages[di] + ": " + str(sorted(dx.items(), key=lambda k_v: k_v[1], reverse=True)))

word_counts = np.zeros(raw_features.shape[1])
all_words = set()

for i, d in enumerate(dictionaries):
    for k, v in d.items():
        word_counts[i] += v
        all_words.add(k)

name = "test_set"
test_x = read_file_with_instance_no('dataset/' + name + '_x.csv')

confusion = list()


def is_subseq(v2, v1):
    """Check whether v2 is a subsequence of v1."""
    it = iter(v1)
    return all(c in it for c in v2)


def process_dict(dictionary, chars, di):
    word_score = 0
    chars = ''.join(chars)
    for word in dictionary:
        if len(word) > len(chars):
            continue
        if chars == word:
            word_score = dictionary[word]
            continue
        if is_subseq(list(word), chars):
            word_score += math.log(dictionary[word] / word_counts[di])
    return word_score


def get_anagrams(chars):
    anagrams = list()
    for word in all_words:
        if is_subseq(list(word), chars):
            anagrams.append(word)

    return anagrams


def is_anagram_of_known_phrase(sorted_chars):
    count = 0
    best_dict = -1
    best_guess = ""
    best_similarity = 0
    for di, dict in enumerate(dictionaries):
        for word in dict:
            similarity = 1 - (editdistance.eval(sorted_chars, word) / len(sorted_chars))
            if similarity > 0.85:
                if similarity < best_similarity:
                    continue
                elif similarity == best_similarity and dict[word] <= count:
                    continue

                best_similarity = similarity
                count = dict[word]
                best_dict = di
                best_guess = word

    return best_dict, best_guess


def get_anagram_score(line):
    lang_scores = np.zeros(5)
    #total_samples = len(anagrams[line])
    random_sample = anagrams[line]
    # if total_samples < 300:
    #     random_sample = random.sample(anagrams[line], len(anagrams[line]))
    # else:
    #     random_sample = random.sample(anagrams[line], 300)
    for anagram in random_sample:
        if anagram == '':
            continue
        lang_scores[np.argmax(p_word_given_lang[word_indxs[anagram]])] += 1

    return np.argmax(lang_scores)


def process_chunk(chunk, i_of_n):
    print("Processing chunk #" + str(i_of_n) + " of length " + str(len(chunk)))
    output_file = open("predicted_" + name + "_y" + str(i_of_n) + ".temp", "w")
    tlta_predictions = np.zeros(5)
    if i_of_n == 0:
        print("Id,Category", file=output_file)
    for chunk_line_no, line in enumerate(chunk):
        y = int(line[0])
        inst = line[1]
        # All empty comments should be Spanish
        if inst.__len__() == 0:
            print(str(y) + ',2', file=output_file)
            continue

        chars = inst.split(' ')
        sorted_chars = sorted(chars)

        contains_unicode = False
        # chars = sorted(chars, key=prioritize)
        score = np.log(p_langs)
        for i, c in enumerate(chars):
            if c not in alphabet:
                # print(c + " was not in alphabet!")
                continue
            if ord(c) > 127:
                contains_unicode = True
            c = c.lower()
            p0 = p_char_given_lang[alphabet[c]]
            p1 = p_lang_given_char[alphabet[c]]
            score = np.add(score, np.log(p0 * p1))

        predictions = np.argsort(-score)
        prediction = predictions[0]
        top_confusion = score[predictions[0]] / score[predictions[1]]

        iaokp_output = False
        if len(sorted_chars) < 20:
            iaokp = is_anagram_of_known_phrase(''.join(sorted_chars))
            if iaokp[0] > -1:
                # if iaokp[1] != ''.join(sorted_chars):
                # print("Anagram of " + languages[iaokp[0]] + " phrase: " + ''.join(chars) + " (" + iaokp[1] + ")")
                prediction = iaokp[0]
                iaokp_output = True

        if not iaokp_output and not contains_unicode and len(sorted_chars) > 3 and anagrams[y] is not None:
            prediction = get_anagram_score(y)
            tlta_predictions[prediction] += 1

        # if prediction == 5 and top_confusion > 1.5 and not contains_unicode:
        #    english += 1
        #    # print(chars)
        #    prediction = 2
        if prediction == 5:
            prediction = predictions[1]

        confusion.append([y, chars, score, top_confusion])

        # print(str(y) + ":" + str(chars[:5]) + "->" + str(languages[prediction]))

        print(str(y) + ',' + str(prediction), file=output_file)

        if y % 1000 == 0:
            print(str(tlta_predictions))
            print(chunk_line_no / len(chunk))


def chunk(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


num_cores = multiprocessing.cpu_count()
chunks = chunk(test_x, math.ceil(len(test_x) / num_cores))

io_pairs = list()
Parallel(n_jobs=num_cores)(delayed(process_chunk)(chunk, ci) for ci, chunk in enumerate(chunks))

confusion_sorted = sorted(confusion, key=lambda tup: tup[-1])

output = open("confusion.csv", "w")
print("Id,Input,Naive Bayes Scores,Top-2 Confusion", file=output)
for i in confusion_sorted:
    print(i, file=output)

print("done")
