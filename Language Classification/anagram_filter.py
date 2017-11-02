import csv

output_file = open("python_filtered_anagrams.csv", "w")
with open('all_filtered.csv', 'rt', encoding='utf8') as csvfile:
    anagram_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in anagram_reader:
        words = list()
        for word in row:
            try:
                word.encode('ascii')
            except UnicodeEncodeError:
                continue
            else:
                pass
            if len(word) > 3:
                words.append(word)
        words.sort(key=len)
        for word in words:
            print(word, end=",", file=output_file)
        print(file=output_file)
