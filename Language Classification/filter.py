import praw,sys,time, re

    
    english_file= open("frequentWords.txt", "r")
    english_file.close()
    common_submission_words = set(english_submission_words[:20]) #0-100
 

def is_correct_language(wordset,filter_set):
    for word in wordsset:
        if word in filter_set:
            return False
    return True

