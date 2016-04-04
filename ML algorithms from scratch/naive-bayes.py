"""
Naive Bayes implemented from scratch in Python.
Data is from http://www.cs.cornell.edu/people/Pabo/movie-review-data/
The NB classifier predicts whether a movie review is positive or 
negative using a dataset set of movie reviews.
"""

__author__ = 'chhavi21'
import os, glob
import string
import random as random
from math import *
import collections

import argparse
import re


# Stop word list
stopWords = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also',
             'am', 'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be',
             'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear',
             'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for',
             'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers',
             'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is',
             'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may',
             'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor',
             'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our',
             'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since',
             'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then',
             'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us',
             've', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
             'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'yet',
             'you', 'your']

def parseArgument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('-d', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args

def filelist(pathspec):
    # give a path, returns the list of files that have non zero size
    filelist = glob.glob(pathspec)
    files = [file for file in filelist if (os.stat(file).st_size != 0)]
    return files

def get_text(file):
    #given a file name returns the text in the file
    f = open(file, 'r')
    lines = f.read()
    f.close()
    return lines

def words(d):
    # Input: Document d
    # Result: non-unique list of words wordlist
    # Replace numbers, punctuation, tab, carriage return, newline with space
    # remove stopwords
    d = re.sub('[0-9]+', ' ', d)
    r = re.compile(r'[\s{}]+'.format(re.escape(string.punctuation)))
    d = r.sub( ' ', d)
    wordlist = d.lower().split()
    wordlist = [word for word in wordlist if (len(word) > 2) and
                                                        (word not in stopWords)]
    return wordlist

def create_indexes(filelist):
    #Input: List of filenames
    #Result: (a Counter object mapping term to frequency map tf for a class,
    # count of number of words in all files of a class class_counter   )
    tf = collections.Counter()
    class_count = 0
    for file in filelist:
        d = get_text(file)
        wordlist = words(d)
        class_count += len(wordlist)
        # calculate total word count for a give class of filelist
        tf += collections.Counter(wordlist)
    return (tf, class_count)

def V(tf1, tf2):
    #takes 2 tf maps, for both pos and neg files
    #returns number of unique words
    return len(tf1 + tf2)

def prob_word_class(tf, wordcount, v):
    #Input: term frequency, number of words in class, number of unique words in
    # class
    #Returns: probablity dict for all terms in class
    p = {}
    for t in tf:
        p[t] = (tf[t] + 1)/float(wordcount + v + 1)
    return p

def prob_pred_class(file, prob, p_class, wordcount, v):
    # Input: filename, probabilty dict, P(class), number of words in class,
    # number of unique words in class
    # Returns probablity score for a document given in a class
    d = get_text(file)
    wordlist = words(d)
    wordcounter = collections.Counter(wordlist)
    p = {}
    for t in wordcounter:
        if t not in prob:
            p[t] = wordcounter[t] * log((0+1)/float(wordcount + v + 1))
            #if the word is UNK assign it probablity P(w|c) and then take log
        if t in prob:
            p[t] = wordcounter[t] * log(prob[t])

    doc_p = log(p_class) + sum(p.values())
    return doc_p

def cv(dir_name):
    # Input a directory name
    # Prints the accuracy for the Naive-Bayes Algo
    # Performs 3 fold CV and returns summary statistic for each iteration

    neg_pathspec = dir_name + "/neg/*"
    neg_files = filelist(neg_pathspec) #read file names for positive files

    pos_pathspec = dir_name + "/pos/*"
    pos_files = filelist(pos_pathspec) #read file names for negative files

    pos_smpl = random.sample(pos_files, len(pos_files)) #shuffle the data
    neg_smpl = random.sample(neg_files, len(neg_files)) #shuffle the data

    # evenly section data
    pos_list = [pos_smpl[ :int(len(pos_files)*0.333)],
                pos_smpl[int(len(pos_files)*0.333) : int(len(pos_files)*0.667)],
                pos_smpl[int(len(pos_files)*0.667): ]]
    neg_list = [neg_smpl[ :int(len(neg_files)*0.333)],
                neg_smpl[int(len(neg_files)*0.333) : int(len(neg_files)*0.667)],
                neg_smpl[int(len(neg_files)*0.667): ]]

    accuracy = [] # to store accuracy values

    for i in range(-1,2):
        train_pos = pos_list[i] + pos_list[i-1] #positive file list for training
        train_neg = neg_list[i] + neg_list[i-1] #negative file list for training
        test_pos = pos_list[i-2] # positive file list for test
        test_neg = neg_list[i-2] # negative file list for test

        # count dictionary and word count for training files
        tf_pos, pos_wordcount = create_indexes(train_pos)
        tf_neg, neg_wordcount = create_indexes(train_neg)

        # number of unique words in training set
        v = V(tf_pos, tf_neg)

        # probabilty score for each class in training set
        p_pos = len(train_pos)/float(len(train_pos+train_neg))
        p_neg = len(train_neg)/float(len(train_pos+train_neg))

        # probailty dictionary for positive and negative files in training set
        pos_prob = prob_word_class(tf_pos, pos_wordcount, v)
        neg_prob = prob_word_class(tf_neg, neg_wordcount, v)

        test_score = [] # to store test scores
        for file in test_pos:
            # generate probability scores for each document in test pos set
            test_score.append(("pos",
                                   (prob_pred_class(file, pos_prob,
                                                    p_pos, pos_wordcount, v)),
                                   (prob_pred_class(file, neg_prob,
                                                    p_neg, neg_wordcount, v))))
        # count number of true positives
        n_pos_correct = sum([1 for j in range(len(test_score))
                                    if (test_score[j][1] > test_score[j][2])])

        test_score = [] # to store test scores
        for file in test_neg:
            #generate probability scores for each document in test neg set
            test_score.append(("neg",
                                   (prob_pred_class(file, pos_prob,
                                                    p_pos, pos_wordcount, v)),
                                   (prob_pred_class(file, neg_prob,
                                                    p_neg, neg_wordcount, v))))
        # count number of true negatives
        n_neg_correct = sum([1 for j in range(len(test_score))
                                    if (test_score[j][1] < test_score[j][2])])

        #Accuracy = (TP + TN)/T
        accuracy.append((n_pos_correct + n_neg_correct) / float(len(test_pos)
                                                            + len(test_neg)))

        print "iterartion %d:" % (i+2)
        print "num_pos_test_docs: %d" % len(test_pos)
        print "num_pos_training_docs: %d" % len(train_pos)
        print "num_pos_correct_docs: %d" %n_pos_correct
        print "num_neg_test_docs: %d" % len(test_neg)
        print "num_neg_training_docs: %d" % len(train_neg)
        print "num_neg_correct_docs: %d" %n_neg_correct
        print "accuracy: %.1f" % (accuracy[-1] * 100)
    print "ave_accuracy: %.1f" % (100 * (sum(accuracy)/3.0))


def main():
    args = parseArgument()
    directory = args['d'][0]
    cv(directory)

main()