"""
The data is taken from http://ai.stanford.edu/~amaas/data/sentiment/. This dataset
contains 25,000 highly polar movie reviews for training, and 25,000 for test-
ing. The following code implements Naive Bayes from Scratch in PySpark and the accuracy
of the algorithm is compared with NaiveBayes Implementation in MLlib. The accuracy with MLlib is
slighly better that from-scratch implementation.
"""

#########################################################################################################
# Naive Bayes in Spark from Scratch
#########################################################################################################

from __future__ import division
import re
import time
from math import *
from collections import Counter

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

stopWords = Counter(stopWords)

def clean_data(line):
	line = re.sub(re.compile(r'[^a-z]'), ' ', line.lower())
	line = line.split()
	line = [l for l in line if l not in stopWords and len(l) > 2]
	return line

# use the following path for running on local
path = '/Users/chhavi21/Box Sync/aclImdb/'


#use the following path for aws
# AWS_ACCESS_KEY_ID = "-"
# AWS_SECRET_ACCESS_KEY = "-"
# sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY_ID)
# sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)
# path = 's3n://amlsparkstuff/'


s = time.time()

train_pos = sc.textFile(path + 'train/pos/*.txt').repartition(16)
train_neg = sc.textFile(path + 'train/neg/*.txt').repartition(16)
test_pos = sc.textFile(path + 'test/pos/*.txt').repartition(16)
test_neg = sc.textFile(path + 'test/neg/*.txt').repartition(16)

## TRAINING THE MODEL

pos_count = train_pos.count()
neg_count = train_neg.count()

pos_prior = pos_count/(pos_count + neg_count)
neg_prior = neg_count/(pos_count + neg_count)

train_pos = train_pos.map(clean_data)
train_neg = train_neg.map(clean_data)
test_pos = test_pos.map(clean_data)
test_neg = test_neg.map(clean_data)

# maping of count of words in all documents in a class
# gives (word, count)
train_pos_word_map = train_pos.map(lambda x: [(m,1) for m in x]).flatMap(lambda x: x).reduceByKey(lambda a,b: a+b).cache()
train_neg_word_map = train_neg.map(lambda x: [(m,1) for m in x]).flatMap(lambda x: x).reduceByKey(lambda a,b: a+b).cache()


# number of words in all pos docs = 1531176
# number of words in all neg docs = 1462341
train_pos_word_count = train_pos_word_map.map(lambda (a,b): b).reduce(lambda a,b: a+b)
train_neg_word_count = train_neg_word_map.map(lambda (a,b): b).reduce(lambda a,b: a+b)

# number of common words in the 2 classes
V = train_pos_word_map.map(lambda x: x[0]).union(train_neg_word_map.map(lambda x: x[0])).count()
#106987

# change this to impletement laplace smoothing.
# gives (word, probabilty)
train_pos_word_map_prob = train_pos_word_map.map(lambda (a,b): (a, (b+1)/(train_pos_word_count + V + 1))).collect()
train_neg_word_map_prob = train_neg_word_map.map(lambda (a,b): (a, (b+1)/(train_neg_word_count + V + 1))).collect()

# dictionary of word probabilities
train_pos_word_map_prob = dict(train_pos_word_map_prob)
train_neg_word_map_prob = dict(train_neg_word_map_prob)


## MAKING PREDICTIONS

def predict(word_map, train_class_word_map_prob, class_prior, train_class_word_count, V):
    #word_map: dict of a cleaned document
    #train_class_word_map_prob: dictionary of P(X|class) Class Conditionals
    #class_prion: Prior probabilty of class
    #train_class_word_count: number of words in the class for training set
    #V: number of common words in pos and neg class
    #returns: probabilty of document belonging to a class
	prob = log(class_prior)
	for word in word_map:
		power = word_map[word]
		if word in train_class_word_map_prob:
			word_prob = train_class_word_map_prob[word]
			prob += power * log(word_prob)
		else:
			prob += power * log((0+1)/float(train_class_word_count + V + 1))
	return prob


# positive set accuracy on train
train_pos_correct = train_pos.map(lambda x: Counter(x)).map(lambda x: (predict(x, train_pos_word_map_prob, pos_prior, train_pos_word_count, V), predict(x, train_neg_word_map_prob, neg_prior, train_neg_word_count, V))).map(lambda x: 1 if x[0]>x[1] else 0).reduce(lambda a,b: a+b)
# negative set accuracy on train
train_neg_correct = train_neg.map(lambda x: Counter(x)).map(lambda x: (predict(x, train_pos_word_map_prob, pos_prior, train_pos_word_count, V), predict(x, train_neg_word_map_prob, neg_prior, train_neg_word_count, V))).map(lambda x: 1 if x[1]>x[0] else 0).reduce(lambda a,b: a+b)

# positive set accuracy on train - 0.89168
train_pos_correct/pos_count
# positive set accuracy on test - 0.93464
train_neg_correct/neg_count

#total test set accuracy - 0.91316
combined_train = (train_pos_correct+train_neg_correct)/(pos_count + neg_count)


# apply decision rule. If prob(pos)>prob(neg) then assign pos class to doc.
test_pos_correct = test_pos.map(lambda x: Counter(x)).map(lambda x: (predict(x, train_pos_word_map_prob, pos_prior, train_pos_word_count, V), predict(x, train_neg_word_map_prob, neg_prior, train_neg_word_count, V))).map(lambda x: 1 if x[0]>x[1] else 0).reduce(lambda a,b: a+b)
test_neg_correct = test_neg.map(lambda x: Counter(x)).map(lambda x: (predict(x, train_pos_word_map_prob, pos_prior, train_pos_word_count, V), predict(x, train_neg_word_map_prob, neg_prior, train_neg_word_count, V))).map(lambda x: 1 if x[1]>x[0] else 0).reduce(lambda a,b: a+b)

test_pos_correct_count = test_pos.count()
test_neg_correct_count = test_neg.count()

# positive set accuracy on test
test_pos_correct/test_pos_correct_count #0.7736
# negative set accuracy on test
test_neg_correct/test_neg_correct_count #0.88144

#total test set accuracy
combined_test = (test_pos_correct+test_neg_correct)/(test_pos_correct_count + test_neg_correct_count)
# 0.82752

print (time.time() - s)/60, " mins"



#########################################################################################################
# Naive Bayes in Spark from MlLIb
#########################################################################################################

from __future__ import division
from collections import Counter
import time
import re
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.regression import LabeledPoint


# use the following path for running on local
path = '/Users/chhavi21/Box Sync/aclImdb/'


#use the following path and access key for aws
# AWS_ACCESS_KEY_ID = "-"
# AWS_SECRET_ACCESS_KEY = "-"

# sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY_ID)
# sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)

# path = 's3n://amlsparkstuff/'


stopWords = ['able', 'about', 'across', 'after', 'all', 'almost', 'also',
             'among', 'and', 'any', 'are', 'because', 'been', 'but', 'can',
             'cannot', 'could', 'dear', 'did', 'does', 'either', 'else',
             'ever', 'every', 'for', 'from', 'get', 'got', 'had', 'has',
             'have', 'her', 'here' 'hers', 'him', 'his', 'how', 'however',
             'into', 'its', 'just', 'least', 'let', 'like', 'likely', 'may',
             'might', 'most', 'must', 'neither', 'nor', 'not', 'off', 'often',
             'only', 'other', 'our', 'own', 'put', 'rather', 'said', 'say',
             'says', 'she', 'should', 'since', 'some', 'such', 'than', 'that',
             'the', 'their', 'them','then', 'there', 'these', 'they', 'this',
             'tis', 'too', 'twas', 'wants', 'was', 'were', 'what', 'when',
             'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with',
             'would', 'yet', 'you', 'your', 'www', 'http', 'women', 'males',
             'each', 'done', 'see', 'before', 'each', 'irs', 'ira', 'hal', 'ham', 'isn']

stopWords = Counter(stopWords)

def clean_lines(line):
    line = re.sub(re.compile(r'[^a-z]'), ' ', line.lower())
    line = line.split()
    line = [l for l in line if l not in stopWords and len(l) > 2]
    return line

def parsePosTrain(line):
    hashingTF1 = HashingTF()
    tf1 = hashingTF1.transform(line)
    return LabeledPoint(1.0, tf1)

def parseNegTrain(line):
    hashingTF2 = HashingTF()
    tf2 = hashingTF2.transform(line)
    return LabeledPoint(0.0, tf2)


s = time.time()


# Positive and negative train data sets
pos_data = sc.wholeTextFiles(path + "train/pos/*.txt").repartition(16)
train_pos = pos_data.map(lambda x: parsePosTrain(clean_lines(x[1])))

neg_data = sc.wholeTextFiles(path + "train/neg/*.txt").repartition(16)
train_neg = neg_data.map(lambda x: parseNegTrain(clean_lines(x[1])))

# Combining positive and negative train data sets and training the model
train = train_pos.union(train_neg).repartition(16)
model = NaiveBayes.train(train)

# Positive and negative test data sets
pos_data = sc.wholeTextFiles(path + "test/pos/*.txt").repartition(16)
test_pos = pos_data.map(lambda x: parsePosTrain(clean_lines(x[1])))

neg_data = sc.wholeTextFiles(path + "test/neg/*.txt").repartition(16)
test_neg = neg_data.map(lambda x: parseNegTrain(clean_lines(x[1])))

test = test_pos.union(test_neg).repartition(16)

pred_train = train.map(lambda p: (model.predict(p.features), p.label))

pred_test = test.map(lambda p: (model.predict(p.features), p.label))

tr_count = train.count()
ts_count = test.count()

accuracy_train = pred_train.filter(lambda (x, v): x == v).count() / tr_count

accuracy_test = pred_test.filter(lambda (x, v): x == v).count() / ts_count

print "Train set accuracy ", accuracy_train, tr_count
# 0.91412 
print "Test set accuracy ", accuracy_test, ts_count
# 0.83236

print (time.time() - s)/60, " mins"

