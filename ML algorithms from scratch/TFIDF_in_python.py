"""
Topic Modeling using TFIDF on Reuters Articles
"""

__author__ = 'chhavi21'
import xml.etree.cElementTree as ET
import os, glob
import string
from math import *
import collections
import re


def filelist(pathspec):
	filelist = glob.glob(pathspec)
	files = [file for file in filelist if (os.stat(file).st_size != 0)]
	return files


def get_text(file):
	tree = ET.ElementTree(file = file)
	title_tag = tree.iter(tag = "title")
	title = title_tag.next()
	title = title.text

	text_tag = tree.iter(tag = "text")
	text = text_tag.next()
	t = ""
	for i in text.itertext():
		t += i

	return title + " " + t


def words(d):
	# Input: Document d
	# Result: non-unique list of words wordlist
	#Replace numbers, punctuation, tab, carriage return, newline with space
	d = re.sub('[0-9]+', ' ', d)
	r = re.compile(r'[\s{}]+'.format(re.escape(string.punctuation)))
	d = r.sub( ' ', d)
	wordlist = d.lower().split()
	wordlist = [word for word in wordlist if len(word) > 2]
	return wordlist


def create_indexes(filelist):
	#Input: List of filenames files
	#Result: (Map document name to Counter object mapping term to
	#frequency map tf_map, Counter object mapping term to
	#document count df )
	df = collections.Counter()
	tf_map = {}
	for file in filelist:
		d = get_text(file)
		wordlist = words(d)
		n = len(wordlist)
		tf = collections.Counter(wordlist)
		# walk unique word list
		for t in tf:
			tf[t] = tf[t]/float(n)
			df[t] += 1
		tf_map[file] = tf

	return (tf_map, df)

def doc_tfidf(tf , df , N):
	#Input: Term to frequency map tf
	#Input: Term to document count map df
	#Input: Number of documents N
	#Result: Map of each term in doc (tf) to TFIDF score
	tfidf = {}
	for t in tf:
		if t in df:
			tfidf[t] = tf[t] * log((N)/float(df[t]))
		else:
			tfidf[t] = tf[t] * log((N+1)/float(df[t]+1))
			# apply smoothing only if the df[t] = 0
	return tfidf


def create_tfidf_map(files):
	"""
	Input: List of xml filenames files
	Result: Map from file to map of term to TFIDF score
	(tf_map, d f ) = create_indexes( files)
	tfidf_map = {}
	foreach f 2 f iles do
	tfidf = doc_tf id f (tf _map[ f ], d f )
	tfidf_map[ f ] = t f id f
	end"""
	(tf_map, df) = create_indexes(files)
	tfidf_map = {}
	N = len(files)
	print N
	for f in files:
		tfidf_map[f] = doc_tfidf(tf_map[f], df, N)
	return tfidf_map