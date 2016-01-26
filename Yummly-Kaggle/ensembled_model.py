from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import re, nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import *
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.feature_extraction import text

########################################################################################
# Data Prep
########################################################################################

train = pd.read_json('train.json')
train.head()

#Initalize a CountVectorizer only considering the top 2000 features.
#Then Extract the ingredients and convert them to a single list of recipes called words_list
vectorizer = CountVectorizer(max_features = 2000)
ingredients = train['ingredients']
words_list = [' '.join(x) for x in ingredients]

#create a bag of words and convert to a array and then print the shape
bag_of_words = vectorizer.fit(words_list)
bag_of_words = vectorizer.transform(words_list).toarray()
print(bag_of_words.shape)

#Now read the test json file in
test = pd.read_json('test.json')
test.head()

#Do the same thing we did with the training set and create a array using the count vectorizer.
test_ingredients = test['ingredients']
test_ingredients_words = [' '.join(x) for x in test_ingredients]
test_ingredients_array = vectorizer.transform(test_ingredients_words).toarray()

########################################################################################
# Random Forest
########################################################################################

#Initilize a random forest classifier with 500 trees and fit it with the bag of words we created
forest = RandomForestClassifier(n_estimators = 500)
forest = forest.fit( bag_of_words, train["cuisine"] )

# Use the random forest to make cusine predictions
result = forest.predict(test_ingredients_array)
result[2]

forest_prob = forest.predict_proba(test_ingredients_array)
forest_prob[2]

########################################################################################
# Logistic Regression
########################################################################################

clf_log = LogisticRegression(C=10)
clf_log = clf_log.fit(bag_of_words, train["cuisine"])
result_clf_log = clf_log.predict(test_ingredients_array)
result_clf_log[1]
result_clf_log_prob = clf_log.predict_proba(test_ingredients_array)
result_clf_log_prob[1]

########################################################################################
# XGB
########################################################################################

train_data_df = pd.read_json("train.json")
test_data_df = pd.read_json("test.json")

train_data_df1 = train_data_df.drop('id',1)
test_data_df1 = test_data_df.drop('id',1)


remove_words = ["oil", "salt", "fresh", "gorund", "water", "sugar",
                "black", "red", "juice", "chop", "green", "dri", "veget",
                "leave", "larg", "allpurpos", "seed", "cook", "grate",
                "past", "unsalt", "boneless", "shred", "yellow", "dice",
                "minc", "crush", "purpl", "heavi", "canola", "peel", "granul",
                "wedg", "coars", "roll", "nonfat", "partskim", "longgrain",
                "refri", "leg", "melt", "instant", "gold", "star", "drain",
                "blend", "evapor", "pure", "english", "top", "pack", "food",
                "base", "rise", "prepar", "soften", "self", "puspos", "old",
                "silver", "unbleached", "refriger", "snow", "skim",
                "reducedfat", "round", "sea", "bone", "countri", "lower",
                "soft", "short", "liquid", "mild", "piec", "concentr",
                "quickcook", "color", "lump", "jumbo", "fri", "unflavor",
                "flavor", "hardboil", "blanch", "seedless", "angel", "raw",
                "steam", "cut", "organ", "stem", "rub", "link", "new", "tabl",
                "substitut", "paper", "brew", "bag", "mash", "skin", "spread",
                "split", "uncook", "gluten", "cane", "natur", "bottl",
                "doubl", "silken", "mein", "center", "back", "best", "chunk",
                "bought", "great", "wholemilk", "northern", "regular", "process",
                "frost", "dust", "real", "tip", "vegan", "simpl", "key", "pink",
                "breakfast", "colour", "lowsodium", "melt", "fatfre", "wheat",
                "drip", "blue", "skirt", "shortgrain", "homemad", "hellman",
                "store", "ovenreadi", "glass", "simpl", "delici", "key",
                "ripen", "thin", "sec", "origin", "neutral", "strip", "colour",
                "fill", "lowsodium", "grand", "wide",
                "full", "loos", "quick", "straw", "splenda", "quarter", "olive",
                "dinner", "nutrit", "glutenfre", "garden", "pound", "world",
                "imit", "stevia", "solid", "lite", "five", "oldfashion",
                "non", "convert", "distil", "fire", "vegetarian", "rapid", "bun",
                "wing", "thick", "neck", "pan", "four", "cuisin", "winter",
                "ring", "equal", "fatback", "well", "wrap", "undrain", "well",
                "sum", "parboil", "kikkoman", "steamer", "stripe", "mrs",
                "thousand", "rich", "purpo", "semi", "home", "greater",
                "moistur",  "rainbow", "protein", "mountain", "mms", "violet",
                "kitchen", "globe", "homestyl", "day", "diet", "aka", "whey",
                "starter", "shape", "quatr", "plus", "mixtur", "compress",
                "beverag", "vitamin", "multigrain", "good", "special",
                "section", "ounc", "one", "nestl", "iced", "herbal", "godiva",
                "fashion", "alphabet", "wholesom", "vegetablefil", "true",
                "straight", "storebought", "stand", "tast", "seven",
                "multipurpos", "milkfat", "grind", "crunch", "big", "better",
                "delux", "dreamfield", "dream", "harvest", "hint", "hershey",
                "bittersweet", "swiss", "salad", "ketchup", "frank", "ground",
                "hot", "flake", "iodiz", "perfect", "yoplait", "valley",
                "truvia", "wishbone", "vay", "tabasco", "sargento","robusto",
                "redhot", "ranch", "ragu", "mazola", "jonshonville", "unbleach",
                "johnsonville", "islands", "heinz", "creations", "crock",
                "crocker", "breyer", "bertolli", "bacardi", "zero", "wish",
                "wishbon", "wholem", "whip", "white", "whole", "warm",
                "unsweeten", "unsmok", "tricolor", "tripl", "textur", "stuf",
                "thickcut", "tenderloin", "sweeten", "sweet", "superior",
                "style", "string", "stoneground", "stone", "steelcut", "squeez",
                "shave", "segment", "scrub", "secret", "rustic", "prime",
                "preserv", "premium", "precook", "prebak", "powder", "pour",
                "pot", "plain", "plain", "nondairi", "light", "less", "lesser",
                "layer", "leftov", "halfandhalf", "half", "fryer", "fume",
                "free", "flesh", "flat", "flatbread", "firm", "farmer",
                "famili", "extract", "essenc", "earth", "eat", "dinosaur",
                "diamond", "deep", "decor", "dash", "deepfri", "cube",
                "edibl", "dress", "dairi", "dew", "dessert", "dish", "creation",
                "crack", "cool", "cooki", "cocacola", "coke", "cola", "clear",
                "chip", "bulk", "bunch", "brown", "breast", "bowl", "bottom",
                "boil", "block", "blade", "blacken", "beaten", "age", "accent",
                "activ", "zest", "young", "wild", "toast", "thigh", "tender",
                "syrup", "sugarcan", "sub", "stir", "sprinkl", "spiral",
                "soak", "smooth", "smoke", "smart", "slim", "slice", "sliver",
                "sodium", "softboil", "sour", "small", "smart", "slab",
                "size", "simpli", "semisoft", "semisweet", "season", "mix",
                "roma", "ripe", "recip", "proactiv", "pulp", "organic",
                "philadelphia", "part", "originals", "nosaltad", "nutella",
                "ornament", "nonstick", "nostick", "nonhydrogen", "mixer",
                "mini", "miniatur", "minicub", "min", "mincemeat", "milk",
                "mccormick", "marbl", "leafi", "leaf", "lean", "lard",
                "lake", "juic", "johnsonvill", "jonshonvill", "island", "ice",
                "halv", "grill", "glucos","glutin", "fullfat", "frozen",
                "fructos", "enrich", "drink", "crumbl", "crumb", "chobani",
                "cholesterol", "butterflavor", "broilerfry", "broil",
                "artisan", "artifici", "smithfield", "smith", "skinless",
                "reduc", "leav", "low", "roast", "fat", "dark", "cold",
                "lowfat", "condens", "sharp", "loin", "grain", "farm", "kraft",
                "fine", "medium", "golden", "extra", "long", "extralean",
                "farmhous", "digest"]

train_data_df1['ingredients'] = [' , '.join(z).strip() for z in train_data_df1['ingredients']]
test_data_df1['ingredients'] = [' , '.join(z).strip() for z in test_data_df1['ingredients']]


def clean_data(strng):
    # correct spelling mistakes
    if 'india' in strng: strng = strng.replace('india', 'indian')
    if 'america' in strng: strng = strng.replace('america', 'american')
    if 'italianstyle' in strng: strng = strng.replace('italianstyle', 'italian')
    if 'chapati' in strng: strng = strng.replace('chapati', 'chapatti')
    if 'cardamon' in strng: strng = strng.replace('cardamon', 'cardamom')
    if 'asafetida' in strng: strng = strng.replace('asafetida', 'asafoetida')
    if 'glace' in strng: strng = strng.replace('asafetida', 'glaze')
    if 'jamaica' in strng: strng = strng.replace('jamaica', 'jamaican')
    if 'linguica' in strng: strng = strng.replace('linguica', 'linguica')
    if 'mayonnais' in strng: strng = strng.replace('mayonnais', 'mayonais')
    if 'linguin' in strng: strng = strng.replace('linguin', 'linguini')
    if 'mexicana' in strng: strng = strng.replace('mexicana', 'mexican')
    if 'mexico' in strng: strng = strng.replace('mexico', 'mexican')
    if 'pepperoncini' in strng: strng = strng.replace('pepperoncini', 'pepperocini')
    if 'peperoncini' in strng: strng = strng.replace('peperoncini', 'pepperocini')
    if 'peperoncino' in strng: strng = strng.replace('peperoncino', 'pepperocini')
    if 'proscuitto' in strng: strng = strng.replace('proscuitto', 'prosciutto')
    if 'portobello' in strng: strng = strng.replace('portobello', 'portabello')
    if 'mozarella' in strng: strng = strng.replace('mozarella', 'mozzarella')
    if 'parmagiano' in strng: strng = strng.replace('parmagiano', 'parmigiano')
    if 'parmigianareggiano' in strng: strng = strng.replace('parmigianareggiano', 'parmigiano')
    if 'parmigianoreggiano' in strng: strng = strng.replace('parmigianoreggiano', 'parmigiano')
    if 'yoghurt' in strng: strng = strng.replace('yoghurt', 'yogurt')
    if 'chili' in strng: strng = strng.replace('chili', 'chilli')
    return strng


train_data_df1['ingredients'] = train_data_df1['ingredients'].apply(lambda x:
                                                                  clean_data(x))
test_data_df1['ingredients'] = test_data_df1['ingredients'].apply(lambda x:
                                                                  clean_data(x))

stemmer = WordNetLemmatizer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.lemmatize(item))
    return stemmed

def tokenize(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub(" +"," ", text)
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


remove_words = remove_words + list(text.ENGLISH_STOP_WORDS)

vectorizer = CountVectorizer(analyzer = 'word',
                             tokenizer = tokenize,
                             lowercase = True,
                             stop_words =remove_words)


corpus_data_features = vectorizer.fit_transform(train_data_df1.ingredients.tolist() + test_data_df1.ingredients.tolist())


le = LabelEncoder()
le.fit(train_data_df.cuisine)
labels_numeric = le.transform(train_data_df.cuisine)


xg_train = xgb.DMatrix(corpus_data_features[0:len(train_data_df1)], label = labels_numeric)
xg_test = xgb.DMatrix(corpus_data_features[len(train_data_df):])

param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.01
param['max_depth'] = 25
param['num_class'] = 20
num_round = 1000

gbm = xgb.train(param, xg_train, num_round)
test_pred = gbm.predict(xg_test)

########################################################################################
# Ensembling tuning on training
########################################################################################

train_forest_prob = forest.predict_proba(bag_of_words)
train_forest_prob[2]
train_clf_log_prob = clf_log.predict_proba(bag_of_words)
train_clf_log_prob[1]

train_pred = gbm.predict(xg_train)

acc_df = pd.DataFrame(columns=["i","j","k","acc"])

wts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for i in wts:
    for j in wts:
        for k in wts:
            if abs(1 - (i + j + k) ) <= 0.01:
                print abs(1 - (i + j + k) )
                train_pred = (i*train_pred +
                              j*train_forest_prob +
                              k*train_clf_log_prob)/3
                train_prediction= le.inverse_transform(train_pred.argmax(
                        axis=1))
                acc = np.mean(train_prediction == train.cuisine)
                acc_df = acc_df.append(pd.DataFrame({"i": i, "j":j, "k":k,
                                                "acc":acc}, index=[0]))


acc_df.sort(['acc'])

########################################################################################
# Ensembling by averaging
########################################################################################

final_pred = (0.5*test_pred + 0.3*forest_prob + 0.2*result_clf_log_prob)/3

pred = le.inverse_transform(final_pred.argmax(axis=1))

# Copy the results to a pandas dataframe with an "id" column and
# a "cusine" column
output = pd.DataFrame( {"id":test["id"], "cuisine":pred} )

# Use pandas to write the comma-separated output file
output[['id', 'cuisine']].to_csv( "submission_ensemble_avg.csv", index=False)

