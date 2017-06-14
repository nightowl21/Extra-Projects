from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction import text
import re, nltk
from sklearn.feature_extraction.text import *
import pandas as pd

REMOVE_WORDS = ["oil", "salt", "fresh", "gorund", "water", "sugar",
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

REMOVE_WORDS += list(text.ENGLISH_STOP_WORDS)

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

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.lemmatize(item))
    return stemmed

def tokenize(text):
    stemmer = WordNetLemmatizer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub(" +"," ", text)
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def create_features(train_file = 'data/train.json',
                    test_file = 'data/test.json'):
    train = pd.read_json(train_file)
    train.head()
    train.columns = ["Y", "id", "ingredients"]
    #Then Extract the ingredients and convert them to a single list of recipes
    #  called words_list

    ingredients = train['ingredients']
    # ingredients = train['ingredients'].append(test['ingredients'])

    words_list = [' '.join(x).strip() for x in ingredients if x not in REMOVE_WORDS]
    words_list = [clean_data(x) for x in words_list]

    vectorizer = CountVectorizer(analyzer = 'word',
                                 tokenizer = tokenize,
                                 lowercase = True,
                                 stop_words=REMOVE_WORDS,
                                 max_features = 2000)

    #create a bag of words and convert to a array
    bag_of_words = vectorizer.fit(words_list)
    train_ingredients = vectorizer.transform(words_list).toarray()
    train_ingredients = pd.DataFrame(train_ingredients,
                                     columns=vectorizer.get_feature_names())

    train = pd.concat([train[["Y", "id"]], train_ingredients], axis=1)

    #Do the same thing we did with the training set and create a array using the
    # count vectorizer.
    test = pd.read_json(test_file)
    test.head()
    ingredients = test['ingredients']
    words_list = [' '.join(x).strip() for x in ingredients if x not in REMOVE_WORDS]
    words_list = [clean_data(x) for x in words_list]
    test_ingredients = vectorizer.transform(words_list).toarray()
    test_ingredients = pd.DataFrame(test_ingredients,
                                    columns=vectorizer.get_feature_names())

    test = pd.concat([test["id"], test_ingredients], axis=1)

    return train, test


