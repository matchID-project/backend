#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# various libs
import re, datetime, hashlib, unicodedata, shutil, simplejson, random
from collections import OrderedDict


#ngram with nltk
from nltk.util import ngrams
#from nltk.tokenize import WhitespaceTokenizer

# geodistance computation
from geopy.distance import vincenty
# from decimal import *
# from fuzzywuzzy import fuzz, process
# from fastcomp import compare
import jellyfish
import jellyfish._jellyfish as py_jellyfish
from log import err
import json



def float_val(x):
    try:
        return float(x)
    except:
        return 0

def deepupdate(original, update):
    """
    Recursively update a dict.
    Subdict's won't be overwritten but also updated.
    """
    for key, value in original.items():
    # python3 for key, value in original.items():
        if key not in update:
            update[key] = value
        elif isinstance(value, dict):
            deepupdate(value, update[key])
    return update

def parsedate(x="",format="%Y%m%d"):
    try:
        return datetime.datetime.strptime(x,format)
    except:
        return None

def jsonDumps(j=None,encoding='utf8'):
    return simplejson.dumps(j, ensure_ascii=False, encoding=encoding,ignore_nan=True)

def toJson(x = None):
    if (x == None):
        return ""
    if ((type(x) != str) & (type(x) != str)):
        return x
    if (x == ""):
        return x
    try:
        return [json.loads(x)]
    except:
        return x

def geopoint(geopoint):
    try:
        return tuple(float(x) for x in geopoint.replace("POINT(","").replace(")","").split(" ")[::-1])
    except:
        return ""

def union(x):
    return list(x)

def distance(a,b):
    try:
        return round(10*vincenty(geopoint(a),geopoint(b)).kilometers)/10
    except:
        return ""

def replace_regex(x,regex):
    if (type(x)==str):
        for r in regex:
            x=r[0].sub(r[1],x)
    elif (type(x)==list):
        x=[replace_regex(z,regex) for z in x]
    elif (type(x)==dict):
        x=dict((k,replace_regex(v,regex)) for (k,v) in list(x.items()))
    return x

def replace_dict(x,dic):
    if (type(x)==str):
        if x in list(dic.keys()):
            return dic[x]
    elif (type(x)==list):
        x=[replace_dict(z,dic) for z in x]
    elif ((type(x)==dict) | (type(x).__name__=="OrderedDict")):
        x=dict((k,replace_dict(v,dic)) for (k,v) in list(x.items()))
    return x

def sha1(row):
    return hashlib.sha1(str(row).encode('utf-8')).hexdigest()

def ngrams(x,n = [3]):
    if (type(x) == list):
        return flatten([ngrams(z, n) for z in x])
    elif (type(x)==str):
        return flatten([[x[i:i+p] for i in range(len(x)-p+1)] for p in n])

def flatten(x):
    if (type(x) == list):
        return [a for i in x for a in flatten(i)]
    else:
        return list([x])

def tokenize (x=None):
    if (type(x)==list):
        return flatten([tokenize(z) for z in x])
    elif (type(x)==str):
        return re.split('\s\s*',x)
    else:
        return tokenize(str(x))

def unicode_safe(x):
    if (type(x) == float) | (type(x) == int):
        return x
    try:
        return str(x)
    except:
        pass
    try:
        return "Ooops: '{}' return an unicode error: {}".format(str(x, "utf8", errors="ignore"),err())
    except:
        pass
    return "Ooops: '{}' return an unicode error: {}".format(str.encode(x, "ascii", errors="ignore").decode('ascii'),err())

def normalize(x=None):
    if (type(x)==str):
        x=unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('ascii')
    if (type(x)==str):
        x=re.sub('[^A-Za-z0-9]+', ' ', x.lower())
        x=re.sub('\s+', ' ', x)
        x=re.sub('^\s+$', '', x)
    elif (type(x)==list):
        x=[_f for _f in [normalize(z) for z in x] if _f]
        # if (len(x)==1):
        # 	x=x[0]
        # elif(len(x)==0):
        # 	x=""
    return x

def jw(s1,s2):
    maxi=0
    if (type(s1)==list):
        for s in s1:
            maxi=max(maxi,jw(s,s2))
        return maxi
    if (type(s2)==list):
        for s in s2:
            maxi=max(maxi,jw(s1,s))
        return maxi
    if (type(s1) == str):
        s1 = str(s1)
    if (type(s2) == str):
        s2 = str(s2)
    return round(100*jellyfish.jaro_winkler(s1,s2))/100

def levenshtein(s1, s2):
    if (not s1):
        s1=""
    if (not s2):
        s2=""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    #choosen
    if len(s2) == 0:
        return len(s1)

    try:
        return jellyfish.damerau_levenshtein_distance(str(s1),str(s2))
    except:
        # workaround for unicode : fallback from c to python version
	    return py_jellyfish.damerau_levenshtein_distance(str(s1),str(s2))
    # cached version
    try:
        return levCache[tuple(s1,s2)]
    except:
        pass

    levCache[tuple([s1,s2])] = jellyfish.levenshtein_distance(str(s1),str(s2))
    return levCache[tuple([s1,s2])]

    #original
    # len(s1) >= len(s2)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def levenshtein_norm(s1,s2):
    if True:
        if (type(s1)==list):
            maxi=0
            for s in s1:
                maxi=max(maxi,levenshtein_norm(s,s2))
            return maxi

        if (type(s2)==list):
            maxi=0
            for s in s2:
                maxi=max(maxi,levenshtein_norm(s1,s))
            return maxi
        maxi=0
        return max(maxi,round(100-100*float(levenshtein(s1,s2))/(1+min(len(s1),len(s2))))/100)

    else:
        return 0

def safeeval(expression=None,row=None,verbose=True,defaut=""):
    cell = None
    locals().update(row)
    try:
        if ('cell' in expression):
            exec(expression)
        else:
            cell = eval(expression)

        return cell
    except:
        if (verbose):
            return "Ooops in exec('{}'): {}".format(expression,err())
        else:
            return default


def match_lv1(x, list_strings):
    best_match = None
    best_score = 3
    for current_string in list_strings:
        current_score = compare(x, current_string)
        if (current_score==0):
            return current_string
        elif (current_score>0 & current_score < best_score):
            best_match=current_string
            best_score=current_score

    if best_score >= 2:
        return None
    return best_match

def match_jw(x, list_strings):
    best_match = None
    best_score = 0

    for current_string in list_strings:
        current_score = jellyfish.jaro_winkler(str(x), str(current_string))
        if(current_score > best_score):
            best_score = current_score
            best_match = current_string

    if (best_score>=0.95):
        return best_match
    else:
        return None

