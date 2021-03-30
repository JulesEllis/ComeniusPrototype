import math
import numpy as np
import os
import spacy
import logging
import json
import math
import random
import numpy as np
import nltk
import spacy
import re
import nltk
import inflect
from copy import copy
from spacy.tokens.token import Token
from spacy.tokens.doc import Doc
from spacy import displacy
from nltk import CFG, Tree, edit_distance
from scipy import stats
from typing import Dict, List, Tuple
from scipy import stats
from typing import Dict, List, Callable, Tuple
from nltk.corpus import wordnet as wn

class Variable:
    def __init__(self, name:str, synonyms:list, node:str):
        self.name = name
        self.synonyms = []
        self.node = node


#nlp = spacy.load('nl')
nlp = spacy.load('en')
sent = nlp('No experiment, so multiple explanations are possible. The primary explanation is that birthplace influences weight. The alternative hypothesis is that birthplace and weight are both caused by an interfering variable. ')
nd = dict([(x.text,x) for x in sent])

print([(x.text,x.dep_) for x in sent])
#print(check_causality(nd['race'],nd['criminality']))
#nld - eng
#synonyms = []
#for syn in wn.synsets("birthplace", lang='eng'): 
#    for l in syn.lemmas(lang='eng'): 
#        synonyms.append(l.name())
#print(synonyms)