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

def check_causality(i:Doc, d:Doc, alternative:bool=False) -> bool:
    print([i.text,i.pos_,i.tag_,i.dep_,i.shape_])
    print([d.text,d.pos_,d.tag_,d.dep_,d.shape_])
    #print(independent.dep_ + '-' + dependent.dep_)
    mes={'L_ENGLISH':False}
    if not mes['L_ENGLISH']: #Dutch
        if not alternative:
            tuples = [('nsubj', 'obj'),('obj', 'ROOT'),('nsubj', 'nmod'),('obl', 'obj'),('ROOT', 'obj'),
                  ('obj', 'nmod'), ('amod', 'obj'), ('obl','obl'),('nsubj','obl'),('obj','obj'),('nsubj','amod'),
                  ('obj','obl'),('nmod','obj'),('obl','ROOT'),('obl','nsubj'),('obl','csubj'),('advmod','obj'),
                  ('advmod','nmod'),('advmod','obj'),('advmod','obl')]
        else: #Add reverse causality and disturbing variable options
            tuples = [('obj','obj'),('obj','nsubj'), ('ROOT','obj'),('nmod','nsubj'),('obj','obl'),('obj','ROOT'),('amod','nsubj'),
                           ('nmod','obj'),('obj','amod'),('obl','nsubj'),('obl','obj'),('obj','nmod'),('ROOT','obl'),('nsubj','obl'),
                           ('obl','obl'), ('csubj','obl')]
    else: #English
        if not alternative:
            tuples = [('nsubj','dobj'),('nsubj','ROOT'),('pobj','nsubjpass'),('nsubj','pobj'),('ROOT','obl'), ('amod','ROOT'),
                      ('compound','ROOT'),('amod','amod'),('obl','amod')]
        else:
            tuples = [('dobj','nsubj'),('ROOT','nsubj'),('nsubjpass','pobj'),('pobj','nsubj'),('obl','ROOT'), ('ROOT','amod'),
                      ('ROOT','compound'),('compound','obl')]
    
    for t in tuples:
        if i.dep_ == t[0] and d.dep_ == t[1]:
            return True
    return False

#nlp = spacy.load('nl')
nlp = spacy.load('nl')
txt = 'Geen experiment, dus er zijn meerdere verklaringen mogelijk. De primaire is dat ras criminaliteit beinvloedt.'
sent = nlp(txt)
nd = dict([(x.text,x) for x in sent])
print(check_causality(nd['ras'],nd['criminaliteit']))
#nld - eng
#synonyms = []
#for syn in wn.synsets("birthplace", lang='eng'): 
#    for l in syn.lemmas(lang='eng'): 
#        synonyms.append(l.name())
#print(synonyms)