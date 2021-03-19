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

def check_causality(independent:Doc, dependent:Doc, alternative:bool=False) -> bool:
    #print(independent.dep_ + '-' + dependent.dep_)
    if not self.mes['L_ENGLISH']: #Dutch
        if not alternative:
            tuples = [('nsubj', 'obj'),('obj', 'ROOT'),('nsubj', 'nmod'),('obl', 'obj'),('ROOT', 'obj'),
                  ('obj', 'nmod'), ('amod', 'obj'), ('obl','obl'),('nsubj','obl'),('obj','obj'),('nsubj','amod'),
                  ('obj','obl'),('nmod','obj'),('obl','ROOT'),('obl','nsubj'),('obl','csubj'),('advmod','obj'),
                  ('advmod','nmod'),('advmod','obj'),('advmod','obl')]
        else: #Add reverse causality and disturbing variable options
            tuples = [('obj','obj'),('obj','nsubj'), ('ROOT','obj'),('nmod','nsubj'),('obj','obl'),('obj','ROOT'),('amod','nsubj'),
                           ('nmod','obj'),('obj','amod'),('obl','nsubj'),('obl','obj'),('obj','nmod'),('ROOT','obl'),('nsubj','obl'),
                           ('obl','obl'), ('csubj','obl')]
    else:
        if not alternative:
            tuples = [('nsubj','dobj'),('nsubj','ROOT'),('pobj','nsubjpass'),('nsubj','pobj')]
        else:
            tuples = [('dobj','nsubj'),('ROOT','nsubj'),('nsubjpass','pobj'),('pobj','nsubj')]
    
    for t in tuples:
        if independent.dep_ == t[0] and dependent.dep_ == t[1]:
            return True
    return False


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