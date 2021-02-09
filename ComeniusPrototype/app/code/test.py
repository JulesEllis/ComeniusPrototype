#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 10:10:33 2021

@author: jelmer
"""
import math
import random
import numpy as np
import os
import spacy
import logging
import nltk
from scipy import stats
from typing import Dict, List, Callable, Tuple

def closest_effect(var:str, words:list, unidirectional = False, weighted = False):
    if not var in words:
        return None
    else:
        index = words.index(var)
        if not unidirectional and not weighted:
            proximityscores = [(words[i], abs(i - index)) for i in range(len(words)) if words[i] in ['klein','matig','groot','zwak','sterk']]
        elif not unidirectional:
            proximityscores = [(words[i], abs(i - index) / int(2 if i > index else 1)) for i in range(len(words)) if words[i] in ['klein','matig','groot','zwak','sterk']]
        else:
            proximityscores = [(words[i], abs(i - index)) for i in range(len(words)) if words[i] in ['klein','matig','groot','zwak','sterk'] and i > index]
        if proximityscores == []:
            return None
        else:
            return min(proximityscores, key = lambda x: x[1])[0]
    
def associate(text:str, method:int):
    nlp = spacy.load('nl')
    doc = nlp(text)
    words = nltk.word_tokenize(doc.text)#[x.text for x in doc]    
    varss = ['within-subject', 'interactie','between-subject'] #['geluid','zicht','reuk']
    golds = ['sterk','klein','matig'] #['klein','groot','matig']
    sortchoice = method
    if sortchoice == 0:
        effects = [x for x in words if x in ['klein','zwak','matig','groot','sterk']]
        answers = effects[:len(golds)]
    elif sortchoice == 1:
        answers = [closest_effect(x, words) for x in varss]
    elif sortchoice == 2:
        answers = [closest_effect(x, words, unidirectional=True) for x in varss]
    elif sortchoice == 3:
        answers = [closest_effect(x, words, weighted=True) for x in varss]
    return sum([answers[i] == golds[i] for i in range(len(golds))]) / len(golds)
        

sent = 'Bij de within-subject multivariate beslissing is er een sterk significant effect (F = 6.89, p = 0.01, eta = 0.65). '\
    'Dit komt door het contrast tussen na en followup (p = 0.0).'\
    'Bij de interactie tussen meting en bloedtype is er een significant effect. Dit effect is klein .'\
    'Bij de between-subject factor bloedtype is er wel een significant effect (F = 10.01 p = 0.0 eta = 0.91). Dit effect is matig. '
print('Volgorde: ' + str(associate(sent, 0)))
print('Symmetrische afstand: ' + str(associate(sent, 1)))
print('Afstand rechts: ' + str(associate(sent, 2)))
print('Gewogen afstand: ' + str(associate(sent, 3)))