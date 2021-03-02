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
from spacy.tokens.token import Token
from spacy.tokens.doc import Doc
from typing import Dict, List, Callable, Tuple
from spacy.pipeline import merge_noun_chunks, merge_entities

def negation_counter(tokens: List[str]) -> int:
    count: int = 0
    for token in tokens:
        if token in ['geen', 'niet']:   # or token[:2] == 'on':
            count += 1
    return count

def descendants(node) -> List[Token]:
    output:list = []
    for child in node.children:
        output.append(child)
        output += descendants(child)
    return output

def detect_primary(sent:Doc, solution:dict, num:int=1) -> List[str]:
    #Define variables
    criteria:list = ['cause', 'ind', 'dep', 'prim', 'neg', 'alignment']
    scorepoints = dict([(x,False) for x in criteria])
    i_key: str = 'independent' + str(num) if num > 1 else 'independent'
    syn_key: str = 'ind_syns' if num == 1 else 'ind' + str(num) + '_syns'
    control: bool = solution['control'] if num < 2 else solution['control'+str(num)]
    rejected = solution['p'][num-1] < 0.05
    tokens = [x.text for x in sent] 
    output:List[str] = []
    
    #Controleer input
    scorepoints['prim'] = 'primaire' in tokens if not control else True
    scorepoints['neg'] = bool(negation_counter(tokens) % 2) != rejected
    causeverbs = [x for x in sent if x.text in ['veroorzaakt', 'heeft', 'beinvloedt', 'beinvloed','verantwoordelijk', 'oorzaak']] 
    if any(causeverbs): #effect_children = descendants(causeverbs[0])
        scorepoints['cause'] = True
    depnode = [x for x in sent if x.text in solution['dep_syns'] + [solution['dependent']]]
    indynode = [x for x in sent if x.text in solution[syn_key] + [solution[i_key]]]
    scorepoints['ind'] = indynode != []
    scorepoints['dep'] = depnode != []
    if scorepoints['ind'] and scorepoints['dep']:
        scorepoints['alignment'] = check_causality(indynode[0],depnode[0])
    
    #Add strings
    if not scorepoints['cause']:
        output.append(' -er wordt niet gesproken over de oorzaak van het effect')
    if not scorepoints['neg']:
        output.append(' -ten onrechte een negatie toegevoegd of weggelaten bij de primaire verklaring')
    if not scorepoints['ind']:
        output.append(' -de onafhankelijke variabele wordt niet genoemd bij de primaire verklaring')
    if not scorepoints['dep']:
        output.append(' -de afhankelijke variabele wordt niet genoemd bij de primaire verklaring')
    if scorepoints['ind'] and scorepoints['dep'] and not scorepoints['alignment']:
        output.append(' -het causale verband tussen de afhankelijke en onafhankelijke variabele is niet juist aangegeven bij de primaire verklaring')
    if not scorepoints['prim']:
        output.append(' -de primaire verlaring wordt niet genoemd')
    return output

def detect_primary_interaction(sent:Doc, solution:dict) -> List[str]:
    criteria:list = ['interaction', 'negation', 'indy1', 'indy2', 'dep', 'level_present', 'both_levels', 'same']
    scorepoints = dict([(x,False) for x in criteria])
    tokens = [x.text for x in sent]
    output:list = []
    var1levels:list[bool] = [solution['levels'][i].lower() in tokens or any([y in tokens for y in solution['level_syns'][i]]) for i in range(len(solution['levels']))]
    var2levels:list[bool] = [solution['levels2'][i].lower() in tokens or any([y in tokens for y in solution['level2_syns'][i]]) for i in range(len(solution['levels2']))]
    rejected = solution['p'][2] < 0.05
    
    # Fill scorepoints
    dep_node = [x for x in sent if x.text in solution['dep_syns'] + [solution['dependent']]]
    indy1node = [x for x in sent if x.text in solution['ind_syns'] + [solution['independent']]]
    indy2node = [x for x in sent if x.text in solution['ind2_syns'] + [solution['independent2']]]
    scorepoints['indy1'] = indy1node != []
    scorepoints['indy2'] = indy2node != []
    scorepoints['dep'] = dep_node != []
    scorepoints['same'] = 'dezelfde' in tokens or 'gelijk' in tokens or 'gelijke' in tokens or 'hetzelfde' in tokens
    scorepoints['negation'] = bool(negation_counter(tokens) % 2) == rejected
    if scorepoints['dep']:
        if check_causality(indy1node[0], dep_node[0]) and any(var2levels):
            scorepoints['interaction'] = True
            scorepoints['level_present'] = any(var2levels)
            scorepoints['both_levels'] = all(var2levels)
        elif check_causality(indy2node[0], dep_node[0]) and any(var1levels):
            scorepoints['interaction'] = True
            scorepoints['level_present'] = any(var1levels) or scorepoints['level_present']
            scorepoints['both_levels'] = all(var1levels) or scorepoints['both_levels']
                
    #Add strings
    if not scorepoints['negation']:
        output.append(' -ten onrechte een negatie toegevoegd of weggelaten bij de primaire verklaring')
    if not scorepoints['indy1']:
        output.append(' -de eerste onafhankelijke variabele wordt niet genoemd bij de primaire verklaring')
    if not scorepoints['indy2']:
        output.append(' -de tweede onafhankelijke variabele wordt niet genoemd bij de primaire verklaring')
    if not scorepoints['dep']:
        output.append(' -de afhankelijke variabele wordt niet genoemd bij de primaire verklaring')
    if not scorepoints['same']:
        output.append(' -niet gesteld of de invloed van een van de factoren op de afhanklijke variabele hetzelfde is bij beide niveaus van de andere factor')
    if scorepoints['dep'] and not scorepoints['interaction']:
        output.append(' -het causale verband tussen de afhankelijke en onafhankelijke variabele is niet juist aangegeven bij de primaire verklaring')
    elif scorepoints['interaction'] and not scorepoints['both_levels']:
        output.append(' -beide niveaus van een van de onafhankelijke variabelen worden nog niet genoemd')
    elif scorepoints['interaction'] and not scorepoints['level_present']:
        output.append(' -de niveaus van een van de onafhankelijke variabelen worden nog niet genoemd')
    return output

def detect_alternative(sent:Doc, solution:dict, num:int=1) -> List[str]:
    #Define variables
    criteria = ['alt','ind','dep','cause','relation_type']
    scorepoints = dict([(x,False) for x in criteria])
    i_key: str = 'independent' + str(num) if num > 1 else 'independent'
    syn_key: str = 'ind_syns' if num == 1 else 'ind' + str(num) + '_syns'
    control: bool = solution['control'] if num < 2 else solution['control'+str(num)]
    #rejected: bool = solution['p'][num-1] < 0.05
    output:List[str] = []
    
    #Controleer input
    scorepoints['alt'] = 'alternatieve' in [x.text for x in sent] if not control else True
    causeverbs = [x for x in sent if x.text in ['veroorzaakt', 'heeft', 'beinvloedt', 'beinvloed','verantwoordelijk', 'oorzaak', 'invloed']] 
    if any(causeverbs):
        scorepoints['cause'] = True
    depnode = [x for x in sent if x.text in solution['dep_syns'] + [solution['dependent']]]
    indynode = [x for x in sent if x.text in solution[syn_key] + [solution[i_key]]]
    scorepoints['ind'] = indynode != []
    scorepoints['dep'] = depnode != []
    if scorepoints['ind'] and scorepoints['dep']:        
        scorepoints['relation_type'] = check_causality(indynode[0], depnode[0], alternative=True)
    else:
        scorepoints['relation_type'] = True
    
    #Add strings
    if not scorepoints['cause']:
        output.append(' -niet duidelijk genoemd hoe het effect wordt veroorzaakt bij de alternatieve verklaring')
    if not scorepoints['ind']:
        output.append(' -de onafhankelijke variabele wordt niet genoemd bij de alternatieve verklaring')
    if not scorepoints['dep']:
        output.append(' -de afhankelijke variabele wordt niet genoemd bij de alternatieve verklaring')
    if not scorepoints['alt']:
        output.append(' -de alternatieve verlaring wordt niet genoemd')
    if not scorepoints['relation_type']:
        output.append(' -de relatie tussen de onafhankelijke en afhankelijke variabele is niet geldig hier omdat dit een alternatieve verklaring is')
    return output

def detect_alternative_interaction(sent:Doc, solution:dict) -> List[str]:
    output:list = []
    tokens = [x.text for x in sent]
    if not ('storende' in tokens and 'variabelen' in tokens) or ('storende' in tokens and 'variabele' in tokens):
        output.append(' -niet gesteld dat storende variabelen een mogelijkheid zijn voor de alternatieve verklaring')
    if not ('omgekeerde' in tokens and 'causaliteit' in tokens):
        output.append(' -niet gesteld dat ongekeerde causaliteit een mogelijkheid is voor de alternatieve verklaring')
    return output

def detect_unk(sent:Doc, solution:dict, num:int=1):
    #Define variables
    criteria:list=['two']#['unk','two']
    scorepoints = dict([(x,False) for x in criteria])
    control = solution['control'] if num < 2 else solution['control' + str(num)] if num < 3 else solution['control2'] or solution['control2']
    tokens = [x.text for x in sent]
    output:List[str] = []
    
    #Controleer input
    #scorepoints['unk'] = 'onbekend' in tokens if not control else True
    scorepoints['two'] = ('een' in tokens) or ('één' in tokens) or ('1' in tokens) if control else \
                    ('twee' in tokens) or ('meerdere' in tokens) or ('2' in tokens)
    
    #Add strings
    if not scorepoints['two']:
        output.append(' -niet juist genoemd hoeveel mogelijke interpretaties er zijn')
    #if not scorepoints['unk']:
    #    output.append(' -niet gesteld dat de oorzaak van het effect onbekend is')
    return output

def scan_interpretation(doc:Doc, solution:dict, anova:bool, num:int=1, prefix=True):
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:'] if prefix else []
    control: bool = solution['control'] if num < 2 else solution['control'+str(num)]
    primary_checks:list = ['primaire','eerste'] if not control else [solution['dependent']]
    unk_sents = [x for x in doc.sents if any([y in [z.text for z in x] for y in ['mogelijk','mogelijke','verklaring','verklaringen']])]
    if unk_sents != []:
        output.extend(detect_unk(unk_sents[0], solution, num))
    else:
        output.append(' -niet genoemd hoeveel mogelijke verklaringen er zijn')
    primair_sents = [x for x in doc.sents if any([z in [y.text for y in x] for z in primary_checks])]
    if primair_sents != []:
        output.extend(detect_primary(primair_sents[0], solution, num))
    else:
        output.append(' -de primaire verklaring wordt niet genoemd')
    if not control:
        alt_sents = [x for x in doc.sents if 'alternatieve' in [y.text for y in x]]
        #displacy.serve(alt_sents[0])
        if alt_sents != []:
            output.extend(detect_alternative(alt_sents[0], solution, num))
        else:
            output.append(' -de alternatieve verklaring wordt niet genoemd')
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return False, 'Mooi, deze interpretatie klopt. ' if prefix else ''
    else:
        return True, '<br>'.join(output)
    
def scan_interpretation_anova(doc:Doc, solution:dict, num:int=3, prefix=True):
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:'] if prefix else []
    control:bool = solution['control'] or solution['control2']
    primary_checks:list = ['primaire','eerste'] if not control else [solution['dependent']]
    unk_sents = [x for x in doc.sents if 'mogelijk' in [y.text for y in x] or 'mogelijke' in [y.text for y in x]]
    if unk_sents != []:
        output.extend(detect_unk(unk_sents[0], solution))
    else:
        output.append(' -niet genoemd hoeveel mogelijke interpretaties er zijn')
    primair_sents = [x for x in doc.sents if any([z in [y.text for y in x] for z in primary_checks])]
    if primair_sents != []:
        output.extend(detect_primary_interaction(primair_sents[0], solution))
    else:
        output.append(' -de primaire verklaring wordt niet genoemd')
    # EXPLICIETE ALTERNATIEVE VERKLARINGEN HOEVEN NIET BIJ INTERACTIE, STATISMogelijke alternatieve verklaringen zijn storende variabelen en omgekeerde causaliteitTIEK VOOR DE PSYCHOLOGIE 3 PAGINA 80
    if not control:
        alt_sents = [x for x in doc.sents if 'alternatieve' in [y.text for y in x]]
        if alt_sents != []:
            output.extend(detect_alternative_interaction(alt_sents[0], solution))
        else:
            output.append(' -de mogelijkheid van alternatieve verklaringen wordt niet genoemd')
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return False, 'Mooi, deze interpretatie klopt. ' if prefix else ''
    else:
        return True, '<br>'.join(output)
    
def check_causality(independent:Doc, dependent:Doc, alternative:bool=False) -> bool:
    #print(independent.dep_ + '-' + dependent.dep_)
    if not alternative:
        tuples = [('nsubj', 'obj'),('obj', 'ROOT'),('nsubj', 'nmod'),('obl', 'obj'),('ROOT', 'obj'),
              ('obj', 'nmod'), ('amod', 'obj'), ('obl','obl'),('nsubj','obl'),('obj','obj'),('nsubj','amod'),
              ('obj','obl'),('nmod','obj'),('obl','ROOT'),('obl','nsubj'),('obl','csubj'),('advmod','obj'),
              ('advmod','nmod'),('advmod','obj')]
    else: #Add reverse causality and disturbing variable options
        tuples = [('obj','obj'),('obj','nsubj'), ('ROOT','obj'),('nmod','nsubj'),('obj','obl'),('obj','ROOT'),('amod','nsubj'),
                       ('nmod','obj'),('obj','amod'),('obl','nsubj'),('obl','obj'),('obj','nmod'),('ROOT','obl'),('nsubj','obl'),
                       ('obl','obl'), ('csubj','obl')]
    for t in tuples:
        if independent.dep_ == t[0] and dependent.dep_ == t[1]:
            return True
    return False

txt = 'Experiment, dus er is een verklaring mogelijk. Dit is dat bloeddruk wordt niet veroorzaakt door weer.'
nlp = spacy.load('nl')
doc = nlp(txt)
print([(x.text, x.dep_) for x in doc])

#solution = {'control':True,'control2':True,'independent':'weer','independent2':'stimulusvorm','dependent':'gewicht','level_syns':[[],[]],
#      'level2_syns':[[],[]],'ind_syns':[],'ind2_syns':[],'dep_syns':[],'levels':['zon','regen'],'levels2':['vierkant','rond'],'p':[1,0,0,1,1,1]}
#print(scan_interpretation_anova(doc,solution)[1].replace('<br>','\n'))

"""
def detect_eq(text:str):
    nlp = spacy.load('nl')
    doc = nlp(text)
    merge_noun_chunks(doc)
    merge_entities(doc)
    for token in doc:
        # We have an attribute and direct object, so check for subject
        if token.dep_ in ("attr", "dobj"):
            subj = [w for w in token.head.lefts if w.dep_ == "nsubj"]
            if subj:
                print(subj[0], "-->", token)
        if token.dep_ in ("amod"):
            subj = [w for w in token.lefts if w.dep_ == "obl"]
            if subj:
                print(subj[0], "-->", token)
        # We have a prepositional object with a preposition
        elif token.dep_ == "pobj" and token.head.dep_ == "prep":
            print(token.head.head, "-->", token)
    

#detect_eq('De winst was meer dan $10.')
detect_eq('Het design heeft woningnood als onafhankelijke variabele en depressie als afhankelijke variabele.')

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
print('Gewogen afstand: ' + str(associate(sent, 3)))"""