#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 23:02:17 2020

@author: jelmer
"""
import math
import random
import numpy as np
import nltk
import spacy
from spacy.tokens.token import Token
from spacy import displacy
from nltk import CFG, Tree
from scipy import stats
from typing import Dict, List, Tuple



def descendants(node) -> List[Token]:
    output:list = []
    for child in node.children:
        output.append(child)
        output += descendants(child)
    return output

def scan_decision_new(text: str, assignment: Dict, solution: Dict, anova: bool=False, num: int=1) -> [bool, str]:
    #SCAN DECISION
    ## H0
    ## bools=rejected, hyp_present
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    criteria = ['hyp_rejected', 'hyp_present', 'right_comparison', 'right_negation',
                'mean_present', 'pop_present', 'level_present', 'both_present',
                'effect_present', 'strength_present', 'right_strength','p_present',
                'p_comparison']
    scorepoints = dict([(name, False) for name in criteria])
    hyp_statement = [x for x in doc if x.text == 'verworpen' or 
                   x.text =='behouden' or x.text == 'verwerpen']
    if any(hyp_statement):
        scorepoints['hyp_rejected'] = not hyp_statement[0].text == 'behouden'
        children = hyp_statement[0].children
        scorepoints['hyp_present'] = any([x for x in children if x.text == 'H0' or x.text == 'nulhypothese'])
        
    ## COMPARING MEANS - 
    ## bools=right_comparison, right_negation, mean_present, pop_present, level_present, both_present
    gold_comp = 'ongelijk' if anova else ['ongelijk','groter','kleiner'][assignment['hypothesis']]
    comparisons = [x for x in doc if x.text == 'groter' or 
                   x.text =='gelijk' or x.text == 'ongelijk' or x.text == 'kleiner']
    if comparisons != []:
        comptree:List = descendants(comparisons[0])
        
        not_present = 'niet' in [x.text for x in comptree]
        scorepoints['right_comparison'] = comparisons[0].text == gold_comp
        if gold_comp == 'ongelijk' and comparisons[0].text == 'ongelijk':
            scorepoints['right_negation'] = not_present != solution['p'][0] < 0.05
        elif gold_comp == 'ongelijk' and comparisons[0].text == 'gelijk':
            scorepoints['right_negation'] = not_present == solution['p'][0] < 0.05
        else:
            scorepoints['right_negation'] = not_present != solution['p'][0] < 0.05
        levels:List = ['Nederlands', 'Duits'] #dummy values
        scorepoints['level_present'] = any([x in comptree for x in levels])
        scorepoints['both_present'] = all([x in comptree for x in levels])
        
        mean = [x for x in comparisons[0].children if x.text == 'gemiddelde']
        mean_2 = [x for x in doc if x.text == 'populatiegemiddelde']
        scorepoints['mean_present'] = any(mean) or any(mean_2)
        if scorepoints['mean_present']:
            meanroot = mean[0] if any(mean) else mean_2[0] if any(mean_2) else None
            meantree:list = descendants(meanroot)
            scorepoints['pop_present'] = any(mean_2) or 'populatie' in [x.text for x in meantree]
    
    ## STRENGTH
    ## bools=effect_present, strength_present, right_strength, 
    gold_strength = 'sterk'
    to_be = [x for x in doc if x.lemma_ in ['is','was'] and 'effect' in [y.text for y in descendants(x)]]
    if to_be != []:
        be_tree = descendants(to_be[0])
        scorepoints['effect_present'] = True
        scorepoints['strength_present'] = any([x in [y.text for y in be_tree] for x in ['klein','matig','sterk']])
        scorepoints['right_strength'] = gold_strength in [x.text for x in be_tree]
        
    ## P-VALUE
    ## bools=p_present, p_comparison
    gold_comp_p = 'groter'
    comparisons_p = [x for x in doc if (x.text == 'groter' or x.text == 'kleiner') and 
                     ('p' in [y.text for y in descendants(x)] or 'p-waarde' in [y.text for y in descendants(x)])]
    if comparisons != []:
        scorepoints['p_present'] = True
        scorepoints['p_comparison'] = comparisons_p[0].text == gold_comp_p
        
    if False in [all(x) for x in list(scorepoints.values())]:
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
        if not scorepoints['hyp_rejected']:
            output += ' -ten onrechte gesteld dat de hypothese wordt verworpen als deze wordt behouden of andersom\n'
        if not scorepoints['hyp_present']:
            output += ' -hypothese niet genoemd\n'
        if not scorepoints['right_comparison']:
            output += ' -niveaus in de populatie niet of niet juist met elkaar vergelijken\n'
        if not scorepoints['right_negation']:
            output += ' -ten onrechte een negatie toegevoegd of weggelaten bij het vergelijken van de niveaus\n'
        if not scorepoints['mean_present']:
            output += ' -niet genoemd dat de beslissing om populatiegemiddelden gaat\n'
        if not scorepoints['pop_present']:
            output += ' -niet gesteld dat de beslissing over de populatie gaat\n'
        if not scorepoints['level_present']:
            output += ' -de niveaus van de onafhankelijke variabele worden niet genoemd\n'
        if not scorepoints['both_present'] and scorepoints['level_present']:
            output += ' -enkele niveaus van de onafhankelijke variabele weggelaten\n'
        if not scorepoints['effect_present']:
            output += ' -het effect wordt niet genoemd'
        if scorepoints['effect_present'] and not scorepoints['strength_present']:
            output += ' -de sterkte van het effect wordt niet genoemd\n'
        elif scorepoints['effect_present'] and not scorepoints['right_strength']:
            output += ' -de sterkte van het effect wordt niet juist genoemd\n'
        if not scorepoints['p_present']:
            output += ' -p wordt niet genoemd'
        if not scorepoints['p_comparison']:
            output += ' -p wordt niet juist vergeleken met 0.05'
    else:
        return False, 'Mooi, deze beslissing klopt. '

def scan_interpretation(text: str, solution: Dict, anova: bool=False, num: int=1) -> [bool, str]:
    tokens: List[str] = nltk.word_tokenize(text.lower().replace('.',''))
    #sol_tokens = solution['interpretation'].split()
    i_key: str = 'independent' + str(num) if num > 1 else 'independent'
    independent = solution[i_key]
    control: bool = solution['control']
    criteria = ['cause', 'effect', 'unk', 'difference', 'var', 'dep', 'prim', 'alt']
    scorepoints = dict([(name, False) for name in criteria])
    
    
    if False in list(scorepoints.values()):
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
        if not scorepoints['cause']:
            output += ' -er wordt niet gesproken over de oorzaak van het effect<br>'
        if not scorepoints['effect']:
            output += ' -het effect wordt niet genoemd<br>'
        if not scorepoints['unk']:
            output += ' -niet gesteld dat de oorzaak van het effect onbekend is<br>'
        if not scorepoints['difference']:
            output += ' -er wordt niet genoemd dat er een verschil is tussen de twee niveaus van de onafhankelijke variabele<br>'
        if not scorepoints['var']:
            output += ' -de onafhankelijke variabele wordt niet genoemd<br>'
        if not scorepoints['dep']:
            output += ' -de afhankelijke variabele wordt niet genoemd<br>'
        if not scorepoints['prim']:
            output += ' -de primaire verklaring wordt niet genoemd<br>'
        if not scorepoints['alt']:
            output += ' -de alternatieve verlaring wordt niet genoemd<br>'
        return True, output
    else:
        return False, 'Mooi, deze causale interpretatie klopt. '