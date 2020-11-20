#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FUNCTIONS FOR SCANNING CAUSAL INTERPRETATION AND DECISION

THESE TAKE THE INPUT TEXT IN DOC INSTEAD OF STRING
"""
import math
import random
import numpy as np
import nltk
import spacy
import re
from copy import copy
from spacy.tokens.token import Token
from spacy.tokens.doc import Doc
from spacy import displacy
from nltk import CFG, Tree
from scipy import stats
from typing import Dict, List, Tuple

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

def detect_h0(sent:Doc, solution:dict, num:int=1) -> List[str]:
    #Define variables
    scorepoints = {'hyp_rejected': False,
                   'hyp_present': False
            }
    output:List[str] = []
    rejected = solution['p'][num-1] < 0.05
    behoud_list = ['behoud', 'behouden']
    verwerp_list = ['verwerp', 'verworpen', 'verwerpen']
    
    #Controleer input
    hyp_statement = [x for x in sent if x.text in ['behoud', 'behouden','verwerp','verworpen','verwerpen']]
    if any(hyp_statement):
        scorepoints['hyp_rejected'] = (hyp_statement[0].text in verwerp_list and rejected) or (hyp_statement[0].text in behoud_list and not rejected)
        children = hyp_statement[0].children
        scorepoints['hyp_present'] = any([x for x in children if x.text == 'h0' or x.text == 'nulhypothese'])
    
    #Add strings
    if not scorepoints['hyp_rejected'] and scorepoints['hyp_present']:
        if num < 2:
            output.append(' -ten onrechte gesteld dat de hypothese wordt verworpen als deze wordt behouden of andersom')
        else:
            output.append(' -ten onrechte gesteld dat de hypothese van factor '+str(num)+' wordt verworpen als deze wordt behouden of andersom')
    if not scorepoints['hyp_present']:
        if num < 2:
            output.append(' -hypothese niet genoemd')
        else:
            output.append(' -hypothese van factor '+str(num)+' niet genoemd')
    return output

def detect_comparison(sent:Doc, solution:dict, anova:bool, num:int) -> List[str]:
    #Define variables
    criteria = ['right_comparison', 'right_negation', 'mean_present', 'pop_present', 'level_present', 'both_present']
    scorepoints = dict([(x,False) for x in criteria])
    output:List[str] = []
    tokens = [y.text for y in sent]
    levels=[x.lower() for x in solution['levels' + str(num) if num > 1 else 'levels']]
    
    #Controleer input
    gold_comp = 'ongelijk' if anova else ['ongelijk','groter','kleiner'][solution['hypothesis']]
    comparisons = [x for x in sent if x.text == 'groter' or 
                   x.text =='gelijk' or x.text == 'ongelijk' or x.text == 'kleiner']
    if comparisons != []:
        comptree:List = descendants(comparisons[0])
        not_present = 'niet' in [x.text for x in comptree]
        scorepoints['right_comparison'] = comparisons[0].text == gold_comp or (gold_comp == 'ongelijk' and comparisons[0].text == 'gelijk')
        if gold_comp == 'ongelijk' and comparisons[0].text == 'ongelijk':
            scorepoints['right_negation'] = not_present != (solution['p'][num-1] < 0.05)
        elif gold_comp == 'ongelijk' and comparisons[0].text == 'gelijk':
            scorepoints['right_negation'] = not_present == (solution['p'][num-1] < 0.05)
        else:
            scorepoints['right_negation'] = not_present != (solution['p'][num-1] < 0.05)
        
        
    mean = [x for x in sent if x.text == 'gemiddelde']
    mean_2 = [x for x in sent if x.text == 'populatiegemiddelde']
    scorepoints['mean_present'] = any(mean) or any(mean_2)
    #if scorepoints['mean_present']:
        #meanroot = mean[0] if any(mean) else mean_2[0] if any(mean_2) else None
        #meantree:list = descendants(meanroot)
    scorepoints['pop_present'] = any(mean_2) or 'populatie' in tokens
    level_bools:list[bool] = [x in tokens for x in levels]
    scorepoints['level_present'] = any(level_bools) #or scorepoints['level_present']
    scorepoints['both_present'] = all(level_bools)# or scorepoints['both_present']
    
    #Add strings:
    if not scorepoints['right_comparison']:
        if num < 2:
            output.append(' -niveaus in de populatie niet of niet juist met elkaar vergeleken')
        else:
            output.append(' -niveaus van factor '+str(num)+' in de populatie niet of niet juist met elkaar vergeleken')
    if not scorepoints['right_negation']:
        if num < 2:
            output.append(' -ten onrechte een negatie toegevoegd of weggelaten bij het vergelijken van de niveaus')
        else:
            output.append(' -ten onrechte een negatie toegevoegd of weggelaten bij het vergelijken van de niveaus van factor '+str(num))
    if not scorepoints['mean_present']:
        if num < 2:
            output.append(' -niet genoemd dat de beslissing om populatiegemiddelden gaat')
        else:
            output.append(' -niet genoemd dat de beslissing van factor '+str(num)+'  om populatiegemiddelden gaat')
    if not scorepoints['pop_present']:
        if num < 2:
            output.append(' -niet gesteld dat de beslissing over de populatie gaat')
        else:
            output.append(' -niet gesteld dat de beslissing van factor '+str(num)+' over de populatie gaat')
    if not scorepoints['level_present']:
        if num < 2:
            output.append(' -de niveaus van de onafhankelijke variabele worden niet genoemd')
        else:
            output.append(' -de niveaus van de onafhankelijke variabele van factor '+str(num)+'  worden niet genoemd')
    if not scorepoints['both_present'] and scorepoints['level_present']:
        if num < 2:
            output.append(' -enkele niveaus van de onafhankelijke variabele weggelaten')
        else:
            output.append(' -enkele niveaus van de onafhankelijke variabele van factor '+str(num)+' weggelaten')
    return output

def detect_comparison_mreg(sent:Doc, solution:dict) -> List[str]:
    #Define variables
    criteria = ['bigger', 'neg', 'sign', 'propvar', 'zero_present', 'conj']
    scorepoints = dict([(x,False) for x in criteria])
    output:List[str] = []
    rejected = solution['p'][0] < 0.05
    tokens = [y.text for y in sent]
    
    #Controleer input
    scorepoints['neg'] = bool(negation_counter(tokens) % 2) != rejected
    scorepoints['sign'] = 'significant' in [x.text for x in sent]
    scorepoints['propvar'] = all([y in [x.text for x in sent] for y in ['proportie','verklaarde','variantie']])
    bigger_present = [x for x in sent if x.text == 'groter']
    if bigger_present != []:
        scorepoints['bigger'] = True
        comptree:List = descendants(bigger_present[0])
        zero = [x for x in comptree if x.text == 'nul' or x.text == '0']
        if zero != []:
            scorepoints['zero_present'] = True
            scorepoints['conj'] = zero[0].dep_ == 'obl'
    
    #Add strings:
    if not scorepoints['bigger']:
        output.append(' -niet genoemd of het proportie verklaarde variantie significant groter dan nul is')
    if not scorepoints['neg'] and scorepoints['bigger']:
        output.append(' -ten onrechte een negatie toegevoegd of weggelaten bij de beslissing over het proportie verklaarde variantie')
    if not scorepoints['sign'] and scorepoints['bigger']:
        output.append(' -niet genoemd of het verschil van het proportie verklaarde variantie met nul significant is')
    if not scorepoints['propvar'] and scorepoints['bigger']:
        output.append(' -het proportie verklaarde variantie word niet genoemd')
    if not scorepoints['zero_present'] and scorepoints['bigger']:
        output.append(' -nul niet genoemd bij de vergelijking van het proportie verklaarde variantie')
    if not scorepoints['conj'] and scorepoints['bigger']: #TODO FIX CONJ
        output.append(' -het proportie verklaarde variantie word niet juist vergeleken met nul')    
    return output

def detect_interaction(sent:Doc, solution:dict, anova:bool) -> List[str]:
    #Define variables
    criteria = ['interactie','indy1','indy2','pop_present','right_negation']
    scorepoints = dict([(x,False) for x in criteria])
    tokens = [x.text for x in sent]
    rejected = solution['p'][-1] < 0.05
    output:List[str] = []
    
    #Controleer input
    interactie_list = [x for x in sent if x.text == 'interactie']
    if interactie_list != []:
        int_descendants = descendants(interactie_list[0])
        scorepoints['interactie'] = True
        scorepoints['indy1'] = solution['independent'].lower() in [x.text for x in int_descendants]
        scorepoints['indy2'] = solution['independent2'].lower() in [x.text for x in int_descendants]
        scorepoints['pop_present'] = 'populatie' in [x.text for x in int_descendants]
        scorepoints['right_negation'] = bool(negation_counter(tokens) % 2) != rejected
        
    #Add strings
    if not scorepoints['interactie']:
        output.append(' -niet gesteld dat de interactiebeslissing over interactie gaat')
    if not scorepoints['right_negation']:
        output.append(' -ten onrechte een negatie toegevoegd of weggelaten bij de interactiebeslissing')
    if not scorepoints['pop_present']:
        output.append(' -niet gesteld dat de interactiebeslissing over de populatie gaat')
    if not scorepoints['indy1'] and not scorepoints['indy2']:
        output.append(' -de onafhankelijke variabelen ontbreken bij de interactiebeslissing')
    elif not scorepoints['indy2'] or not scorepoints['indy2']:
        output.append(' -een van de onafhankelijke variabelen ontbreekt bij de interactiebeslissing')
    return output

def detect_true_scores(sent:Doc, solution:dict) -> List[str]:
    #Define variables
    criteria:list = ['right_comparison', 'right_negation', 'mean_present', 'pop_present','jacked']
    scorepoints = dict([(x,False) for x in criteria])
    rejected = solution['p'][-1] < 0.05
    output:List[str] = []
    
    #Controleer inpur
    comparisons = [x for x in sent if x.text =='gelijk' or x.text == 'ongelijk']
    if comparisons != []:
        comptree:List = descendants(comparisons[0])
        not_present = 'niet' in [x.text for x in comptree]
        scorepoints['right_comparison'] = comparisons[0].text in ['gelijk','ongelijk']
        if comparisons[0].text == 'ongelijk':
            scorepoints['right_negation'] = not_present != rejected
        elif comparisons[0].text == 'gelijk':
            scorepoints['right_negation'] = not_present == rejected
    scorepoints['jacked'] = 'opgevoerde' in [x.text for x in sent]
        
    mean = [x for x in sent if x.text == 'gemiddelde']
    mean_2 = [x for x in sent if x.text == 'populatiegemiddelde']
    scorepoints['mean_present'] = any(mean) or any(mean_2)
    scorepoints['pop_present'] = any(mean_2) or 'populatie' in [x.text for x in sent]
            
    #Add strings
    if not scorepoints['right_comparison']:
        output.append(' -niveaus in de populatie niet of niet juist met elkaar vergeleken')
    if not scorepoints['right_negation']:
        output.append(' -ten onrechte een negatie toegevoegd of weggelaten bij het vergelijken van de niveaus')
    if not scorepoints['mean_present']:
        output.append(' -niet genoemd dat de beslissing om gemiddelden gaat')
    if not scorepoints['pop_present']:
        output.append(' -niet gesteld dat de beslissing over de populatie gaat')
    if not scorepoints['jacked']:
        output.append(' -niet gesteld dat het over de opgevoerde gemiddelden gaat')
    return output
    
def detect_strength(sent:Doc, solution:dict, anova:bool, num:int) -> List[str]:
    #Define variables
    criteria:list = ['effect_present', 'strength_present', 'right_strength']
    scorepoints = dict([(x,False) for x in criteria])
    output:List[str] = []
    
    #Controleer input
    if solution['assignment_type'] in [1,2]: #T-test
        sterkte = solution['relative_effect'][0]
        gold_strength: str = 'sterk' if sterkte > 0.8 else 'matig' if sterkte > 0.5 else 'klein'
    else:#One-way ANOVA
        if solution['assignment_type'] == 3: #One-way ANOVA
            sterkte: float = solution['r2'][0]
        else: #Two-way and within-subject ANOVA
            sterkte: float = solution['r2'][num-1]
        gold_strength: str = 'sterk' if sterkte > 0.2 else 'matig' if sterkte > 0.1 else 'klein'
    effect = [x for x in sent if x.lemma_ == 'effect']
    if effect != []:
        e_tree = descendants(effect[0])
        scorepoints['effect_present'] = True
        scorepoints['strength_present'] = any([x in [y.text for y in e_tree] for x in ['klein','matig','sterk']])
        scorepoints['right_strength'] = gold_strength in [x.text for x in e_tree]
    if solution['p'][num - 1] > 0.05 or math.isnan(solution['p'][num - 1]):
        for x in ['effect_present','strength_present','right_strength']:
            scorepoints[x] = True
            
    #Add strings
    if not scorepoints['effect_present']:
        if num < 2:
            output.append(' -de effectgrootte wordt niet genoemd')
        else:
            output.append(' -de effectgrootte van factor '+str(num)+' wordt niet genoemd')
    if scorepoints['effect_present'] and not scorepoints['strength_present']:
        if num < 2:
             output.append(' -de sterkte van het effect wordt niet genoemd')
        else:
             output.append(' -de sterkte van het effect van factor '+str(num)+' wordt niet genoemd')
    elif scorepoints['effect_present'] and not scorepoints['right_strength']:
        if num < 2:
            output.append(' -de sterkte van het effect wordt niet juist genoemd')
        else:
            output.append(' -de sterkte van het effect van factor '+str(num)+' wordt niet juist genoemd')
    return output

def detect_unk(sent:Doc, solution:dict):
    #Define variables
    criteria:list=['unk','two']
    scorepoints = dict([(x,False) for x in criteria])
    control = solution['control']
    tokens = [x.text for x in sent]
    output:List[str] = []
    
    #Controleer input
    scorepoints['unk'] = 'onbekend' in tokens if not control else True
    scorepoints['two'] = ('een' in tokens) or ('één' in tokens) if control else ('twee' in tokens) or ('meerdere' in tokens)
    
    #Add strings
    if not scorepoints['two']:
        output.append(' -niet juist genoemd hoeveel mogelijke interpretaties er zijn')
    if not scorepoints['unk']:
        output.append(' -niet gesteld dat de oorzaak van het effect onbekend is')
    return output
    
def detect_primary(sent:Doc, solution:dict, num:int=1) -> List[str]:
    #Define variables
    criteria:list = ['cause', 'ind', 'dep', 'prim', 'neg', 'alignment']
    scorepoints = dict([(x,False) for x in criteria])
    i_key: str = 'independent' + str(num) if num > 1 else 'independent'
    independent = solution[i_key].lower()
    dependent = solution['dependent'].lower()
    control: bool = solution['control']
    rejected = solution['p'][num-1] < 0.05
    tokens = [x.text for x in sent] 
    output:List[str] = []
    
    #Controleer input
    scorepoints['prim'] = 'primaire' in tokens if not control else True
    scorepoints['neg'] = bool(negation_counter(tokens) % 2) != rejected
    causeverbs = [x for x in sent if x.text in ['veroorzaakt', 'heeft', 'beinvloedt', 'beinvloed','verantwoordelijk', 'oorzaak']] 
    if any(causeverbs):
        #effect_children = descendants(causeverbs[0])
        scorepoints['cause'] = True
    scorepoints['dep'] = dependent in [x.text for x in sent]
    scorepoints['ind'] = independent in [x.text for x in sent]
    if scorepoints['ind'] and scorepoints['dep']:
        indynode = sent[tokens.index(independent.lower())]
        depnode = sent[tokens.index(dependent.lower())]
        if indynode.dep_ == 'nsubj' and depnode.dep_ == 'obj': #Normale causaliteit
            scorepoints['alignment'] = True
        if indynode.dep_ == 'obj' and depnode.dep_ == 'ROOT': #Consistent verkeerde parse in spacy
            scorepoints['alignment'] = True
        if indynode.dep_ == 'nsubj' and depnode.dep_ == 'nmod': #Consistent verkeerde parse in spacy
            scorepoints['alignment'] = True
        if indynode.dep_ == 'obl' and depnode.dep_ == 'obj': #Consistent verkeerde parse in spacy
            scorepoints['alignment'] = True
        if indynode.dep_ == 'ROOT' and depnode.dep_ == 'obj': #Consistent verkeerde parse in spacy
            scorepoints['alignment'] = True
        if indynode.dep_ == 'amod' and depnode.dep_ == 'obj': #Consistent verkeerde parse in spacy
            scorepoints['relation_type'] = True
        if indynode.dep_ == 'obj' and depnode.dep_ == 'nmod': #Consistent verkeerde parse in spacy
            scorepoints['alignment'] = True
    
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
    var1levels:list[bool] = [x in tokens for x in [x.lower() for x in solution['levels']]]
    var2levels:list[bool] = [x in tokens for x in [x.lower() for x in solution['levels2']]]
    rejected = solution['p'][2] < 0.05
    
    # Fill scorepoints
    scorepoints['dep'] = solution['dependent'].lower() in tokens
    scorepoints['indy1'] = solution['independent'].lower() in tokens
    scorepoints['indy2'] = solution['independent2'].lower() in tokens
    scorepoints['same'] = 'dezelfde' in tokens or 'gelijk' in tokens or 'gelijke' in tokens
    scorepoints['negation'] = bool(negation_counter(tokens) % 2) == rejected
    if scorepoints['dep']:
        dep_node = sent[tokens.index(solution['dependent'].lower())]    
        if scorepoints['indy1']:
            indy1node = sent[tokens.index(solution['independent'].lower())]
            if (indy1node.dep_ == 'nsubj' and dep_node.dep_ == 'obj') or (indy1node.dep_ == 'obj' and dep_node.dep_ == 'ROOT') or (indy1node.dep_ == 'nsubj' and dep_node.dep_ == 'nmod'):
                scorepoints['interaction'] = True
                scorepoints['level_present'] = any(var2levels)
                scorepoints['both_levels'] = all(var2levels)
        if scorepoints['indy2']:
            indy2node = sent[tokens.index(solution['independent2'].lower())]
            if (indy2node.dep_ == 'nsubj' and dep_node.dep_ == 'obj') or (indy2node.dep_ == 'obj' and dep_node.dep_ == 'ROOT') or (indy1node.dep_ == 'nsubj' and dep_node.dep_ == 'nmod'):
                scorepoints['interaction'] = True
                scorepoints['level_present'] = any(var1levels)
                scorepoints['both_levels'] = all(var1levels)
        
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
    if scorepoints['interaction'] and not scorepoints['level_present']:
        output.append(' -de niveaus van een van de onafhankelijke variabelen worden nog niet genoemd')
    if scorepoints['interaction'] and not scorepoints['both_levels']:
        output.append(' -beide niveaus van een van de onafhankelijke variabelen worden nog niet genoemd')
    return output

def detect_alternative(sent:Doc, solution:dict, num:int=1) -> List[str]:
    #Define variables
    criteria = ['alt','ind','dep','cause','relation_type']
    scorepoints = dict([(x,False) for x in criteria])
    i_key: str = 'independent' + str(num) if num > 1 else 'independent'
    independent = solution[i_key].lower()
    dependent = solution['dependent'].lower()
    control: bool = solution['control']
    #rejected: bool = solution['p'][num-1] < 0.05
    tokens = [x.text for x in sent]
    output:List[str] = []
    
    #Controleer input
    scorepoints['alt'] = 'alternatieve' in [x.text for x in sent] if not control else True
    causeverbs = [x for x in sent if x.text in ['veroorzaakt', 'heeft', 'beinvloedt', 'beinvloed','verantwoordelijk', 'oorzaak']] 
    if any(causeverbs):
        effect_children = descendants(causeverbs[0])
        tokens2 = [x.text for x in effect_children]
        scorepoints['cause'] = True
        scorepoints['ind'] = independent in tokens2
        scorepoints['dep'] = dependent in tokens2
    scorepoints['dep'] = scorepoints['dep'] or dependent in [x.text for x in sent]
    if scorepoints['ind'] and scorepoints['dep']:
        indynode = sent[tokens.index(independent.lower())]
        depnode = sent[tokens.index(dependent.lower())]
        if indynode.dep_ == 'obj' and depnode.dep_ == 'nsubj': #Omgekeerde causaliteit
            scorepoints['relation_type'] = True
        if indynode.dep_ == 'ROOT' and depnode.dep_ == 'obj': #Consistent verkeerde parse in spacy
            scorepoints['relation_type'] = True
        if indynode.dep_ == 'nmod' and depnode.dep_ == 'nsubj': #Consistent verkeerde parse in spacy
            scorepoints['relation_type'] = True
        if indynode.dep_ == 'obj' and depnode.dep_ == 'obl': #Consistent verkeerde parse in spacy
            scorepoints['relation_type'] = True
        if indynode.dep_ == 'obj' and depnode.dep_ == 'ROOT': #Consistent verkeerde parse in spacy
            scorepoints['relation_type'] = True
        if indynode.dep_ == 'nmod' and depnode.dep_ == 'obj': #Consistent verkeerde parse in spacy
            scorepoints['relation_type'] = True
        if indynode.dep_ == 'obj' and depnode.dep_ == 'amod': #Consistent verkeerde parse in spacy
            scorepoints['relation_type'] = True
        if indynode.dep_ == 'obj' and depnode.dep_ == 'obj': #Storende variabele
            scorepoints['relation_type'] = True
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

def detect_report_stat(doc:Doc, stat:str, value:str, aliases:list=[], num:int=1, margin=0.01) -> List[str]:
    tokens:list[str] = [x.text for x in doc]
    for i in range(len(tokens) - 2):
        t3:str = tokens[i+2]
        if t3.replace('.','').replace('-','').isdigit():
            t1:str = tokens[i]
            t2:str = tokens[i+1]
            if t1 == stat.lower() or t1 in aliases: 
                if t2 in ['==','='] and float(t3) < value + margin and float(t3) > value - margin:
                    return []
    if num < 2:        
        return [' -de juiste waarde van '+stat+' wordt niet genoemd']
    else:
        return [' -de juiste waarde van '+stat+' voor factor '+str(num)+' wordt niet genoemd']

def detect_name(doc:Doc, names:List[str], label:str) -> List[str]:
    tokens = [x.text for x in doc]
    if all([x.lower() in tokens for x in names]):
        return ['']
    else:
        return [' -'+label+' niet genoemd']

def scan_decision(doc:Doc, solution:dict, anova:bool, num:int=1, prefix=True) -> [bool, List[str]]:
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:'] if prefix else []
    output.extend(detect_h0(doc, solution, num))
    output.extend(detect_comparison(doc, solution, anova, num))
    output.extend(detect_strength(doc, solution, anova, num))
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return False, 'Mooi, deze interpretatie klopt. ' if prefix else ''
    else:
        return True, '<br>'.join(output)
    
def scan_decision_anova(doc:Doc, solution:dict, num:int=1, prefix=True) -> [bool, List[str]]:
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:'] if prefix else []
    output.extend(detect_h0(doc, solution, num))
    output.extend(detect_interaction(doc, solution, True))
    #output.extend(detect_strength(doc, solution, True, num))
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return False, 'Mooi, deze interpretatie klopt. ' if prefix else ''
    else:
        return True, '<br>'.join(output)
    
def scan_decision_rmanova(doc:Doc, solution:dict, num:int=1, prefix=True) -> [bool, List[str]]:
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:'] if prefix else []
    output.extend(detect_h0(doc, solution, 2))
    output.extend(detect_true_scores(doc, solution))
    output.extend(detect_strength(doc, solution, True, num))
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return False, 'Mooi, deze interpretatie klopt. ' if prefix else ''
    else:
        return True, '<br>'.join(output)
    
def scan_interpretation(doc:Doc, solution:dict, anova:bool, num:int=1, prefix=True):
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:'] if prefix else []
    unk_sents = [x for x in doc.sents if any([y in [z.text for z in x] for y in ['mogelijk','mogelijke','verklaring','verklaringen']])]
    if unk_sents != []:
        output.extend(detect_unk(unk_sents[0], solution))
    else:
        output.append(' -niet genoemd hoeveel mogelijke verklaringen er zijn')
    primair_sents = [x for x in doc.sents if 'primaire' in [y.text for y in x]]
    if primair_sents != []:
        output.extend(detect_primary(primair_sents[0], solution, num))
    else:
        output.append(' -de primaire verklaring wordt niet genoemd')
    if not solution['control']:
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
    unk_sents = [x for x in doc.sents if 'mogelijk' in [y.text for y in x] or 'mogelijke' in [y.text for y in x]]
    if unk_sents != []:
        output.extend(detect_unk(unk_sents[0], solution))
    else:
        output.append(' -niet genoemd hoeveel mogelijke interpretaties er zijn')
    primair_sents = [x for x in doc.sents if 'primaire' in [y.text for y in x]]
    if primair_sents != []:
        output.extend(detect_primary_interaction(primair_sents[0], solution))
    else:
        output.append(' -de primaire verklaring wordt niet genoemd')
    # EXPLICIETE ALTERNATIEVE VERKLARINGEN HOEVEN NIET BIJ INTERACTIE, STATISMogelijke alternatieve verklaringen zijn storende variabelen en omgekeerde causaliteitTIEK VOOR DE PSYCHOLOGIE 3 PAGINA 80
    if not solution['control']:
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

def scan_predictors(doc:Doc, solution:dict, prefix:bool=True):
    tokens = [x.text for x in doc]
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:'] if prefix else []
    varnames = [x.lower() for x in solution['data']['varnames'][1:]]
    for x in varnames:
        if ' ' in x:
            names = x.split()
            if not all([y in tokens for y in names]):
                output.append(' -predictor ' + x + ' niet genoemd.')
        else:
            if not x in tokens:
                output.append(' -predictor ' + x + ' niet genoemd.')
    for i in range(len(varnames)):
        if solution['predictor_p'][i] < 0.05:
            output.extend(detect_report_stat(doc, 'de p-waarde van '+varnames[i], solution['predictor_p'][i+1]))
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return False, 'Mooi, deze interpretatie klopt. ' if prefix else ''
    else:
        return True, '<br>'.join(output)

#def scan_design(doc:Doc, solution:dict, prefix:bool=True) -> [bool, List[str]]:
#    print(solution['assignment_type'])
#    return False, 'Mooi, deze interpretatie klopt.' if prefix else ''

def split_grade_ttest(text: str, solution:dict, between_subject:bool) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    label2:str = ['between','-','subject'] if between_subject else ['within','-','subject']
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc, label2+['t','-','test'], 'naam van de analyse'))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'T', solution['T'][0]))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'p', solution['p'][0]))
    output += '<br>' + scan_decision(doc, solution, anova=False, prefix=False)[1]
    #if solution['p'][0] < 0.05:
    #output += '<br>' + scan_interpretation(doc, solution, anova=False, prefix=False)[1]
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)
    
    
def split_grade_anova(text: str, solution:dict, two_way:bool) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    label2:str = ['two','-','way'] if two_way else ['one','-','way']
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc, label2 + ['anova'], 'naam van de analyse'))
    if not two_way:
        output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F'][0]))
        output += '<br>'+'<br>'.join(detect_report_stat(doc, 'p', solution['p'][0]))
        output += '<br>'+'<br>'.join(detect_report_stat(doc, 'R<sup>2</sup>', solution['r2'][0], aliases=['r2','r','r^2']))
        output += '<br>' + scan_decision(doc, solution, anova=True, prefix=False)[1]
    else:
        for i in range(3):
            output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F'][0], num=i+1))
            output += '<br>'+'<br>'.join(detect_report_stat(doc, 'p', solution['p'][0], num=i+1))
            output += '<br>'+'<br>'.join(detect_report_stat(doc, 'R<sup>2</sup>', solution['r2'][0], aliases=['r2'], num=i+1))
            if i < 2:
                output += '<br>' + scan_decision(doc, solution, anova=True, num=i+1, prefix=False)[1]
            else:
                output += '<br>' + scan_decision_anova(doc, solution, anova=True, num=i+1, prefix=False)[1]
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)
        
def split_grade_rmanova(text: str, solution:dict) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc, ['repeated-measures', 'anova'], 'naam van de analyse'))
    output += '<br>'+'<br>'.join(detect_comparison_mreg(doc, solution))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F'][0]))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'p', solution['p'][0]))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'R<sup>2</sup>', solution['r2'][0], aliases=['r2']))
    output += '<br>' + scan_decision(doc, solution, anova=True, num=1, prefix=False)[1]
    output += '<br>' + scan_decision_rmanova(doc, solution, anova=True, num=2, prefix=False)[1]
    #if solution['p'][0] < 0.05:
    #output += '<br>' + scan_interpretation(doc, solution, anova=True, prefix=False)[1]
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)
        
def split_grade_mregression(text:str, solution:dict) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc, ['regressieanalyse'], 'naam van de analyse'))
    output += '<br>'+'<br>'.join(detect_comparison_mreg(doc, solution))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F'][0]))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'p', solution['p'][0]))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'R<sup>2</sup>', solution['r2'][0], aliases=['r2']))
    output += '<br>' + scan_predictors(doc, solution, prefix=False)[1]
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)
    
import spacy
def print_dissection(text:str):
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text)
    print([(x.text, x.dep_) for x in doc])
        