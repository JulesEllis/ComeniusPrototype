#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:37:19 2020

@author: jelmer
"""
import math
import random
import numpy as np
import nltk
import spacy
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
    if not scorepoints['hyp_rejected']:
        output.append(' -ten onrechte gesteld dat de hypothese wordt verworpen als deze wordt behouden of andersom')
    if not scorepoints['hyp_present']:
        output.append(' -hypothese niet genoemd')
    return output

def detect_comparison(sent:Doc, solution:dict, anova:bool, num:int) -> List[str]:
    #Define variables
    criteria = ['right_comparison', 'right_negation', 'mean_present', 'pop_present', 'level_present', 'both_present']
    scorepoints = dict([(x,False) for x in criteria])
    output:List[str] = []
    levels=solution['levels' + str(num) if num > 1 else 'levels']
    
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
        scorepoints['level_present'] = any([x in [y.text for y in comptree] for x in levels])
        scorepoints['both_present'] = all([x in [y.text for y in comptree] for x in levels])
        
        
    mean = [x for x in sent if x.text == 'gemiddelde']
    mean_2 = [x for x in sent if x.text == 'populatiegemiddelde']
    scorepoints['mean_present'] = any(mean) or any(mean_2)
    #if scorepoints['mean_present']:
        #meanroot = mean[0] if any(mean) else mean_2[0] if any(mean_2) else None
        #meantree:list = descendants(meanroot)
    scorepoints['pop_present'] = any(mean_2) or 'populatie' in [x.text for x in sent]
    scorepoints['level_present'] = any([x in [y.text for y in sent] for x in levels]) or scorepoints['level_present']
    scorepoints['both_present'] = all([x in [y.text for y in sent] for x in levels]) or scorepoints['both_present']
    
    #Add strings:
    if not scorepoints['right_comparison']:
        output.append(' -niveaus in de populatie niet of niet juist met elkaar vergeleken')
    if not scorepoints['right_negation']:
        output.append(' -ten onrechte een negatie toegevoegd of weggelaten bij het vergelijken van de niveaus')
    if not scorepoints['mean_present']:
        output.append(' -niet genoemd dat de beslissing om populatiegemiddelden gaat')
    if not scorepoints['pop_present']:
        output.append(' -niet gesteld dat de beslissing over de populatie gaat')
    if not scorepoints['level_present']:
        output.append(' -de niveaus van de onafhankelijke variabele worden niet genoemd')
    if not scorepoints['both_present'] and scorepoints['level_present']:
        output.append(' -enkele niveaus van de onafhankelijke variabele weggelaten')
    return output

def detect_interaction(sent:Doc, solution:dict, anova:bool) -> List[str]:
    #Define variables
    criteria = ['interactie','indy1','indy2','pop_present','right_negation']
    scorepoints = dict([(x,False) for x in criteria])
    tokens = [x.text for x in sent]
    rejected = solution['p'][-1] < 0.05
    output:List[str] = []
    
    #Controleer input
    interactie_list = [x for x in doc if x.text == 'interactie']
    if interactie_list != []:
        int_descendants = descendants(interactie_list[0])
        scorepoints['interactie'] = True
        scorepoints['indy1'] = solution['independent'] in [x.text for x in int_descendants]
        scorepoints['indy2'] = solution['independent2'] in [x.text for x in int_descendants]
        scorepoints['pop_present'] = 'populatie' in [x.text for x in int_descendants]
        scorepoints['right_negation'] = bool(negation_counter(tokens) % 2) != rejected
        
    #Add strings
    if not scorepoints['interactie']:
        output.append(' -niet gesteld dat deze beslissing over interactie gaat')
    if not scorepoints['right_negation']:
        output.append('')
    if not scorepoints['pop_present']:
        output.append(' -niet gesteld dat deze beslissing over de populatie gaat')
    if not scorepoints['indy1'] and not scorepoints['indy2']:
        output.append(' -de onafhankelijke variabelen ontbreken')
    elif not scorepoints['indy2'] or not scorepoints['indy2']:
        output.append(' -een van de onafhankelijke variabelen ontbreekt')
    return output

def detect_true_scores(sent:Doc, solution:dict) -> List[str]:
    #Define variables
    criteria:list = ['right_comparison', 'right_negation', 'mean_present', 'pop_present','jacked']
    scorepoints = dict([(x,False) for x in criteria])
    rejected = solution['p'][-1] < 0.05
    output:List[str] = []
    
    #Controleer inpur
    gold_comp = 'ongelijk'
    comparisons = [x for x in doc if x.text == 'groter' or 
                   x.text =='gelijk' or x.text == 'ongelijk' or x.text == 'kleiner']
    if comparisons != []:
        comptree:List = descendants(comparisons[0])
        
        not_present = 'niet' in [x.text for x in comptree]
        scorepoints['right_comparison'] = comparisons[0].text == gold_comp
        if comparisons[0].text == 'ongelijk':
            scorepoints['right_negation'] = not_present != rejected
        elif comparisons[0].text == 'gelijk':
            scorepoints['right_negation'] = not_present == rejected
        scorepoints['jacked'] = 'opgevoerde' in [x.text for x in comptree]
        
        mean = [x for x in comparisons[0].children if x.text == 'gemiddelde']
        mean_2 = [x for x in comparisons[0].children if x.text == 'populatiegemiddelde']
        scorepoints['mean_present'] = any(mean) or any(mean_2)
        if scorepoints['mean_present']:
            meanroot = mean[0] if any(mean) else mean_2[0] if any(mean_2) else None
            meantree:list = descendants(meanroot)
            scorepoints['pop_present'] = any(mean_2) or 'populatie' in [x.text for x in meantree]
            
    #Add strings
    if not scorepoints['right_comparison']:
        output += ' -niveaus in de populatie niet of niet juist met elkaar vergeleken\n'
    if not scorepoints['right_negation']:
        output += ' -ten onrechte een negatie toegevoegd of weggelaten bij het vergelijken van de niveaus\n'
    if not scorepoints['mean_present']:
        output += ' -niet genoemd dat de beslissing om populatiegemiddelden gaat\n'
    if not scorepoints['pop_present']:
        output += ' -niet gesteld dat de beslissing over de populatie gaat\n'
    if not scorepoints['jacked']:
        output += ' -niet gesteld dat het over de opgevoerde gemiddelden gaat'
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
            sterkte: float = solution['r2'][num]
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
        output.append(' -de effectgrootte wordt niet genoemd')
    if scorepoints['effect_present'] and not scorepoints['strength_present']:
        output.append(' -de sterkte van het effect wordt niet genoemd')
    elif scorepoints['effect_present'] and not scorepoints['right_strength']:
        output.append(' -de sterkte van het effect wordt niet juist genoemd')
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
    scorepoints['two'] = 'twee' in tokens or 'meerdere' in tokens if not control else ('een' in tokens or 'één' in tokens)
    
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
    rejected = solution['p'][num-1]
    tokens = [x.text for x in sent] 
    output:List[str] = []
    
    #Controleer input
    scorepoints['prim'] = 'primaire' in tokens if not control else True
    scorepoints['neg'] = bool(negation_counter(tokens) % 2) != rejected
    causeverbs = [x for x in sent if x.text in ['veroorzaakt', 'heeft', 'beinvloedt', 'beinvloed','verantwoordelijk', 'oorzaak']] 
    if any(causeverbs):
        effect_children = descendants(causeverbs[0])
        print([x.text for x in effect_children])
        scorepoints['cause'] = True
        scorepoints['ind'] = independent in [x.text for x in effect_children]
        scorepoints['dep'] = dependent in [x.text for x in effect_children]
    if scorepoints['ind'] and scorepoints['dep']:
        indynode = sent[tokens.index(independent)]
        depnode = sent[tokens.index(dependent)]
        if indynode.dep_ == 'nsubj' and depnode.dep_ == 'obj': #Omgekeerde causaliteit
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
        output.append(' -het causale verband tussen de afhankelijke en onafhankelijke variabele is niet juist aangegeven')
    if not scorepoints['prim']:
        output.append(' -de primaire verlaring wordt niet genoemd')
    return output

def detect_alternative(sent:Doc, solution:dict, num:int=1) -> List[str]:
    #Define variables
    criteria = ['alt','ind','dep','content','relation_type']
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
    if scorepoints['ind'] and scorepoints['dep']:
        indynode = sent[tokens.index(independent)]
        depnode = sent[tokens.index(dependent)]
        if indynode.dep_ == 'obj' and depnode.dep_ == 'nsubj': #Omgekeerde causaliteit
            scorepoints['relation_type'] = True
        if indynode.dep_ == 'obj' and depnode.dep_ == 'obj': #Storende variabele
            scorepoints['relation_type'] = True
        #cause2 = [x for x in descendants(causeverbs[])]
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

def scan_decision(doc:Doc, solution:dict, anova:bool, num:int=1) -> [bool, List[str]]:
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:']
    output.extend(detect_h0(doc, solution, num))
    output.extend(detect_comparison(doc, solution, anova, num))
    output.extend(detect_strength(doc, solution, anova, num))
    if len(output) == 1:
        return False, 'Mooi, deze beslissing klopt.'
    else:
        return True, '\n'.join(output)
    
def scan_decision_anova(doc:Doc, solution:dict, anova:bool, num:int=1) -> [bool, List[str]]:
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:']
    output.extend(detect_h0(doc, solution, num))
    output.extend(detect_interaction(doc, solution, anova))
    output.extend(detect_strength(doc, solution, anova, num))
    if len(output) == 1:
        return False, 'Mooi, deze beslissing klopt.'
    else:
        return True, '\n'.join(output)
    
def scan_decision_rmanova(doc:Doc, solution:dict, anova:bool, num:int=1) -> [bool, List[str]]:
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:']
    output.extend(detect_h0(doc, solution, num))
    output.extend(detect_true_scores(doc, solution))
    output.extend(detect_strength(doc, solution, anova, num))
    if len(output) == 1:
        return False, 'Mooi, deze beslissing klopt.'
    else:
        return True, '\n'.join(output)
    
def scan_interpretation(doc:Doc, solution:dict, anova:bool, num:int=1):
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:']
    unk_sents = [x for x in doc.sents if 'mogelijk' in [y.text for y in x]]
    if unk_sents != []:
        output.extend(detect_unk(unk_sents[0], solution))
    else:
        output.append(' -niet genoemd hoeveel mogelijke interpretaties er zijn')
    primair_sents = [x for x in doc.sents if 'primaire' in [y.text for y in x]]
    if primair_sents != []:
        output.extend(detect_primary(primair_sents[0], solution))
    else:
        output.append(' -de primaire verklaring wordt niet genoemd')
    if not solution['control']:
        alt_sents = [x for x in doc.sents if 'alternatieve' in [y.text for y in x]]
        #displacy.serve(alt_sents[0])
        if alt_sents != []:
            output.extend(detect_alternative(alt_sents[0], solution))
        else:
            output.append(' -de alternatieve verklaring wordt niet genoemd')
    if len(output) == 1:
        return False, 'Mooi, deze interpretatie klopt.'
    else:
        return True, '\n'.join(output)

nl_nlp = spacy.load('nl')
doc = nl_nlp("Geen experiment, dus de echte oorzaak is onbekend en er zijn meerdere verklaringen mogelijk. "\
             'De primaire verklaring is dat nationaliteit invloed heeft op gewicht. De alternatieve verklaring '\
             'is dat huidskleiur invloed heeft op gewicht en huidskleur ook invloed heeft op nationaliteit'.lower())
feedback = scan_interpretation(doc,
              {'relative_effect':[0.9],'p':[0.01],'levels':['nederlands','duits'],'hypothesis':0,'independent':'nationaliteit','dependent':'gewicht','control':False},
              False)[1]
print(feedback)