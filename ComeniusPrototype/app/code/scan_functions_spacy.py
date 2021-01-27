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

def check_causality(independent:Doc, dependent:Doc, alternative:bool=False) -> bool:
    #print(independent.dep_ + '-' + dependent.dep_)
    if not alternative:
        tuples = [('nsubj', 'obj'),('obj', 'ROOT'),('nsubj', 'nmod'),('obl', 'obj'),('ROOT', 'obj'),
              ('obj', 'nmod'), ('amod', 'obj'), ('obl','obl'),('nsubj','obl'),('obj','obj'),('nsubj','amod'),
              ('obj','obl'),('nmod','obj'),('obl','ROOT'),('obl','nsubj'),('obl','csubj')]
    else: #Add reverse causality and disturbing variable options
        tuples = [('obj','obj'),('obj','nsubj'), ('ROOT','obj'),('nmod','nsubj'),('obj','obl'),('obj','ROOT'),('amod','nsubj'),
                       ('nmod','obj'),('obj','amod'),('obl','nsubj'),('obl','obj'),('obj','nmod'),('ROOT','obl'),('nsubj','obl'),
                       ('obl','obl'), ('csubj','obl')]
    for t in tuples:
        if independent.dep_ == t[0] and dependent.dep_ == t[1]:
            return True
    return False
    
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
        base = hyp_statement[num-1] if len(hyp_statement) >= num else hyp_statement[0]
        scorepoints['hyp_rejected'] = (base.text in verwerp_list and rejected) or (base.text in behoud_list and not rejected)
        scorepoints['hyp_present'] = any([x for x in base.children if x.text == 'h0' or x.text == 'nulhypothese'])
    
    #Add strings
    if not scorepoints['hyp_rejected'] and scorepoints['hyp_present']:
        if num < 2:
            output.append(' -ten onrechte gesteld dat de hypothese wordt verworpen als deze wordt behouden of andersom')
        if num > 2:
            output.append(' -ten onrechte gesteld dat de interactiehypothese wordt verworpen als deze wordt behouden of andersom')
        else:
            output.append(' -ten onrechte gesteld dat de hypothese van factor '+str(num)+' wordt verworpen als deze wordt behouden of andersom')
    if not scorepoints['hyp_present']:
        if num < 2:
            output.append(' -hypothese niet genoemd')
        elif num < 3:
            if solution['assignment_type'] != 5:
                output.append(' -hypothese van factor '+str(num)+' niet genoemd')
            else:
                output.append(' -hypothese van de subjecten niet genoemd')
        else:
            output.append(' -de interactiehypothese wordt niet genoemd')
    return output

def detect_significance(doc:Doc, solution:dict, num:int=1) -> List[str]:
    scorepoints = {'effect': False,
                   'sign': False,
                   'neg': False
                   }
    output:List[str] = []
    rejected:bool = solution['p'][num-1] < 0.05
    h0_output:list = detect_h0(doc, solution, num)
    if h0_output == []:
        return []
    
    difference = [sent for sent in doc.sents if any([y in [x.text for x in sent] for y in ['verschil','effect']]) 
            and not any([y in [x.text for x in sent] for y in ['matig','klein','sterk']])]
    if difference != []:
        d_root = difference[num - 1] if num <= len(difference) else difference[num-1 - (3 - len(difference))]
        scorepoints['effect'] = True
        tokens:List[str] = [x.text for x in d_root]
        scorepoints['sign'] = 'significant' in tokens
        scorepoints['neg'] = bool(negation_counter(tokens) % 2) != rejected
    if num < 2:
        if not scorepoints['effect']:
            output.append(' -niet genoemd of het effect significant is')
        if not scorepoints['sign'] and scorepoints['effect']:
            output.append(' -niet genoemd of het effect significant is')
        if not scorepoints['neg']:
            output.append(' -ten onrechte een negatie toegevoegd of weggelaten bij de beschrijving van het effect')
    else:
        appendix:str = ' bij factor ' + str(num) if num < 3 else ' bij de interactie '
        if not scorepoints['effect']:
            output.append(' -niet genoemd of het effect significant is'+appendix)
        if not scorepoints['sign'] and scorepoints['effect']:
            output.append(' -niet genoemd of het effect significant is'+appendix)
        if not scorepoints['neg']:
            output.append(' -ten onrechte een negatie toegevoegd of weggelaten bij de beschrijving van het effect'+appendix)
    return output

def detect_comparison(sent:Doc, solution:dict, anova:bool, num:int) -> List[str]:
    #Define variables
    criteria = ['right_comparison', 'right_negation', 'mean_present', 'pop_present', 'level_present', 'both_present','contrasign']
    scorepoints = dict([(x,False) for x in criteria])
    output:List[str] = []
    tokens = [y.text for y in sent]
    levels=[x.lower() for x in solution['levels' + str(num) if num > 1 else 'levels']]
    level_syns = solution['level_syns'] if num == 1 else solution['level' + str(num) + '_syns']
    
    #Controleer input
    gold_comp = 'ongelijk' if anova else ['ongelijk','groter','kleiner'][solution['hypothesis']]
    comparisons = [x for x in sent if x.text in ['groter','ongelijk','gelijk','kleiner','anders','verschillend']]
    if comparisons != []:
        comproot = comparisons[num-1] if len(comparisons) >= num else comparisons[0]
        comptree:List = descendants(comproot)
        not_present = 'niet' in [x.text for x in comptree]
        scorepoints['right_comparison'] = comproot.text == gold_comp or (gold_comp == 'ongelijk' and comproot.text in ['gelijk','verschillend','anders'])
        if gold_comp == 'ongelijk' and comproot.text in ['ongelijk','anders','verschillend']:
            scorepoints['right_negation'] = not_present != (solution['p'][num-1] < 0.05)
        elif gold_comp == 'ongelijk' and comproot.text == 'gelijk':
            scorepoints['right_negation'] = not_present == (solution['p'][num-1] < 0.05)
        else:
            scorepoints['right_negation'] = not_present != (solution['p'][num-1] < 0.05)
    else:
        scorepoints['right_negation'] = True
    
    mean = [x for x in sent if x.text == 'gemiddelde' or x.text == 'gemiddelden' or x.text == 'gemiddeld']
    mean_2 = [x for x in sent if x.text == 'populatiegemiddelde' or x.text == 'populatiegemiddelden']
    scorepoints['mean_present'] = any(mean) or any(mean_2)
    scorepoints['pop_present'] = any(mean_2) or 'populatie' in tokens or any([x in tokens for x in ['significant','significante']])
    level_bools:list[bool] = [levels[i] in tokens or any([y in tokens for y in level_syns[i]]) for i in range(2)]#len(levels))]
    scorepoints['level_present'] = any(level_bools) #or scorepoints['level_present']
    scorepoints['both_present'] = all(level_bools)# or scorepoints['both_present']
    scorepoints['contrasign'] = not ((any(mean_2) or 'populatie' in tokens) and any([x in tokens for x in ['significant','significante']]))
    
    #Add strings:
    appendix:str = '' if num < 2 else 'bij factor ' + str(num) if num < 3 else ' bij de interactie '
    if not scorepoints['contrasign']:
        output.append(' -'+appendix+'zowel "populatie" en "significant" genoemd, haal een van de twee weg')
    if not scorepoints['right_comparison']:
        output.append(' -niveaus '+appendix+'in de populatie niet of niet juist met elkaar vergeleken')
    if not scorepoints['right_negation']:
        output.append(' -ten onrechte een negatie toegevoegd of weggelaten bij het vergelijken van de niveaus '+appendix) 
    if not scorepoints['mean_present']:
        output.append(' -niet genoemd dat de beslissing '+appendix+'om populatiegemiddelden gaat')
    if not scorepoints['pop_present']:
        output.append(' -niet gesteld dat de beslissing '+appendix+'over de populatie gaat')
    if not scorepoints['level_present']:
        output.append(' -de niveaus van de onafhankelijke variabele '+appendix+'worden niet genoemd')
    if not scorepoints['both_present'] and scorepoints['level_present']:
        output.append(' -enkele niveaus van de onafhankelijke variabele '+appendix+'weggelaten')
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

def detect_interaction(doc:Doc, solution:dict, anova:bool) -> List[str]:
    #Define variables
    criteria = ['interactie','indy1','indy2','pop_present','right_negation', 'contrasign']
    scorepoints = dict([(x,False) for x in criteria])
    rejected = solution['p'][-1] < 0.05
    tokens = [y.text for y in doc]
    output:List[str] = []
    
    #Controleer input
    i_sents = [sent for sent in doc.sents if any([x in tokens for x in ['interactie','populatie','populatiegemiddelde','populatiegemiddelden']])]
    if i_sents != []:
        int_descendants = i_sents[0]    
        tokens = [x.text for x in int_descendants]
        scorepoints['interactie'] = True
        scorepoints['indy1'] = solution['independent'].lower() in [x.text for x in int_descendants]
        scorepoints['indy2'] = solution['independent2'].lower() in [x.text for x in int_descendants]
        scorepoints['pop_present'] = 'populatie' in [x.text for x in int_descendants] or any([x in tokens for x in ['significant','significante']])
        scorepoints['right_negation'] = bool(negation_counter(tokens) % 2) != rejected    
        scorepoints['contrasign'] = not (('populatie' in tokens) and any([x in tokens for x in ['significant','significante']]))
        
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
    if not scorepoints['contrasign']:
        output.append(' -zowel "populatie" en "significant" genoemd, haal een van de twee weg')
    return output

def detect_decision_ancova(sent:Doc, solution:dict) -> List[str]:
    rejected:bool = solution['p'][-1] < 0.05
    tokens:list = [x.text for x in sent]
    scorepoints:dict = {'sign_val': 'significant voorspellende waarde' in sent.text,
        'indep': solution['independent'] in sent.text or any([x in sent.text for x in solution['ind_syns']]),
        'cov1': solution['predictor_names'][0] in sent.text or any([x in sent.text for x in solution['predictor_syns'][0]]),
        'cov2': solution['predictor_names'][1] in sent.text or any([x in sent.text for x in solution['predictor_syns'][1]]),
        'dep': solution['dependent'] in sent.text or any([x in sent.text for x in solution['dep_syns']]),
        'neg': bool(negation_counter(tokens) % 2) != rejected }
    
    output:List[str] = []
    if not scorepoints['sign_val']:
        output.append(' -significant voorspellende waarde niet genoemd in de hoofdbeslissing')
    if not scorepoints['indep']:
        output.append(' -onafhankelijke factor niet genoemd in de hoofdbeslissing')
    if not scorepoints['cov1'] and not scorepoints['cov2']:
        output.append(' -beide covariaten niet genoemd in de hoofdbeslissing')
    elif not scorepoints['cov1'] or not scorepoints['cov2']:
        output.append(' -een van de covariaten niet genoemd in de hoofdbeslissing')
    if not scorepoints['dep']:
        output.append(' -afhankelijke variabele niet genoemd in de hoofdbeslissing')
    if not scorepoints['neg']:
        output.append(' -ten onrechte een negatie toegevoegd of weggelaten bij de hoofdbeslissing')
    return output

def detect_decision_manova(sent:Doc, solution:dict, variable:str, synonyms:list, p:float, eta:float, num:int) -> List[str]:
    rejected:bool = p < 0.05
    tokens:list = [x.text for x in sent]
    #print(rejected)
    #print(tokens)
    #print(negation_counter(tokens))
    scorepoints:dict = {'sign_effect': 'significant' in sent.text,
        'indep': solution['independent'] in sent.text or any([x in sent.text for x in solution['ind_syns']]),
        'dep': variable in sent.text or any([x in sent.text for x in synonyms]) or 'multivariate' in variable,
        'neg': bool(negation_counter(tokens) % 2) != rejected,
        'effect_present': p > 0.05,
        'strength_present': p > 0.05,
        'right_strength': p > 0.05}
    gold_strength: int = 2 if eta > 0.2 else 1 if eta > 0.1 else 0
    effect = [x for x in sent if x.lemma_ in ['klein','zwak','matig','groot','sterk']]
    if effect != []:
        n_effects:int = len(effect)
        e_root = effect[num-1] if num <= n_effects else effect[num-1 - (3 - n_effects)]
        e_tree = descendants(e_root)
        scorepoints['effect_present'] = e_root.head.text == 'effect' or 'effect' in [x.text for x in e_tree]
        scorepoints['strength_present'] = True #any([x in [y.text for y in e_tree] for x in ['klein','matig','sterk']]) or e_root.head.text in ['klein','matig','sterk']
        scorepoints['right_strength'] = e_root.text in ['sterk','groot'] if gold_strength == 2 else e_root.text in ['matig'] if gold_strength == 1 else e_root.text in ['klein','zwak']
    
    output:List[str] = []
    if not scorepoints['sign_effect']:
        output.append(' -niet genoemd bij de beslissing van '+variable+' of het effect significant is')
    if not scorepoints['indep']:
        output.append(' -onafhankelijke factor niet genoemd bij de beslissing van '+variable)
    if not scorepoints['dep']:
        output.append(' -afhankelijke variabele niet genoemd bij de beslissing van '+variable)
    if not scorepoints['neg']:
        output.append(' -ten onrechte een negatie toegevoegd of weggelaten bij de beslissing van '+variable)
    if not scorepoints['effect_present'] and scorepoints['strength_present']:
        output.append(' -de effectgrootte van '+variable+' wordt niet genoemd')
    if not scorepoints['strength_present']:
        output.append(' -de sterkte van het effect '+variable+' wordt niet genoemd')
    elif scorepoints['effect_present'] and not scorepoints['right_strength']:
        output.append(' -de sterkte van het effect '+variable+' wordt niet juist genoemd')
    return output
    
def detect_true_scores(sent:Doc, solution:dict, num=2) -> List[str]:
    #Define variables
    criteria:list = ['right_comparison', 'right_negation', 'mean_present', 'pop_present','jacked','contrasign']
    scorepoints:dict = dict([(x,False) for x in criteria])
    rejected:bool = solution['p'][-1] < 0.05
    tokens:list = [x.text for x in sent]
    output:List[str] = []
    
    #Controleer input
    comparisons = [x for x in sent if x.text in ['gelijk','ongelijk','anders','verschillend']]
    if comparisons != []:
        comproot = comparisons[num-1] if len(comparisons) >= num else comparisons[0]
        #comptree:List = descendants(comproot)
        not_present = bool(negation_counter(tokens) % 2)
        scorepoints['right_comparison'] = comproot.text in ['gelijk','ongelijk','anders','verschillend']
        if comproot.text in ['ongelijk','anders','verschillend']:
            scorepoints['right_negation'] = not_present != rejected
        elif comproot.text == 'gelijk':
            scorepoints['right_negation'] = not_present == rejected
    else:
        scorepoints['right_negation'] = True
    scorepoints['jacked'] = 'opgevoerde' in [x.text for x in sent]
        
    mean = [x for x in sent if x.text == 'gemiddelde' or x.text == 'gemiddelden' or x.text == 'gemiddeld']
    mean_2 = [x for x in sent if x.text == 'populatiegemiddelde' or x.text == 'populatiegemiddelden']
    scorepoints['mean_present'] = any(mean) or any(mean_2)
    scorepoints['pop_present'] = any(mean_2) or 'populatie' in [x.text for x in sent] or any([x in tokens for x in ['significant','significante']])
    scorepoints['contrasign'] = not ((any(mean_2) or 'populatie' in tokens) and any([x in tokens for x in ['significant','significante']]))
    print(scorepoints)
    
    #Add strings
    if not scorepoints['right_comparison']:
        output.append(' -niveaus in de populatie niet of niet juist met elkaar vergeleken')
    if not scorepoints['right_negation']:
        output.append(' -ten onrechte een negatie toegevoegd of weggelaten bij het vergelijken van de niveaus in de beslissing van de subjecten')
    if not scorepoints['mean_present']:
        output.append(' -niet genoemd dat de beslissing om gemiddelden gaat')
    if not scorepoints['pop_present']:
        output.append(' -niet gesteld dat de beslissing over de populatie gaat')
    if not scorepoints['jacked']:
        output.append(' -niet gesteld dat het over de opgevoerde gemiddelden gaat')
    if not scorepoints['contrasign']:
        output.append(' -zowel "populatie" en "significant" genoemd, haal een van de twee weg')
    return output
    
def detect_strength(sent:Doc, solution:dict, anova:bool, num:int) -> List[str]:
    #Define variables
    criteria:list = ['effect_present', 'strength_present', 'right_strength']
    scorepoints = dict([(x,False) for x in criteria])
    output:List[str] = []
    
    #Controleer input
    if solution['assignment_type'] in [1,2]: #T-test
        sterkte = solution['relative_effect'][0]
        gold_strength: int = 2 if sterkte > 0.8 else 1 if sterkte > 0.5 else 0
    else:#One-way ANOVA
        if solution['assignment_type'] == 3: #One-way ANOVA
            sterkte: float = solution['r2'][0]
        else: #Two-way and within-subject ANOVA
            sterkte: float = solution['r2'][num-1]
        gold_strength: int = 2 if sterkte > 0.2 else 1 if sterkte > 0.1 else 0
    effect = [x for x in sent if x.lemma_ in ['klein','zwak','matig','groot','sterk']]
    if effect != []:
        n_effects:int = len(effect)
        e_root = effect[num-1] if num <= n_effects else effect[num-1 - (3 - n_effects)]
        e_tree = descendants(e_root)
        scorepoints['effect_present'] = e_root.head.text == 'effect' or 'effect' in [x.text for x in e_tree]
        scorepoints['strength_present'] = True #any([x in [y.text for y in e_tree] for x in ['klein','matig','sterk']]) or e_root.head.text in ['klein','matig','sterk']
        scorepoints['right_strength'] = e_root.text in ['sterk','groot'] if gold_strength == 2 else e_root.text in ['matig'] if gold_strength == 1 else e_root.text in ['klein','zwak']
    
    #Add strings
    appendix:str = '' if num < 2 else 'bij factor ' + str(num)+' ' if num == 2 and solution['assignment_type'] != 5 else 'bij de subjecten ' if num < 3 else ' bij de interactie '
    if not scorepoints['effect_present'] and scorepoints['strength_present']:
        output.append(' -de effectgrootte '+appendix+'wordt niet genoemd')
    if not scorepoints['strength_present']:
        output.append(' -de sterkte van het effect '+appendix+'wordt niet genoemd')
    elif scorepoints['effect_present'] and not scorepoints['right_strength']:
        output.append(' -de sterkte van het effect '+appendix+'wordt niet juist genoemd')
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
    var1levels:list[bool] = [solution['levels'][i] in tokens or any([y in tokens for y in solution['level_syns'][i]]) for i in range(len(solution['levels']))]
    var2levels:list[bool] = [solution['levels2'][i] in tokens or any([y in tokens for y in solution['level2_syns'][i]]) for i in range(len(solution['levels2']))]
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
        if scorepoints['indy1']:
            scorepoints['indy2'] = True
            if check_causality(indy1node[0], dep_node[0]):
                scorepoints['interaction'] = True
                scorepoints['level_present'] = any(var2levels)
                scorepoints['both_levels'] = all(var2levels)
        if scorepoints['indy2']:
            scorepoints['indy1'] = True
            if check_causality(indy2node[0], dep_node[0]):
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
    if scorepoints['interaction'] and not scorepoints['both_levels']:
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

def detect_report_stat(doc:Doc, stat:str, value:float, aliases:list=[], num:int=1, margin=0.01, appendix=None) -> List[str]:
    tokens:list[str] = [x.text for x in doc]
    for i in range(len(tokens) - 2):
        t3:str = tokens[i+2]
        if t3.replace('.','').replace('-','').isdigit():
            t1:str = tokens[i]
            t2:str = tokens[i+1]
            if t1 == stat.lower() or t1 in aliases:
                if t2 in ['==','='] and float(t3) < value + margin and float(t3) > value - margin:
                    return []
    if appendix == None:
        appendix:str = '' if num < 2 else 'bij factor ' + str(num) if num < 3 else 'bij de interactie '
    return [' -de juiste waarde van '+stat+' '+appendix+'wordt niet genoemd']

def detect_p(doc:Doc, value:float, num:int=1, label:str=None, margin=0.01) -> List[str]:
    tokens:list[str] = [x.text for x in doc]
    for i in range(len(tokens) - 2):
        t3:str = tokens[i+2]
        if t3.replace('.','').replace('-','').isdigit():
            t1:str = tokens[i]
            t2:str = tokens[i+1]
            if t1 == 'p': 
                if t2 in ['==','='] and float(t3) < value + margin and float(t3) > value - margin:
                    return []
            #if t2 == '<' and t3 == '0.05' and TODO: < 0.05 en > 0.05 ook goedrekenen
    if label != None:
        return [' -de juiste p-waarde van '+label+' wordt niet genoemd']
    else:
        appendix:str = '' if num < 2 else 'bij factor ' + str(num) if num < 3 else ' bij de interactie '
        return [' -de juiste p-waarde '+appendix+'wordt niet genoemd']
    
def detect_name(doc:Doc, solution:Dict) -> List[str]:
    names:list = []
    if solution['assignment_type'] == 1:
        names = [('t','-','toets','voor','onafhankelijke','variabelen'),('between','-','subject','t','-','test'),('t','-','toets','voor','onafhankelijke','subjecten')]
    if solution['assignment_type'] == 2:
        names = [('t','-','toets','voor','gekoppelde','paren'),('within','-','subject','t','-','test')]
    if solution['assignment_type'] == 3:
        names = [('one','-','way','anova'), ('1-factor','anova')]
    if solution['assignment_type'] == 4:
        names = [('two','-','way','anova'), ('2-factor','anova')]
    if solution['assignment_type'] == 5:
        names = [('repeated','-','measures','anova'), ('repeated','-','measures','-','anova')]
    if solution['assignment_type'] == 6:
        names = [('regressieanalyse'),('multiple','-','regression'),('multipele', 'regressie')]
    if solution['assignment_type'] == 11:
        names = [('manova')]
    if solution['assignment_type'] == 12:
        names = [('ancova')]
    if any([all([x in doc.text for x in y]) for y in names]):
        return ['']
    else:
        return [' -naam van de analyse niet juist genoemd']

def scan_decision(doc:Doc, solution:dict, anova:bool, num:int=1, prefix=True, elementair=True) -> [bool, List[str]]:
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:'] if prefix else []
    if elementair:
        output.extend(detect_h0(doc, solution, num))
    else:
        output.extend(detect_significance(doc, solution, num))
    output.extend(detect_comparison(doc, solution, anova, num))
    if not (solution['p'][num - 1] > 0.05 or math.isnan(solution['p'][num - 1])):
        output.extend(detect_strength(doc, solution, anova, num))
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return False, 'Mooi, deze interpretatie klopt. ' if prefix else ''
    else:
        return True, '<br>'.join(output)
    
def scan_decision_anova(doc:Doc, solution:dict, num:int=3, prefix=True, elementair=True) -> [bool, List[str]]:
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:'] if prefix else []
    if elementair:
        output.extend(detect_h0(doc, solution, num))
    else:
        output.extend(detect_significance(doc, solution, num))
    output.extend(detect_interaction(doc, solution, True))
    output.extend(detect_strength(doc, solution, True, num))
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return False, 'Mooi, deze interpretatie klopt. ' if prefix else ''
    else:
        return True, '<br>'.join(output)
    
def scan_decision_rmanova(doc:Doc, solution:dict, num:int=1, prefix=True, elementair=True) -> [bool, List[str]]:
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:'] if prefix else []
    if elementair:
        output.extend(detect_h0(doc, solution, num))
    else:
        output.extend(detect_significance(doc, solution, num))
    output.extend(detect_true_scores(doc, solution, 2))
    if solution['p'][1] < 0.05:
        output.extend(detect_strength(doc, solution, True, num))
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return False, 'Mooi, deze interpretatie klopt. ' if prefix else ''
    else:
        return True, '<br>'.join(output)
    
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

def scan_predictors(doc:Doc, solution:dict, prefix:bool=True):
    tokens = [x.text for x in doc]
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:'] if prefix else []
    varnames = [x.lower() for x in solution['data']['predictoren'][1:]]
    for x in varnames:
        if ' ' in x:
            names = x.split()
            if not all([y in tokens for y in names]):
                output.append(' -predictor ' + x + ' niet genoemd.')
        else:
            if not x in tokens:
                output.append(' -predictor ' + x + ' niet genoemd.')
    for i in range(len(varnames)):
        if solution['predictor_p'][i+1] < 0.05:
            output.extend(detect_p(doc, solution['predictor_p'][i+1], label=varnames[i]))
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return False, 'Mooi, deze interpretatie klopt. ' if prefix else ''
    else:
        return True, '<br>'.join(output)

def scan_design(doc:Doc, solution:dict, prefix:bool=True) -> [bool, List[str]]:
    criteria = ['ind', 'indcorrect','ind2','ind2correct','dep','depcorrect']
    scorepoints = dict([(x,False) for x in criteria])
    output:List[str] = []
    indeps = [x for x in doc if x.text == solution['independent'].lower()]
    if indeps != []:
        scorepoints['ind'] = True
        if indeps[0].dep_ == 'conj':
            indep_span = descendants(indeps[0].head.head)
        else:
            indep_span = descendants(indeps[0].head)
        scorepoints['indcorrect'] = 'onafhankelijke' in [x.text for x in indep_span] and not 'afhankelijke' in [x.text for x in indep_span] 
    if solution['assignment_type'] == 2:    
        indeps2 = [x for x in doc if x.text == solution['independent'].lower()]
        if indeps2 != []:
            scorepoints['ind2'] = True
            if indeps2[0].dep_ == 'conj':
                indep2_span = descendants(indeps2[0].head.head)
            else:
                indep2_span = descendants(indeps2[0].head)
            scorepoints['ind2correct'] = 'onafhankelijke' in [x.text for x in indep2_span] and not 'afhankelijke' in [x.text for x in indep2_span] 
    else:
        scorepoints['ind2'] = True;scorepoints['ind2correct'] = True
    deps = [x for x in doc if x.text == solution['dependent'].lower()]
    if deps != []:
        scorepoints['dep'] = True
        dep_span = descendants(deps[0].head)
        scorepoints['depcorrect'] = 'afhankelijke' in [x.text for x in dep_span] and not 'onafhankelijke' in [x.text for x in dep_span] 
    
    #Add feedback text
    if not scorepoints['ind']:
        output.append(' -de onafhankelijke variabele wordt niet genoemd in het design')
    if not scorepoints['indcorrect'] and scorepoints['ind']:
        output.append(' -de rol van de onafhankelijke variabele wordt niet juist genoemd in het design')
    if not scorepoints['ind2']:
        output.append(' -de tweede onafhankelijke variabele wordt niet genoemd in het design')
    if not scorepoints['ind2correct'] and scorepoints['ind2']:
        output.append(' -de rol van de tweede onafhankelijke variabele wordt niet juist genoemd in het design')
    if not scorepoints['dep']:
        output.append(' -de afhankelijke variabele wordt niet genoemd in het design')
    if not scorepoints['depcorrect'] and scorepoints['dep']:
        output.append(' -de rol van de afhankelijke variabele wordt niet juist genoemd in het design')
    if not False in list(scorepoints.values()):        
        return False, 'Mooi, dit design klopt.' if prefix else ''
    else:
        return True, '<br>'.join(output)
    
def scan_design_manova(doc:Doc, solution:dict, prefix:bool=True):
    text = doc.text
    scorepoints = {'indcorrect':False,
                   #'levels1':all([x in text for x in solution['levels']]),
                   'mes':all([solution[x] in text for x in ['dependent','dependent2','dependent3']]),
                   'dep1':False,
                   'dep2':False,
                   'dep3':False
                   }
    scorepoints['indcorrect'] = any([True if solution['independent'] in sent.text and ('factor' in sent.text \
                                    or 'onafhankelijke' in sent.text) else False for sent in doc.sents])
    deps = [x for x in doc if x.text == solution['dependent'].lower()]
    if deps != []:
        dep_span = descendants(deps[0].head)
        scorepoints['dep1'] = 'afhankelijke' in [x.text for x in dep_span] #and not 'onafhankelijke' in [x.text for x in dep_span]
    deps2 = [x for x in doc if x.text == solution['dependent'].lower()]
    if deps2 != []:
        dep2_span = descendants(deps2[0].head)
        scorepoints['dep2'] = 'afhankelijke' in [x.text for x in dep2_span] #and not 'onafhankelijke' in [x.text for x in dep2_span] 
    deps3 = [x for x in doc if x.text == solution['dependent'].lower()]
    if deps3 != []:
        dep3_span = descendants(deps3[0].head)
        scorepoints['dep3'] = 'afhankelijke' in [x.text for x in dep3_span] #and not 'onafhankelijke' in [x.text for x in dep3_span] 
    
    output:List[str] = []
    if not scorepoints['dep1']:
        output.append(' -eerste afhankelijke variabele niet juist genoemd')
    if not scorepoints['dep2']:
        output.append(' -tweede afhankelijke variabele niet juist genoemd')
    if not scorepoints['dep3']:
        output.append(' -derde afhankelijke variabele niet juist genoemd')
    if not scorepoints['indcorrect']:
        output.append(' -de onafhankelijke variabele wordt niet juist genoemd')
    elif not scorepoints['mes']:
        output.append(' -niet alle aparte afhankelijke variabelen juist genoemd')
    if not False in list(scorepoints.values()):
        return False, 'Mooi, dit design klopt.' if prefix else ''
    else:
        return True, '<br>'.join(output)

def split_grade_ttest(text: str, solution:dict, between_subject:bool) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc,solution))
    output += '<br>' + scan_design(doc, solution, prefix=False)[1]
    if True: #solution['p'][0] < 0.05:
        output += '<br>'+'<br>'.join(detect_report_stat(doc, 'T', solution['T'][0], aliases=['T(' + solution['independent'] + ')']))
        output += '<br>'+'<br>'.join(detect_report_stat(doc, 'p', solution['p'][0]))
    output += '<br>' + scan_decision(doc, solution, anova=False, prefix=False, elementair=False)[1]
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)
    
    
def split_grade_anova(text: str, solution:dict, two_way:bool) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc,solution))
    output += '<br>' + scan_design(doc, solution, prefix=False)[1]
    if not two_way:
        if solution['p'][0] < 0.05:
            output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F'][0], aliases=['F(' + solution['independent'] + ')']))
            output += '<br>'+'<br>'.join(detect_p(doc, solution['p'][0]))
            output += '<br>'+'<br>'.join(detect_report_stat(doc, 'R<sup>2</sup>', solution['r2'][0], aliases=['r2','r','kwadraat']))
        output += '<br>' + scan_decision(doc, solution, anova=True, prefix=False, elementair=False)[1]
    else:
        for i in range(3):
            if solution['p'][i] < 0.05:
                f_aliases = ['F(' + solution['independent' + str(i+1)] + ')'] if i > 0 and i < 2 else []
                output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F'][i], aliases=f_aliases, num=i+1))
                output += '<br>'+'<br>'.join(detect_p(doc, solution['p'][i], num=i+1))
                output += '<br>'+'<br>'.join(detect_report_stat(doc, 'R<sup>2</sup>', solution['r2'][i], aliases=['r2','r','kwadraat'], num=i+1))
            if i < 2:
                output += '<br>' + scan_decision(doc, solution, anova=True, num=i+1, prefix=False, elementair=False)[1]
            else:
                output += '<br>' + scan_decision_anova(doc, solution, num=i+1, prefix=False, elementair=False)[1]
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)
        
def split_grade_rmanova(text: str, solution:dict) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc,solution))
    output += '<br>' + scan_design(doc, solution, prefix=False)[1]
    if solution['p'][0] < 0.05:
        output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F'][0], aliases=['F(' + solution['independent'] + ')']))
        output += '<br>'+'<br>'.join(detect_p(doc, solution['p'][0]))
        output += '<br>'+'<br>'.join(detect_report_stat(doc, 'R<sup>2</sup>', solution['r2'][0], aliases=['r2','r','kwadraat']))
    output += '<br>' + scan_decision(doc, solution, anova=True, num=1, prefix=False, elementair=False)[1]
    output += '<br>' + scan_decision_rmanova(doc, solution, num=2, prefix=False, elementair=False)[1]
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)
        
def split_grade_mregression(text:str, solution:dict) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc,solution))
    output += '<br>'+'<br>'.join(detect_comparison_mreg(doc, solution))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, ' F', solution['F'][0]))
    output += '<br>'+'<br>'.join(detect_p(doc, solution['p'][0]))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'R<sup>2</sup>', solution['r2'][0], aliases=['r2','r','kwadraat']))
    output += '<br>' + scan_predictors(doc, solution, prefix=False)[1]
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)

def split_grade_ancova(text:str, solution:dict) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc,solution))
    output += '<br>' + scan_design(doc, solution, prefix=False)[1]
    output += '<br>' + scan_predictors(doc, solution, prefix=False)[1]
    
    output += '<br>'+'<br>'.join(detect_decision_ancova(doc, solution))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F'][3]))
    output += '<br>'+'<br>'.join(detect_p(doc, solution['p'][3]))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'eta<sup>2</sup>', solution['eta'][3], aliases=['eta2','eta']))
    #print(str(solution['F'][3]) + ' - '+ str(solution['p'][3]) + ' - '+ str(solution['eta'][3]))
    if(solution['p'][2] < 0.05):
        output += '<br>'+'<br>'.join(detect_p(doc, solution['p'][2], label=solution['independent']))
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)
 
def split_grade_manova(text:str, solution:dict) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc,solution))
    output += '<br>' + scan_design_manova(doc, solution, prefix=False)[1]
    for i in range(3):
        var_key = 'dependent' if i < 1 else 'dependent' + str(i+1)
        rangwoorden = ['eerste','tweede','derde']
        if solution['p_multivar'] < 0.05 and solution['p_multivar'] < 0.05:
            decision_sent = [x for x in doc.sents if solution[var_key] in x.text and ('significant' in x.text or 'effect' in x.text)]
            if decision_sent != []:
                output += '<br>'+'<br>'.join(detect_decision_manova(decision_sent[0],solution, variable=solution[var_key], synonyms=[], p=solution['p'][i][0], eta=solution['eta'][i][0], num=i+1))
            else:
                output += '<br> -de beslissing van '+solution[var_key]+' wordt niet genoemd'
        if solution['p'][i][0] < 0.05 and solution['p_multivar'] < 0.05:
            output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F'][i][0], appendix='bij de '+rangwoorden[i] + ' afhankelijke variabele '))
            output += '<br>'+'<br>'.join(detect_p(doc, solution['p'][i][0], label='de '+rangwoorden[i] + ' afhankelijke variabele '))
            output += '<br>'+'<br>'.join(detect_report_stat(doc, 'eta<sup>2</sup>', solution['eta'][i][0], aliases=['eta','eta2','eta-kwadraat'],appendix='bij de '+rangwoorden[i] + ' afhankelijke variabele '))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F_multivar'], appendix='bij de multivariate beslissing '))
    output += '<br>'+'<br>'.join(detect_p(doc, solution['p_multivar'], label='de multivariate beslissing '))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'eta<sup>2</sup>', solution['eta_multivar'], aliases=['eta','eta2','eta-kwadraat'], appendix='bij de multivariate beslissing '))
    decision_sent = [x for x in doc.sents if (solution['sumdependent'] in x.text or 'multivariate' in x.text or 'multivariaat' in x.text) \
                         and ('significant' in x.text or 'effect' in x.text)]
    if decision_sent != []:
        output += '<br>'+'<br>'.join(detect_decision_manova(doc,solution,variable=solution['sumdependent'],synonyms=['multivariate'], p=solution['p_multivar'], eta=solution['eta_multivar'], num=0))
    else:
        output += '<br> -de multivariate beslissing wordt niet genoemd'
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)
    
def split_grade_multirm(text:str, solution:dict) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc,solution))
    
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)
    
"""
FUNCTIONS FOR TESTING
"""
def print_dissection(text:str):
    import spacy
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text)
    print([(x.text, x.dep_) for x in doc])
    
def load_dissection(text:str):
    import spacy
    from spacy import displacy
    nlp = spacy.load('nl')
    doc = nlp(text)
    displacy.serve(doc, style='dep')
        