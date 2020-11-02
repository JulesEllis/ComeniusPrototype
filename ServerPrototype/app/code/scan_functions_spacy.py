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

def scan_decision(text: str, assignment: Dict={'hypothesis':0}, solution: Dict={'relative_effect':[0.5],'p':[0.10],'levels':['nederlands','duits']}, anova: bool=False, num: int=1) -> [bool, str]:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    criteria = ['hyp_rejected', 'hyp_present', 'right_comparison', 'right_negation',
                'mean_present', 'pop_present', 'level_present', 'both_present',
                'effect_present', 'strength_present', 'right_strength']#,'p_present',
               # 'p_comparison']
    l_key: str = 'levels' + str(num) if num > 1 else 'levels'
    levels = [x.lower() for x in solution[l_key]]
    scorepoints = dict([(name, False) for name in criteria])
    
    ## H0
    ## bools=rejected, hyp_present
    rejected = solution['p'][num - 1] < 0.05
    hyp_statement = [x for x in doc if x.text == 'verworpen' or 
                   x.text =='behouden' or x.text == 'verwerpen']
    if any(hyp_statement):
        scorepoints['hyp_rejected'] = (not hyp_statement[0].text == 'behouden' and rejected) or (hyp_statement[0].text == 'behouden' and not rejected)
        children = hyp_statement[0].children
        scorepoints['hyp_present'] = any([x for x in children if x.text == 'h0' or x.text == 'nulhypothese'])
        
    ## COMPARING MEANS - 
    ## bools=right_comparison, right_negation, mean_present, pop_present, level_present, both_present
    gold_comp = 'ongelijk' if anova else ['ongelijk','groter','kleiner'][assignment['hypothesis']]
    comparisons = [x for x in doc if x.text == 'groter' or 
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
        
        
    mean = [x for x in doc if x.text == 'gemiddelde']
    mean_2 = [x for x in doc if x.text == 'populatiegemiddelde']
    scorepoints['mean_present'] = any(mean) or any(mean_2)
    #if scorepoints['mean_present']:
        #meanroot = mean[0] if any(mean) else mean_2[0] if any(mean_2) else None
        #meantree:list = descendants(meanroot)
    scorepoints['pop_present'] = any(mean_2) or 'populatie' in [x.text for x in doc]
    scorepoints['level_present'] = any([x in [y.text for y in doc] for x in levels]) or scorepoints['level_present']
    scorepoints['both_present'] = all([x in [y.text for y in doc] for x in levels]) or scorepoints['both_present']
        
    ## STRENGTH
    ## bools=effect_present, strength_present, right_strength, 
    if not anova: #T-test
        sterkte = solution['relative_effect'][0]
        gold_strength: str = 'sterk' if sterkte > 0.8 else 'matig' if sterkte > 0.5 else 'klein'
    else:#One-way ANOVA
        if not assignment['two_way'] and not 'jackedmeans' in list(assignment['data'].keys()):
            sterkte: float = solution['r2'][0]
        else: #Two-way and within-subject ANOVA
            sterkte: float = solution['r2'][num]
        gold_strength: str = 'sterk' if sterkte > 0.2 else 'matig' if sterkte > 0.1 else 'klein'
    effect = [x for x in doc if x.lemma_ == 'effect']
    if effect != []:
        e_tree = descendants(effect[0])
        scorepoints['effect_present'] = True
        scorepoints['strength_present'] = any([x in [y.text for y in e_tree] for x in ['klein','matig','sterk']])
        scorepoints['right_strength'] = gold_strength in [x.text for x in e_tree]
    if solution['p'][num - 1] > 0.05 or math.isnan(solution['p'][num - 1]):
        for x in ['effect_present','strength_present','right_strength']:
            scorepoints[x] = True
    
    ## P-VALUE
    ## bools=p_present, p_comparison
    if False in list(scorepoints.values()):
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:\n'
        if not scorepoints['hyp_rejected']:
            output += ' -ten onrechte gesteld dat de hypothese wordt verworpen als deze wordt behouden of andersom\n'
        if not scorepoints['hyp_present']:
            output += ' -hypothese niet genoemd\n'
        if not scorepoints['right_comparison']:
            output += ' -niveaus in de populatie niet of niet juist met elkaar vergeleken\n'
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
            output += ' -de effectgrootte wordt niet genoemd\n'
        if scorepoints['effect_present'] and not scorepoints['strength_present']:
            output += ' -de sterkte van het effect wordt niet genoemd\n'
        elif scorepoints['effect_present'] and not scorepoints['right_strength']:
            output += ' -de sterkte van het effect wordt niet juist genoemd\n'
        return True, output
    else:
        return False, 'Mooi, deze beslissing klopt. '

#text = 'H0 verworpen, het populatiegemiddelde van Nederlands is niet gelijk aan dat van Duits, er is een klein effect'
#print(scan_decision(text)[1])
#displacy.serve(nl_nlp(text))

def scan_decision_anova(text: str, assignment:Dict={}, solution: Dict={'independent':'nationaliteit',
                        'independent2':'geslacht','p':[None,None,0.10],'r2':[None,None,0.40]}) -> [bool, str]:
    #Define important variables necessary for checking the answer's components
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    tokens = [x.text for x in doc]
    criteria = ['hyp_rejected', 'hyp_present', 'right_negation',
                'mean_present', 'pop_present', 'indy1', 'indy2',
                'strength_present', 'right_strength','effect_present']
    scorepoints = dict([(name, False) for name in criteria])
    rejected = solution ['p'][2] < 0.05
    
    #Interactie
    interactie_list = [x for x in doc if x.text == 'interactie']
    if interactie_list != []:
        int_descendants = descendants(interactie_list[0])
        scorepoints['interactie'] = True
        scorepoints['indy1'] = solution['independent'] in [x.text for x in int_descendants]
        scorepoints['indy2'] = solution['independent2'] in [x.text for x in int_descendants]
        scorepoints['pop_present'] = 'populatie' in [x.text for x in int_descendants]
        scorepoints['right_negation'] = bool(negation_counter(tokens) % 2) != rejected
        
    #Sterkte
    sterkte: float = solution['r2'][2]
    gold_strength: str = 'sterk' if sterkte > 0.2 else 'matig' if sterkte > 0.1 else 'klein'
    effect = [x for x in doc if x.lemma_ == 'effect']
    if effect != []:
        e_tree = descendants(effect[0])
        scorepoints['effect_present'] = True
        scorepoints['strength_present'] = any([x in [y.text for y in e_tree] for x in ['klein','matig','sterk']])
        scorepoints['right_strength'] = gold_strength in [x.text for x in e_tree]
    if solution['p'][2] > 0.05:
        for x in ['effect_present','strength_present','right_strength']:
            scorepoints[x] = True
    
    #H0
    rejected = solution['p'][2] < 0.05
    hyp_statement = [x for x in doc if x.text in ['verworpen','verwerpen','verwerp','behouden','behoud']]
    if any(hyp_statement):
        scorepoints['hyp_rejected'] = rejected != (hyp_statement[0].text in ['behouden', 'behoud'])
        children = hyp_statement[0].children
        scorepoints['hyp_present'] = any([x for x in children if x.text == 'h0' or x.text == 'nulhypothese'])
    
    if False in list(scorepoints.values()):
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:\n'
        if not scorepoints['hyp_rejected']:
            output += ' -ten onrechte gesteld dat de hypothese wordt verworpen als deze wordt behouden of andersom\n'
        if not scorepoints['hyp_present']:
            output += ' -hypothese niet genoemd\n'
        if not scorepoints['right_negation']:
            output += ' -ten onrechte een negatie toegevoegd of weggelaten bij het vergelijken van de niveaus\n'
        if not scorepoints['mean_present']:
            output += ' -niet genoemd dat de beslissing om populatiegemiddelden gaat\n'
        if not scorepoints['pop_present']:
            output += ' -niet gesteld dat de beslissing over de populatie gaat\n'
        if not scorepoints['indy1'] and not scorepoints['indy2']:
            output += ' -de onafhankelijke variabelen ontbreken\n'
        elif not scorepoints['indy2'] or not scorepoints['indy2']:
            output += ' -een van de onafhankelijke variabelen ontbreekt\n'
        if not scorepoints['effect_present']:
            output += ' -de effectgrootte wordt niet genoemd\n'
        if scorepoints['effect_present'] and not scorepoints['strength_present']:
            output += ' -de sterkte van het effect wordt niet genoemd\n'
        elif scorepoints['effect_present'] and not scorepoints['right_strength']:
            output += ' -de sterkte van het effect wordt niet juist genoemd\n'
        return True, output
    else:
        return False, 'Mooi, deze beslissing klopt. ' 

def scan_decision_rmanova(text: str, assignment: Dict, solution: Dict={'independent':'nationaliteit','independent2':'geslacht','p':[0.10,0.10],'r2':[0.30,0.40]}) -> [bool, str]:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    rejected = solution['p'][1] < 0.05
    
    criteria = ['hyp_rejected', 'hyp_present', 'right_comparison', 'right_negation',
                'mean_present', 'pop_present',
                'effect_present', 'strength_present', 'right_strength','jacked']
    scorepoints = dict([(name, False) for name in criteria])
    
    ## H0
    ## bools=hyp_rejected, hyp_present
    hyp_statement = [x for x in doc if x.text == 'verworpen' or 
                   x.text =='behouden' or x.text == 'verwerpen']
    if any(hyp_statement):
        scorepoints['hyp_rejected'] = (not hyp_statement[0].text == 'behouden' and rejected) or (hyp_statement[0].text == 'behouden' and not rejected)
        children = hyp_statement[0].children
        scorepoints['hyp_present'] = any([x for x in children if x.text == 'h0' or x.text == 'nulhypothese'])
        
    ## COMPARING MEANS - 
    ## bools=right_comparison, right_negation, mean_present, pop_present, level_present, both_present
    gold_comp = 'ongelijk'
    comparisons = [x for x in doc if x.text == 'groter' or 
                   x.text =='gelijk' or x.text == 'ongelijk' or x.text == 'kleiner']
    if comparisons != []:
        comptree:List = descendants(comparisons[0])
        
        not_present = 'niet' in [x.text for x in comptree]
        scorepoints['right_comparison'] = comparisons[0].text == gold_comp
        if comparisons[0].text == 'ongelijk':
            scorepoints['right_negation'] = not_present != solution['p'][1] < 0.05
        elif comparisons[0].text == 'gelijk':
            scorepoints['right_negation'] = not_present == solution['p'][1] < 0.05
        scorepoints['jacked'] = 'opgevoerde' in [x.text for x in comptree]
        
        mean = [x for x in comparisons[0].children if x.text == 'gemiddelde']
        mean_2 = [x for x in comparisons[0].children if x.text == 'populatiegemiddelde']
        scorepoints['mean_present'] = any(mean) or any(mean_2)
        if scorepoints['mean_present']:
            meanroot = mean[0] if any(mean) else mean_2[0] if any(mean_2) else None
            meantree:list = descendants(meanroot)
            scorepoints['pop_present'] = any(mean_2) or 'populatie' in [x.text for x in meantree]
        
    ## STRENGTH
    ## bools=effect_present, strength_present, right_strength, 
    sterkte: float = solution['r2'][1]
    gold_strength: str = 'sterk' if sterkte > 0.2 else 'matig' if sterkte > 0.1 else 'klein'
    effect = [x for x in doc if x.lemma_ == 'effect']
    if effect != []:
        e_tree = descendants(effect[0])
        scorepoints['effect_present'] = True
        scorepoints['strength_present'] = any([x in [y.text for y in e_tree] for x in ['klein','matig','sterk']])
        scorepoints['right_strength'] = gold_strength in [x.text for x in e_tree]
    if solution['p'][1] > 0.05:
        for x in ['effect_present','strength_present','right_strength']:
            scorepoints[x] = True
    
    #Detect whether the right phrases appear in the decision
    if False in list(scorepoints.values()):
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:\n'
        if not scorepoints['hyp_rejected']:
            output += ' -ten onrechte gesteld dat de hypothese wordt verworpen als deze wordt behouden of andersom\n'
        if not scorepoints['hyp_present']:
            output += ' -hypothese niet genoemd\n'
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
        if not scorepoints['effect_present']:
            output += ' -de effectgrootte wordt niet genoemd\n'
        if scorepoints['effect_present'] and not scorepoints['strength_present']:
            output += ' -de sterkte van het effect wordt niet genoemd\n'
        elif scorepoints['effect_present'] and not scorepoints['right_strength']:
            output += ' -de sterkte van het effect wordt niet juist genoemd\n'
        return True, output
    else:
        return False, 'Mooi, deze beslissing klopt. '

def scan_interpretation(text: str, solution: Dict={'control':False, 'dependent':'gewicht','independent':'nationaliteit'}, 
                        anova: bool=False, num: int=1) -> [bool, str]:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    
    i_key: str = 'independent' + str(num) if num > 1 else 'independent'
    independent = solution[i_key]
    dependent = solution['dependent']
    control: bool = solution['control']
    criteria = ['cause', 'effect', 'unk', 'var', 'dep', 'prim', 'alt', 'cause_alignment'] #Replace effect with verschil?
    scorepoints = dict([(name, False) for name in criteria])
    scorepoints['prim'] = 'primaire' in [x.text for x in doc] if not control else True
    scorepoints['alt'] = 'alternatieve' in [x.text for x in doc] if not control else True
    scorepoints['unk'] = 'onbekend' in [x.text for x in doc] if not control else True
    #TODO: verbind 'onbekend' met 'oorzaak is' en daaran verwandte uitdrukkingen
    
    causeverbs = [x for x in doc if x.text in ['veroorzaakt', 'heeft', 'beinvloedt', 'beinvloed','verantwoordelijk', 'oorzaak']]
    effect = [x for x in doc if x.text == 'effect']    
    if any(causeverbs) or any(effect):
        if any(causeverbs) and any(effect):
            effect_children = descendants(causeverbs[0]) + descendants(effect[0])
            ancestors = [x for x in causeverbs[0].ancestors] + [x for x in effect[0].ancestors]
        elif any(causeverbs):
            effect_children = descendants(causeverbs[0])
            ancestors = [x for x in causeverbs[0].ancestors]
        else:
            effect_children = descendants(effect[0])
            ancestors = [x for x in effect[0].ancestors]
        scorepoints['cause'] = causeverbs != []
        scorepoints['effect'] = effect != []
        scorepoints['var'] = independent in [x.text for x in effect_children]
        scorepoints['dep'] = dependent in [x.text for x in effect_children]
        if ancestors != []:
            if [x for x in [y for y in ancestors[0].children]] != []:
                verklaring_children = [x.text for x in [y for y in ancestors][0].children]
                if control: 
                    scorepoints['cause_alignment'] = 'primaire' in verklaring_children and not 'alternatieve' in verklaring_children 
                else:
                    scorepoints['cause_alignment'] = not 'primaire' in verklaring_children and 'alternatieve' in verklaring_children
    
    if False in list(scorepoints.values()):
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:\n'
        if not scorepoints['cause']:
            output += ' -er wordt niet gesproken over de oorzaak van het effect\n'
        if not scorepoints['effect']:
            output += ' -het effect wordt niet genoemd\n'
        if not scorepoints['unk']:
            output += ' -niet gesteld dat de oorzaak van het effect onbekend is\n'
        if not scorepoints['var']:
            output += ' -de onafhankelijke variabele wordt niet genoemd\n'
        if not scorepoints['dep']:
            output += ' -de afhankelijke variabele wordt niet genoemd\n'
        if not scorepoints['prim']:
            output += ' -de primaire verklaring wordt niet genoemd\n'
        if not scorepoints['alt']:
            output += ' -de alternatieve verlaring wordt niet genoemd\n'
        if not scorepoints['cause_alignment'] and scorepoints['prim'] and scorepoints['alt']:
            output += ' -de alternatieve verlaring en primaire verklaring zijn omgekeerd aangegeven\n'
        return True, output
    else:
        return False, 'Mooi, deze causale interpretatie klopt. '

def scan_interpretation_anova(text: str, solution: Dict={'independent':'nationaliteit', 'control':True,
                                                         'independent2':'geslacht','dependent':'gewicht',
                                                         'levels':['nederlands','duits'],'levels2':['man','vrouw']}) -> [bool, str]:
    #sol_tokens = solution['interpretation'].split()
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    tokens = [x.text for x in doc]
    
    control: bool = solution['control']
    levels=[x.lower() for x in solution['levels']]
    levels2=[x.lower() for x in solution['levels2']]
    criteria = ['unk','prim','alt','dep','indylevels', 'indy2']
    scorepoints = dict([(name, False) for name in criteria])
    
    #Check answer elements
    scorepoints['unk'] = 'onbekend' in tokens if not control else True
    scorepoints['prim'] = 'primaire' in tokens if not control else True
    scorepoints['alt'] = 'alternatieve' in tokens if not control else True
    dependents = [x for x in doc if x.text == solution['dependent']]
    ind_nr = 0
    if dependents != []:
        scorepoints['dep'] = True
        dep_children = [x.text for x in descendants(dependents[0])]
        if levels2[0] in dep_children and levels2[1] in dep_children:
            ind_nr = 1; scorepoints['indylevels'] = True
        elif levels[0] in dep_children and levels[1] in dep_children:
            ind_nr = 2; scorepoints['indylevels'] = True
        scorepoints['indy2'] = solution['independent'] in tokens if ind_nr == 2 else solution['independent2'] in tokens if ind_nr == 1 else False
    
    if False in list(scorepoints.values()):
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:\n'
        if not scorepoints['prim']:
            output += ' -de primaire verklaring wordt niet genoemd\n'
        if not scorepoints['alt']:
            output += ' -de alternatieve verlaring wordt niet genoemd\n'
        if not scorepoints['unk']:
            output += ' -niet gesteld dat de echte oorzaak van het effect onbekend is\n'
        if not scorepoints['dep']:
            output += ' -afhankelijke variabele niet genoemd is\n'
        if not scorepoints['indylevels']:
            output += ' -één van de onafhankelijke variabelen of de niveaus daarvan worden niet genoemd\n'
        if not scorepoints['indy2'] and scorepoints['indylevels']:
            output += ' -ten minste één van de onafhankelijke variabelen wordt niet genoemd'
        return True, output
    else:
        return False, 'Mooi, deze causale interpretatie klopt. '

def split_grade_ttest(text: str) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    tokens = [x.text for x in doc]
    sents = list(doc.sents)
    return scan_decision(text)[1] + '\n' + scan_interpretation(text)[1]
    
    
#nl_nlp = spacy.load('nl')
#txt = 'H0 behouden, het populatiegemiddelde van nederlands is gelijk aan dat van duits in de populatie'\
#' experiment, dus er is slechts een verklaring mogelijk: de alternatieve verklaring is dat de onafhankelijke variabele nationaliteit'\
#' invloed heeft op de afhankelijke variabele gewicht. Het effect is sterk'
#print(split_grade_ttest(txt))
    

#text = 'nationaliteit heeft dezelfde invloed op gewicht bij man als bij vrouw in de variabele geslacht'
#print(scan_interpretation_anova(text)[1])
#displacy.serve(nl_nlp(text))

#text = 'Het is een experiment, de echte verklaring is onbekend en er zijn meerdere mogelijk: De alternatieve verklaring'\
#' is dat de onafhankelijke variabele nationaliteit verantwoordelijk is voor het verschil in '\
#'effect van de afhankelijke variabele gewicht. De primaire verklaring is niks'
#print(scan_interpretation(text)[1])
#displacy.serve(nl_nlp(text))