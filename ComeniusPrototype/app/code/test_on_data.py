#    This project contains CASWIC: Coaching App for Statistical Writing in Introductory Course.
#    Copyright (C) 2023 Jules Ellis and Jelmer Jansen
#
#    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
#

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:24:55 2021

@author: jelmer
"""
import math
import numpy as np
import os
import spacy
import math
import random
import numpy as np
import nltk
import spacy
import re
import nltk
import openpyxl
import pandas as pd
import matplotlib.pyplot as plt
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
from language import LanguageInterface

"""
Scan functions \/
"""
def descendants(node) -> List[Token]:
    output:list = []
    for child in node.children:
        output.append(child)
        output += descendants(child)
    return output

def negation_counter(tokens: List[str]) -> int:
    count: int = 0
    for token in tokens:
        if token in ['geen', 'niet', 'no', 'not', "isn't"]:   # or token[:2] == 'on':
            count += 1
    return count

def detect_strength(sent:Doc, solution:dict, anova:bool, num:int) -> List[str]:

    #Define variables
    sizes = ['medium','moderate','small','strong','large','tiny'] if mes['L_ENGLISH'] else ['klein','zwak','matig','groot','sterk']
    large = ['strong','large'] if mes['L_ENGLISH'] else ['sterk','groot']
    med = ['medium','moderate'] if mes['L_ENGLISH'] else ['matig']
    small = ['small','weak'] if mes['L_ENGLISH'] else ['klein','zwak']
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
    effect = [x for x in sent if x.lemma_ in sizes]
    if effect != []:
        n_effects:int = len(effect)
        e_root = effect[num-1] if num <= n_effects else effect[num-1 - (3 - n_effects)]
        e_tree = descendants(e_root)
        scorepoints['effect_present'] = e_root.head.text == 'effect' or 'effect' in [x.text for x in e_tree]
        scorepoints['strength_present'] = True #any([x in [y.text for y in e_tree] for x in ['klein','matig','sterk']]) or e_root.head.text in ['klein','matig','sterk']
        scorepoints['right_strength'] = e_root.text in large if gold_strength == 2 else e_root.text in med if gold_strength == 1 else e_root.text in small
    
    #Add strings
    appendix:str = '' if num < 2 else mes['S_FORFAC2'] + str(num)+' ' if num == 2 and solution['assignment_type'] != 5 else mes['S_SUBS'] if num < 3 else mes['S_INTERACT']
    if not scorepoints['effect_present'] and scorepoints['strength_present']:
        output.append(mes['F_NOSIZE']+appendix+mes['S_NONAME'])
    if not scorepoints['strength_present']:
        output.append(mes['F_STRENGTH']+appendix+mes['S_NONAME'])
    elif scorepoints['effect_present'] and not scorepoints['right_strength']:
        output.append(mes['F_STRENGTH']+appendix+mes['S_NONAMES'])
    return output

def detect_comparison(sent:Doc, solution:dict, anova:bool, num:int) -> List[str]:
    #Define language
    comp1 = ['unequal','larger','smaller'] if mes['L_ENGLISH'] else ['ongelijk','groter','kleiner']
    comp2 = ['larger','unequal','smaller','equal','different','differing'] if mes['L_ENGLISH'] else ['groter','ongelijk','kleiner','gelijk','anders','verschillend']
    avgs1 = ['average','averages','mean','means'] if mes['L_ENGLISH'] else ['gemiddelde','gemiddelden','gemiddeld']
    avgs2 = ['population average', 'population mean'] if mes['L_ENGLISH'] else ['populatiegemiddelde','populatiegemiddelden']
    pop = 'population' if mes['L_ENGLISH'] else 'populatie'
    nott = 'not' if mes['L_ENGLISH'] else 'niet'
    
    #Define variables
    criteria = ['right_comparison', 'right_negation', 'mean_present', 'pop_present', 'level_present', 'both_present','contrasign']
    scorepoints = dict([(x,False) for x in criteria])
    output:List[str] = []
    tokens = [y.text for y in sent]
    levels=[x.lower() for x in solution['levels' + str(num) if num > 1 else 'levels']]
    level_syns = solution['level_syns'] if num == 1 else solution['level' + str(num) + '_syns']
    
    #Controleer input
    gold_comp = comp1[0] if anova else comp1[solution['hypothesis']]
    comparisons = [x for x in sent if x.text in comp2]
    if comparisons != []:
        comproot = comparisons[num-1] if len(comparisons) >= num else comparisons[0]
        comptree:List = descendants(comproot)
        not_present = nott in [x.text for x in comptree]
        scorepoints['right_comparison'] = comproot.text == gold_comp or (gold_comp == comp1[0] and comproot.text in comp2[3:])
        if gold_comp == comp1[0] and comproot.text in [comp1[0],comp2[4],comp2[5]]:
            scorepoints['right_negation'] = not_present != (solution['p'][num-1] < 0.05)
        elif gold_comp == comp1[0] and comproot.text == comp2[3]:
            scorepoints['right_negation'] = not_present == (solution['p'][num-1] < 0.05)
        else:
            scorepoints['right_negation'] = not_present != (solution['p'][num-1] < 0.05)
    else:
        scorepoints['right_negation'] = True
    
    mean = [x in sent.text for x in avgs1]
    mean_2 = [x in sent.text for x in avgs2]
    scorepoints['mean_present'] = any(mean) or any(mean_2)
    scorepoints['pop_present'] = any(mean_2) or pop in tokens or any([x in tokens for x in ['significant','significante']])
    level_bools:list[bool] = [levels[i] in tokens or any([y in tokens for y in level_syns[i]]) for i in range(2)]#len(levels))]
    scorepoints['level_present'] = any(level_bools) #or scorepoints['level_present']
    scorepoints['both_present'] = all(level_bools)# or scorepoints['both_present']
    scorepoints['contrasign'] = not ((any(mean_2) or pop in tokens) and any([x in tokens for x in ['significant','significante']]))
    
    #Add strings:
    appendix:str = '' if num < 2 else mes['S_FORFAC'] + str(num) if num < 3 else mes['S_WITHINT']
    if not scorepoints['contrasign']:
        output.append(mes['F_BOTHPOP'] + appendix)
    if not scorepoints['right_comparison']:
        output.append(mes['F_JUSTLEVELS']+appendix+mes['S_BADCOMPARED'])
    if not scorepoints['right_negation']:
        output.append(mes['F_INTNEG']+mes['S_COMPARING']+appendix) 
    if not scorepoints['mean_present']:
        output.append(mes['F_DECISION']+appendix+' '+mes['S_POPAVGS'])
    if not scorepoints['pop_present']:
        output.append(mes['F_DECISION'] + appendix + ' '+mes['S_OVERPOP'])
    if not scorepoints['level_present']:
        output.append(mes['F_INDEPLEVELS']+appendix+mes['S_NONAMES'][1:])
    if not scorepoints['both_present'] and scorepoints['level_present']:
        output.append(mes['F_SOMELEVEL']+appendix+mes['S_LEFTOUT'])
    return output

def detect_significance(doc:Doc, solution:dict, num:int=1) -> List[str]:
    scorepoints = {'effect': False,
                   'sign': False,
                   'neg': False}
    diff_words = ['difference','effect'] if mes['L_ENGLISH'] else ['verschil','effect']
    size_words = ['medium','moderate','small','strong','large','tiny'] if mes['L_ENGLISH'] else ['zwak','matig','klein','sterk','groot']
    output:List[str] = []
    rejected:bool = solution['p'][num-1] < 0.05
    h0_output:list = detect_h0(doc, solution, num)
    if h0_output == []:
        return []
    
    if hasattr(doc, 'sents'):
        difference = [sent for sent in doc.sents if any([y in sent.text for y in diff_words]) 
            and not any([y in [x.text for x in sent] for y in size_words])]
    else:
        difference = [doc]
    
    if difference != []:
        d_root = difference[num - 1] if num <= len(difference) else difference[num-1 - (3 - len(difference))]
        scorepoints['effect'] = True
        tokens:List[str] = [x.text for x in d_root]
        scorepoints['sign'] = 'significant' in tokens
        scorepoints['neg'] = bool(negation_counter(tokens) % 2) != rejected
    if num < 2:
        if not scorepoints['effect']:
            output.append(mes['F_EFFECTSIGN'])
        if not scorepoints['sign'] and scorepoints['effect']:
            output.append(mes['F_EFFECTSIGN'])
        if not scorepoints['neg']:
            output.append(mes['F_NEGEFFECT'])
    else:
        appendix:str = mes['F_FORFAC'] + str(num) if num < 3 else mes['S_WITHINT']
        if not scorepoints['effect']:
            output.append(mes['F_EFFECTSIGN']+appendix)
        if not scorepoints['sign'] and scorepoints['effect']:
            output.append(mes['F_EFFECTSIGN']+appendix)
        if not scorepoints['neg']:
            output.append(mes['F_NEGEFFECT']+appendix)
    return output
    
def detect_h0(sent:Doc, solution:dict, num:int=1) -> List[str]:
    #Define variables
    scorepoints = {'hyp_rejected': False,
                   'hyp_present': False
            }
    output:List[str] = []
    rejected = solution['p'][num-1] < 0.05
    behoud_list = ['retain','retained'] if mes['L_ENGLISH'] else ['behoud', 'behouden']
    verwerp_list = ['reject','rejected'] if mes['L_ENGLISH'] else ['verwerp', 'verworpen', 'verwerpen']
    h0_list = ['h0','hypothesis'] if mes['L_ENGLISH'] else ['h0','nulhypothese']
    
    #Controleer input
    scorepoints['hyp_rejected'] = any([x in sent.text for x in verwerp_list]) if rejected else any([x in sent.text for x in behoud_list])
    scorepoints['hyp_present'] = any([x in sent.text for x in h0_list])
    
    #Add strings
    if not scorepoints['hyp_rejected'] and scorepoints['hyp_present']:
        if num < 2:
            output.append(mes['F_HYPSWITCH'])
        if num > 2:
            output.append(mes['F_INTSWITCH'])
        else:
            output.append(mes['F_HYPFACTOR']+str(num)+mes['S_SWITCHEROO'])
    if not scorepoints['hyp_present']:
        if num < 2:
            output.append(mes['F_NOHYP'])
        elif num < 3:
            if solution['assignment_type'] != 5:
                output.append(mes['F_HYPFACTOR']+str(num)+' '+mes['S_LACKING1'])
            else:
                output.append(mes['F_NOSUBJ'])
        else:
            output.append(mes['F_INTHYP'])
    return output

def detect_unk(sent:Doc, solution:dict, num:int=1):
    #Define variables
    criteria:list=['two']#['unk','two']
    ones = ['one','1'] if mes['L_ENGLISH'] else ['één','een','1']
    twos = ['two','multiple','several','2'] if mes['L_ENGLISH'] else ['twee','meerdere','2']
    scorepoints = dict([(x,False) for x in criteria])
    control = solution['control'] if num < 2 else solution['control' + str(num)] if num < 3 else solution['control2'] or solution['control2']
    tokens = [x.text for x in sent]
    output:List[str] = []
    
    #Controleer input
    #scorepoints['unk'] = 'onbekend' in tokens if not control else True
    scorepoints['two'] = any([x in ones for x in tokens]) if control else any([x in twos for x in tokens])
    
    #Add strings
    if not scorepoints['two']:
        output.append(mes['F_XINTS'])
    return output

def detect_primary(sent:Doc, solution:dict, num:int=1) -> List[str]:
    #Define variables
    suffix = ' for the primary explanation' if mes['L_ENGLISH'] else ' bij de primaire verklaring'
    prim = 'primary' if mes['L_ENGLISH'] else 'primaire'
    criteria:list = ['cause', 'ind', 'dep', 'prim', 'neg', 'alignment']
    scorepoints = dict([(x,False) for x in criteria])
    i_key: str = 'independent' + str(num) if num > 1 else 'independent'
    syn_key: str = 'ind_syns' if num == 1 else 'ind' + str(num) + '_syns'
    control: bool = solution['control'] if num < 2 else solution['control'+str(num)]
    rejected = solution['p'][num-1] < 0.05
    tokens = [x.text for x in sent] 
    output:List[str] = []
    
    #Controleer input
    scorepoints['prim'] = prim in tokens if not control else True
    scorepoints['neg'] = bool(negation_counter(tokens) % 2) != rejected
    if mes['L_ENGLISH']:
        causeverbs:list = [x for x in sent if x.text in ['causes', 'influences', 'influenced', 'responsible','cause', 'creates']]
    else:
        causeverbs:list = [x for x in sent if x.text in ['veroorzaakt', 'heeft', 'beinvloedt', 'beinvloed','verantwoordelijk', 'oorzaak', 'invloed']] 
    if any(causeverbs): #effect_children = descendants(causeverbs[0])
        scorepoints['cause'] = True
    deptag = 'mate' if solution['dependent'] == 'mate van positief zelfbeeld' else 'attitude' if solution['dependent'] == 'positieve attitude t.a.v. Koningshuis' else solution['dependent']
    depnode = [x for x in sent if x.text in solution['dep_syns'] + [deptag]]
    indynode = [x for x in sent if x.text in solution[syn_key] + [solution[i_key]]]
    scorepoints['ind'] = solution[i_key] in sent.text #indynode != []
    scorepoints['dep'] = solution['dependent'] in sent.text #depnode != []
    if scorepoints['ind'] and scorepoints['dep']:
        scorepoints['alignment'] = check_causality(indynode[0],depnode[0])
    
    #Add strings
    if not scorepoints['cause']:
        output.append(mes['F_EFFECTCAUSE'])
    if not scorepoints['neg']:
        output.append(mes['F_INTNEG']+suffix)
    if not scorepoints['ind']:
        output.append(mes['F_NOIND']+suffix)
    if not scorepoints['dep']:
        output.append(mes['F_NODEP']+suffix)
    if scorepoints['ind'] and scorepoints['dep'] and not scorepoints['alignment']:
        output.append(mes['F_CAUSEVARS']+suffix)
    if not scorepoints['prim']:
        output.append(mes['F_NOPRIMVAR'])
    return output

def detect_alternative(sent:Doc, solution:dict, num:int=1) -> List[str]:
    #Define variables
    #print([(x.text,x.dep_) for x in sent])
    criteria = ['alt','ind','dep','cause','relation_type']
    scorepoints = dict([(x,False) for x in criteria])
    i_key: str = 'independent' + str(num) if num > 1 else 'independent'
    syn_key: str = 'ind_syns' if num == 1 else 'ind' + str(num) + '_syns'
    control: bool = solution['control'] if num < 2 else solution['control'+str(num)]
    #rejected: bool = solution['p'][num-1] < 0.05
    output:List[str] = []
    
    #Controleer input
    alt = 'alternative' if mes['L_ENGLISH'] else 'alternatieve'
    scorepoints['alt'] = alt in [x.text for x in sent] if not control else True
    suffix = ' for the alternative explanation' if mes['L_ENGLISH'] else ' bij de alternatieve verklaring'
    if mes['L_ENGLISH']:
        causeverbs:list = [x for x in sent if x.text in ['causes', 'influences', 'influenced', 'responsible','cause', 'creates','has','caused']]
    else:
        causeverbs:list = [x for x in sent if x.text in ['veroorzaakt', 'heeft', 'beinvloedt', 'beinvloed','verantwoordelijk', 'oorzaak', 'invloed']] 
    if any(causeverbs):
        scorepoints['cause'] = True
    deptag = 'mate' if solution['dependent'] == 'mate van positief zelfbeeld' else 'attitude' if solution['dependent'] == 'positieve attitude t.a.v. Koningshuis' else solution['dependent']
    depnode = [x for x in sent if x.text in solution['dep_syns'] + [deptag]]
    indynode = [x for x in sent if x.text in solution[syn_key] + [solution[i_key]]]
    scorepoints['ind'] = indynode != []
    scorepoints['dep'] = depnode != []
    if scorepoints['ind'] and scorepoints['dep']:        
        scorepoints['relation_type'] = check_causality(indynode[0], depnode[0], alternative=True)
    else:
        scorepoints['relation_type'] = True
    
    #Add strings
    if not scorepoints['cause']:
        output.append(mes['F_NOCAUSE']+suffix)
    if not scorepoints['ind']:
        output.append(mes['F_NOIND']+suffix)
    if not scorepoints['dep']:
        output.append(mes['F_NODEP']+suffix)
    if not scorepoints['alt']:
        output.append(mes['F_NOALT'])
    if not scorepoints['relation_type']:
        output.append(mes['F_WRONGCAUSATION'])
    return output

def check_causality(independent:Doc, dependent:Doc, alternative:bool=False) -> bool:
    #print(independent.dep_ + '-' + dependent.dep_)
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
        if independent.dep_ == t[0] and dependent.dep_ == t[1]:
            return True
    return False

def scan_decision(doc:Doc, solution:dict, anova:bool, num:int=1, prefix=True, elementair=True) -> [bool, List[str]]:
    output = []
    if elementair:
        output.extend(detect_h0(doc, solution, num))
    else:
        output.extend(detect_significance(doc, solution, num))
    output.extend(detect_comparison(doc, solution, anova, num))
    #if solution['p'][num - 1] < 0.05:
    #    output.extend(detect_strength(doc, solution, anova, num))
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return []
    else:
        return output
    
def scan_interpretation(doc:Doc, solution:dict, anova:bool, num:int=1, prefix=True):
    output = []
    control: bool = solution['control'] if num < 2 else solution['control'+str(num)]
    if mes['L_ENGLISH']:
        primary_checks:list = ['primary','first'] if not control else [solution['dependent']]
        second_check:str = 'alternative'
        unk_checks:list = ['possible','interpretation','interpretations','']
    else:
        primary_checks:list = ['primaire','eerste'] if not control else [solution['dependent']]
        second_check:str = 'alternatieve'
        unk_checks:list = ['mogelijk','mogelijke','verklaring','verklaringen']
    
    unk_sents = [x for x in doc.sents if any([y in [z.text for z in x] for y in unk_checks])]
    if unk_sents != []:
        output.extend(detect_unk(unk_sents[0], solution, num))
    else:
        output.append(mes['F_MANEXP'])
    primair_sents = [x for x in doc.sents if any([z in x.text for z in primary_checks])]
    if primair_sents != []:
        output.extend(detect_primary(primair_sents[0], solution, num))
    else:
        output.append(mes['F_PRIMEXP'])
    if not control:
        alt_sents = [x for x in doc.sents if second_check in [y.text for y in x]]
        #displacy.serve(alt_sents[0])
        if alt_sents != []:
            output.extend(detect_alternative(alt_sents[0], solution, num))
        else:
            output.append(mes['F_ALTEXP'])
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return []
    else:
        return output

file = 'data.xlsx'
nldata = pd.read_excel(file, sheet_name='NL T-within')
endata = pd.read_excel(file, sheet_name='EN T-within')
li = LanguageInterface()
mes = li.get_messages(False)

decisions = list(nldata['decision'])
interpretations = list(nldata['interpretation'])
ldict = {'film2':['horror','romance'],
         'behandeling':['voor','na'],
         'behandeling2':['voor','na'],
         'emotie1':['angst','verassing'],
         'emotie2':['angst','boosheid'],
         'beloning':['spelen','koekjes'],
         'beloning2':['wel','niet'],
         'leeftijd1':['20','30'],
         'leeftijd2':['20','21'],
         'eenzaamheid2':['24','12'],
         'eenzaamheid':['samen','alleen']}
inds = list(nldata['independent'])
deps = list(nldata['dependent'])
hyp  = list(nldata['hyp'])
controls = [bool(int(x)) for x in list(nldata['control'])]
p = list(nldata['p'])
r2 = [0.25 if x == 'groot' else 0.15 if x == 'matig' else 0.05 if x == 'klein' else None for x in list(nldata['strength'])]
N = len([x for x in hyp if x in ['gelijk','groter','kleiner']])

nlp = spacy.load('nl')
mdict = {'Totaal aantal opgaven':N}
for i in range(len(decisions)):
    assignment = {'independent':inds[i],
                  'dependent':deps[i],
                  'levels':ldict[inds[i]],
                  'control':controls[i],
                  'ind_syns':[],
                  'dep_syns':[],
                  'level_syns':[[],[]],
                  'hypothesis':0 if hyp[i] == 'gelijk' else 1 if hyp[i] == 'groter' else 2 if hyp[i] == 'kleiner' else None}
    solution = {**{'p':[p[i]],'r2':[r2[i]]}, **assignment}
    if assignment['hypothesis'] == None:
        continue
    
    mistakes = scan_decision(nlp(decisions[i]), solution, anova=False) + scan_interpretation(nlp(interpretations[i]), solution, anova=False)
    for m in mistakes:
        if not m in list(mdict.keys()):
            mdict[m] = 0
        else:
            mdict[m] += 1

i=0
mdict = dict(sorted(list(mdict.items()),key=lambda x:x[1],reverse=True))
for key, value in list(mdict.items()):
    print(str(i)+' - ('+str(value) + '):' + key)
    i += 1
            
y_pos = np.arange(len(mdict))
plt.bar(y_pos, list(mdict.values()), align='center', alpha=0.5)
plt.xticks(y_pos, [str(x) for x in range(len(mdict))])
plt.ylabel('Mistake')
plt.title('Within-subject T-test mistake types')
plt.show()

"""
{'instruction': 'De reactietijd van een aantal subjecten wordt gemeten. De subjecten krijgen twee stimuli te zien, een vierkant en een cirkel. <br><br>Maak een elementair rapport van onderstaande data voor de hypothese dat de score bij rond gemiddeld groter is dan die bij vierkant. De personen van elke groep zijn willekeurig geselecteerd. "Voer je antwoorden alsjeblieft tot op 2 decimalen in, en gebruik dezelfde vergelijking van de gemiddelden in je antwoord als in de vraagstelling staat (e.g. ""groter"" of ""kleiner""). "', 'hypothesis': 1, 'between_subject': True, 'control': True, 'dependent': 'reactietijd', 'dep_syns': ['reactietijden'], 'assignment_type': 1, 'independent': 'stimulusvorm', 'levels': ['rond', 'vierkant'], 'ind_syns': ['stimulusvormen'], 'level_syns': [['ronde'], ['vierkante']], 'A': [108.95, 128.74, 114.87, 121.36, 122.99, 116.64, 115.87, 124.89, 110.6, 115.92, 117.74], 'B': [188.34, 179.62, 176.6, 177.79, 162.83, 178.02, 174.29, 172.96, 180.96, 192.18, 178.38, 173.6, 162.96, 190.86]}
ldict = {'emotie3':['angst','verassing'],
         'emotie4':['angst','boosheid'],
         'geslacht':['man','vrouw'],
         'feedback':['positief','negatief'],
         'inkomen':['wel','geen'],
         'ras':['zwart','wit'],
         'strategie':['coöperatief','competief'],
         'beroep':['arts','leraar'],
         'verdekking':['toedekkend','verwekkend']}
{'hypothesis': 1, 'assignment_type': 1, 'independent': 'stimulusvorm', 'ind_syns': ['stimulusvormen'], 'levels': ['rond', 'vierkant'], 'level_syns': [['ronde'], ['vierkante']], 'dependent': 'reactietijd', 'dependent_measure': 'kwantitatief', 'independent_measure': 'kwalitatief', 'dep_syns': ['reactietijden'], 'null': 'h0: mu(rond) <= mu(vierkant)', 'control': True, 'means': [118.05181818181818, 177.81357142857146], 'stds': [5.968414895400317, 8.776157041529192], 'ns': [11, 14], 'df': [23], 'raw_effect': [-59.76175324675329], 'sp': 7.682537751756973, 'relative_effect': [-7.778907852823236], 'T': [-19.306741555154368], 'p': [1.0], 'decision': 'H0 behouden, het populatiegemiddelde van rond is niet groter vergeleken met dat vanvierkant. ', 'interpretation': 'Experiment, dus er is slechts een verklaring mogelijk. Dit is namelijk dat stimulusvorm geen invloed heeft op reactietijd.'}
"""
