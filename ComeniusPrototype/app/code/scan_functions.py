#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:28:48 2020

@author: jelmer
"""
import math
import random
import numpy as np
import nltk
import re
from nltk import CFG, Tree
from scipy import stats
from typing import Dict, List, Tuple

def parse_yes_no(input_text: str) -> str:
    return input_text.lower() in ['ja','jaa','yes','yeah','yep','sure','positive','1','true']

def negation_counter(tokens: List[str]) -> int:
    count: int = 0
    for token in tokens:
        if token in ['geen', 'niet']:   # or token[:2] == 'on':
            count += 1
    return count

def fancy_names(stat: str) -> str:
    fancynames: Dict[str,str] = {'means':'gemiddelde', 'stds':'standaarddeviatie','ns':'n', 'df':'het aantal vrijheidsgraden',
                  'T':'T', 'p':'p', 'ss':'sum of squares','ms':'mean squares','F':'F','r2':'de correlatie'}
    return fancynames[stat]

def scan_yesno(input_text : str) -> [bool, bool]:
    return False, parse_yes_no(input_text)

def scan_dummy(input_Text : str)  -> [bool, str]:
    return False, ''

def scan_protocol_choice(input_text : str):
    if input_text not in ['1','2','3','4','5','6']:
        return True, 'Sorry, voer uw antwoord opnieuw in in de vorm van een cijfer.'
    else:
        return False, 'Ok, hier is je opgave:<br>'

def scan_report_choice(input_text : str):
    if input_text not in ['1','2','3','4','5','6']:
        return True, 'Sorry, voer uw antwoord opnieuw in in de vorm van een cijfer.'
    else:
        return False, 'Ok, hier is je opgave:<br>'

def scan_indep(text :str, solution :Dict) -> [bool, str]:
    #Determine which of the necessary elements are present in the answer
    text: List[str] = nltk.word_tokenize(text.lower())
    scorepoints: Dict[str, bool] = {'var': False, 'measure': False, 'level1': False, 'level2': False}
    pairs = [(solution['independent'].lower(),'var'),(solution['independent_measure'].lower(),'measure'),(solution['levels'][0].lower(),'level1'),(solution['levels'][1].lower(),'level2')]
    for pair in pairs:
        if pair[0] in text:
            scorepoints[pair[1]] = True
        if pair[1] == 'measure':
            if pair[0][:-1] + 've' in text:
                scorepoints[pair[1]] = True
    scorepoints['var'] = scorepoints['var'] or any([x in text for x in solution['ind_syns']])
    
    #Determine the response of the chatbot
    if False in list(scorepoints.values()):
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
        if not scorepoints['var']:
            output += ' -de juiste onafhankelijke variabele<br>'
        if not scorepoints['measure']:
            output += ' -het juiste meetniveau<br>'
        if scorepoints['level1'] != scorepoints['level2']:
            output += ' -één niveau van de onafhankelijke variabele<br>'
        elif not scorepoints['level1'] and not scorepoints['level2']:
            output += ' -beide niveaus van de onafhankelijke variabele<br>'
        return True, output
    else:
        return False, 'Mooi, deze beschrijving klopt. '
    
def scan_indep_anova(text: str, solution: Dict, num: int=1, between_subject=True) -> [bool, str]:
    #Determine which of the necessary elements are present in the answer
    text: List[str] = nltk.word_tokenize(text.lower())
    n_key: str = 'independent' if num == 1 else 'independent' + str(num)
    syn_key: str = 'ind_syns' if num == 1 else 'ind' + str(num) + '_syns'
    l_key: str = 'levels' if num == 1 else 'levels' + str(num)
    scorepoints: Dict = {'factor': any(x in text for x in ['factor', 'between-subjectfactor','within-subjectfactor']), 
                   'domain': any(x in text for x in ['between-subject', 'between-subjectfactor']) if between_subject 
                   else any(x in text for x in ['within-subject', 'within-subjectfactor']), 
                   'name': solution[n_key].lower() in text or any([x in text for x in solution[syn_key]]), 
                   'levels': [level.lower() in text for level in solution[l_key]]}
    
    #Determine the response of the chatbot
    if False in list(scorepoints.values()):
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
        if not scorepoints['factor']:
            output += ' -de uitspraak dat deze variabele een factor is<br>'
        if not scorepoints['domain']:
            output += ' -het domein van de variabele<br>'
        if not scorepoints['name']:
            output += ' -de naam van de variabele<br>'
        if True not in scorepoints['levels']:
            output += ' -alle niveaus van de onafhankelijke variabele<br>'
        elif False in scorepoints['levels']:
            output += ' -enkele niveaus van de onafhankelijke variabele<br>'
        return True, output
    else:
        return False, 'Mooi, deze beschrijving klopt. '

def scan_dep(text: str, solution: Dict) -> [bool, str]:
    #Determine which of the necessary elements are present in the answer
    text: List[str] = nltk.word_tokenize(text.lower())
    scorepoints: Dict[str, bool] = {'var': False, 'measure': False}
    pairs: List[Tuple[str,str]] = [(solution['dependent'].lower(),'var'),(solution['dependent_measure'],'measure')]
    for pair in pairs:
        if pair[0] in text:
            scorepoints[pair[1]] = True
        if pair[1] == 'measure':
            if pair[0][:-1] + 've' in text:
                scorepoints[pair[1]] = True
    scorepoints['var'] = scorepoints['var'] or any([x in text for x in solution['dep_syns']])
    
    #Determine the response of the chatbot
    if False in list(scorepoints.values()):
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
        if not scorepoints['var']:
            output += ' -de juiste afhankelijke variabele<br>'
        if not scorepoints['measure']:
            output += ' -het juiste meetniveau<br>'
        return True, output
    else:
        return False, 'Mooi, deze beschrijving klopt. '
    
def scan_control(text: str, solution: Dict, num:int=1) -> [bool, str]:
    tokens: List[str] = nltk.word_tokenize(text.lower())
    control: float = solution['control'] if num < 2 else solution['control'+str(num)]
    n_negations: int = negation_counter(tokens)
    scorepoints: Dict[str, bool] = {'experiment': 'experiment' in tokens, 
                     'negations': control != bool(n_negations % 2) if 'experiment' in tokens else True}
    if 'experiment' in tokens or 'experimenteel' in tokens:
        if control != bool(n_negations % 2):
            return False, 'Mooi, deze beschrijving klopt. '
        else:
            if control:
                return True, 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>-ten onrechte gesteld dat het onderzoek geen experiment is<br>'
            else:
                return True, 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>-ten onrechte gesteld dat het onderzoek een experiment is<br>'
    elif 'passief-observerend' in tokens or ('passief' in tokens) or ('passief-geobserveerd' in tokens) : #and 'observerend' in tokens):
        if control == bool(n_negations % 2):
            return False, 'Mooi, deze beschrijving klopt. '
        else:
            if control:
                return True, 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>-ten onrechte gesteld dat het onderzoek passief-observerend is<br>'
            else:
                return True, 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>-ten onrechte gesteld dat het onderzoek niet passief-observerend is<br>'
    else:
        return True, 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>-de uitspraak of het onderzoek een experiment is of niet<br>'
        
    
    #Determine the response of the chatbot
    if False in list(scorepoints.values()):
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
        if not scorepoints['experiment']:
            output += ' -de uitspraak of het onderzoek een experiment is of niet<br>'
        if not scorepoints['negations'] and control:
            output += ' -je hebt ten onrechte gesteld dat het onderzoek geen experiment is<br>'
        if not scorepoints['negations'] and not control:
            output += ' -je hebt ten onrechte gesteld dat het onderzoek een experiment is<br>'
        return True, output
    else:
        return False, 'Mooi, deze beschrijving klopt. '

def scan_numbers(text: str, stat: str, solution: Dict, margin: float) -> [bool, str]:
    tokens: List[str] = nltk.word_tokenize(text.lower())
    numbers: List[float] = []
    for t in tokens:
        try:
            numbers.append(float(t))
        except ValueError:
            pass
            
    #Compare the received numbers to the gold standards from the solution
    goldnums: List[float] = solution[stat]
    nums: List[List[float]] = [[n for n in numbers if goldnums[i] - margin < n and n < goldnums[i] + margin] for i in range(len(goldnums))]
    scorepoints: List[int] = [1 if num !=[] else 0 for num in nums] #Binary representation for each number in the solution whether it is present in the answer
    
    #Determine the response of the chatbot
    if 0 in scorepoints:
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
        if 1 not in scorepoints:
            output += ' -Alle juiste waarden van ' + fancy_names(stat) + ' ontbreken<br>'
        else:
            output += ' -Een juiste waarde van ' + fancy_names(stat) + ' ontbreekt<br>'
        return True, output
    else:
        return False, 'Mooi, deze cijfers kloppen. '
    
def scan_number(text: str, stat: str, solution: Dict, margin: float=0.01) -> [bool, str]:
    fancynames: Dict[str, str] = {'df':'het aantal vrijheidsgraden', 'raw_effect': 'het ruwe effect',
                  'T':'T', 'p':'p', 'relative_effect': 'het relatieve effect', 'F': 'F','r2':'de correlatie','ns':'N'}
    tokens: List[str] = nltk.word_tokenize(text.lower())
    numbers: List[float] = []
    for t in tokens:
        try:
            numbers.append(float(t))
        except ValueError:
            output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'\
            ' -de juiste waarde van ' + fancynames[stat] + '<br>'
            return True, output
            
    #Compare the received numbers to the gold standard from the solution
    gold: float = solution[stat][0]
    right_number: List[bool] = [n for n in numbers if gold - margin < n and n < gold + margin]
    
    #Determine the response of the chatbot
    if right_number == []:
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'\
        ' -de juiste waarde van ' + fancynames[stat] + '<br>'
        return True, output
    else:
        return False, 'Mooi, dit cijfer klopt. '
    
def scan_p(text:str, solution: Dict, margin: float=0.01) -> [bool, str]:
    tokens: List[str] = nltk.word_tokenize(text.lower())
    numbers: List[float] = []
    for t in tokens:
        try:
            numbers.append(float(t))
        except ValueError:
            pass
    gold = solution['p'][0] 
    right_number: bool = [n for n in numbers if gold - margin < n and n < gold + margin] != []
    boundary_100: bool = '0.10' in tokens and round(gold, 2) != 0.01
    boundary_050: bool = '0.05' in tokens and round(gold, 2) != 0.05
    boundary_010: bool = '0.01' in tokens and round(gold, 2) != 0.05
    boundary_005: bool = '0.005' in tokens and round(gold, 2) != 0.05
    boundary_001: bool = '0.001' in tokens and round(gold, 2) != 0.05
    
    if numbers != []:
        if right_number and len(numbers) == len(tokens):#Als er alleen cijfers in het veld staan
            return False, "Mooi, deze waarde van p klopt! "
        elif any([x in tokens for x in ['<','minder','kleiner']]):
            if boundary_100 and gold < 0.10 and gold > 0.05:
                return False, "Mooi, deze waarde van p klopt! "
            elif boundary_050 and gold < 0.05 and gold > 0.01:
                return False, "Mooi, deze waarde van p klopt! "
            elif boundary_010 and gold < 0.01 and gold > 0.005:
                return False, "Mooi, deze waarde van p klopt! "
            elif boundary_005 and gold < 0.005 and gold > 0.001:
                return False, "Mooi, deze waarde van p klopt! "
            elif boundary_001 and gold < 0.001:
                return False, "Mooi, deze waarde van p klopt! "
            else: return True, 'Er ontbreekt nog iets aan je antwoord, namelijk:<br> -de juiste waarde van p'
        elif any([x in tokens for x in ['>','meer','groter']]):
            if boundary_100 and gold > 0.10:
                return False, "Mooi, deze waarde van p klopt! "
            elif boundary_050 and gold > 0.05 and gold < 0.10:
                return False, "Mooi, deze waarde van p klopt! "
            elif boundary_010 and gold > 0.01 and gold < 0.05:
                return False, "Mooi, deze waarde van p klopt! "
            elif boundary_005 and gold > 0.005 and gold < 0.01:
                return False, "Mooi, deze waarde van p klopt! "
            elif boundary_001 and gold > 0.001 and gold < 0.005:
                return False, "Mooi, deze waarde van p klopt! "
            else: return True, 'Er ontbreekt nog iets aan je antwoord, namelijk:<br> -de juiste waarde van p'
        else: 
            return True, 'Er ontbreekt nog iets aan je antwoord, namelijk:<br> -de juiste waarde van p'
    else:
        return True, 'Er ontbreekt nog iets aan je antwoord, namelijk:<br> -de juiste waarde van p'
        
def scan_hypothesis(text: str, solution: Dict, num: int=1) -> [bool, str]:
    #Remove potential dots to avoid confusion
    l_key: str = 'levels' if num < 2 else 'levels' + str(num)
    sign:str = ['==','<=','>='][solution['hypothesis']] if 'hypothesis' in list(solution.keys()) else '=='
    tokens: List[str] = text.lower().replace('.','').split() #nltk.word_tokenize(text.lower())
    avgs: List[str] = ['mu(' + avg.lower() + ')' for avg in solution[l_key]]
    mus: List[bool] = [avg in tokens for avg in avgs]
    scorepoints: Dict[str, bool] = {'H0': 'h0:' in tokens,
                   'sign': sign in tokens, 
                   'order': True}
    if sign == '==' and '=' in tokens:
        scorepoints['sign'] = True
    
    if False in list(scorepoints.values()) + mus:
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
        if not scorepoints['H0']:
            output += ' -het "H0:" teken<br>'
        if not scorepoints['sign']:
            output += ' -het juiste vergelijkingsteken<br>'
        if not scorepoints['order']:
            output += ' -de symbolen staan niet in de juiste volgorde<br>'
        if True not in mus:
            output += ' -de populatiegemiddelden<br>'
        elif False in mus:
            output += ' -ten minste één van de populatiegemiddelden<br>'
        return True, output
    else:
        return False, 'Mooi, deze hypothese klopt. '

#Interaction hypothesis for two-way ANOVA
def scan_hypothesis_anova(text: str, solution: Dict) -> [bool, str]:
    #Remove potential dots to avoid confusion
    levels = solution['levels']; levels2 = solution['levels2']
    criteria:list = ['h0', 'mu1some','mu1all','mu2some','mu2all']
    scorepoints = dict([(x,False) for x in criteria])
    scorepoints['h0'] = bool(re.search(r'h0\(' + solution['independent'] + '( )*(x|\*|(en))( )*' + solution['independent2'] + '\):', text, re.IGNORECASE))
    mu1present = [bool(re.search(x, text)) for x in [r'mu\('+re.escape(levels[0]) + r'( )*(&|,|(en))( )*' + re.escape(levels2[0])+'\)', 'mu\('+re.escape(levels[0])+'\)','mu\('+re.escape(levels2[0])+'\)','mu\(totaal\)']]
    mu2present = [bool(re.search(x, text)) for x in [r'mu\('+re.escape(levels[-1]) + r'( )*(&|,|(en))( )*' + re.escape(levels2[-1])+'\)','mu\('+re.escape(levels2[-1])+'\)','mu\(totaal\)']]
    scorepoints['mu1some'] = any(mu1present); scorepoints['mu1all'] = all(mu1present)
    scorepoints['mu2some'] = any(mu2present); scorepoints['mu2all'] = all(mu2present)
    scorepoints['mu1order'] = bool(re.search(r'mu\('+re.escape(levels[0])+r'( )*(&|,)( )*' +re.escape(levels2[0])+r'\)( )*\=( )*mu\('+re.escape(levels[0])+r'\)( )*\+( )*mu\('+re.escape(levels2[0])+r'\)( )*\-( )*mu\(totaal\)',text))
    scorepoints['mu2order'] = bool(re.search(r'mu\('+re.escape(levels[-1])+r'( )*(&|,)( )*'+re.escape(levels2[-1])+r'\)( )*\=( )*mu\('+re.escape(levels[-1])+r'\)( )*\+( )*mu\('+re.escape(levels2[-1])+r'\)( )*\-( )*mu\(totaal\)',text))  
    
    if False in list(scorepoints.values()):
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
        if not scorepoints['h0']:
            output += ' -het "H0():" teken<br>'
        if not scorepoints['mu1some'] and not scorepoints['mu1all']:
            output += ' -enkele tekens voor populatiegemiddelden bij de eerste vergelijking<br>'
        elif not scorepoints['mu1all']:
            output += ' -alle tekens voor populatiegemiddelden bij de eerste vergelijking<br>'
        if not scorepoints['mu2some'] and not scorepoints['mu2all']:
            output += ' -enkele tekens voor populatiegemiddelden bij de laatste vergelijking<br>'
        elif not scorepoints['mu2all']:
            output += ' -alle tekens voor populatiegemiddelden bij de laatste vergelijking<br>'
        if not scorepoints['mu1order']:
            output += ' -de populatiegemiddelden worden niet juist met elkaar vergeleken bij de eerste vergelijking<br>'
        if not scorepoints['mu2order']:
            output += ' -de populatiegemiddelden worden niet juist met elkaar vergeleken bij de laatste vergelijking<br>'
        return True, output
    else:
        return False, 'Mooi, deze hypothese klopt. '

#Between-person hypothesis for within-subject ANOVA
def scan_hypothesis_rmanova(text: str, solution: Dict) -> [bool, str]:
    #TODO: ADD INTERACTION HYPOTHESIS
    tokens: List[str] = nltk.word_tokenize(text.lower())
    scorepoints: Dict[str, bool] = {'score': 'score' in tokens or 'scores' in tokens,
                   'equal': 'gelijk' in tokens or 'gelijke' in tokens,
                   'jacked': 'opgevoerde' in tokens, 
                   'pop': 'populatie' in tokens}
    
    if False in list(scorepoints.values()):
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
        if not scorepoints['score']:
            output += ' -niet gesteld dat het om de scores van de personen gaat<br>'
        if not scorepoints['equal']:
            output += ' -niet gesteld dat de personen gelijke scores hebben<br>'
        if not scorepoints['jacked']:
            output += ' -niet gesteld dat het om de opgevoerde metingen gaat<br>'
        if not scorepoints['pop']:
            output += ' -er wordt niet aangegeven dat het om de populatie gaat<br>'
        return True, output
    else:
        return False, 'Mooi, deze hypothese klopt.'

def scan_table_ttest(textfields: Dict, solution: Dict, margin:float=0.01) -> [bool, str]:
    if type(textfields) != dict:
        exit()
        print('Wrong data sent to table scan function')
    #Set all empty fields to zero to prevent type errors
    for t in list(textfields.items()):
        if textfields[t[0]] == '':
            textfields[t[0]] = 0.0
        else:
            if textfields[t[0]] == None:
                textfields[t[0]] = ''
            elif textfields[t[0]].replace('.','').isdigit():
                textfields[t[0]] = float(textfields[t[0]])
            else: 
                textfields[t[0]] = 0.0
            
    #Compare input with gold standard
    meaninput :List = [textfields['mean' + str(i+1)] for i in range(len(solution['means']))]
    stdinput :List  = [textfields['std' + str(i+1)] for i in range(len(solution['stds']))]
    ninput :List  = [textfields['n' + str(i+1)] for i in range(len(solution['ns']))]
    scorepoints :Dict = {'mean': sim(solution['means'], meaninput, margin),
                   'std': sim(solution['stds'], stdinput, margin),
                   'n': sim(solution['ns'], ninput, margin),
                   }
    tags = ['eerste', 'tweede']
    if False in [all(x) for x in list(scorepoints.values())]:
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
        if not any(scorepoints['mean']):
            output += ' -alle juiste gemiddelden<br>'
        elif not all(scorepoints['mean']):    
            output += ' -één juiste gemiddelde, namelijk de '+[tags[i] for i in range(2) if not scorepoints['mean'][i]][0]+' van boven<br>'
        if not any(scorepoints['std']):
            output += ' -alle juiste standaardeviaties<br>'
        elif not all(scorepoints['std']):    
            output += ' -één juiste standaardeviatie, namelijk de '+[tags[i] for i in range(2) if not scorepoints['std'][i]][0]+' van boven<br>'
        if not any(scorepoints['n']):
            output += ' -alle juiste waarden van N<br>'
        elif not all(scorepoints['n']):    
            output += ' -één juiste waarde van N, namelijk de '+[tags[i] for i in range(2) if not scorepoints['n'][i]][0]+' van boven<br>'
        return True, output
    else:
        return False, 'Mooi, deze tabel klopt. '
    
def sim_p(solution: Dict, texts: List[str], margin: float=0.01) -> [bool, str]:
    output:list[bool] = []
    for i in range(len(texts)):
        text = texts[i]
        tokens: List[str] = nltk.word_tokenize(str(text).lower())
        numbers: List[float] = []
        for t in tokens:
            if t.replace('.','').replace('-','').isdigit():
                numbers.append(float(t))
        gold = solution['p'][i] 
        right_number: bool = [n for n in numbers if gold - margin < n and n < gold + margin] != []
        boundary_100: bool = '0.10' in tokens and round(gold, 2) != 0.01
        boundary_050: bool = '0.05' in tokens and round(gold, 2) != 0.05
        boundary_010: bool = '0.01' in tokens and round(gold, 2) != 0.05
        boundary_005: bool = '0.005' in tokens and round(gold, 2) != 0.05
        boundary_001: bool = '0.001' in tokens and round(gold, 2) != 0.05
        
        if numbers != []:
            if right_number and len(numbers) == len(tokens):#Als er alleen cijfers in het veld staan
                output.append(True)
            elif any([x in tokens for x in ['<','minder','kleiner']]):
                if boundary_100 and gold < 0.10 and gold > 0.05:
                    output.append(True)
                elif boundary_050 and gold < 0.05 and gold > 0.01:
                    output.append(True)
                elif boundary_010 and gold < 0.01 and gold > 0.005:
                    output.append(True)
                elif boundary_005 and gold < 0.005 and gold > 0.001:
                    output.append(True)
                elif boundary_001 and gold < 0.001:
                    output.append(True)
                else: output.append(False)
            elif any([x in tokens for x in ['>','meer','groter']]):
                if boundary_100 and gold > 0.10:
                    output.append(True)
                elif boundary_050 and gold > 0.05 and gold < 0.10:
                    output.append(True)
                elif boundary_010 and gold > 0.01 and gold < 0.05:
                    output.append(True)
                elif boundary_005 and gold > 0.005 and gold < 0.01:
                    output.append(True)
                elif boundary_001 and gold > 0.001 and gold < 0.05:
                    output.append(True)
                else: output.append(False)
            else: 
                output.append(False)
        else:
            output.append(False)
    return output
    
def sim(gold_numbers :List, numbers :List, margin:float) -> True: #Return true if there is a similar number to num in the given list/float/integer in the solution
    return [gold_numbers[i] - margin < numbers[i] and numbers[i] < gold_numbers[i] + margin for i in range(len(gold_numbers))]

def scan_table(textfields: Dict, solution: Dict, margin:float=0.01) -> [bool, str]:
    #Convert fieldinput to float
    if type(textfields) != dict:
        exit()
        print('Wrong data sent to table scan function')
    #Set all empty fields to zero to prevent type errors
    for t in list(textfields.items()):
        if textfields[t[0]] == '' and t[0][0] != 'p':
            textfields[t[0]] = 0.0
        else:
            if textfields[t[0]] == None:
                textfields[t[0]] = ''
            elif textfields[t[0]].replace('.','').isdigit():
                textfields[t[0]] = float(textfields[t[0]])
            elif t[0][0] == 'p':
                pass
            else: 
                textfields[t[0]] = 0.0
    
    #Compare input with gold standard
    dfinput :List = [textfields['df' + str(i+1)] for i in range(len(solution['df']))]
    ssinput :List  = [textfields['ss' + str(i+1)] for i in range(len(solution['ss']))]
    msinput :List  = [textfields['ms' + str(i+1)] for i in range(len(solution['ms']))]
    finput :List  = [textfields['f' + str(i+1)] for i in range(len(solution['F']))]
    pinput :List  = [textfields['p' + str(i+1)] for i in range(len(solution['p']))]
    r2input :List  = [textfields['r2' + str(i+1)] for i in range(len(solution['r2']))]
    scorepoints :Dict = {'df': sim(solution['df'], dfinput, margin),
                   'ss': sim(solution['ss'], ssinput, margin),
                   'ms': sim(solution['ms'], msinput, margin),
                   'F':sim(solution['F'], finput, margin),
                   'p': sim_p(solution, pinput, margin),
                   'r2': sim(solution['r2'], r2input, margin)
                   }
    nametags:list = ['eerste','tweede','derde','vierde','vijfde','zesde']
    if False in [all(x) for x in list(scorepoints.values())]:
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
        if not any(scorepoints['df']):
            output += ' -alle juiste waarden van df<br>'
        elif not all(scorepoints['df']):
            output += ' -ten minste één waarde van df, waaronder de '+' en '.join([nametags[i] for i in range(len(scorepoints['df'])) if not scorepoints['df'][i]])+' van boven<br>'
        if not any(scorepoints['ss']):
            output += ' -alle juiste waarden van ss<br>'
        elif not all(scorepoints['ss']):
            output += ' -ten minste één waarde van ss, waaronder de '+' en '.join([nametags[i] for i in range(len(scorepoints['ss'])) if not scorepoints['ss'][i]])+' van boven<br>'
        if not any(scorepoints['ms']):
            output += ' -alle juiste waarden van ms<br>'
        elif not all(scorepoints['ms']):
            output += ' -ten minste één waarde van ms, waaronder de '+' en '.join([nametags[i] for i in range(len(scorepoints['ms'])) if not scorepoints['ms'][i]])+' van boven<br>'
        if not any(scorepoints['F']):
            output += ' -alle juiste waarden van F<br>'
        elif not all(scorepoints['F']):
            output += ' -ten minste één waarde van F, waaronder de '+' en '.join([nametags[i] for i in range(len(scorepoints['F'])) if not scorepoints['F'][i]])+' van boven<br>'
        if not any(scorepoints['p']):
            output += ' -alle juiste waarden van p<br>'
        elif not all(scorepoints['p']):
            output += ' -ten minste één waarde van p, waaronder de '+' en '.join([nametags[i] for i in range(len(scorepoints['p'])) if not scorepoints['p'][i]])+' van boven<br>'
        if not any(scorepoints['r2']):
            output += ' -alle juiste waarden van R<sup>2</sup><br>'
        elif not all(scorepoints['r2']):
            output += ' --ten minste één waarde van R<sup>2</sup><br>, waaronder de '+' en '.join([nametags[i] for i in range(len(scorepoints['r2'])) if not scorepoints['r2'][i]])+' van boven<br>'
        return True, output
    else:
        return False, 'Mooi, deze tabel klopt. '

#Flexible scan function that checks the given answer for a list of keywords. These are given as a dictionary so
#that they can be dynamically defined in the protocol function in interface.py where this function is called
#If the given value for a keyword in keywords is false, this keyword is ignored and will no longer affect
#the given feedback
def scan_custom(text: str, solution: Dict, keywords: dict):
    tokens: List[str] = nltk.word_tokenize(text.lower().replace('.',''))
    scorepoints = dict([(key, key in tokens) for key in list(keywords.keys()) if keywords[key]])
    
    if False in list(scorepoints.values()):
        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
        for key in list(scorepoints.keys()):
            if not scorepoints[key]:
                output += ' -de juist waarde van ' + key + ' wordt niet genoemd<br>'
        
        return True, output
    else:
        return False, 'Mooi, dit antwoord klopt. '

