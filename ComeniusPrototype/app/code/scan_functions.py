#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:28:37 2021

@author: jelmer

STANDARD FUNCTIONS:
"""
import math
import random
import numpy as np
import nltk
import spacy
import re
from nltk import edit_distance
from copy import copy
from spacy.tokens.token import Token
from spacy.tokens.doc import Doc
from nltk import CFG, Tree
from scipy import stats
from typing import Dict, List, Tuple
from copy import copy
from app.code.assignments import Variable

"""
SCAN CLASS, THROUGH WHICH ALL SCAN FUNCTIONS CAN ACCESS THE LANGUAGE DICTIONARY "mes"
"""
class ScanFunctions:
    def __init__(self, mes:dict=None):
        self.mes = mes
    
    def set_messages(self, mes:dict):
        self.mes = mes
    
    """
    SCAN FUNCTIONS (WITH STRING INPUT)
    """
    def scan_yesno(self, input_text : str) -> [bool, bool]:
        return False, parse_yes_no(input_text)
    
    def scan_dummy(self, input_Text : str)  -> [bool, str]:
        return False, ''
    
    def scan_indep(self, text :str, solution :Dict) -> [bool, str]:
        #Determine which of the necessary elements are present in the answer
        texts: List[str] = nltk.word_tokenize(text)
        altmarkers = ['qualitative'] if self.mes['L_ENGLISH'] else ['kwalitatieve','kwalitatief']
        scorepoints: Dict[str, bool] = {'var': lef(solution['independent'].get_all_syns(),texts),#any([x in text for x in solution['independent'].get_all_syns()]), 
                         'measure': any([x in text for x in altmarkers]), 
                         'level1': any([x in text for x in solution['independent'].get_all_level_syns()[0]]), 
                         'level2': any([x in text for x in solution['independent'].get_all_level_syns()[1]])}
        
        #Determine the response of the chatbot
        if False in list(scorepoints.values()):
            output: str = self.mes['F_INCOMPLETE']+'<br>' #'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
            if not scorepoints['var']:
                output += self.mes['F_IND']+'<br>' #' -de juiste onafhankelijke variabele<br>'
            if not scorepoints['measure']:
                output += self.mes['F_MEASURE']+'<br>' #' -het juiste meetniveau<br>'
            if scorepoints['level1'] != scorepoints['level2']:
                output += self.mes['F_1LEVEL']+'<br>' #' -één niveau van de onafhankelijke variabele<br>'
            elif not scorepoints['level1'] and not scorepoints['level2']:
                output += self.mes['F_2LEVEL']+'<br>' #' -beide niveaus van de onafhankelijke variabele<br>'
            return True, output
        else:
            return False, self.mes['F_CORRECT'] #'Mooi, deze beschrijving klopt. '
        
    def scan_indep_anova(self, text: str, solution: Dict, num: int=1, between_subject=True) -> [bool, str]:
        #Determine which of the necessary elements are present in the answer
        texts: List[str] = nltk.word_tokenize(text.lower())
        n_key: str = 'independent' if num == 1 else 'independent' + str(num)
        scorepoints: Dict = {'factor': any(x in text for x in ['factor', 'between-subjectfactor','within-subjectfactor']), 
                       'domain': any(x in text for x in ['between','between-subject', 'between-subjectfactor']) if between_subject 
                       else any(x in text for x in ['within','within-subject', 'within-subjectfactor']), 
                       'name': lef(solution[n_key].get_all_syns(),texts), 
                       'levels': [any([x in text for x in y]) for y in solution[n_key].get_all_level_syns()]
                       }
        #Determine the response of the chatbot
        if False in list(scorepoints.values()):
            output: str = self.mes['F_INCOMPLETE']+'<br>' #'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
            if not scorepoints['factor']:
                output += self.mes['F_ISFACTOR']+'<br>' #' -de uitspraak dat deze variabele een factor is<br>'
            if not scorepoints['domain']:
                output += self.mes['F_DOMAIN']+'<br>' #' -het domein van de variabele<br>'
            if not scorepoints['name']:
                output += self.mes['F_VARNAME']+'<br>' #' -de naam van de variabele<br>'
            if True not in scorepoints['levels']:
                output += self.mes['F_INDEPLEVELS']+'<br>' #' -alle niveaus van de onafhankelijke variabele<br>'
            elif False in scorepoints['levels']:
                output += self.mes['F_INDEPLEVEL']+'<br>' #' -enkele niveaus van de onafhankelijke variabele<br>'
            return True, output
        else:
            return False, self.mes['F_CORRECT'] #'Mooi, deze beschrijving klopt. '
    
    def scan_dep(self, text: str, solution: Dict) -> [bool, str]:
        #Determine which of the necessary elements are present in the answer
        texts: List[str] = nltk.word_tokenize(text.lower())
        altmarkers = ['quantitative'] if self.mes['L_ENGLISH'] else ['kwantitatieve','kwantitatief']
        scorepoints: Dict[str, bool] = {'var': lef(solution['dependent'].get_all_syns(),texts),#any([x in text for x in solution['dependent'].get_all_syns()]), 
                                     'measure': any([x in text for x in altmarkers])}
        
        #Determine the response of the chatbot
        if False in list(scorepoints.values()):
            output: str = self.mes['F_INCOMPLETE']+'<br>' #'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
            if not scorepoints['var']:
                output += self.mes['F_DEP']+'<br>' #' -de juiste afhankelijke variabele<br>'
            if not scorepoints['measure']:
                output += self.mes['F_MEASURE']+'<br>' #' -het juiste meetniveau<br>'
            return True, output
        else:
            return False, self.mes['F_CORRECT'] #'Mooi, deze beschrijving klopt. '
        
    def scan_control(self, text: str, solution: Dict, num:int=1) -> [bool, str]:
        tokens: List[str] = nltk.word_tokenize(text.lower())
        control: float = solution['control'] if num < 2 else solution['control'+str(num)]
        n_negations: int = negation_counter(tokens)
        scorepoints: Dict[str, bool] = {'experiment': 'experiment' in tokens, 
                         'negations': control != bool(n_negations % 2) if 'experiment' in tokens else True}
        keywords = [['experiment','experimenteel'],['passief','passief-observerend','passief-geobserveerd']] if not\
                                    self.mes['L_ENGLISH'] else [['experiment','experimental'],['passive','passive-observing']]
        if any([x in tokens for x in keywords[0]]):
            if control != bool(n_negations % 2):
                return False, self.mes['F_CORRECT']
            else:
                if control:
                    return True, self.mes['F_INCOMPLETE']+'<br>'+self.mes['F_NOX']
                else:
                    return True, self.mes['F_INCOMPLETE']+'<br>'+self.mes['F_EX']
        elif any([x in tokens for x in keywords[1]]): #and 'observerend' in tokens):
            if control == bool(n_negations % 2):
                return False, self.mes['F_CORRECT']
            else:
                if control:
                    return True, self.mes['F_INCOMPLETE']+'<br>'+self.mes['F_PAS']
                else:
                    return True, self.mes['F_INCOMPLETE']+'<br>'+self.mes['F_NPAS']
        else:
            return True, self.mes['F_INCOMPLETE']+'<br>'+self.mes['F_X']
            
        #Determine the response of the program
        if False in list(scorepoints.values()):
            output: str = self.mes['F_INCOMPLETE']+'<br>' #'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
            if not scorepoints['experiment']:
                output += self.mes['F_X']+'<br>' #' -de uitspraak of het onderzoek een experiment is of niet<br>'
            if not scorepoints['negations'] and control:
                output += self.mes['F_NOX']+'<br>' #' -je hebt ten onrechte gesteld dat het onderzoek geen experiment is<br>'
            if not scorepoints['negations'] and not control:
                output += self.mes['F_EX']+'<br>' #' -je hebt ten onrechte gesteld dat het onderzoek een experiment is<br>'
            return True, output
        else:
            return False, self.mes['F_CORRECT']
    
    def scan_numbers(self, text: str, stat: str, solution: Dict, margin: float) -> [bool, str]:
        tokens: List[str] = nltk.word_tokenize(text.lower())
        numbers: List[float] = []
        for t in tokens:
            try:
                numbers.append(float(t))
            except ValueError:
                pass
        
        #Define naming dictionary
        if self.mes['L_ENGLISH']:
            fancynames: Dict[str, str] = {'df':'the degrees of freedom (df)', 'raw_effect': 'the raw effect',
                      'T':'T', 'p':'p', 'relative_effect': 'the relative effect', 'F': 'F','r2':'the correlation','ns':'N'}
        else:
            fancynames: Dict[str, str] = {'df':'het aantal vrijheidsgraden', 'raw_effect': 'het ruwe effect',
                      'T':'T', 'p':'p', 'relative_effect': 'het relatieve effect', 'F': 'F','r2':'de correlatie','ns':'N'}
        
        #Compare the received numbers to the gold standards from the solution
        goldnums: List[float] = solution[stat]
        nums: List[List[float]] = [[n for n in numbers if goldnums[i] - margin < n and n < goldnums[i] + margin] for i in range(len(goldnums))]
        scorepoints: List[int] = [1 if num !=[] else 0 for num in nums] #Binary representation for each number in the solution whether it is present in the answer
        
        #Determine the response of the chatbot
        if 0 in scorepoints:
            output: str = self.mes['F_INCOMPLETE']+'<br>'
            if 1 not in scorepoints:
                output += self.mes['F_ALLVALS'] + fancynames(stat) + self.mes['S_LACKING']+'<br>' #Alle juiste waarden ontbreken
            else:
                output += self.mes['F_ONEVAL'] + fancynames(stat) + self.mes['S_LACKING']+'<br>' #Sommige juiste waarden ontbreken
            return True, output
        else:
            return False, self.mes['F_CSCORRECT']
        
    def scan_number(self, text: str, stat: str, solution: Dict, margin: float=0.01) -> [bool, str]:
        if self.mes['L_ENGLISH']:
            fancynames: Dict[str, str] = {'df':'the degrees of freedom (df)', 'raw_effect': 'the raw effect',
                      'T':'T', 'p':'p', 'relative_effect': 'the relative effect', 'F': 'F','r2':'the correlation','ns':'N'}
        else:
            fancynames: Dict[str, str] = {'df':'het aantal vrijheidsgraden', 'raw_effect': 'het ruwe effect',
                      'T':'T', 'p':'p', 'relative_effect': 'het relatieve effect', 'F': 'F','r2':'de correlatie','ns':'N'}
        tokens: List[str] = nltk.word_tokenize(text.lower())
        numbers: List[float] = []
        for t in tokens:
            try:
                numbers.append(float(t))
            except ValueError:
                output: str = self.mes['F_INCOMPLETE']+'<br> -'+\
                    self.mes['S_RIGHTVALUE'] + fancynames[stat] + '<br>'
                return True, output
                
        #Compare the received numbers to the gold standard from the solution
        gold: float = solution[stat][0]
        right_number: List[bool] = [n for n in numbers if gold - margin < n and n < gold + margin]
        
        #Determine the response of the chatbot
        if right_number == []:
            output: str = self.mes['F_INCOMPLETE']+'<br> -'+\
                self.mes['S_RIGHTVALUE'] + fancynames[stat] + '<br>'
            return True, output
        else:
            return False, self.mes['F_CCORRECT']
        
    def scan_p(self, text:str, solution: Dict, margin: float=0.01) -> [bool, str]:
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
                return False, self.mes['F_PCORRECT']
            elif any([x in tokens for x in ['<','minder','kleiner']]):
                if boundary_100 and gold < 0.10 and gold > 0.05:
                    return False, self.mes['F_PCORRECT']
                elif boundary_050 and gold < 0.05 and gold > 0.01:
                    return False, self.mes['F_PCORRECT']
                elif boundary_010 and gold < 0.01 and gold > 0.005:
                    return False, self.mes['F_PCORRECT']
                elif boundary_005 and gold < 0.005 and gold > 0.001:
                    return False, self.mes['F_PCORRECT']
                elif boundary_001 and gold < 0.001:
                    return False, self.mes['F_PCORRECT']
                else: return True, self.mes['F_INCOMPLETE']+'<br>'+self.mes['F_NOPHERE']
            elif any([x in tokens for x in ['>','meer','groter']]):
                if boundary_100 and gold > 0.10:
                    return False,self.mes['F_PCORRECT']
                elif boundary_050 and gold > 0.05 and gold < 0.10:
                    return False, self.mes['F_PCORRECT']
                elif boundary_010 and gold > 0.01 and gold < 0.05:
                    return False, self.mes['F_PCORRECT']
                elif boundary_005 and gold > 0.005 and gold < 0.01:
                    return False, self.mes['F_PCORRECT']
                elif boundary_001 and gold > 0.001 and gold < 0.005:
                    return False, self.mes['F_PCORRECT']
                else: return True, self.mes['F_INCOMPLETE']+'<br>'+self.mes['F_NOPHERE']
            else: 
                return True, self.mes['F_INCOMPLETE']+'<br>'+self.mes['F_NOPHERE']
        else:
            return True, self.mes['F_INCOMPLETE']+'<br>'+self.mes['F_NOPHERE']
            
    def scan_hypothesis(self, text: str, solution: Dict, num: int=1) -> [bool, str]:
        #Remove potential dots to avoid confusion
        i_key: str = 'independent' if num < 2 else 'independent' + str(num)
        sign:str = ['==','<=','>='][solution['hypothesis']] if 'hypothesis' in list(solution.keys()) else '=='
        tokens: List[str] = text.lower().replace('.','').split() #nltk.word_tokenize(text.lower())
        mus: List[bool] = [lef(['mu('+y+')' for y in x],tokens) for x in solution[i_key].get_all_level_syns()]#[any(['mu('+x+')' in text for x in y]) for y in solution[i_key].get_all_level_syns()]
        scorepoints: Dict[str, bool] = {'H0': 'h0:' in tokens or 'h0' in tokens,
                       'sign': sign in tokens, 
                       'order': True}
        if sign == '==' and '=' in tokens:
            scorepoints['sign'] = True
        
        if False in list(scorepoints.values()) + mus:
            output: str = self.mes['F_INCOMPLETE']+'<br>'
            if not scorepoints['H0']:
                output += self.mes['F_H0SIGN']+'<br>'
            if not scorepoints['sign']:
                output += self.mes['F_COMPARISON']+'<br>'
            if not scorepoints['order']:
                output += self.mes['F_ORDER']+'<br>'
            if True not in mus:
                output += self.mes['F_AVGSMULTI']+'<br>'
            elif False in mus:
                output += self.mes['F_AVGSONE']+'<br>'
            return True, output
        else:
            return False, self.mes['F_HYPCORRECT']
    
    #Interaction hypothesis for two-way ANOVA
    def scan_hypothesis_anova(self, text: str, solution: Dict) -> [bool, str]:
        #Remove potential dots to avoid confusion
        levels = solution['independent'].levels; levels2 = solution['independent2'].levels
        markers:list = ['and','total'] if self.mes['L_ENGLISH'] else ['en','totaal'] #Dutch and English words for the different versions
        criteria:list = ['h0', 'mu1some','mu1all','mu2some','mu2all']
        scorepoints = dict([(x,False) for x in criteria])
        scorepoints['h0'] = bool(re.search(r'h0\(' + re.escape(str(solution['independent'].name)) + r'( )*(x|\*|('+re.escape(markers[0])+'))( )*' + re.escape(str(solution['independent2'].name)) + r'\)', text))
        mu1present = [bool(re.search(x, text)) for x in [r'mu\('+re.escape(levels[0]) + r'( )*(&|,|('+re.escape(markers[0])+'))( )*' + re.escape(levels2[0])+r'\)', r'mu\('+re.escape(levels[0])+r'\)',r'mu\('+re.escape(levels2[0])+r'\)',r'mu\('+re.escape(markers[1])+'\)']]
        mu2present = [bool(re.search(x, text)) for x in [r'mu\('+re.escape(levels[-1]) + r'( )*(&|,|('+re.escape(markers[0])+'))( )*' + re.escape(levels2[-1])+r'\)',r'mu\('+re.escape(levels2[-1])+r'\)',r'mu\('+re.escape(markers[1])+'\)']]
        scorepoints['mu1some'] = any(mu1present); scorepoints['mu1all'] = all(mu1present)
        scorepoints['mu2some'] = any(mu2present); scorepoints['mu2all'] = all(mu2present)
        scorepoints['mu1order'] = bool(re.search(r'mu\('+re.escape(levels[0])+r'( )*(&|,)( )*' +re.escape(levels2[0])+r'\)( )*\=( )*mu\('+re.escape(levels[0])+r'\)( )*\+( )*mu\('+re.escape(levels2[0])+r'\)( )*\-( )*mu\('+re.escape(markers[1])+'\)',text))
        scorepoints['mu2order'] = bool(re.search(r'mu\('+re.escape(levels[-1])+r'( )*(&|,)( )*'+re.escape(levels2[-1])+r'\)( )*\=( )*mu\('+re.escape(levels[-1])+r'\)( )*\+( )*mu\('+re.escape(levels2[-1])+r'\)( )*\-( )*mu\('+re.escape(markers[1])+'\)',text))  
        
        if False in list(scorepoints.values()):
            output: str = self.mes['F_INCOMPLETE']+'<br>'
            if not scorepoints['h0']:
                output += self.mes['F_H0SIGN']+'<br>'
            if not scorepoints['mu1some'] and not scorepoints['mu1all']:
                output += self.mes['F_ALLSIGNS'] + self.mes['S_COMP1'] + '<br>' #' -alle tekens voor populatiegemiddelden bij de eerste vergelijking<br>'
            elif not scorepoints['mu1all']:
                output += self.mes['F_SOMESIGNS'] + self.mes['S_COMP1'] + '<br>' #' -enkele tekens voor populatiegemiddelden bij de eerste vergelijking<br>'
            if not scorepoints['mu2some'] and not scorepoints['mu2all']:
                output += self.mes['F_ALLSIGNS'] + self.mes['S_COMP2'] + '<br>' #' -alle tekens voor populatiegemiddelden bij de laatste vergelijking<br>'
            elif not scorepoints['mu2all']:
                output += self.mes['F_SOMESIGNS'] + self.mes['S_COMP2'] + '<br>' #' -enkele tekens voor populatiegemiddelden bij de laatste vergelijking<br>'
            if not scorepoints['mu1order']:
                output += self.mes['F_WRONGCOMP1'] + '<br>' #' -de populatiegemiddelden worden niet juist met elkaar vergeleken bij de eerste vergelijking<br>'
            if not scorepoints['mu2order']:
                output += self.mes['F_WRONGCOMP2'] + '<br>' #' -de populatiegemiddelden worden niet juist met elkaar vergeleken bij de laatste vergelijking<br>'
            return True, output
        else:
            return False, self.mes['F_HYPCORRECT']
    
    #Between-person hypothesis for within-subject ANOVA
    def scan_hypothesis_rmanova(self, text: str, solution: Dict) -> [bool, str]:
        #TODO: ADD INTERACTION HYPOTHESIS
        tokens: List[str] = nltk.word_tokenize(text.lower())
        scorepoints: Dict[str, bool] = {'H0': 'h0:' in text or 'h0' in text,
                       'equal': '==' in tokens or '=' in tokens,
                       'tau1': 'tau(subject 1)' in text, 
                       'tau2': 'tau(subject 2)' in text,
                       'taun': 'tau(subject '+str(solution['n_subjects'])+')' in text}
        
        if False in list(scorepoints.values()):
            output: str = self.mes['F_INCOMPLETE']+'<br>' #'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
            if not scorepoints['H0']:
                output += self.mes['F_H0SIGN']+'<br>' #' -H0 teken niet genoemd<br>'
            if not scorepoints['equal']:
                output += self.mes['F_COMPARISON']+'<br>'#' -Het juiste vergelijkingsteken<br>'
            if not scorepoints['tau1']:
                output += self.mes['F_TAU1']+'<br>'
            if not scorepoints['tau2']:
                output += self.mes['F_TAU2']+'<br>'
            if not scorepoints['taun']:
                output += self.mes['F_TAUN']+'<br>'
            return True, output
        else:
            return False, self.mes['F_HYPCORRECT']
    
    def scan_table_ttest(self, textfields: Dict, solution: Dict, margin:float=0.01) -> [bool, str]:
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
        tags = ['first', 'second'] if self.mes['L_ENGLISH'] else ['eerste', 'tweede']
        if False in [all(x) for x in list(scorepoints.values())]:
            output: str = self.mes['F_INCOMPLETE']+'<br>'
            if not any(scorepoints['mean']):
                output += self.mes['F_ALLAVG']+'<br>'
            elif not all(scorepoints['mean']):    
                output += self.mes['F_ONEAVG']+[tags[i] for i in range(2) if not scorepoints['mean'][i]][0]+self.mes['F_FROMABOVE']+'<br>'
            if not any(scorepoints['std']):
                output += self.mes['F_ALLSTD']+'<br>'
            elif not all(scorepoints['std']):    
                output += self.mes['F_ONESTD']+[tags[i] for i in range(2) if not scorepoints['std'][i]][0]+self.mes['F_FROMABOVE']+'<br>'
            if not any(scorepoints['n']):
                output += self.mes['F_ALLN']+'<br>'
            elif not all(scorepoints['n']):    
                output += self.mes['F_ONEN']++[tags[i] for i in range(2) if not scorepoints['n'][i]][0]+self.mes['F_FROMABOVE']+'<br>'
            return True, output
        else:
            return False, self.mes['F_TABLECORRECT']
    
    def scan_table(self, textfields: Dict, solution: Dict, margin:float=0.01) -> [bool, str]:
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
        nametags:list = ['first','second','third','vierde','fifth','sixth'] if self.mes['L_ENGLISH'] else ['eerste','tweede','derde','vierde','vijfde','zesde']
        if False in [all(x) for x in list(scorepoints.values())]:
            output: str = self.mes['F_INCOMPLETE']+'<br>'
            if not any(scorepoints['df']):
                output += self.mes['F_ALLVALS']+'df<br>'
            elif not all(scorepoints['df']):
                output += self.mes['F_ONEPLUS']+'df'+self.mes['S_INCL']+self.mes['S_AND'].join([nametags[i] for i in range(len(scorepoints['df'])) if not scorepoints['df'][i]])+self.mes['S_FROMABOVE']+'<br>'
            if not any(scorepoints['ss']):
                output += self.mes['F_ALLVALS']+'ss<br>'
            elif not all(scorepoints['ss']):
                output += self.mes['F_ONEPLUS']+'ss'+self.mes['S_INCL']+self.mes['S_AND'].join([nametags[i] for i in range(len(scorepoints['ss'])) if not scorepoints['ss'][i]])+self.mes['S_FROMABOVE']+'<br>'
            if not any(scorepoints['ms']):
                output += self.mes['F_ALLVALS']+'ms<br>'
            elif not all(scorepoints['ms']):
                output += self.mes['F_ONEPLUS']+'ms'+self.mes['S_INCL']+self.mes['S_AND'].join([nametags[i] for i in range(len(scorepoints['ms'])) if not scorepoints['ms'][i]])+self.mes['S_FROMABOVE']+'<br>'
            if not any(scorepoints['F']):
                output += self.mes['F_ALLVALS']+'F<br>'
            elif not all(scorepoints['F']):
                output += self.mes['F_ONEPLUS']+'F'+self.mes['S_INCL']+self.mes['S_AND'].join([nametags[i] for i in range(len(scorepoints['F'])) if not scorepoints['F'][i]])+self.mes['S_FROMABOVE']+'<br>'
            if not any(scorepoints['p']):
                output += self.mes['F_ALLVALS']+'p<br>'
            elif not all(scorepoints['p']):
                output += self.mes['F_ONEPLUS']+'p'+self.mes['S_INCL']+self.mes['S_AND'].join([nametags[i] for i in range(len(scorepoints['p'])) if not scorepoints['p'][i]])+self.mes['S_FROMABOVE']+'<br>'
            if not any(scorepoints['r2']):
                output += self.mes['F_ALLVALS']+'R<sup>2</sup><br>'
            elif not all(scorepoints['r2']):
                output += self.mes['F_ONEPLUS']+'R<sup>2</sup>'+self.mes['S_INCL']+self.mes['S_AND'].join([nametags[i] for i in range(len(scorepoints['r2'])) if not scorepoints['r2'][i]])+self.mes['S_FROMABOVE']+'<br>'
            return True, output
        else:
            return False, self.mes['F_TABLECORRECT']
    
    #Flexible scan function that checks the given answer for a list of keywords. These are given as a dictionary so
    #that they can be dynamically defined in the protocol function in interface.py where this function is called
    #If the given value for a keyword in keywords is false, this keyword is ignored and will no longer affect
    #the given feedback
    #def scan_custom(self, text: str, solution: Dict, keywords: dict):
    #    tokens: List[str] = nltk.word_tokenize(text.lower().replace('.',''))
    #    scorepoints = dict([(key, key in tokens) for key in list(keywords.keys()) if keywords[key]])
    #    
    #    if False in list(scorepoints.values()):
    #        output: str = 'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
    #        for key in list(scorepoints.keys()):
    #            if not scorepoints[key]:
    #                output += ' -de juist waarde van ' + key + ' wordt niet genoemd<br>'
    #        
    #        return True, output
    #    else:
    #        return False, 'Mooi, dit antwoord klopt. '
    
    """
    FUNCTIONS FOR SCANNING CAUSAL INTERPRETATION AND DECISION
    
    THESE TAKE THE INPUT TEXT IN DOC INSTEAD OF STRING
    """
    
    def scan_decision(self, doc:Doc, solution:dict, anova:bool, num:int=1, prefix=True, elementair=True) -> [bool, List[str]]:
        output = [self.mes['F_INCOMPLETE']] if prefix else []
        if elementair:
            output.extend(self.detect_h0(doc, solution, num))
        else:
            output.extend(self.detect_significance(doc, solution, num))
        output.extend(self.detect_comparison(doc, solution, anova, num))
        vartag = 'independent' if num == 1 else 'independent2'
        if solution['p'][num - 1] < 0.05 and solution['assignment_type'] > 2: #self, sent:Doc, solution:dict, variable:str, p:float, eta:float) -> List[str]
            output.extend(self.detect_effect(doc, solution, solution[vartag].name, solution['p'][num-1], solution['r2'][num-1]))
        correct:bool = len(output) == 1 if prefix else output == []
        if correct:
            return False, self.mes['F_DECCORRECT'] if prefix else ''
        else:
            return True, '<br>'.join(output)
        
    def scan_decision_anova(self, doc:Doc, solution:dict, num:int=3, prefix=True, elementair=True) -> [bool, List[str]]:
        output = [self.mes['F_INCOMPLETE']] if prefix else []
        if elementair:
            output.extend(self.detect_h0(doc, solution, num))
        else:
            output.extend(self.detect_significance(doc, solution, num))
        output.extend(self.detect_interaction(doc, solution, True))
        vartag = 'the interaction' if self.mes['L_ENGLISH'] else 'de interactie'
        output.extend(self.detect_effect(doc, solution, vartag, solution['p'][num-1], solution['eta'][num-1]))
        correct:bool = len(output) == 1 if prefix else output == []
        if correct:
            return False, self.mes['F_DECCORRECT'] if prefix else ''
        else:
            return True, '<br>'.join(output)
        
    def scan_decision_rmanova(self, doc:Doc, solution:dict, num:int=2, prefix=True, elementair=True) -> [bool, List[str]]:
        output = [self.mes['F_INCOMPLETE']] if prefix else []
        if elementair:
            output.extend(self.detect_h0(doc, solution, num))
        else:
            output.extend(self.detect_significance(doc, solution, num))
        output.extend(self.detect_true_scores(doc, solution, 2))
        vartag = 'the subjects' if self.mes['L_ENGLISH'] else 'de subjecten'
        if solution['p'][1] < 0.05: 
            output.extend(self.detect_effect(doc, solution, vartag, solution['p'][num-1], solution['r2'][num-1]))
        correct:bool = len(output) == 1 if prefix else output == []
        if correct:
            return False, self.mes['F_DECCORRECT'] if prefix else ''
        else:
            return True, '<br>'.join(output)
        
    def scan_interpretation(self, doc:Doc, solution:dict, anova:bool, num:int=1, prefix=True):
        output = [self.mes['F_INCOMPLETE']] if prefix else []
        control: bool = solution['control'] if num < 2 else solution['control'+str(num)]
        if self.mes['L_ENGLISH']:
            primary_checks:list = ['primary','first'] if not control else solution['dependent'].get_all_syns()
            second_check:str = 'alternative'
            unk_checks:list = ['possible','interpretation','interpretations','']
        else:
            primary_checks:list = ['primaire','eerste'] if not control else solution['dependent'].get_all_syns()
            second_check:str = 'alternatieve'
            unk_checks:list = ['mogelijk','mogelijke','verklaring','verklaringen']
        
        unk_sents = [x for x in doc.sents if any([y in [z.text for z in x] for y in unk_checks])]
        if unk_sents != []:
            output.extend(self.detect_unk(unk_sents[0], solution, num))
        else:
            output.append(self.mes['F_MANEXP'])
        primair_sents = [x for x in doc.sents if any([z in x.text for z in primary_checks])]
        if primair_sents != []:
            output.extend(self.detect_primary(primair_sents[0], solution, num))
        else:
            output.append(self.mes['F_PRIMEXP'])
        if not control:
            alt_sents = [x for x in doc.sents if second_check in [y.text for y in x]]
            #displacy.serve(alt_sents[0])
            if alt_sents != []:
                output.extend(self.detect_alternative(alt_sents[0], solution, num))
            else:
                output.append(self.mes['F_ALTEXP'])
        correct:bool = len(output) == 1 if prefix else output == []
        if correct:
            return False, self.mes['F_INTCORRECT'] if prefix else ''
        else:
            return True, '<br>'.join(output)
        
    def scan_interpretation_anova(self, doc:Doc, solution:dict, num:int=3, prefix=True):
        output = [self.mes['F_INCOMPLETE']] if prefix else []
        control:bool = solution['control'] or solution['control2']
        if self.mes['L_ENGLISH']:
            primary_checks:list = ['primary','first'] if not control else solution['dependent'].get_all_syns()
            second_check:str = 'alternative'
            unk_checks:list = ['possible','interpretation','interpretations','']
        else:
            primary_checks:list = ['primaire','eerste'] if not control else solution['dependent'].get_all_syns()
            second_check:str = 'alternatieve'
            unk_checks:list = ['mogelijk','mogelijke','verklaring','verklaringen']
            
        unk_sents = [x for x in doc.sents if lef(unk_checks,[y for y in x])]
        if unk_sents != []:
            output.extend(self.detect_unk(unk_sents[0], solution))
        else:
            output.append(self.mes['F_MANEXP'])
        primair_sents = [x for x in doc.sents if lef(primary_checks,[y for y in x])]
        if primair_sents != []:
            output.extend(self.detect_primary_interaction(primair_sents[0], solution))
        else:
            output.append(self.mes['F_PRIMEXP'])
        # EXPLICIETE ALTERNATIEVE VERKLARINGEN HOEVEN NIET BIJ INTERACTIE, STATISMogelijke alternatieve verklaringen zijn storende variabelen en omgekeerde causaliteitTIEK VOOR DE PSYCHOLOGIE 3 PAGINA 80
        if not control:
            alt_sents = [x for x in doc.sents if lef(second_check,[y for y in x])]
            if alt_sents != []:
                output.extend(self.detect_alternative_interaction(alt_sents[0], solution))
            else:
                output.append(self.mes['F_MANALT'])
        correct:bool = len(output) == 1 if prefix else output == []
        if correct:
            return False, self.mes['F_INTCORRECT'] if prefix else ''
        else:
            return True, '<br>'.join(output)
    
    def scan_predictors(self, doc:Doc, solution:dict, prefix:bool=True):
        tokens = [x.text for x in doc]
        output = [self.mes['F_INCOMPLETE']] if prefix else []
        varnames = [x.lower() for x in solution['data']['predictoren'][1:]] if solution['assignment_type'] == 6 \
                        else [x.lower() for x in solution['data']['predictoren']]
        for x in varnames:
            if ' ' in x:
                names = x.split()
                if not all([y in tokens for y in names]):
                    output.append(' -predictor ' + x + self.mes['S_NONAME'])
            else:
                if not x in tokens:
                    output.append(' -predictor ' + x + self.mes['S_NONAME'])
        for i in range(len(varnames)):
            if varnames[0] == 'Intercept':
                index = i - 1
            else:
                index = i
            if solution['predictor_p'][i] < 0.05:
                output.extend(self.detect_p(doc, solution['predictor_p'][i], label=varnames[index]))
        correct:bool = len(output) == 1 if prefix else output == []
        if correct:
            return False, self.mes['F_INTCORRECT'] if prefix else ''
        else:
            return True, '<br>'.join(output)
    
    def scan_design(self, doc:Doc, solution:dict, prefix:bool=True) -> [bool, List[str]]:
        criteria = ['ind', 'indcorrect','ind2','ind2correct','dep','depcorrect','factor1','factor2']
        marker_ind = 'independent' if self.mes['L_ENGLISH'] else 'onafhankelijke'
        scorepoints = dict([(x,False) for x in criteria])
        if solution['assignment_type'] != 13:
            scorepoints['factor1'] = True;scorepoints['factor2'] = True
        output:List[str] = []
        factor_roles:list = ['independent','dependent'] if self.mes['L_ENGLISH'] else ['onafhankelijke', 'afhankelijke']
        
        indeps = [x for x in doc.sents if lef(solution['independent'].get_all_syns(),[y.text for y in x])]#if x.text == solution['independent']]
        if indeps != []:
            scorepoints['ind'] = True
            indep_span = indeps[0]
            scorepoints['indcorrect'] = factor_roles[0] in indep_span.text or 'factor' in indep_span.text
            if solution['assignment_type'] == 5 or solution['assignment_type'] == 13:
                scorepoints['factor1'] = 'within-subject' in indep_span.text or 'within' in indep_span.text
        if solution['assignment_type'] == 13 or solution['assignment_type'] == 4:    
            indeps2 = [x for x in doc.sents if lef(solution['independent2'].get_all_syns(),[y.text for y in x])]
            if indeps2 != []:
                scorepoints['ind2'] = True
                indep2_span = indeps2[0]
                scorepoints['ind2correct'] = factor_roles[0] in indep2_span.text or 'factor' in indep2_span.text 
                if solution['assignment_type'] == 13:
                    scorepoints['factor2'] = 'between-subject' in indep2_span.text or 'between' in indep2_span.text
        else:
            scorepoints['ind2'] = True;scorepoints['ind2correct'] = True
        deps = [x for x in doc.sents if lef(solution['dependent'].get_all_syns(),[y.text for y in x])]
        if deps != []:
            scorepoints['dep'] = True
            dep_span = deps[0]
            scorepoints['depcorrect'] = factor_roles[1] in dep_span.text and not marker_ind in dep_span.text
        
        #Add feedback text
        if not scorepoints['ind']:
            output.append(self.mes['F_IND'] + self.mes['S_INDES'])
        if not scorepoints['indcorrect'] and scorepoints['ind']:
            output.append(self.mes['F_INDROLE'] + self.mes['S_INDES'])
        if not scorepoints['ind2']:
            output.append(self.mes['F_IND2'] + self.mes['S_INDES'])
        if not scorepoints['ind2correct'] and scorepoints['ind2']:
            output.append(self.mes['F_IND2ROLE'] + self.mes['S_INDES'])
        if not scorepoints['dep']:
            output.append(self.mes['F_DEPROLE'] + self.mes['S_INDES'])
        if not scorepoints['factor1']:
            output.append(self.mes['F_WHATFACTOR1'])
        if not scorepoints['factor2']:
            output.append(self.mes['F_WHATFACTOR2'])
        if not scorepoints['depcorrect'] and scorepoints['dep']:
            output.append(self.mes['F_DEPROLE'] + self.mes['S_INDES'])
        if not False in list(scorepoints.values()):        
            return False, self.mes['F_NICEDES'] if prefix else ''
        else:
            return True, '<br>'.join(output)
        
    def scan_design_manova(self, doc:Doc, solution:dict, prefix:bool=True):
        text = doc.text
        factor_roles:list = ['independent','dependent'] if self.mes['L_ENGLISH'] else ['onafhankelijke', 'afhankelijke']
        scorepoints = {'indcorrect':False,
                       #'levels1':all([x in text for x in solution['levels']]),
                       'mes':all([any([y in text for y in solution[x].get_all_syns()]) for x in ['dependent','dependent2','dependent3']]),
                       'dep1':lef(solution['dependent'].get_all_syns(),[x.text for x in doc]),#distance(words, [factor_roles[1]], solution['dependent'].get_all_syns()) < distance(words, [factor_roles[0], 'factor'], solution['dependent'].get_all_syns()),
                       'dep2':lef(solution['dependent2'].get_all_syns(),[x.text for x in doc]),#distance(words, [factor_roles[1]], solution['dependent2'].get_all_syns()) < distance(words, [factor_roles[0], 'factor'], solution['dependent2'].get_all_syns()),
                       'dep3':lef(solution['dependent3'].get_all_syns(),[x.text for x in doc]),#distance(words, [factor_roles[1]], solution['dependent3'].get_all_syns()) < distance(words, [factor_roles[0], 'factor'], solution['dependent3'].get_all_syns()) 
                       }
        scorepoints['indcorrect'] = [x for x in doc.sents if lef(solution['independent'].get_all_syns(),[y.text for y in x])]
        any([True if lef(solution['independent'].get_all_syns(),[x.text for x in doc]) and ('factor' in sent.text \
                                        or factor_roles[0] in doc.text) else False for sent in doc.sents])
        
        output:List[str] = []
        if not scorepoints['dep1']:
            output.append(self.mes['F_DEP1'])
        if not scorepoints['dep2']:
            output.append(self.mes['F_DEP2'])
        if not scorepoints['dep3']:
            output.append(self.mes['F_DEP3'])
        if not scorepoints['indcorrect']:
            output.append(self.mes['F_IND'])
        elif not scorepoints['mes']:
            output.append(self.mes['F_ALLDEPS'])
        if not False in list(scorepoints.values()):
            return False, self.mes['F_NICEDES'] if prefix else ''
        else:
            return True, '<br>'.join(output)
    
    """
    SPLIT GRADE FUNCTIONS: THESE ARE CALLED ON THE INPUT FOR A SHORT REPORT
    """
    
    def split_grade_ttest(self, text: str, solution:dict, between_subject:bool) -> str:
        if self.mes['L_ENGLISH']:
            nl_nlp = spacy.load('en_core_web_sm') 
        else: 
            nl_nlp = spacy.load('nl_core_news_sm')
        doc = nl_nlp(text.lower())
        output:str = ''
        output += '<br>'+'<br>'.join(self.detect_name(doc,solution))
        output += '<br>' + self.scan_design(doc, solution, prefix=False)[1]
        #if solution['p'][0] < 0.05:
        output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'T', solution['T'][0], aliases=[')' + solution['independent'].name + ')']))
        output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'p', solution['p'][0]))
        output += '<br>' + self.scan_decision(doc, solution, anova=False, prefix=False, elementair=False)[1]
        if output.replace('<br>','') == '':
            return self.mes['F_NICEREP']
        else:
            return self.mes['F_INCOMPLETE'] + re.sub(r'<br>(<br>)+', '<br>', output)
        
    def split_grade_anova(self, text: str, solution:dict, two_way:bool) -> str:
        if self.mes['L_ENGLISH']:
            nl_nlp = spacy.load('en_core_web_sm') 
        else: 
            nl_nlp = spacy.load('nl_core_news_sm')
        doc = nl_nlp(text.lower())
        markers = ['squared','interaction'] if self.mes['L_ENGLISH'] else ['kwadraat','interactie']
        
        output:str = ''
        output += '<br>'+'<br>'.join(self.detect_name(doc,solution))
        output += '<br>' + self.scan_design(doc, solution, prefix=False)[1]
        if not two_way:
            if solution['p'][0] < 0.05:
                output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'F', solution['F'][0], aliases=['F(' + solution['independent'].name + ')']))
                output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'p', solution['p'][0]))
                output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'R<sup>2</sup>', solution['r2'][0], aliases=['r2','r^2','r',markers[0]]))
            output += '<br>' + self.scan_decision(doc, solution, anova=True, prefix=False, elementair=False)[1]
        else:
            for i in range(3):
                if solution['p'][i] < 0.05:
                    f_aliases = ['F(' + solution['independent' + str(i+1)].name + ')'] if i > 0 and i < 2 else []
                    output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'F', solution['F'][i], aliases=f_aliases, num=i+1))
                    output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'p', solution['p'][i], num=i+1))
                    output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'R<sup>2</sup>', solution['r2'][i], aliases=['r2','r^2','r',markers[0]], num=i+1))
                #Find right decision
                varss = [solution['independent'],solution['independent2'],'interactie']
                levels = [solution['independent'].levels,solution['independent2'].levels,['interactie']]
                if self.mes['L_ENGLISH']:
                    comparisons = ['unequal','equal','different','same'] if i < 2 else ['']
                else:
                    comparisons = ['ongelijk','gelijk','anders','verschillend'] if i < 2 else ['']
                decision_sent = [x for x in doc.sents if any([y in x.text for y in comparisons]) \
                                 and all([y.lower() in x.text for y in levels[i]])]
                if decision_sent != []: 
                    if i < 2:
                        output += '<br>' + self.scan_decision(decision_sent[0], solution, anova=True, num=i+1, prefix=False, elementair=False)[1]
                    else:
                        output += '<br>' + self.scan_decision_anova(decision_sent[0], solution, num=i+1, prefix=False, elementair=False)[1]
                else:
                    varss = [solution['independent'].name,solution['independent2'].name,markers[1]]
                    output += '<br>'+self.mes['F_NODEC']+ varss[i] + self.mes['S_LACKING1']
                
        if output.replace('<br>','') == '':
            return self.mes['F_NICEREP']
        else:
            return self.mes['F_INCOMPLETE'] + re.sub(r'<br>(<br>)+', '<br>', output)
            
    def split_grade_rmanova(self, text: str, solution:dict) -> str:
        if self.mes['L_ENGLISH']:
            nl_nlp = spacy.load('en_core_web_sm') 
        else: 
            nl_nlp = spacy.load('nl_core_news_sm')
        doc = nl_nlp(text.lower())
        output:str = ''
        output += '<br>'+'<br>'.join(self.detect_name(doc,solution))
        output += '<br>' + self.scan_design(doc, solution, prefix=False)[1]
        if solution['p'][0] < 0.05:
            output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'F', solution['F'][0], aliases=['F(' + solution['independent'].name + ')']))
            output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'p', solution['p'][0]))
            output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'R<sup>2</sup>', solution['r2'][0], aliases=['r2','r^2','r','kwadraat']))
        markers = ['h0','significant','subjecten','de subjecten','opgevoerde'] if not self.mes['L_ENGLISH'] else ['h0','significant','subjects','the subjects','stepped-up']
        d1_sents = [sent for sent in doc.sents if markers[0] in sent.text or markers[1] in sent.text]
        if d1_sents != []:
            output += '<br>' + self.scan_decision(d1_sents[0], solution, anova=True, num=1, prefix=False, elementair=False)[1]
        else:
            output += '<br>'+self.mes['F_NODEC'] + solution['independent'].name + self.mes['S_NONAME']
        d2_sents = [sent for sent in doc.sents if (markers[0] in sent.text or markers[1] in sent.text) and (markers[2] in sent.text or markers[4] in sent.text)]
        if d2_sents != []:
            output += '<br>' + self.scan_decision_rmanova(d2_sents[0], solution, num=2, prefix=False, elementair=False)[1]
        else:
            output += '<br>'+self.mes['F_NODEC'] + markers[3] + self.mes['S_NONAME']
        if output.replace('<br>','') == '':
            return self.mes['F_NICEREP']
        else:
            return self.mes['F_INCOMPLETE'] + re.sub(r'<br>(<br>)+', '<br>', output)
            
    def split_grade_mregression(self, text:str, solution:dict) -> str:
        if self.mes['L_ENGLISH']:
            nl_nlp = spacy.load('en_core_web_sm') 
        else: 
            nl_nlp = spacy.load('nl_core_news_sm')
        doc = nl_nlp(text.lower())
        output:str = ''
        output += '<br>'+'<br>'.join(self.detect_name(doc,solution))
        output += '<br>'+'<br>'.join(self.detect_comparison_mreg(doc, solution))
        output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'F', solution['F'][0]))
        output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'p', solution['p'][0]))
        output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'R<sup>2</sup>', solution['r2'][0], aliases=['r2','r^2','r','kwadraat']))
        output += '<br>'+'<br>'.join(self.detect_criterium(doc,solution))
        output += '<br>'+self.scan_predictors(doc, solution, prefix=False)[1]
        if output.replace('<br>','') == '':
            return self.mes['F_NICEREP']
        else:
            return self.mes['F_INCOMPLETE'] + re.sub(r'<br>(<br>)+', '<br>', output)
     
    def split_grade_manova(self, text:str, solution:dict) -> str:
        if self.mes['L_ENGLISH']:
            nl_nlp = spacy.load('en_core_web_sm') 
        else: 
            nl_nlp = spacy.load('nl_core_news_sm')
        doc = nl_nlp(text.lower())
        markers = ['eta-squared','for ','the multivariate decision ','for the multivariate decision ','multivariate','for the decision of '] if self.mes['L_ENGLISH'] \
                    else ['eta-kwadraat','bij ','de multivariate beslissing','bij de multivariate beslissing ','multivariaat','voor de beslissing van ']
        
        output:str = ''
        output += '<br>'+'<br>'.join(self.detect_name(doc,solution))
        output += '<br>' + self.scan_design_manova(doc, solution, prefix=False)[1]
        for i in range(3):
            var_key = 'dependent' if i < 1 else 'dependent' + str(i+1)
            if solution['p_multivar'] < 0.05 and solution['p_multivar'] < 0.05:
                decision_sent = [x for x in doc.sents if lef(solution[var_key].get_all_syns(),[y.text for y in x]) and ('significant' in x.text or 'effect' in x.text)]
                if decision_sent != []:
                    output += '<br>'+'<br>'.join(self.detect_decision_manova(decision_sent[0],solution, variable=solution[var_key].name, synonyms=[], p=solution['p'+str(i)][0], eta=solution['eta'+str(i)][0], num=i+1))
                    if solution['p'+str(i)][0] < 0.05:
                        output += '<br>'+'<br>'.join(self.detect_effect(decision_sent[0],solution, variable=markers[1]+solution[var_key].name, p=solution['p'+str(i)][0], eta=solution['eta'+str(i)][0]))
                else:
                    output += '<br>'+self.mes['F_NODEC']+ solution[var_key].name + self.mes['S_LACKING1']
            if solution['p'+str(i)][0] < 0.05 and solution['p_multivar'] < 0.05:
                output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'F', solution['F'+str(i)][0], appendix=markers[5]+solution[var_key].name+' '))
                output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'p', solution['p'+str(i)][0], appendix=markers[5]+solution[var_key].name+' '))
                output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'eta<sup>2</sup>', solution['eta'+str(i)][0], aliases=['eta','eta2',markers[0]],appendix=markers[5]+solution[var_key].name+' '))
        output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'F', solution['F_multivar'], appendix=markers[3]))
        output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'p', solution['p_multivar'], appendix=markers[3]))
        #output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'eta<sup>2</sup>', solution['eta_multivar'], aliases=['eta','eta2',markers[0]], appendix=markers[3]))
        decision_sent = [x for x in doc.sents if ('multivariate' in x.text or 'multivariaat' in x.text) \
                             and ('significant' in x.text or 'effect' in x.text)]
        if decision_sent != []:
            output += '<br>'+'<br>'.join(self.detect_decision_manova(decision_sent[0],solution,variable=markers[2],synonyms=['multivariate'], p=solution['p_multivar'], eta=solution['eta_multivar'], num=0))
            output += '<br>'+'<br>'.join(self.detect_effect(decision_sent[0],solution, variable=markers[3], p=solution['p_multivar'], eta=solution['eta_multivar']))
        else:
            output += '<br>' + self.mes['F_NOMULTIVAR']
        if output.replace('<br>','') == '':
            return self.mes['F_NICEREP']
        else:
            return self.mes['F_INCOMPLETE'] + re.sub(r'<br>(<br>)+', '<br>', output)
    
    def split_grade_ancova(self, text:str, solution:dict) -> str:
        if self.mes['L_ENGLISH']:
            nl_nlp = spacy.load('en_core_web_sm') 
        else: 
            nl_nlp = spacy.load('nl_core_news_sm')
        doc = nl_nlp(text.lower())
        markers = ['predictive value','predictive'] if self.mes['L_ENGLISH'] else ['voorspellende waarde','voorspellend']
        
        output:str = ''
        output += '<br>'+'<br>'.join(self.detect_name(doc,solution))
        output += '<br>' + self.scan_design(doc, solution, prefix=False)[1]
        output += '<br>' + self.scan_predictors(doc, solution, prefix=False)[1]
        
        multivar_sent = [x for x in doc.sents if markers[0] in x.text]
        if multivar_sent != []:
            output += '<br>'+'<br>'.join(self.detect_decision_ancova(multivar_sent[0], solution))
            if(solution['p'][3] < 0.05):
                output += '<br>'+'<br>'.join(self.detect_effect(multivar_sent[0],solution, variable='multivariate', p=solution['p'][3], eta=solution['eta'][3]))
        else:
            output += '<br>'+self.mes['F_NOPRED']
        output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'F', solution['F'][3]))
        output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'p', solution['p'][3]))
        output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'eta<sup>2</sup>', solution['eta'][3], aliases=['eta2','eta']))
        
        between_sent = [x for x in doc.sents if (lef(solution['independent'].get_all_syns(),[y.text for y in x]) or 'between-subject' in x.text) and ('significant' in x.text or 'effect' in x.text) and not markers[1] in x.text]
        if between_sent != []:
            output += '<br>'+'<br>'.join(self.detect_decision_multirm(between_sent[0], solution, solution['independent'].name, ['between-subject'], solution['p'][2],solution['eta'][2]))
            if(solution['p'][2] < 0.05):
                output += '<br>'+'<br>'.join(self.detect_effect(between_sent[0],solution, variable=solution['independent'].name, p=solution['p'][2], eta=solution['eta'][2]))
                output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'p', solution['p'][2], appendix=solution['independent'].name))
        else:
            output += '<br>'+self.mes['F_NOBTFAC']
        if output.replace('<br>','') == '':
            return self.mes['F_NICEREP']
        else:
            return self.mes['F_INCOMPLETE'] + re.sub(r'<br>(<br>)+', '<br>', output)
    
    def split_grade_multirm(self, text:str, solution:dict) -> str:
        if self.mes['L_ENGLISH']:
            nl_nlp = spacy.load('en_core_web_sm') 
        else: 
            nl_nlp = spacy.load('nl_core_news_sm')
        doc = nl_nlp(text.lower())
        markers = ['eta-squared','interaction'] if self.mes['L_ENGLISH'] else ['eta-kwadraat','interactie']
        
        output:str = '';num:int = 0
        output += '<br>'+'<br>'.join(self.detect_name(doc,solution))
        output += '<br>'+self.scan_design(doc,solution,prefix=False)[1]
        levels = solution['independent'].levels
        
        #Multivar within subject
        decision_sent = [x for x in doc.sents if (lef(solution['independent'].get_all_syns(),[y.text for y in x]) or 'within-subject' in x.text) \
                             and ('significant' in x.text or 'effect' in x.text) and markers[1] not in x.text]
        if decision_sent != []: 
            num += 1
            user_given_name:str = solution['independent'].name if solution['independent'].name in decision_sent[0].text else 'within-subject'
            output += '<br>'+'<br>'.join(self.detect_decision_multirm(decision_sent[0],solution,variable=user_given_name,synonyms=['multivariate within-subject'], p=solution['p0'][0], eta=solution['eta0'][0]))
            output += '<br>'+'<br>'.join(self.detect_effect(decision_sent[0],solution, variable=solution['independent'].name, p=solution['p0'][0], eta=solution['eta0'][0]))
        else:
            output += '<br>'+self.mes['F_NOMULTIVARWS']
        if solution['p0'][0] < 0.05:
            output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'F', solution['F0'][0], appendix=self.mes['S_WITHINFACTOR']))
            output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'p', solution['p0'][0], appendix=self.mes['S_WITHINFACTOR']))
            output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'eta<sup>2</sup>', solution['eta0'][0], aliases=['eta','eta2',markers[0]],appendix=self.mes['S_WITHINFACTOR']))
            if solution['p1'][0] < 0.05:
                output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'p', solution['p1'][0], appendix=self.mes['S_CONTRAST']+levels[0]+self.mes['S_AND']+levels[1]+' '))
            if solution['p1'][1] < 0.05:
                output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'p', solution['p1'][1], appendix=self.mes['S_CONTRAST']+levels[1]+self.mes['S_AND']+levels[2]+' '))
        
        #Multivar interaction
        decision_sent2 = [x for x in doc.sents if (markers[1] in x.text) and ('significant' in x.text or 'effect' in x.text)]
        if decision_sent2 != []:
            num += 1
            output += '<br>'+'<br>'.join(self.detect_decision_multirm(decision_sent2[0],solution,variable=markers[1],synonyms=[], p=solution['p0'][1], eta=solution['eta0'][1]))
            output += '<br>'+'<br>'.join(self.detect_effect(decision_sent2[0],solution, variable=markers[1], p=solution['p0'][1], eta=solution['eta0'][1]))
        else:
            output += '<br>'+self.mes['F_NOMULTIVARINT']
        if solution['p0'][1] < 0.05:
            output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'F', solution['F0'][1], appendix=self.mes['S_INTERACT']))
            output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'p', solution['p0'][1], appendix=self.mes['S_INTERACT']))
            output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'eta<sup>2</sup>', solution['eta0'][1], aliases=['eta','eta2',markers[0]],appendix=self.mes['S_INTERACT']))
            if solution['p1'][2] < 0.05:
                output += '<br>'+'<br>'.join(self.detect_p(doc, 'p', solution['p1'][2], appendix=self.mes['S_CONTRAST']+levels[0]+self.mes['S_AND']+levels[1]+' '+self.mes['S_INTERACT']))
            if solution['p1'][3] < 0.05:
                output += '<br>'+'<br>'.join(self.detect_p(doc, 'p', solution['p1'][3], appendix=self.mes['S_CONTRAST']+levels[1]+self.mes['S_AND']+levels[2]+' '+self.mes['S_INTERACT']))
        
        #Between-subject
        decision_sent3 = [x for x in doc.sents if (lef(solution['independent2'].get_all_syns(),[y.text for y in x]) or 'between-subject' in x.text) and ('significant' in x.text or 'effect' in x.text) and markers[1] not in x.text]
        print(decision_sent3)
        if decision_sent3 != []:
            num += 1
            user_given_name:str = solution['independent2'].name if solution['independent2'].name in decision_sent3[0].text else 'between-subject'
            output += '<br>'+'<br>'.join(self.detect_decision_multirm(decision_sent3[0],solution,variable=user_given_name,synonyms=['multivariate between-subject'], p=solution['p'][1], eta=solution['eta'][1]))
            output += '<br>'+'<br>'.join(self.detect_effect(decision_sent3[0], solution, variable=solution['independent2'].name, p=solution['p'][1], eta=solution['eta'][1]))
        else:
            output += '<br>'+self.mes['F_NOMULTIVARBS']
        if solution['p'][1] < 0.05:
            output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'F', solution['F'][1], appendix=self.mes['S_BETWEENFACTOR']))
            output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'p', solution['p'][1], appendix=self.mes['S_BETWEENFACTOR']))
            output += '<br>'+'<br>'.join(self.detect_report_stat(doc, 'eta<sup>2</sup>', solution['eta'][1], aliases=['eta','eta2',markers[0]],appendix=self.mes['S_BETWEENFACTOR']))
        if output.replace('<br>','') == '':
            return self.mes['F_NICEREP']
        else:
            return self.mes['F_INCOMPLETE'] + re.sub(r'<br>(<br>)+', '<br>', output)
    
    def split_grade_multirm2(self, text:str, solution:dict) -> str:
        if self.mes['L_ENGLISH']:
            nl_nlp = spacy.load('en_core_web_sm') 
        else: 
            nl_nlp = spacy.load('nl_core_news_sm')
        doc = nl_nlp(text.lower())
        output:str = ''
        output += '<br>'+'<br>'.join(self.detect_name(doc,solution))
        if output.replace('<br>','') == '':
            return self.mes['F_NICEREP']
        else:
            return self.mes['F_INCOMPLETE'] + re.sub(r'<br>(<br>)+', '<br>', output)
    
    """
    DETECTION FUNCTIONS (THESE ARE CALLED TO SCAN THE DECISION, INTERPRETATION AND SHORT REPORTS)
    
    THEY TAKE DOC (TEXT OBJECTS ANNOTATED WITH SPACY) AS INPUT
    """
    def detect_h0(self, sent:Doc, solution:dict, num:int=1) -> List[str]:
        #Define variables
        scorepoints = {'hyp_rejected': False,
                       'hyp_present': False
                }
        output:List[str] = []
        rejected = solution['p'][num-1] < 0.05
        behoud_list = ['retain','retained'] if self.mes['L_ENGLISH'] else ['behoud', 'behouden']
        verwerp_list = ['reject','rejected'] if self.mes['L_ENGLISH'] else ['verwerp', 'verworpen', 'verwerpen']
        h0_list = ['h0','hypothesis'] if self.mes['L_ENGLISH'] else ['h0','nulhypothese']
        
        #Controleer input
        scorepoints['hyp_rejected'] = any([x in sent.text for x in verwerp_list]) if rejected else any([x in sent.text for x in behoud_list])
        scorepoints['hyp_present'] = any([x in sent.text for x in h0_list])
        
        #Add strings
        if not scorepoints['hyp_rejected'] and scorepoints['hyp_present']:
            if num < 2:
                output.append(self.mes['F_HYPSWITCH'])
            if num > 2:
                output.append(self.mes['F_INTSWITCH'])
            else:
                output.append(self.mes['F_HYPFACTOR']+str(num)+self.mes['S_SWITCHEROO'])
        if not scorepoints['hyp_present']:
            if num < 2:
                output.append(self.mes['F_NOHYP'])
            elif num < 3:
                if solution['assignment_type'] != 5:
                    output.append(self.mes['F_HYPFACTOR']+str(num)+' '+self.mes['S_LACKING1'])
                else:
                    output.append(self.mes['F_NOSUBJ'])
            else:
                output.append(self.mes['F_INTHYP'])
        return output
    
    def detect_significance(self, doc:Doc, solution:dict, num:int=1) -> List[str]:
        scorepoints = {'effect': False,
                       'sign': False,
                       'neg': False
                       }
        diff_words = ['difference','effect'] if self.mes['L_ENGLISH'] else ['verschil','effect']
        size_words = ['medium','moderate','small','strong','large','tiny'] if self.mes['L_ENGLISH'] else ['zwak','matig','klein','sterk','groot']
        output:List[str] = []
        rejected:bool = solution['p'][num-1] < 0.05
        h0_output:list = self.detect_h0(doc, solution, num)
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
                output.append(self.mes['F_EFFECTSIGN'])
            if not scorepoints['sign'] and scorepoints['effect']:
                output.append(self.mes['F_EFFECTSIGN'])
            if not scorepoints['neg']:
                output.append(self.mes['F_NEGEFFECT'])
        else:
            appendix:str = ' '+self.mes['S_SUBJECTS'] if solution['assignment_type'] == 5 and num == 2 else self.mes['S_FORFAC'] + str(num) if num < 3 else self.mes['S_WITHINT']
            if not scorepoints['effect']:
                output.append(self.mes['F_EFFECTSIGN']+appendix)
            if not scorepoints['sign'] and scorepoints['effect']:
                output.append(self.mes['F_EFFECTSIGN']+appendix)
            if not scorepoints['neg']:
                output.append(self.mes['F_NEGEFFECT']+appendix)
        return output
    
    def detect_true_scores(self, sent:Doc, solution:dict, num=None) -> List[str]:
        #Define variables
        criteria:list = ['right_comparison', 'right_negation', 'mean_present', 'pop_present','jacked','contrasign']
        pop = 'population' if self.mes['L_ENGLISH'] else 'population'
        popavgs = ['population average','population averages','population mean','population means'] if self.mes['L_ENGLISH'] else ['populatiegemiddelden','populatiegemiddelde']
        avgs = ['average','mean','averages','means'] if self.mes['L_ENGLISH'] else ['gemiddeld','gemiddelde','gemiddelden']
        equal = ['gelijk'] if not self.mes['L_ENGLISH'] else ['equal','same']
        diff = ['ongelijk','anders','verschillend'] if not self.mes['L_ENGLISH'] else ['different','unequal']
        stepmark = ['stepped-up'] if self.mes['L_ENGLISH'] else ['opgevoerde']
        scorepoints:dict = dict([(x,False) for x in criteria])
        rejected:bool = solution['p'][-1] < 0.05
        tokens:list = [x.text for x in sent]
        output:List[str] = []
        
        #Controleer input
        comparisons = [x for x in sent if x.text in equal+diff]
        if comparisons != []:
            comproot = comparisons[num-1] if len(comparisons) >= num else comparisons[0]
            #comptree:List = descendants(comproot)
            not_present = bool(negation_counter(tokens) % 2)
            scorepoints['right_comparison'] = comproot.text in equal+diff
            if comproot.text in diff:
                scorepoints['right_negation'] = not_present != rejected
            elif comproot.text in equal:
                scorepoints['right_negation'] = not_present == rejected
        else:
            scorepoints['right_negation'] = True
        scorepoints['jacked'] = any([x in sent.text for x in stepmark])
            
        mean = [x in sent.text for x in avgs]
        mean_2 = [x in sent.text for x in popavgs]
        scorepoints['mean_present'] = any(mean) or any(mean_2)
        scorepoints['pop_present'] = any(mean_2) or pop in [x.text for x in sent] or any([x in tokens for x in ['significant','significante']])
        scorepoints['contrasign'] = not ((any(mean_2) or pop in tokens) and any([x in tokens for x in ['significant','significante']]))
        
        #Add strings
        if not scorepoints['right_comparison']:
            output.append(self.mes['F_POPLEVELS'])
        if not scorepoints['right_negation']:
            output.append(self.mes['F_INTNEG']+self.mes['S_SUBDEC'])
        if not scorepoints['mean_present']:
            output.append(self.mes['F_DECAVGS'])
        if not scorepoints['pop_present']:
            output.append(self.mes['F_DECPOP'])
        if not scorepoints['jacked']:
            output.append(self.mes['F_JACKEDMEANS'])
        if not scorepoints['contrasign']:
            output.append(self.mes['F_BOTHPOP'])
        return output
    
    def detect_criterium(self, doc:Doc, solution:dict, prefix:bool=True):
        output = []
        if (not 'criterium' in doc.text and not 'afhankelijke' in doc.text and not 'dependent' in doc.text) or \
                        not lef(solution['independent'].get_all_syns(),[x.text for x in doc]):
            output.append(self.mes['F_DEP'])
        return output
    
    def detect_comparison(self, sent:Doc, solution:dict, anova:bool, num:int) -> List[str]:
        #Define language
        comp1 = ['unequal','larger','smaller'] if self.mes['L_ENGLISH'] else ['ongelijk','groter','kleiner']
        comp2 = ['larger','unequal','smaller','equal','different','differing'] if self.mes['L_ENGLISH'] else ['groter','ongelijk','kleiner','gelijk','anders','verschillend']
        avgs1 = ['average','averages','mean','means'] if self.mes['L_ENGLISH'] else ['gemiddelde','gemiddelden','gemiddeld']
        avgs2 = ['population average', 'population mean'] if self.mes['L_ENGLISH'] else ['populatiegemiddelde','populatiegemiddelden']
        pop = 'population' if self.mes['L_ENGLISH'] else 'populatie'
        nott = 'not' if self.mes['L_ENGLISH'] else 'niet'
        
        #Define variables
        criteria = ['right_comparison', 'right_negation', 'mean_present', 'pop_present', 'level_present', 'both_present','contrasign']
        scorepoints = dict([(x,False) for x in criteria])
        output:List[str] = []
        level_syns = solution['independent'].get_all_level_syns() if num == 1 else solution['independent2'].get_all_level_syns()
        
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
        scorepoints['pop_present'] = any(mean_2) or pop in sent.text or any([x in sent.text for x in ['significant','significante']])
        level_bools:list[bool] = [lef(level_syns[i],[x.text for x in sent]) for i in range(2)]#len(levels))]
        scorepoints['level_present'] = any(level_bools) #or scorepoints['level_present']
        scorepoints['both_present'] = all(level_bools)# or scorepoints['both_present']
        scorepoints['contrasign'] = not ((any(mean_2) or pop in sent.text) and any([x in sent.text for x in ['significant','significante']]))
        subjtext = 'of the subjects' if self.mes['L_ENGLISH'] else 'van de subjecten'
        
        #Add strings:
        appendix:str = '' if num < 2 else self.mes['S_FORFAC'] + str(num) if num < 3 else subjtext if solution['assignment_type']==5 and num==2 else self.mes['S_WITHINT']
        if not scorepoints['contrasign']:
            output.append(self.mes['F_BOTHPOP'] + appendix)
        if not scorepoints['right_comparison']:
            output.append(self.mes['F_JUSTLEVELS']+appendix+self.mes['S_BADCOMPARED'])
        if not scorepoints['right_negation']:
            output.append(self.mes['F_INTNEG']+self.mes['S_COMPARING']+appendix) 
        if not scorepoints['mean_present']:
            output.append(self.mes['F_DECISION']+appendix+' '+self.mes['S_POPAVGS'])
        if not scorepoints['pop_present']:
            output.append(self.mes['F_DECISION'] + appendix + ' '+self.mes['S_OVERPOP'])
        if not scorepoints['level_present']:
            output.append(self.mes['F_INDEPLEVELS']+appendix+self.mes['S_NONAMES'][1:])
        if not scorepoints['both_present'] and scorepoints['level_present']:
            output.append(self.mes['F_SOMELEVEL']+appendix+self.mes['S_LEFTOUT'])
        return output
    
    def detect_comparison_mreg(self, sent:Doc, solution:dict) -> List[str]:
        #Define variables
        criteria = ['bigger', 'neg', 'sign', 'propvar', 'zero_present', 'conj']
        comp_markers = [['proportion','declared','variance'],['proportion','explained','variance'],['R2'],['R^2'],['R','squared'],['R-squared']] \
                        if self.mes['L_ENGLISH'] \
                        else [['proportie','verklaarde','variantie']]
        markers = [['larger','bigger'],['zero','null','0']] if self.mes['L_ENGLISH'] else [['groter'],['nul','0']]
        scorepoints = dict([(x,False) for x in criteria])
        output:List[str] = []
        rejected = solution['p'][0] < 0.05
        tokens = [y.text for y in sent]
        
        #Controleer input
        scorepoints['neg'] = bool(negation_counter(tokens) % 2) != rejected
        scorepoints['sign'] = 'significant' in sent.text
        scorepoints['propvar'] = any(all([y in sent.text for y in comp_markers[i]]) for i in range(len(comp_markers)))
        bigger_present = [x for x in sent if x.text in markers[0]]
        if bigger_present != []:
            scorepoints['bigger'] = True
            zero = [x for x in sent if x.text in markers[1]]
            if zero != []:
                scorepoints['zero_present'] = True
                scorepoints['conj'] = zero[0].dep_ in ['obl','nummod','attr','pobj']
        
        #Add strings:
        if not scorepoints['bigger']:
            output.append(self.mes['F_PROPVARNONULL'])
        if not scorepoints['neg'] and scorepoints['bigger']:
            output.append(self.mes['F_PROPVARNEG'])
        if not scorepoints['sign'] and scorepoints['bigger']:
            output.append(self.mes['F_PROPVARSIGN'])
        if not scorepoints['propvar'] and scorepoints['bigger']:
            output.append(self.mes['F_PROPVAR'])
        if not scorepoints['zero_present'] and scorepoints['bigger']:
            output.append(self.mes['F_PROPVARCOMP'])
        if not scorepoints['conj'] and scorepoints['bigger']: #TODO FIX CONJ
            output.append(self.mes['F_PROPVARNULL'])    
        return output
    
    def detect_interaction(self, doc:Doc, solution:dict, anova:bool) -> List[str]:
        #Define variables
        pop = 'population' if self.mes['L_ENGLISH'] else 'populatie'
        dec_markers = ['interaction'] if self.mes['L_ENGLISH'] else ['interactie']
        criteria = ['interactie','indy1','indy2','pop_present','right_negation', 'contrasign']
        scorepoints = dict([(x,False) for x in criteria])
        rejected = solution['p'][-1] < 0.05
        tokens = [y.text for y in doc]
        output:List[str] = []
        
        #Controleer input
        if hasattr(doc, 'sents'):
            i_sents = [sent for sent in doc.sents if any([x in tokens for x in dec_markers])]
        else:
            i_sents = [doc]
        if i_sents != []:
            int_descendants = i_sents[0]    
            tokens = [x.text for x in int_descendants]
            scorepoints['interactie'] = True
            scorepoints['indy1'] = lef(solution['independent'].get_all_syns(),[x.text for x in doc])
            scorepoints['indy2'] = lef(solution['independent2'].get_all_syns(),[x.text for x in doc])
            scorepoints['pop_present'] = pop in [x.text for x in int_descendants] or any([x in tokens for x in ['significant','significante']])
            scorepoints['right_negation'] = bool(negation_counter(tokens) % 2) != rejected    
            scorepoints['contrasign'] = not ((pop in tokens) and any([x in tokens for x in ['significant','significante']]))
            
        #Add strings
        if not scorepoints['interactie']:
            output.append(self.mes['F_INTINT'])
        if not scorepoints['right_negation']:
            output.append(self.mes['F_INTNEGG'])
        if not scorepoints['pop_present']:
            output.append(self.mes['F_INTPOP'])
        if not scorepoints['indy1'] and not scorepoints['indy2']:
            output.append(self.mes['F_INDINT'])
        elif not scorepoints['indy2'] or not scorepoints['indy2']:
            output.append(self.mes['F_IND1INT'])
        if not scorepoints['contrasign']:
            output.append(self.mes['F_BOTHPOP'])
        return output
    
    def detect_decision_ancova(self, sent:Doc, solution:dict) -> List[str]:
        rejected:bool = solution['p'][3] < 0.05
        suffix = ' in de hoofdbeslissing' if self.mes['L_ENGLISH'] else ' for the main decision'
        marker = 'significant predictive value' if self.mes['L_ENGLISH'] else 'significant voorspellende waarde'
        tokens:list = [x.text for x in sent]
        scorepoints:dict = {'sign_val': marker in sent.text,
            'indep': lef(solution['independent'].get_all_syns(),[x.text for x in sent]),
            'cov1': solution['predictor_names'][0] in sent.text or any([x in sent.text for x in solution['predictor_syns'][0]]),
            'cov2': solution['predictor_names'][1] in sent.text or any([x in sent.text for x in solution['predictor_syns'][1]]),
            'dep': any([x in sent.text for x in solution['dependent'].get_all_syns()]),
            'neg': bool(negation_counter(tokens) % 2) != rejected }
        
        output:List[str] = []
        if not scorepoints['sign_val']:
            output.append(self.mes['F_PRED'])
        if not scorepoints['indep']:
            output.append(self.mes['F_NOIND']+suffix)
        if not scorepoints['cov1'] and not scorepoints['cov2']:
            output.append(self.mes['F_2COV'])
        elif not scorepoints['cov1'] or not scorepoints['cov2']:
            output.append(self.mes['F_1COV'])
        if not scorepoints['dep']:
            output.append(self.mes['F_NODEC']+suffix)
        if not scorepoints['neg']:
            output.append(self.mes['F_INTNEG']+suffix[1:])
        return output
    
    def detect_decision_manova(self, sent:Doc, solution:dict, variable:str, synonyms:list, p:float, eta:float, num:int) -> List[str]:
        rejected:bool = p < 0.05
        dep_key = 'dependent' if num < 2 else 'dependent' + str(num)
        tokens:list = [x.text for x in sent]
        suffix = ' bij de beslissing van ' if not self.mes['L_ENGLISH'] else ' for the decision of '
        scorepoints:dict = {'sign_effect': 'significant' in sent.text,
            'indep': lef(solution['independent'].get_all_syns(),[x.text for x in sent]),
            'dep': lef(solution[dep_key].get_all_syns(),[x.text for x in sent]) or 'multivariate' in variable,
            'neg': bool(negation_counter(tokens) % 2) != rejected}
        
        output:List[str] = []
        if not scorepoints['sign_effect']:
            output.append(' -'+self.mes['S_NANDEC']+variable+self.mes['S_SIGNEFFECT'])
        if not scorepoints['indep']:
            output.append(self.mes['F_NOIND']+suffix+variable)
        if not scorepoints['dep']:
            output.append(self.mes['F_NODEP']+suffix+variable)
        if not scorepoints['neg']:
            output.append(self.mes['F_NONEG']+variable)
        return output
    
    def detect_decision_multirm(self, sent:Doc, solution:dict, variable:str, synonyms:list, p:float, eta:float) -> List[str]:
        rejected:bool = p < 0.05
        tokens:list = [x.text for x in sent]
        scorepoints:dict = {'sign_effect': 'significant' in sent.text,
            'var': variable in sent.text or any([x in sent.text for x in synonyms]),
            'neg': bool(negation_counter(tokens) % 2) != rejected}
        
        output:List[str] = []
        if not scorepoints['sign_effect']:
            output.append(' -'+self.mes['S_NANDEC']+variable+self.mes['S_SIGNEFFECT'])
        if not scorepoints['var']:
            output.append(' -'+variable+' '+self.mes['S_NANDEC']+variable)
        if not scorepoints['neg']:
            output.append(self.mes['F_NONEG']+variable)
        return output
        
    def fres(self, sent:Doc, solution:dict, num=None) -> List[str]:
        #Define variables
        criteria:list = ['right_comparison', 'right_negation', 'mean_present', 'pop_present','jacked','contrasign']
        pop = 'population' if self.mes['L_ENGLISH'] else 'population'
        popavgs = ['population average','population averages','population mean','population means'] if self.mes['L_ENGLISH'] else ['populatiegemiddelden','populatiegemiddelde']
        avgs = ['average','mean','averages','means'] if self.mes['L_ENGLISH'] else ['gemiddeld','gemiddelde','gemiddelden']
        equal = ['gelijk'] if not self.mes['L_ENGLISH'] else ['equal','same']
        diff = ['ongelijk','anders','verschillend'] if not self.mes['L_ENGLISH'] else ['different','unequal']
        stepmark = ['stepped-up'] if self.mes['L_ENGLISH'] else ['opgevoerde']
        scorepoints:dict = dict([(x,False) for x in criteria])
        rejected:bool = solution['p'][-1] < 0.05
        tokens:list = [x.text for x in sent]
        output:List[str] = []
        
        #Controleer input
        comparisons = [x for x in sent if x.text in equal+diff]
        if comparisons != []:
            comproot = comparisons[num-1] if len(comparisons) >= num else comparisons[0]
            #comptree:List = descendants(comproot)
            #not_present = bool(negation_counter(tokens) % 2)
            scorepoints['right_comparison'] = comproot.text in equal+diff
            scorepoints['right_negation'] = bool(comproot.text in diff) == rejected
            #if comproot.text in diff:
            #    scorepoints['right_negation'] = not_present != rejected
            #elif comproot.text in equal:
            #    scorepoints['right_negation'] = not_present == rejected
        else:
            scorepoints['right_negation'] = True
        scorepoints['jacked'] = any([x in sent.text for x in stepmark])
            
        mean = [x in sent.text for x in avgs]
        mean_2 = [x in sent.text for x in popavgs]
        scorepoints['mean_present'] = any(mean) or any(mean_2)
        scorepoints['pop_present'] = any(mean_2) or pop in [x.text for x in sent] or any([x in tokens for x in ['significant','significante']])
        scorepoints['contrasign'] = not ((any(mean_2) or pop in tokens) and any([x in tokens for x in ['significant','significante']]))
        
        #Add strings
        suffix = ' for the decision of subjects' if self.mes['L_ENGLISH'] else ' voor de beslissing van de subjecten'
        if not scorepoints['right_comparison']:
            output.append(self.mes['F_POPLEVELS']+suffix)
        if not scorepoints['right_negation']:
            output.append(self.mes['F_INTNEG']+suffix)
        if not scorepoints['mean_present']:
            output.append(self.mes['F_DECAVGS']+suffix)
        if not scorepoints['pop_present']:
            output.append(self.mes['F_DECPOP']+suffix)
        if not scorepoints['jacked']:
            output.append(self.mes['F_JACKEDMEANS']+suffix)
        if not scorepoints['contrasign']:
            output.append(self.mes['F_BOTHPOP']+suffix)
        return output
    
    def detect_effect(self, sent:Doc, solution:dict, variable:str, p:float, eta:float) -> List[str]:
        output:List[str] = []
        if p > 0.05:
            return output
        scorepoints:dict ={'effect_present': False,
            'strength_present': False,
            'right_strength': False,
            'no_wrongs': False}
        gold_strength: int = 2 if eta > 0.2 else 1 if eta > 0.1 else 0
        sizes = ['medium','moderate','small','strong','large','tiny'] if self.mes['L_ENGLISH'] else ['klein','zwak','matig','groot','sterk']
        large = ['strong','large'] if self.mes['L_ENGLISH'] else ['sterk','groot']
        med = ['medium','moderate'] if self.mes['L_ENGLISH'] else ['matig']
        small = ['small','weak'] if self.mes['L_ENGLISH'] else ['klein','zwak']
        effect = [x.text for x in sent if x.text in sizes]
        scorepoints['effect_present'] = 'effect' in sent.text
        scorepoints['strength_present'] = any([x in sent.text for x in sizes])
        scorepoints['right_strength'] = any([x in large if gold_strength == 2 else x in med if gold_strength == 1 else x in small for x in effect])
        scorepoints['no_wrongs'] = not any([x in large or x in med for x in effect]) if gold_strength == 0 else \
                                   not any([x in large or x in small for x in effect]) if gold_strength == 1 else \
                                   not any([x in med or x in small for x in effect])
        if not scorepoints['effect_present'] and scorepoints['strength_present']:
            output.append(self.mes['F_NOSIZE']+variable+self.mes['S_NONAME'])
        if not scorepoints['strength_present']:
            output.append(self.mes['F_STRENGTH']+variable+self.mes['S_NONAME'])
        elif scorepoints['effect_present'] and not scorepoints['right_strength']:
            output.append(self.mes['F_STRENGTH']+variable+self.mes['S_NONAME'])
        elif scorepoints['effect_present'] and not scorepoints['no_wrongs']:
            output.append(self.mes['F_STRENGTH']+variable)
        return output
    
    def detect_unk(self, sent:Doc, solution:dict, num:int=1):
        #Define variables
        criteria:list=['two']#['unk','two']
        ones = ['one','1'] if self.mes['L_ENGLISH'] else ['één','een','1']
        twos = ['two','multiple','several','2'] if self.mes['L_ENGLISH'] else ['twee','meerdere','2']
        scorepoints = dict([(x,False) for x in criteria])
        control = solution['control'] if num < 2 else solution['control' + str(num)] if num < 3 else solution['control2'] or solution['control2']
        tokens = [x.text for x in sent]
        output:List[str] = []
        
        #Controleer input
        #scorepoints['unk'] = 'onbekend' in tokens if not control else True
        scorepoints['two'] = any([x in ones for x in tokens]) if control else any([x in twos for x in tokens])
        
        #Add strings
        if not scorepoints['two']:
            output.append(self.mes['F_XINTS'])
        return output
        
    def detect_primary(self, sent:Doc, solution:dict, num:int=1) -> List[str]:
        #Define variables
        suffix = ' for the primary explanation' if self.mes['L_ENGLISH'] else ' bij de primaire verklaring'
        prim = 'primary' if self.mes['L_ENGLISH'] else 'primaire'
        criteria:list = ['cause', 'ind', 'dep', 'prim', 'neg', 'alignment']
        scorepoints = dict([(x,False) for x in criteria])
        i_key: str = 'independent' + str(num) if num > 1 else 'independent'
        control: bool = solution['control'] if num < 2 else solution['control'+str(num)]
        rejected = solution['p'][num-1] < 0.05
        tokens = [x.text for x in sent] 
        output:List[str] = []
        
        #Controleer input
        scorepoints['prim'] = prim in tokens if not control else True
        scorepoints['neg'] = bool(negation_counter(tokens) % 2) != rejected
        if self.mes['L_ENGLISH']:
            causeverbs:list = [x for x in sent if x.text in ['causes', 'influences', 'influenced', 'responsible','cause', 'creates']]
        else:
            causeverbs:list = [x for x in sent if x.text in ['veroorzaakt', 'heeft', 'beinvloedt', 'beinvloed','verantwoordelijk', 'oorzaak', 'invloed']] 
        if any(causeverbs): #effect_children = descendants(causeverbs[0])
            scorepoints['cause'] = True
        depnode = [x for x in sent if edit_distance(x.text,solution['dependent'].node) < 2]
        indynode = [x for x in sent if edit_distance(x.text,solution[i_key].node) < 2]
        scorepoints['ind'] = indynode != [] #any([x in sent.text for x in solution[i_key]]) #indynode != []
        scorepoints['dep'] = depnode != []#any([x in sent.text for x in solution['dependent']]) #depnode != []
        if scorepoints['ind'] and scorepoints['dep']:
            scorepoints['alignment'] = self.check_causality(indynode[0],depnode[0])
        
        #Add strings
        if not scorepoints['cause']:
            output.append(self.mes['F_EFFECTCAUSE'])
        if not scorepoints['neg']:
            output.append(self.mes['F_INTNEG']+suffix)
        if not scorepoints['ind']:
            output.append(self.mes['F_NOIND']+suffix)
        if not scorepoints['dep']:
            output.append(self.mes['F_NODEP']+suffix)
        if scorepoints['ind'] and scorepoints['dep'] and not scorepoints['alignment']:
            output.append(self.mes['F_CAUSEVARS']+suffix)
        if not scorepoints['prim']:
            output.append(self.mes['F_NOPRIMVAR'])
        return output
    
    def detect_primary_interaction(self, sent:Doc, solution:dict) -> List[str]:
        smarkers = ['same','equal'] if self.mes['L_ENGLISH'] else ['dezelfde','hetzelfde','gelijk','gelijke']
        dmarkers = ['ongelijk','verschillend','verschillende'] if not self.mes['L_ENGLISH'] else ['different', 'unequal']
        suffix = ' bij de primaire verklaring' if not self.mes['L_ENGLISH'] else ' for the primary explanation'
        criteria:list = ['interaction', 'negation', 'indy1', 'indy2', 'dep', 'level_present', 'both_levels', 'same']
        scorepoints = dict([(x,False) for x in criteria])
        tokens = [x.text for x in sent]
        output:list = []
        var1levels:list[bool] = [any([y in tokens for y in solution['independent'].get_all_level_syns()[i]]) for i in range(solution['independent'].nlevels)]
        var2levels:list[bool] = [any([y in tokens for y in solution['independent2'].get_all_level_syns()[i]]) for i in range(solution['independent2'].nlevels)]
        rejected = solution['p'][2] < 0.05
        
        # Fill scorepoints
        dep_node = [x for x in sent if edit_distance(x.text,solution['dependent'].node) < 2]
        indy1node = [x for x in sent if edit_distance(x.text,solution['independent'].node) < 2]
        indy2node = [x for x in sent if edit_distance(x.text,solution['independent2'].node) < 2]
        scorepoints['indy1'] = indy1node != []
        scorepoints['indy2'] = indy2node != []
        scorepoints['dep'] = dep_node != []
        scorepoints['same'] = any([x in tokens for x in smarkers]) or any([x in tokens for x in dmarkers])
        if scorepoints['same']:
            if any([x in tokens for x in smarkers]):
                scorepoints['negation'] = bool(negation_counter(tokens) % 2) == rejected
            else:
                scorepoints['negation'] = bool(negation_counter(tokens) % 2) != rejected
        else:
            scorepoints['negation'] = True
        if scorepoints['dep']:
            if self.check_causality(indy1node[0], dep_node[0]) and any(var2levels):
                scorepoints['interaction'] = True
                scorepoints['level_present'] = any(var2levels)
                scorepoints['both_levels'] = all(var2levels)
            elif self.check_causality(indy2node[0], dep_node[0]) and any(var1levels):
                scorepoints['interaction'] = True
                scorepoints['level_present'] = any(var1levels) or scorepoints['level_present']
                scorepoints['both_levels'] = all(var1levels) or scorepoints['both_levels']
                    
        #Add strings
        if not scorepoints['negation']:
            output.append(self.mes['F_INTNEG']+suffix)
        if not scorepoints['indy1']:
            output.append(self.mes['F_NOIND1']+suffix)
        if not scorepoints['indy2']:
            output.append(self.mes['F_NOIND2']+suffix)
        if not scorepoints['dep']:
            output.append(self.mes['F_NODEP']+suffix)
        if scorepoints['dep'] and ((not scorepoints['interaction'] and not (any(var1levels) or any(var2levels))) or not scorepoints['same']):
            output.append(self.mes['F_SAMEEFFECT']+suffix)
        elif scorepoints['dep'] and not scorepoints['interaction']:
            output.append(self.mes['F_CAUSEVARS']+suffix)
        elif scorepoints['interaction'] and not scorepoints['both_levels']:
            output.append(self.mes['F_NOLEVELS'])
        elif scorepoints['interaction'] and not scorepoints['level_present']:
            output.append(self.mes['F_NOLEVELS1'])
        return output
    
    def detect_alternative(self, sent:Doc, solution:dict, num:int=1) -> List[str]:
        #Define variables
        criteria = ['alt','ind','dep','cause','relation_type']
        scorepoints = dict([(x,False) for x in criteria])
        i_key: str = 'independent' + str(num) if num > 1 else 'independent'
        control: bool = solution['control'] if num < 2 else solution['control'+str(num)]
        #rejected: bool = solution['p'][num-1] < 0.05
        output:List[str] = []
        
        #Controleer input
        alt = 'alternative' if self.mes['L_ENGLISH'] else 'alternatieve'
        scorepoints['alt'] = alt in [x.text for x in sent] if not control else True
        suffix = ' for the alternative explanation' if self.mes['L_ENGLISH'] else ' bij de alternatieve verklaring'
        if self.mes['L_ENGLISH']:
            causeverbs:list = [x for x in sent if x.text in ['causes', 'influences', 'influenced', 'responsible','cause', 'creates','has','caused']]
        else:
            causeverbs:list = [x for x in sent if x.text in ['veroorzaakt', 'heeft', 'beinvloedt', 'beinvloed','verantwoordelijk', 'oorzaak', 'invloed']] 
        if any(causeverbs):
            scorepoints['cause'] = True
        depnode = [x for x in sent if edit_distance(x.text,solution['dependent'].node) < 2]
        indynode = [x for x in sent if edit_distance(x.text,solution[i_key].node) < 2]
        scorepoints['ind'] = indynode != []
        scorepoints['dep'] = depnode != []
        if scorepoints['ind'] and scorepoints['dep']:        
            scorepoints['relation_type'] = self.check_causality(indynode[0], depnode[0], alternative=True)
        else:
            scorepoints['relation_type'] = True
        
        #Add strings
        if not scorepoints['cause']:
            output.append(self.mes['F_NOCAUSE']+suffix)
        if not scorepoints['ind']:
            output.append(self.mes['F_NOIND']+suffix)
        if not scorepoints['dep']:
            output.append(self.mes['F_NODEP']+suffix)
        if not scorepoints['alt']:
            output.append(self.mes['F_NOALT'])
        if not scorepoints['relation_type']:
            output.append(self.mes['F_WRONGCAUSATION'])
        return output
    
    def detect_alternative_interaction(self, sent:Doc, solution:dict) -> List[str]:
        markers = ['interfer','variabl','reverse','causality'] if self.mes['L_ENGLISH'] \
                            else ['storend','variabel','variabele','omgekeerde','causaliteit']
        output:list = []
        if not (markers[0] in sent.text and markers[1] in sent.text):
            output.append(self.mes['F_NODISTURBANCE'])
        if not (markers[2] in sent.text and markers[3] in sent.text):
            output.append(self.mes['F_NOREVERSECAUSE'])
        return output
    
    def detect_report_stat(self, doc:Doc, stat:str, value:float, aliases:list=[], num:int=1, margin=0.01, appendix=None) -> List[str]:
        tokens:list[str] = [x.text for x in doc]
        for i in range(len(tokens)):
            if i < len(tokens) - 2:
                t3:str = tokens[i+2]
                if t3.replace('.','').replace('-','').isdigit():
                    t1:str = tokens[i]
                    t2:str = tokens[i+1]
                    stat_present = t1 == stat.lower() or t1 in aliases if stat not in ['F','T'] else t1 == stat.lower() or t1 in aliases or ')' in t1
                    if stat_present:
                        if t2 in ['==','='] and float(t3) < value + margin and float(t3) > value - margin:
                            return []
            if '=' in tokens[i]:
                if tokens[i][0] == '=' and i > 0:
                    t3:str = tokens[i][1:]; t1 = tokens[i-1]
                    if i > 0 and t3.replace('.','').replace('-','').isdigit():
                        stat_present = t1 == stat.lower() or t1 in aliases if stat not in ['F','T'] else t1 == stat.lower() or t1 in aliases or ')' in t1
                        if float(t3) < value + margin and float(t3) > value - margin and stat_present:
                            return []
                elif tokens[i][-1] == '=' and i < len(tokens):
                    t3:str = tokens[i+1]; t1 = tokens[i][:-1]
                    if t3.replace('.','').replace('-','').isdigit():    
                        stat_present = t1 == stat.lower() or t1 in aliases if stat not in ['F','T'] else t1 == stat.lower() or t1 in aliases or ')' in t1
                        if float(t3) < value + margin and float(t3) > value - margin and stat_present:
                            return []
                else:
                    split = tokens[i].split('=')
                    t1 = split[0]; t3 = split[1]
                    if t3.replace('.','').replace('-','').isdigit():
                        stat_present = t1 == stat.lower() or t1 in aliases if stat not in ['F','T'] else t1 == stat.lower() or t1 in aliases or ')' in t1
                        if float(t3) < value + margin and float(t3) > value - margin and stat_present:
                            return []
        if appendix == None:
            appendix:str = '' if num < 2 else self.mes['S_FORFAC'] + str(num) if num < 3 else self.mes['S_WITHINT']
        return [self.mes['F_RIGHTVAL']+stat+' '+appendix+' '+self.mes['S_NONAME']]
    
    def detect_p(self, doc:Doc, value:float, num:int=1, label:str=None, margin=0.01) -> List[str]:
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
            return [self.mes['F_RIGHTP']+label+' '+self.mes['S_NONAME']]
        else:
            appendix:str = '' if num < 2 else self.mes['S_FORFAC'] + str(num) if num < 3 else self.mes['S_WITHINT']
            return [self.mes['F_NOPHERE']+appendix+' '+self.mes['S_NONAME']]
        
    def detect_name(self, doc:Doc, solution:Dict) -> List[str]:
        names:list = []
        if solution['assignment_type'] == 1:
            names = [('t','-','toets','voor','onafhankelijke','variabelen'),('between','-','subject','t','-','test'),('t','-','toets','voor','onafhankelijke','subjecten'),
                     ('t','-','test','for','independent','samples')]
        if solution['assignment_type'] == 2:
            names = [('t','-','toets','voor','gekoppelde','paren'),('within','-','subject','t','-','test'),('t','-','test','for','paired','samples')]
        if solution['assignment_type'] == 3:
            names = [('one','-','way','anova'), ('1-factor','anova'),('1','-','way','anova')]
        if solution['assignment_type'] == 4:
            names = [('two','-','way','anova'), ('2-factor','anova'),('2','-','way','anova')]
        if solution['assignment_type'] == 5:
            names = [('repeated','-','measures','anova'), ('repeated','-','measures','-','anova'),('rmanova')]
        if solution['assignment_type'] == 6:
            names = [('regressieanalyse'),('multiple','-','regression'),('multipele', 'regressie')]
        if solution['assignment_type'] == 11:
            names = [('manova')]
        if solution['assignment_type'] == 12:
            names = [('ancova')]
        if solution['assignment_type'] == 13:
            names = [('multipele','repeated-measures','anova'),('multipele','rmanova'),('multivariate','repeated-measures-anova'),
                     ('multivariate','repeated-measures','anova'),('multivariate','variantieanalyse'),('multivariate','rmanova'),
                     ('multiple','rmanova'),('multiple','repeated-measures','anova')]
        if any([all([x in doc.text for x in y]) for y in names]):
            return ['']
        else:
            return [self.mes['F_ANALYSIS']]

    """
    HELP FUNCTIONS
    """
    def check_causality(self, independent:Doc, dependent:Doc, alternative:bool=False) -> bool:
        if not self.mes['L_ENGLISH']: #Dutch
            if not alternative:
                tuples = [('nsubj', 'obj'),('obj', 'ROOT'),('nsubj', 'nmod'),('obl', 'obj'),('ROOT', 'obj'),
                      ('obj', 'nmod'), ('amod', 'obj'), ('obl','obl'),('nsubj','obl'),('obj','obj'),('nsubj','amod'),
                      ('obj','obl'),('nmod','obj'),('obl','ROOT'),('obl','nsubj'),('obl','csubj'),('advmod','obj'),
                      ('advmod','nmod'),('advmod','obj'),('advmod','obl')]
            else: #Add reverse causality and disturbing variable options
                tuples = [('obj','obj'),('obj','nsubj'), ('ROOT','obj'),('nmod','nsubj'),('obj','obl'),('obj','ROOT'),('amod','nsubj'),
                               ('nmod','obj'),('obj','amod'),('obl','nsubj'),('obl','obj'),('obj','nmod'),('ROOT','obl'),('nsubj','obl'),
                               ('obl','obl'), ('csubj','obl'),('nsubj','conj')]
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

def distance(words:list, w1:list, w2:list) -> int:
    if not any([x in words for x in w1]):
        return 9999
    if not any([x in words for x in w2]):
        return 9999
    else:
        index1 = words.index([x for x in words if x in w1][0])
        index2 = words.index([x for x in words if x in w2][0])
        return abs(index1 - index2)
    
def negation_counter(tokens: List[str]) -> int:
    count: int = 0
    for token in tokens:
        if token in ['geen', 'niet', 'no', 'not', "isn't"]:   # or token[:2] == 'on':
            count += 1
    return count

def parse_yes_no(input_text: str) -> str:
    return input_text.lower() in ['ja','jaa','yes','yeah','yep','sure','positive','1','true']

def descendants(node) -> List[Token]:
    output:list = []
    for child in node.children:
        output.append(child)
        output += descendants(child)
    return output

def lef(synonyms:list,texts:list) -> float:
    fusedtext = ' '.join(texts)
    for t in texts:
        for s in synonyms:
            if edit_distance(t,s) < 2 or s in fusedtext:
                return True
    return False

def sim(gold_numbers :List, numbers :List, margin:float) -> True: #Return true if there is a similar number to num in the given list/float/integer in the solution
    return [gold_numbers[i] - margin < numbers[i] and numbers[i] < gold_numbers[i] + margin for i in range(len(gold_numbers))]
    
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
            elif any([x in tokens for x in ['<']]):
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
            elif any([x in tokens for x in ['>']]):
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