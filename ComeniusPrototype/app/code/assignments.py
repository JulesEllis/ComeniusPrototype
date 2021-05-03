#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:03:09 2020

@author: jelmer
"""
import math
import random
import numpy as np
import os
import copy
from scipy import stats
from scipy.stats.distributions import chi2
from typing import Dict, List, Tuple
from random import shuffle
#import typing
#from typing import *
def format_table(terms:list) -> str:
    return '<tr><td>' + '</td><td>'.join([str(round(x,2)) if type(x) != str else x for x in terms]) + '</td></tr>'

def cap(term:str) -> str:
    return term[0].upper() + term[1:]

def uncap(term:str):
    return term[0].lower() + term[1:]

#Variable, dependent or independent
class Variable:
    def __init__(self, name:str, synonyms:list, node:str):
        self.name = name
        self.synonyms = synonyms
        self.node = node
        self.levels = None
        self.nlevels = None
        self.level_syns = None
        
    def set_levels(self, levels:list, level_syns:list):
        self.levels = levels
        self.nlevels = len(levels)
        self.level_syns = level_syns
        
    def get_all_syns(self) -> list:
        return [x.lower() for x in [self.name] + self.synonyms]
    
    def get_all_level_syns(self) -> list:
        return [[x.lower() for x in [self.levels[i]] + self.level_syns[i]] for i in range(self.nlevels)]
    
    def get_varnames(self) -> list:
        return [cap(self.name)] + [cap(x) for x in self.levels]

#Object to create different assignment
class Assignments:
    #Create object either from an existing dictionary of (Dutch or English) messages or a new object
    def __init__(self, mes:dict=None):
        if mes == None:
            self.mes = None
        else:
            self.mes = mes
        
        #Import variables
        path = '/home/jelmer/Github/ComeniusPrototype/ComeniusPrototype/app/code/messages/variables.csv' if 'Github' in os.getcwd() else '/var/www/ComeniusPrototype/ComeniusPrototype/app/code/messages/variables.csv'
        self.variables = []
        with open(path, encoding='utf-8', errors='ignore') as file:
            for line in file.readlines()[1:]:
                parts = line.split(';')
                self.variables.append({'name':parts[0],'synonyms':[] if parts[4] == '' else parts[4].split(','),'node':parts[5],'levels':parts[6].split(','),
                                       'levelsyns':[parts[i].split(',') if parts[i] != '' else [] for i in range(7,11)],
                            'type':parts[1],'english':bool(int(parts[2])),'control':bool(int(parts[3])),'intro':parts[11],'intro2':parts[12][:-1]})
        
    def deserialize(self, inp:dict) -> Variable:
        output = {}
        if inp == {}:
            return None
        for x, y in list(inp.items()):
            if not 'dependent' in x:
                output[x] = y
            else:
                output[x] = Variable(y[0],y[1],y[2])
                if 'in' in x:
                    output[x].set_levels(y[3],y[4])
        return output
        
    def serialize(self, inp:dict) -> dict:
        output = {}
        if inp == None:
            return output
        for x, y in list(inp.items()):
            if not 'dependent' in x:
                output[x] = y
            else:
                output[x] = [y.name,y.synonyms,y.node,y.levels,y.level_syns]
        return output
    
    #Checks the nature of the given assignment and returns the output of the right print function
    def print_assignment(self, assignment: Dict) -> str:
        if assignment['assignment_type'] == 7:
            return self.print_report(assignment) #Beknopt rapport
        elif assignment['assignment_type'] in [1,2]:
            return self.print_ttest(assignment) #T-Test
        elif assignment['assignment_type'] in [3,4]:
            return self.print_anova(assignment) #Repeated-measures ANOVA
        elif assignment['assignment_type'] == 5:
            return self.print_rmanova(assignment) #Repeated-measures ANOVA
        elif assignment['assignment_type'] == 6:
            return self.print_mregression(assignment) #Repeated-measures ANOVA
        else:
            print('ERROR: ASSIGNMENT TYPE NOT RECOGNIZED')
            return None
    
    #Sets the message dictionary
    def set_messages(self, mes:dict):
        self.mes = mes
    
    #Returns an independent variable with the given specifications
    def get_factor(self, within_subject:bool = False, control:bool = False, multirm:bool = False, ttest:bool=False, second=False, avoid=[]) -> Tuple:
        t:str = 'BETWEEN' if not within_subject else 'WITHIN'
        nc:int = 2 if ttest else random.randint(2,4) if not multirm else 3
        varis = copy.deepcopy(self.variables) #Create a deep copy as to not permanently remove variables from the pool when taking a second factor
        if avoid != []: #Remove first independent variable from pool
            for x in avoid:
                for y in varis:
                    if y['name'] == x:
                        varis.remove(y)
                        
        var:tuple = random.choice([x for x in varis if x['type'] == t and x['english'] == self.mes['L_ENGLISH'] and x['control'] == control])
        #Get wordnet synonyms
        from nltk.corpus import wordnet as wn
        lan:str = 'eng' if self.mes['L_ENGLISH'] else 'nld'
        wn_syns = []
        for syn in wn.synsets(var['name'], lang=lan): 
            for l in syn.lemmas(lang=lan): 
                name = l.name()
                if not name in var['name'] and not '_' in name:
                    wn_syns.append(l.name())
        text:str = var['intro'] if not second else var['intro2']
        if var['synonyms'] == '':
            var['synonyms'] = []
        output = Variable(var['name'], var['synonyms'], var['node'])
        output.set_levels(var['levels'][:nc], var['levelsyns'][:nc] + wn_syns)
        return output, text
    
    #Returns a dependent variable with the given specifications
    def get_dependent(self) -> Tuple:
        var:tuple = random.choice([x for x in self.variables if x['english'] == self.mes['L_ENGLISH'] and x['type'] == 'DEPENDENT'])
        
        #Get wordnet synonyms
        from nltk.corpus import wordnet as wn
        lan:str = 'eng' if self.mes['L_ENGLISH'] else 'nld'
        wn_syns = []
        for syn in wn.synsets(var['name'], lang=lan): 
            for l in syn.lemmas(lang=lan): 
                wn_syns.append(l.name())
        
        if var['synonyms'] == '':
            var['synonyms'] = []
        output = Variable(var['name'], var['synonyms']+wn_syns, var['node'])
        return output, var['intro']
    
    #Returns lists of covariates (names and synonyms) with the given specifications
    def get_covariates(self, n:int, intercept:bool=False) -> list:
        varss:list = [x for x in self.variables if x['english'] == self.mes['L_ENGLISH'] and x['type'] == 'COVARIATE']
        
        #Get wordnet synonyms
        from nltk.corpus import wordnet as wn
        for var in varss:
            lan:str = 'eng' if self.mes['L_ENGLISH'] else 'nld'
            for syn in wn.synsets(var['name'], lang=lan): 
                for l in syn.lemmas(lang=lan): 
                    var['synonyms'].append(l.name())
                var['synonyms'] = list(set(var['synonyms']))
            
        shuffle(varss)
        output = [x['name'] for x in varss[:n]]
        outputsyns = [x['synonyms'] for x in varss[:n]]
        if intercept:
            return ['Intercept']+output[:-1], [[]]+outputsyns[:-1]
        else:
            return output, outputsyns
        
    #Creates the assignment's data as a tuple of floats
    #If the assignment is a within-subject T-test, n1 and n2 have the same number of samples
    def create_ttest(self, between_subject: bool, hyp_type: int, control: bool, elementary:bool=True) -> Dict:         
        #Number of datapoints for the both types of the independent variable
        n1: int = random.randint(9,16)
        if between_subject:
            n2: int = random.randint(9,16) 
        else: 
            n2: int = n1
        
        #Take random means and standard deviations for both types of the independent variable
        mean1: float = random.uniform(50, 200)
        std1: float = random.uniform(5,15) if between_subject else random.uniform(1,2) 
        mean2: float = random.uniform(50, 200)
        std2: float = random.uniform(5,15) if between_subject else random.uniform(1,2)
        
        #Generate the datapoints as a dictionary of where the list of float entries is 
        #described by the variable name key
        dependent, depintro = self.get_dependent()
        if between_subject:
            independent, intro = self.get_factor(within_subject=False, control=control,ttest=True)
        else:
            independent, intro = self.get_factor(within_subject=True, control=control,ttest=True)
        
        #Create the assignment description
        report_type = self.mes['S_ELEM'] if elementary else self.mes['S_SHORT']
        instruction: str = depintro + intro + '<br><br>'
        instruction += self.mes['I_CREATE']+report_type+self.mes['S_DATAHYP']
        if hyp_type == 0:
            instruction += independent.levels[0] + self.mes['S_UNEQUAL'] + independent.levels[1] + '. '
        if hyp_type == 1:
            instruction += independent.levels[0] + self.mes['S_BIGGER'] + independent.levels[1] + '. '
        if hyp_type == 2:
            instruction += independent.levels[0] + self.mes['S_SMALLER'] + independent.levels[1] + '. '
        if control:
            if between_subject:
                instruction += self.mes['I_RANDOMBETWEEN'] 
            else:
                instruction += self.mes['I_RANDOMWITHIN']
        instruction += self.mes['I_DECIMALS']
        
        #Generate datapoints: Floats are rounded to 2 decimals
        return{'instruction': instruction,
               'hypothesis': hyp_type,
               'between_subject': between_subject,
               'control': control,
               'dependent': dependent,
               'assignment_type':1 if between_subject else 2,
               'independent':independent,
               'A': [round(random.gauss(mean1,std1), 2) for i in range(n1)], 
               'B': [round(random.gauss(mean2,std2), 2) for i in range(n2)]
               }
    
    #Calculate internally all of the numbers and string values the student has to present
    def solve_ttest(self, assignment: Dict, solution: Dict) -> Dict:
        numbers: List = [assignment['A'], assignment['B']]
        names: List[str] = assignment['independent'].levels
        between_subject: bool = assignment['between_subject']
        solution['hypothesis'] = assignment['hypothesis']
        solution['assignment_type'] = assignment['assignment_type']
        
        if not between_subject:
            #Differential scores
            diff: List[float] = [numbers[0][i] - numbers[1][i] for i in range(len(numbers[0]))]
        
        #Determine variable names and types
        solution['independent'] = assignment['independent']
        solution['dependent'] = assignment['dependent']
        solution['dep_measure']: str = 'quantitative' if self.mes['L_ENGLISH'] else 'kwantitatief'
        solution['ind_measure']: str = 'qualitative' if self.mes['L_ENGLISH'] else 'kwalitatief'
        
        #Determine null hypothesis and control measure
        sign: List[str] = ['=','<=','>='][assignment['hypothesis']]
        solution['null']: str = 'h0: mu(' + names[0] + ') ' + sign + ' mu(' + names[1] + ')'
        solution['control']: str = assignment['control']#'experiment' if assignment['control'] else 'geen experiment'
        
        #Calculate numerical aggregates of datapoints
        solution['means']: List = [np.mean(numbers[0]), np.mean(numbers[1])] if between_subject else [np.mean(diff)]
        solution['stds']: List = [np.std(numbers[0], ddof=1), np.std(numbers[1],ddof=1)] if between_subject else [np.std(diff, ddof=1)]
        solution['ns']: List = [len(numbers[0]), len(numbers[1])] if between_subject else [len(diff)]
        
        #Calculate intermediate variables for the T-test
        if between_subject:
            solution['df']: List = [sum(solution['ns']) - 2]
            solution['raw_effect']: List = [solution['means'][0] - solution['means'][1]]
            sp: float = math.sqrt(
                    ((solution['ns'][0] - 1) * solution['stds'][0] ** 2 + (solution['ns'][1] - 1) * solution['stds'][1] ** 2)
                    / solution['df'][0])
            solution['sp'] = sp
            solution['relative_effect']: List = [solution['raw_effect'][0] / sp]
            solution['T']: List = [solution['relative_effect'][0] * math.sqrt(1 / (1 / solution['ns'][0] + 1 / solution['ns'][1]))]
        else:
            solution['df']: List = [len(diff) - 1]
            solution['raw_effect']: List = [np.mean(diff)]
            solution['relative_effect']: List = [solution['raw_effect'][0] / np.std(diff, ddof=1)]
            solution['T']: List = [solution['relative_effect'][0] / math.sqrt(len(diff))]
            
        #Calculate p
        solution['p']: List = None
        if assignment['hypothesis'] == 1 and solution['raw_effect'][0] < 0 or assignment['hypothesis'] == 2 and solution['raw_effect'][0] > 0:
            solution['p'] = [1.0]
        else:
            solution['p']= [stats.t.sf(np.abs(solution['T'][0]), solution['df'][0])]
        if assignment['hypothesis'] == 0:
            solution['p'][0] = min(solution['p'][0] * 2, 1.0)
            
        #Determine textual conclusions
        #Decision
        sterkte:str=self.mes['S_STRONG'] if solution['relative_effect'][0] > 0.8 else self.mes['S_MEDIUM'] if solution['relative_effect'][0] > 0.5 else self.mes['S_SMALL']
        if solution['p'][0] < 0.05:
            decision: Tuple[str] = (self.mes['S_REJECTED'],'', self.mes['S_EFFECTIS'] + sterkte + '. ')
        else:
            decision: Tuple[str] = (self.mes['S_KEPT'],self.mes['S_NOT'], '')
        #Comparison
        if self.mes['L_ENGLISH']:
            comparison: str = ['unequal','larger','smaller'][assignment['hypothesis']]
        else:
            comparison: str = ['ongelijk','groter','kleiner'][assignment['hypothesis']]
        solution['decision']: str = 'H0 ' + decision[0] + ', '+self.mes['S_AVGIS']+' ' + names[0] + ' is ' + decision[1] + comparison + self.mes['S_THAN'] + names[1] + '. ' + decision[2]
        
        #Causal interpretation
        if solution['p'][0] < 0.05:
            if assignment['control']:
                solution['interpretation']: str = self.mes['A_ONEINT'] + solution['independent'].name + self.mes['S_INFLUENCES'] + solution['dependent'].name + '.'
            else:
                solution['interpretation']: str = self.mes['A_MULTINT'] + solution['independent'].name + self.mes['S_INFLUENCES'] + solution['dependent'].name + '. '+\
                self.mes['A_ALTINT'] + solution['dependent'].name + self.mes['S_INFLUENCES'] + solution['independent'].name + '. '
        else:
            if assignment['control']: 
                solution['interpretation']: str = self.mes['A_ONEINT'] + solution['independent'].name + self.mes['S_NINFLUENCES'] + solution['dependent'].name + '.'
            else:
                solution['interpretation']: str = self.mes['A_MULTINT'] + solution['independent'].name + self.mes['S_NINFLUENCES'] + solution['dependent'].name + '. '+\
                self.mes['A_ALTINT'] + solution['independent'].name + self.mes['S_INFLUENCES'] + solution['dependent'].name + self.mes['A_NOTICEABLE']
        return solution
    
    #Create ANOVA assignment
    def create_anova(self, two_way: bool, control: bool, control2:bool=False, elementary:bool=True) -> Dict:
        output = {'two_way':two_way, 'control':control}
        output['instruction']: str = None
        output['assignment_type']: int = 4 if two_way else 3
        
        output['independent'], intro = self.get_factor(within_subject=False, control=control,ttest=False)
        output['dependent'], depintro = self.get_dependent()
        if two_way:
            output['independent2'], intro2 = self.get_factor(within_subject=False, control=control2, ttest=False, second=True, avoid=[output['independent'].name])
            output['control2'] = control2
        
        #Decide the variable names
        report_type = self.mes['S_ELEM'] if elementary else self.mes['S_SHORT']
        if two_way:
            output['instruction']:str = depintro + intro + intro2 + '<br><br>' 
        else: 
            output['instruction']:str = depintro + intro + '<br><br>'
        output['instruction'] += self.mes['I_CREATE']+report_type+self.mes['S_DATAHYP2']
        if not two_way:
            if control:
                output['instruction'] += self.mes['I_ONERANDOM']
        else:
            if control and output['control2']:
                output['instruction'] += self.mes['I_FULLRANDOM']
            elif control:
                output['instruction'] += self.mes['I_RANDOMFACTOR']+output['independent'].name+'. '
            elif output['control2']:
                output['instruction'] += self.mes['I_RANDOMFACTOR']+output['independent2'].name+'. '
        output['instruction'] += self.mes['I_DECIMALS2']
        
        #Generate summary statistics
        n: int = random.randint(9,16)
        if not two_way:
            output['data'] = {
                  'means':[round(random.uniform(50,120),2) for i in range(2)],
                  'stds':[round(random.uniform(5,15),2) for i in range(2)],
                  'ns': [n for i in range(2)]}
        else:
            output['data'] = {
                  'means':[round(random.uniform(50,120),2) for i in range(4)],
                  'stds':[round(random.uniform(5,15),2) for i in range(4)],
                  'ns': [n for i in range(4)]}
        return output
    
    #Create standard answer for the given ANVOA assignment
    def solve_anova(self, assignment: Dict, solution: Dict) -> Dict:
        data: Dict = assignment['data']
        two_way: bool = assignment['two_way']
        solution['assignment_type'] = assignment['assignment_type']
        solution['independent'] = assignment['independent']
        solution['dependent'] = assignment['dependent']
        solution['dep_n_measure']: int = 1 #Aantal metingen per persoon
        solution['dep_measure']: str = 'quantitative' if self.mes['L_ENGLISH'] else 'kwantitatief'
        solution['ind_measure']: str = 'qualitative' if self.mes['L_ENGLISH'] else 'kwalitatief'
        solution['control']: bool = assignment['control']
        if two_way:
            solution['control2']: bool = assignment['control2']
            solution['independent2'] = assignment['independent2']
        
        #One-way statistics
        #mean: float = np.mean(data['means'])
        if not two_way:
            #Intermediary statistics order: Between-group, within-group, total
            #ssm: float = sum([(data['means'][i] - mean) ** 2 for i in range(len(data['ns']))]) * (sum(data['ns']) - 1)
            ssm: float = np.std([data['means'][0] for i in range(data['ns'][0])]+[data['means'][1] for i in range(data['ns'][1])], ddof=1) ** 2 * (sum(data['ns']) - 1)
            sse: float = sum([(data['ns'][i] - 1) * (data['stds'][i]) ** 2 for i in range(len(data['ns']))])
            solution['ss']: List[float] = [ssm, sse, ssm + sse]
            solution['df']: List[float] = [len(data['ns']) - 1, abs(len(data['ns']) - sum(data['ns'])), sum(data['ns']) - 1]
            solution['ms']: List[float] = [solution['ss'][i]/solution['df'][i] for i in range(2)]
            solution['F']: List[float] = [solution['ms'][0] / solution['ms'][1]]
            solution['p']: List[float] = [1 - stats.f.cdf(abs(solution['F'][0]),solution['df'][0],solution['df'][1])]
            solution['r2']: List[float] = [solution['ss'][0]/solution['ss'][2]]
            
            #Verbal parts of the report
            if self.mes['L_ENGLISH']:
                rejected: Tuple[str] = ('rejected','unequal',' ') if solution['p'][0] < 0.05 else ('maintained', 'equal', ' not ')
            else:
                rejected: Tuple[str] = ('verworpen','ongelijk',' ') if solution['p'][0] < 0.05 else ('behouden', 'gelijk', ' niet ')
            levels = assignment['independent'].levels
            solution['null']: str = 'H0: ' + ' = '.join(['mu(' + l + ')' for l in solution['independent'].levels])
            sterkte:str=self.mes['S_STRONG'] if solution['r2'][0] > 0.2 else self.mes['S_MEDIUM'] if solution['r2'][0] > 0.1 else self.mes['S_SMALL']
            solution['decision']: str = 'H0 ' + rejected[0] + self.mes['S_AVGSARE'] + levels[0] +self.mes['S_AND']+levels[1]+self.mes['S_AREAVG']+rejected[1]+'. '
            if solution['p'][0] < 0.05:
                solution['decision'] += self.mes['S_EFFECTIS']+sterkte+'.'
            if assignment['control']:
                solution['interpretation']: str = self.mes['A_ONEINT']+solution['independent'].name+self.mes['S_INFLUENCES']+solution['dependent'].name +rejected[2] +'. '
            else:
                solution['interpretation']: str = self.mes['A_MULTINT']+solution['independent'].name+self.mes['S_INFLUENCES']+solution['dependent'].name + \
                '. '+self.mes['A_ALTINT'] + solution['independent'].name + self.mes['S_AND'] + solution['dependent'].name + self.mes['A_BOTHINT']
                
        else: #Two-way statistics
            #Intermediary statistics order: Between, A, B, AB, Within, Total
            #Degrees of freedom
            l1: int = assignment['independent'].nlevels
            l2: int = assignment['independent2'].nlevels
            levels = assignment['independent'].levels
            levels2 = assignment['independent2'].levels
            lt: int = l1 * l2
            N: int = sum(data['ns'])
            solution['df']: List[int] = [lt - 1, l1 - 1, l2 - 1, (l1-1) * (l2-1), sum(data['ns']) - lt, sum(data['ns']) - 1]
            
            #Numerical parts of the report
            ssbetween: float = (N-1) * np.std([val for sublist in [[data['means'][j] for i in range(data['ns'][j])] for j in range(lt)] for val in sublist], ddof=1) ** 2
            ssa: float = (N-1) * np.std([np.mean([data['means'][0],data['means'][1]]) for i in range(data['ns'][0] + data['ns'][1])]+[np.mean([data['means'][2],data['means'][3]]) for i in range(data['ns'][2] + data['ns'][3])], ddof=1) ** 2
            ssb: float = (N-1) * np.std([np.mean([data['means'][0],data['means'][2]]) for i in range(data['ns'][0] + data['ns'][2])]+[np.mean([data['means'][1],data['means'][3]]) for i in range(data['ns'][1] + data['ns'][3])], ddof=1) ** 2
            sswithin: float = sum([data['stds'][i] ** 2 * (data['ns'][i] - 1) for i in range(lt)])
            solution['ss']: List[float] = [ssbetween, ssa, ssb, ssbetween - ssa - ssb, sswithin, ssbetween + sswithin]
            solution['ms']: List[float] = [solution['ss'][i] / solution['df'][i] for i in range(5)]
            solution['F']: List[float] = [solution['ms'][1] / solution['ms'][4], solution['ms'][2] / solution['ms'][4], solution['ms'][3] / solution['ms'][4]]
            solution['p']: List[float] = [1 - stats.f.cdf(abs(solution['F'][i]),solution['df'][i + 1],solution['df'][4]) for i in range(3)]
            solution['r2']: List[float] = [solution['ss'][1] / solution['ss'][5], solution['ss'][2] / solution['ss'][5], solution['ss'][3] / solution['ss'][5]]
            
            #Verbal parts of the report
            total:str = 'total' if self.mes['L_ENGLISH'] else 'totaal'
            rejected: Tuple[str] = (self.mes['S_REJECTED'],self.mes['S_UNQ']) if solution['p'][0] < 0.05 else (self.mes['S_KEPT'],self.mes['S_EQ'])
            rejected2: Tuple[str] = (self.mes['S_REJECTED'],self.mes['S_UNQ']) if solution['p'][1] < 0.05 else (self.mes['S_KEPT'],self.mes['S_EQ'])
            rejected3: Tuple[str] = (self.mes['S_REJECTED'],self.mes['S_BOOLINT1']) if solution['p'][2] < 0.05 else (self.mes['S_KEPT'],self.mes['S_BOOLINT2'])
            sterkte:str = self.mes['S_STRONG'] if solution['r2'][0] > 0.2 else self.mes['S_MEDIUM'] if solution['r2'][0] > 0.1 else self.mes['S_SMALL']
            sterkte2:str = self.mes['S_STRONG'] if solution['r2'][1] > 0.2 else self.mes['S_MEDIUM'] if solution['r2'][1] > 0.1 else self.mes['S_SMALL']
            sterkte3:str = self.mes['S_STRONG'] if solution['r2'][2] > 0.2 else self.mes['S_MEDIUM'] if solution['r2'][2] > 0.1 else self.mes['S_SMALL']
            solution['control2'] = assignment['control2']
            solution['null']: str = 'H0: mu(' + levels[0] + ') = mu(' + levels[1] + ')'
            solution['null2']: str =  'H0: mu(' + levels2[0] + ') = mu(' + levels2[1] + ')'
            solution['null3']: str = 'H0(' + solution['independent'].name + ' x ' + solution['independent2'].name + '): mu('+levels[0] + ' & ' + levels2[0]+') = mu('+levels[0]+') + mu('+levels2[0]+') - mu('+total+') [...]'\
                    +self.mes['S_AND']+'mu('+levels[-1] + ' & ' + levels2[-1]+') = mu('+levels[-1]+') + mu('+levels2[-1]+') - mu('+total+')'
            solution['decision']: str = 'H0 ' + rejected[0]+self.mes['S_AVGSARE'] + levels[0]+self.mes['S_AND']+levels[1]+self.mes['S_AREAVG']+rejected[1] + '. '
            solution['decision2']: str = 'H0 ' + rejected2[0]+self.mes['S_AVGSARE'] + levels2[0]+self.mes['S_AND']+levels2[1]+self.mes['S_AREAVG']+rejected2[1] + '. '
            solution['decision3']: str = 'H0 ' + rejected3[0]+self.mes['S_THEREIS'] + rejected3[1] + self.mes['S_INBETWEEN'] + solution['independent'].name +self.mes['S_AND']+solution['independent2'].name + self.mes['S_INPOP']
            if solution['p'][0] < 0.05: solution['decision'] += self.mes['S_EFFECTIS'] + sterkte + '.'
            if solution['p'][1] < 0.05: solution['decision2'] += self.mes['S_EFFECTIS'] + sterkte2 + '.'
            if solution['p'][2] < 0.05: solution['decision3'] += self.mes['S_EFFECTIS'] + sterkte3 + '.'
            n1 = self.mes['S_INFLUENCES'] if solution['p'][0] < 0.05 else self.mes['S_NINFLUENCES']
            if assignment['control']:
                solution['interpretation']: str = self.mes['A_ONEINT']+solution['independent'].name+n1+solution['dependent'].name+'.'
            else:
                solution['interpretation']: str = self.mes['A_MULTINT']+solution['independent'].name+n1+solution['dependent'].name+ '. '+\
                    self.mes['A_ALTIS'] + solution['independent'].name + self.mes['S_AND'] + solution['dependent'].name + self.mes['A_BOTHINT']
                    
            n2 = self.mes['S_INFLUENCES'] if solution['p'][1] < 0.05 else self.mes['S_NINFLUENCES']
            if assignment['control2']:
                solution['interpretation2']: str = self.mes['A_ONEINT']+solution['independent2'].name+n2+solution['dependent'].name+'.'
            else:
                solution['interpretation2']: str = self.mes['A_MULTINT']+solution['independent2'].name+n1+solution['dependent'].name + '. '+\
                    self.mes['A_ALTIS'] + solution['independent2'].name + self.mes['S_AND'] + solution['dependent'].name + self.mes['A_BOTHINT']
                    
            n3 = self.mes['A_NOTSAMEINF'] if solution['p'][2] < 0.05 else self.mes['A_SAMEINF']
            if assignment['control'] or assignment['control2']:
                solution['interpretation3']: str = self.mes['A_ONEINT']+solution['independent'].name+n3+solution['dependent'].name+self.mes['A_FORLEVELS']+self.mes['S_AND'].join(levels2) + self.mes['A_OFFACTOR'] + solution['independent2'].name + '.'
            else:
                solution['interpretation3']: str = self.mes['A_MULTINT']+solution['independent'].name+n3+solution['dependent'].name+self.mes['A_FORLEVELS']+self.mes['S_AND'].join(levels2) + self.mes['A_OFFACTOR'] + solution['independent2'].name + '. '+\
                    self.mes['S_DISTURBANCE2']
        return solution
    
    #Create an RMANVOA assignment for the given parameters
    def create_rmanova(self, control: bool, elementary:bool=True) -> Dict:
        #Determine variable shape and names
        output = {'control': control, 'two_way':False, 'assignment_type':5}
        n_conditions = random.randint(2,4)
        n_subjects = int(random.uniform(8,15))
        
        output['independent'], intro = self.get_factor(within_subject=True, control=True,ttest=False)
        output['dependent'], depintro = self.get_dependent()
        
        report_type = self.mes['S_ELEM'] if elementary else self.mes['S_SHORT']
        output['instruction']: str = intro + depintro + '<br><br>' + self.mes['I_CREATE']+report_type+self.mes['S_DATAHYP2']
        output['instruction'] += self.mes['I_DECIMALS2']
            
        true_means = [int(random.uniform(50,120)) for i in range(n_conditions)]
        true_stds = [int(random.uniform(5,20)) for i in range(n_conditions)]
        output['data'] = {
                  'scores':[[round(random.gauss(true_means[i], true_stds[i]),2) for j in range(n_subjects)] for i in range(n_conditions)]
                  }
        output['data']['means']: List = [round(np.mean(output['data']['scores'][i]),2) for i in range(n_conditions)]
        output['data']['stds']: List = [round(np.std(output['data']['scores'][i], ddof=1),2) for i in range(n_conditions)]
        output['data']['jackedmeans']: List = [round(np.mean([output['data']['scores'][i][j] for i in range(n_conditions)]),2) for j in range(n_subjects)]
        output['data']['n_subjects'] = n_subjects
        output['data']['n_conditions'] = n_conditions
        return output
    
    #Create standard answer for the given RMANVOA assignment
    def solve_rmanova(self, assignment: Dict, solution: Dict) -> Dict: 
        data: Dict = assignment['data']
        n_conditions = len(data['means'])
        solution['independent'] = assignment['independent']
        solution['dependent'] = assignment['dependent']
        solution['dep_measure']: str = 'kwantitatief'
        solution['dep_n_measure']: int = n_conditions #Aantal metingen per persoon
        solution['ind_measure']: str = 'kwalitatief'
        solution['control']: bool = assignment['control']
        solution['assignment_type'] = assignment['assignment_type']
        solution['n_subjects'] = data['n_subjects']
        
        #Numerical parts of the report
        #Order of rows: Kwartaal, Persoon, Interactie, Totaal
        nc: int = data['n_conditions']
        ns: int = data['n_subjects']
        N: int = nc * ns
        solution['df']: List[int] =  [nc - 1, ns - 1, (nc-1) * (ns-1), N - 1]
        ssk: float = (N-1) * np.std(data['means'] * ns, ddof=1) ** 2
        ssp: float = (N-1) * np.std(data['jackedmeans'] * nc, ddof=1) ** 2
        sstotal: float = (N-1) * np.std([x for y in [data['scores']] for x in y], ddof=1) ** 2
        ssi = sstotal - ssk - ssp
        solution['ss'] = [ssk, ssp, ssi, sstotal]
        solution['ms']: List[float] = [solution['ss'][i] / solution['df'][i] for i in range(4)]
        solution['F']: List[float] = [solution['ms'][0] / solution['ms'][2], solution['ms'][1] / solution['ms'][2]]
        solution['p']: List[float] = [1 - stats.f.cdf(solution['F'][i],solution['df'][i], solution['df'][2]) for i in range(2)]
        solution['r2']: List[float] = [solution['ss'][0] / solution['ss'][3], solution['ss'][1] / solution['ss'][3]]
        
        #Textual parts of the report
        solution['null']: str = 'H0: ' + ' = '.join(['mu(' + x + ')' for x in assignment['independent'].levels])
        solution['null2']: str = 'H0: tau(subject 1) = tau(subject 2) [...] = tau(subject '+str(data['n_subjects'])+')'
        rejected: Tuple[str] = (self.mes['S_REJECTED'],self.mes['S_UNQ']) if solution['p'][0] < 0.05 else (self.mes['S_KEPT'],self.mes['S_EQ'])
        solution['decision']: str = 'H0 ' + rejected[0] + self.mes['S_AVGSARE']+solution['dependent'].name+self.mes['A_FORLEVELS']+ self.mes['S_AND'].join(assignment['independent'].levels) + self.mes['S_AREAVG'] + rejected[1] + '. '
        if solution['p'][0]:
            sterkte:str = self.mes['S_STRONG'] if solution['r2'][0] > 0.2 else self.mes['S_MEDIUM'] if solution['r2'][0] > 0.1 else self.mes['S_SMALL']
            solution['decision'] += self.mes['S_EFFECTIS']+sterkte+'.'
        
        rejected2: Tuple[str] = (self.mes['S_REJECTED'],self.mes['S_UNQ']) if solution['p'][1] < 0.05 else (self.mes['S_KEPT'],self.mes['S_EQ'])
        solution['decision2']: str = 'H0 ' + rejected2[0] + self.mes['A_BMEANS'] + rejected2[1] + '. '
        if solution['p'][1] < 0.05:
            sterkte2:str = self.mes['S_STRONG'] if solution['r2'][1] > 0.2 else self.mes['S_MEDIUM'] if solution['r2'][1] > 0.1 else self.mes['S_SMALL']
            solution['decision2'] += self.mes['S_EFFECTIS']+sterkte2+'. '    
            
        n1 = self.mes['S_INFLUENCES'] if solution['p'][0] < 0.05 else self.mes['S_NINFLUENCES']
        if assignment['control']:
            solution['interpretation']: str = self.mes['A_ONEINT']+solution['independent'].name+n1+solution['dependent'].name
        else:
            solution['interpretation']: str = self.mes['A_MULTINT']+solution['independent'].name+n1+solution['dependent'].name + '. '+\
                self.mes['A_ALTINT'] + solution['dependent'].name + self.mes['S_INFLUENCES'] + solution['dependent'].name
        return solution
    
    #Create a multiple regression assignment for the given parameters
    def create_mregression(self, control: bool, elementary:bool=False):
        report_type = self.mes['S_ELEM'] if elementary else self.mes['S_SHORT']
        output = {'assignment_type':6}
        N = 50 + int(150 * random.random())
        output['ns'] = [N]
        n_predictors = random.choice([3,4,5,6])
        output['n_predictors'] = n_predictors
        
        p = random.random()
        s = 3 * random.random()
        output['var_obs'] = (1 + chi2.ppf(p, df=10)) * 10 ** s
        r2 = random.random() ** 2
        output['var_pred'] = output['var_obs'] * r2
        output['levels'], output['level_syns'] = self.get_covariates(n_predictors, intercept=True)
        output['data']: dict={'predictoren':output['levels']}
        output['dependent'], depintro = self.get_dependent()
        #output['correlations'] = [random.random() for i in range(int(((n_predictors + 1) ** 2 - n_predictors - 1) * 0.5))]
        output['instruction'] = self.mes['I_CREATE']+report_type+self.mes['S_VARSARE2']+self.mes['S_AND'].join(output['data']['predictoren'][1:])+self.mes['S_PREDICTORS']+self.mes['S_AND']+\
            output['dependent'].name+self.mes['S_CRITERIUM']+'. '+self.mes['I_DECSHORT']
        return output
    
    #Create a standard answer for the given multiple regression assignment
    def solve_mregression(self, assignment: Dict, solution:Dict) -> Dict:
        N = assignment['ns'][0]
        solution['assignment_type'] = assignment['assignment_type']
    
        #compute ANOVA table
        ssreg = (N-1) * assignment['var_pred']
        sstotal = (N-1) * assignment['var_obs']
        solution['df']: list[int] = [3, N-3-1, N-1]
        solution['ss']: list[float] = [ssreg, sstotal - ssreg, sstotal]
        solution['ms']: List[float] = [solution['ss'][i] / solution['df'][i] for i in range(2)]
        solution['F']: List[float] = [solution['ms'][0] / solution['ms'][1]]
        solution['p']: List[float] = [1-stats.f.cdf(solution['F'][0],solution['df'][0], solution['df'][1])]
        solution['r2']: List[float] = [min(ssreg/sstotal,1.00)]
        
        #Compute p-values
        n_predictors = assignment['n_predictors']
        solution['predictor_p'] = [random.random() * 0.05 if random.choice([True, False]) else random.random() * 0.95 + 0.05 for x in range(n_predictors+1)]
        for i in range(len(solution['predictor_p'])):
            if round(solution['predictor_p'][i], 2) == 0.05:
                solution['predictor_p'][i] += 0.01
        solution['predictor_t'] = [stats.t.isf(solution['predictor_p'][i],1,N - 3 - 1) for i in range(n_predictors+1)]
        solution['predictor_beta'] = [np.mean([random.uniform(60,120)])] + [abs(random.gauss(0,0.5)) for i in range(n_predictors)]
        solution['predictor_b'] = [x * np.sqrt(assignment['var_pred']) for x in solution['predictor_beta']]
        solution['predictor_se'] = [solution['predictor_b'][i]/solution['predictor_t'][i] for i in range(n_predictors+1)]
        
        #Verbal answers
        solution['null'] = 'H0: ' + ' = '.join(['beta(' + str(i) + ')' for i in range(1,4)]) + ' = 0'
        return solution
    
    #Create an ANCOVA assignment for the given parameters
    def create_ancova(self, control: bool, elementary:bool=False):
        report_type = self.mes['S_ELEM'] if elementary else self.mes['S_SHORT']
        output = {'assignment_type':12}
        output['independent'], intro = self.get_factor(within_subject=False, control=control,ttest=False)
        N = N = 50 + int(150 * random.random())
        output['ns'] = [N]
        output['n_predictors'] = 2
        
        p = random.random() * 0.05 if random.choice([True, False]) else random.random() * 0.95 + 0.05
        s = 3 * random.random()
        output['var_obs'] = (1 + chi2.ppf(p, df=10)) * 10 ** s
        r2 = random.random() ** 2
        output['var_pred'] = output['var_obs'] * r2
        output['predictor_names'], output['predictor_syns'] = self.get_covariates(2, intercept=False)
        output['data']: dict={'predictoren':output['predictor_names']}
        output['dependent'], depintro = self.get_dependent()
        #output['correlations'] = [random.random() for i in range(int(((n_predictors + 1) ** 2 - n_predictors - 1) * 0.5))]
        output['instruction'] = self.mes['I_CREATE']+report_type+self.mes['S_REPSHORT']+self.mes['I_INDEP']+output['independent'].name+self.mes['S_AND']+' '+uncap(self.mes['I_DEP'])+output['dependent'].name+\
            self.mes['I_METWITH']+' '+self.mes['S_AND'].join(output['data']['predictoren'])+self.mes['S_PREDICTORS']+'. '+self.mes['I_DECSHORT']
        return output
    
    #Create a standard answer for the given ANCOVA assignment
    def solve_ancova(self, assignment: Dict, solution:Dict) -> Dict:
        N = assignment['ns'][0]
        solution['assignment_type'] = assignment['assignment_type']
    
        #compute ANOVA table
        n_predictors = len(assignment['predictor_names'])
        ssreg = (N-1) * assignment['var_pred']
        pred_ss = [random.random() * 0.25 * ssreg, random.random() * 0.25 * ssreg]
        sstotal = (N-1) * assignment['var_obs']
        solution['df']: list[int] = [1,1,2,4,N-5,N-1,1,N]
        solution['ss']: list[float] = [pred_ss[0], pred_ss[1], ssreg - sum(pred_ss), ssreg, sstotal - ssreg, sstotal, random.random() * sstotal, sstotal+10*sstotal*random.random()]
        solution['ms']: List[float] = [solution['ss'][i] / solution['df'][i] for i in range(8)]
        solution['F']: List[float] = [solution['ms'][i] / solution['ms'][5] for i in range(8)]
        solution['p']: List[float] = [1-stats.f.cdf(solution['F'][i],solution['df'][i], solution['df'][5]) for i in range(8)]
        solution['eta']: List[float] = [solution['ss'][i] / solution['ss'][5] for i in range(8)]
        #solution['eta']: List[float] = [solution['ss'][i] / (solution['ss'][i]+solution['ss'][4]) for i in range(4)]
        
        #Compute p-values
        n_predictors = assignment['n_predictors']
        solution['predictor_p'] = [random.random() * 0.05 if random.choice([True, False]) else random.random() * 0.95 + 0.05 for x in range(n_predictors+1)]
        for i in range(len(solution['predictor_p'])):
            if round(solution['predictor_p'][i], 2) == 0.05:
                solution['predictor_p'][i] += 0.01
        solution['predictor_t'] = [stats.t.isf(solution['predictor_p'][i],1,N - 3 - 1) for i in range(n_predictors+1)]
        solution['predictor_beta'] = [np.mean([random.uniform(60,120)])] + [abs(random.gauss(0,0.5)) for i in range(n_predictors)]
        solution['predictor_b'] = [x * np.sqrt(assignment['var_pred']) for x in solution['predictor_beta']]
        solution['predictor_se'] = [solution['predictor_b'][i]/solution['predictor_t'][i] for i in range(n_predictors+1)]
        for i in range(n_predictors): #Set predictor p-values to predictor p-values
            solution['p'][i] = solution['predictor_p'][i]
        
        #Verbal answers
        solution['null'] = 'H0: ' + ' = '.join(['beta(' + str(i) + ')' for i in range(1,4)]) + ' = 0'
        return solution

    #Create an MANOVA assignment for the given parameters
    def create_manova(self, control: bool, control2:bool=False, elementary:bool=False):
        output = {'assignment_type':12}
        report_type = self.mes['S_ELEM'] if elementary else self.mes['S_SHORT']
        p = random.random() * 0.05 if random.choice([True, False]) else random.random() * 0.95 + 0.05
        s = 3 * random.random()
        output['var_obs'] = (1 + chi2.ppf(p, df=10)) * 10 ** s
        output['var_pred'] = [output['var_obs'] * random.random() ** 2 for i in range(3)]
        
        output['independent'], intro = self.get_factor(within_subject=False, control=control,ttest=False)
        output['control'] = control
        if not self.mes['L_ENGLISH']:
            output['dependent'] = Variable('gewicht',['gewichten'],'gewicht')
            output['dependent2'] = Variable('lengte',['leeftijden'],'lengte')
            output['dependent3'] = Variable('leeftijd',['leeftijden'],'leeftijd')
        else:
            output['dependent'] = Variable('weight',['weights'],'weight')
            output['dependent2'] = Variable('length',['lengths'],'length')
            output['dependent3'] = Variable('age',['ages'],'age')
        dependents = [output['dependent'].name, output['dependent2'].name, output['dependent3'].name]
        N = 50 + int(150 * random.random()); output['ns'] = [N]
        
        output['instruction'] = self.mes['I_CREATE']+report_type+self.mes['I_WITHINDEP']+\
            output['independent'].name+' ('+', '.join(output['independent'].levels)+')'\
            '. '+self.mes['I_MULTIVAR']+self.mes['S_AND'].join(dependents)+'. '+self.mes['I_DECSHORT']
        return output
    
    #Create a standard answer for the given MANOVA assignment
    def solve_manova(self, assignment: Dict, solution:Dict) -> Dict:
        solution = {}
        N = assignment['ns'][0]
        for key, value in list(assignment.items()):
            solution[key] = value
        for j in range(3):
            ssreg = (N-1) * assignment['var_pred'][j]
            pred_ss = random.random() * 0.5 * ssreg
            sstotal = (N-1) * assignment['var_obs']
            levels = assignment['independent'].levels
            solution['df'+str(j)]: List[float] = [len(levels) - 1, N - len(levels), N - 1]
            solution['ss'+str(j)]: List[float] = [pred_ss, sstotal-pred_ss, sstotal]
            solution['ms'+str(j)]: List[float] = [solution['ss'+str(j)][i]/solution['df'+str(j)][i] for i in range(2)]
            solution['F'+str(j)]: List[float] = [solution['ms'+str(j)][0] / solution['ms'+str(j)][1]]
            solution['p'+str(j)]: List[float] = [1 - stats.f.cdf(abs(solution['F'+str(j)][0]),solution['df'+str(j)][0],solution['df'+str(j)][1])]
            solution['eta'+str(j)]: List[float] = [solution['ss'+str(j)][0] / solution['ss'+str(j)][2]] #+solution['ss'][j][2])]
        solution['F_multivar'] = np.mean([solution['F'+str(j)][0] for i in range(3)])
        solution['p_multivar'] = np.mean([solution['p'+str(j)][0] for i in range(3)])
        solution['eta_multivar'] = np.mean([solution['eta'+str(j)][0] for i in range(3)])
        
        #Fill table 1 vars
        #eigenvalues = []
        v0 = random.random()*120; solution['value'] = [random.random(),random.random(),v0,v0]+[random.random(),random.random(),random.random()*2.5,random.random()*2.5]
        f0 = random.random()*500; solution['F_t1'] = [f0 for i in range(4)] + [solution['F_multivar'] for i in range(4)]
        solution['p_t1'] = [0.0 for i in range(4)] + [solution['p_multivar'] for i in range(4)]
        intr2 = 0.9*random.random()*0.10; solution['eta_t1'] = [intr2 for i in range(4)] + [solution['eta_multivar'] for i in range(4)]
        solution['hdf'] = [assignment['independent'].nlevels - 1 for i in range(8) for i in range(8)]
        solution['edf'] = [N - assignment['independent'].nlevels for i in range(8) for i in range(8)]
        
        #Fill table 2 vars
        intercepts = [30000 + 2000 * random.random() for i in range(3)]
        solution['ss_t2'] = [solution['ss'+str(i)][0] for i in range(3)]+intercepts+[solution['ss'+str(i)][0] for i in range(3)]+\
                           [solution['ss'+str(i)][1] for i in range(3)]+[solution['ss'+str(i)][2]+intercepts[i] for i in range(3)]+\
                           [solution['ss'+str(i)][2] for i in range(3)]
        nt = sum(assignment['ns']); nl = assignment['independent'].nlevels #Total subjects #Number of levels per factor
        dfs = [nl-1,nl-1,nl-1,1,1,1,nl-1,nl-1,nl-1,nt-nl,nt-nl,nt-nl,nt,nt,nt,nt-1,nt-1,nt-1]
        solution['ms_t2'] = [solution['ss_t2'][i] / dfs[i] for i in range(12)]
        solution['F_t2'] = [solution['ms_t2'][i] / solution['ms'+str(i%3)][1] for i in range(9)]
        solution['p_t2'] = [1-stats.f.cdf(solution['F_t2'][i],dfs[i],nt-nl-1) for i in range(9)]
        solution['eta_t2'] = [solution['ss_t2'][i] / solution['ss'+str(i%3)][2] for i in range(9)]
        return solution
    
    #Create a multiple repeated-measures ANOVA assignment for the given parameters
    def create_multirm(self, control: bool, control2:bool=False, elementary:bool=False) -> dict:
        output = {'assignment_type':13}
        report_type = self.mes['S_ELEM'] if elementary else self.mes['S_SHORT']
        p = random.random() * 0.05 if random.choice([True, False]) else random.random() * 0.95 + 0.05
        s = 3 * random.random()
        output['ns'] = [int(random.random() * 65) + 10, int(random.random() * 65) + 10]
        output['var_obs'] = (1 + chi2.ppf(p, df=10)) * 10 ** s
        output['var_pred'] = output['var_obs'] * random.random() ** 2
        
        output['independent'], intro1 = self.get_factor(within_subject=True, control=True,ttest=False, multirm=True)
        output['control'] = control
        output['independent2'], intro2 = self.get_factor(within_subject=False, control=control2, ttest=False, second=True, multirm=True)
        output['control2'] = control2
        
        output['dependent'], depintro = self.get_dependent()
        output['instruction'] = self.mes['I_CREATE']+report_type+self.mes['I_WITHFACTORS']+\
            output['independent'].name+' ('+', '.join(output['independent'].levels)+')'+self.mes['S_AND'] + output['independent2'].name + ' ('+', '.join(output['independent2'].levels)+')'\
            '. '+self.mes['I_DEP']+output['dependent'].name+'. '+self.mes['I_DECSHORT']
        return output
    
    #Create a standard answer for the given multiple repeated-measures ANOVA assignment
    def solve_multirm(self, assignment: Dict, solution:Dict) -> Dict:
        solution = {}
        N = sum(assignment['ns']); 
        ntimes = assignment['independent'].nlevels; 
        nlev = assignment['independent2'].nlevels
        for key, value in list(assignment.items()):
            solution[key] = value
        # Tests of Between-Subject Effects
        ssm = (N-1) * assignment['var_pred']
        sstotal = (N-1) * assignment['var_obs']
        solution['df']: List[float] = [1,nlev - 1,N - nlev]
        solution['p'] = [random.random() * 0.05 if random.choice([True, False]) else random.random() * 0.95 + 0.05 for i in range(2)]
        solution['F'] = [stats.f.isf(abs(solution['p'][i]),solution['df'][i],solution['df'][2]) for i in range(2)]
        mse = (sstotal - ssm) * solution['df'][2]
        solution['ms'] = [solution['F'][i] * mse for i in range(2)] + [mse]
        solution['ss'] = [solution['ms'][i] * solution['df'][i] for i in range(3)]
        solution['eta'] = [solution['ss'][i]/(solution['ss'][i] + mse) for i in range(2)]
        
        # Multivariate Tests
        solution['value'] = [random.random() for i in range(8)]
        solution['hdf'] = [(ntimes-1) * 2 for i in range(2)]
        solution['edf'] = [N - 1 - (ntimes-1) * 2 for i in range(2)]
        solution['F0'] = [assignment['var_obs'] * random.random() ** 2 for i in range(2)]
        solution['p0'] = [random.random() * 0.05 if random.choice([True, False]) else random.random() * 0.95 + 0.05 for i in range(2)]
        solution['eta0'] = [solution['value'][4*i] for i in range(2)]
        
        # Tests of Within-Subjects Contrasts
        solution['df1'] = [1 for i in range(4)] + [N-2 for i in range(2)]
        solution['p1'] = [random.random() * 0.05 if random.choice([True, False]) else random.random() * 0.95 + 0.05 for i in range(4)]
        solution['F1'] = [stats.f.isf(abs(solution['p1'][i]),solution['df1'][i],solution['df1'][4+i%2]) for i in range(4)]
        mse = [random.random() * 40 for i in range(2)]
        solution['ms1'] = [solution['F1'][i] * mse[i%2] for i in range(4)] + mse
        solution['ss1'] = [solution['ms1'][i] * solution['df1'][i] for i in range(6)]
        solution['eta1'] = [solution['ss1'][i] / (solution['ss1'][i] + solution['ss1'][4+i%2]) for i in range(4)]
        return solution
        
    def create_multirm2(self, control: bool, control2:bool=False, elementary:bool=False):
        output = {'assignment_type':13}
        return output
    
    def solve_multirm2(self, assignment: Dict, solution:Dict) -> Dict:
        solution = {}
        N = sum(assignment['ns']); ntimes = len(assignment['independent'].levels); nlevels = len(assignment['independent2'].levels)
        for key, value in list(assignment.items()):
            solution[key] = value
        
        # Tests of Between-Subject Effects
        ssm = [(N-1) * assignment['var_pred'][j] for j in range(2)]
        sstotal = [(N-1) * assignment['var_obs'][j] for j in range(2)]
        nlev = assignment['independent2'].nlevels
        
        solution['df']: List[float] = [1,1,nlev - 1, nlev - 1,N - nlev, N - nlev]
        solution['p'] = [random.random() * 0.05 if random.choice([True, False]) else random.random() * 0.95 + 0.05 for i in range(4)]
        solution['F'] = [stats.f.isf(abs(solution['p'][i]),solution['df'][i],solution['df'][(i+4) % 2]) for i in range(4)]
        mse = [(sstotal[i] - ssm[i]) * solution['df'][i+4] for i in range(2)]
        solution['ms'] = [solution['F'][i] * mse[i%2] for i in range(4)] + mse
        solution['ss'] = [solution['ms'][i] * solution['df'][i] for i in range(6)]
        solution['eta'] = [solution['ss'][i]/(solution['ss'][i] + mse[i%2]) for i in range(4)]
        
        # Multivariate Tests
        solution['value'] = [random.random() for i in range(16)]
        solution['hdf'] = [(nlevels-1) * 2 for i in range(2)] + [(ntimes-1) * 2 for i in range(2)]
        solution['edf'] = [N - 1 - (nlevels-1) * 2 for i in range(2)] + [N - 1 - (ntimes-1) * 2 for i in range(2)]
        solution['F0'] = [assignment['var_obs'][i%2] * random.random() ** 2 for i in range(4)]
        solution['p0'] = [random.random() * 0.05 if random.choice([True, False]) else random.random() * 0.95 + 0.05 for i in range(4)]
        solution['eta0'] = [solution['value'][4*i] for i in range(4)]
        
        # Tests of Within-Subjects Effects
        solution['df1'] = [1 for i in range(8)] + [N-2 for i in range(4)]
        solution['p1'] = [random.random() * 0.05 if random.choice([True, False]) else random.random() * 0.95 + 0.05 for i in range(8)]
        solution['F1'] = [stats.f.isf(abs(solution['p1'][i]),solution['df1'][i],solution['df1'][(i+4) % 2]) for i in range(8)]
        mse = [random.random() * 40 for i in range(4)]
        solution['ms1'] = [solution['F1'][i] * mse[i%4] for i in range(8)] + mse
        solution['ss1'] = [solution['ms1'][i] * solution['df1'][i] for i in range(12)]
        solution['eta1'] = [solution['ss1'][i] / (solution['ss1'][i] + solution['ss1'][4+i%2]) for i in range(8)]
        return solution
	
    #Create a merged assignment/solution dictionary for the given assignment type
    def create_report(self, control: bool, choice: int=0):
        hyp_type = random.choice([0,1,2])
        if choice == 1:
            assignment = self.create_ttest(True, hyp_type, control, False)
            output = {**assignment, **self.solve_ttest(assignment, {})}
        if choice == 2:
            assignment = self.create_ttest(False, hyp_type, True, False)
            output = {**assignment, **self.solve_ttest(assignment, {})}
        if choice == 3:
            assignment = self.create_anova(False, control, False)
            output = {**assignment, **self.solve_anova(assignment, {})}
        if choice == 4:
            assignment = self.create_anova(True, control, False)
            output = {**assignment, **self.solve_anova(assignment, {})}
        if choice == 5:
            assignment = self.create_rmanova(control, False)
            output = {**assignment, **self.solve_rmanova(assignment, {})}
        if choice == 6:    
            assignment = self.create_mregression(control, False)
            output = {**assignment, **self.solve_mregression(assignment, {})}
        if choice == 11:    
            assignment = self.create_manova(control, False)
            output = {**assignment, **self.solve_manova(assignment, {})}
        if choice == 12:    
            assignment = self.create_ancova(control, False)
            output = {**assignment, **self.solve_ancova(assignment, {})}
        if choice == 13:    
            assignment = self.create_multirm(control, False)
            output = {**assignment, **self.solve_multirm(assignment, {})}
        if choice == 14:    
            assignment = self.create_multirm2(control, False)
            output = {**assignment, **self.solve_multirm2(assignment, {})}
        output['assignment_type'] = choice
        return output
            
    def print_ttest(self, assignment: Dict) -> str:
        output_text = assignment['instruction'] + '<br>'
        varnames: List[str] = assignment['independent'].get_varnames()
        data: List = [assignment['A'], assignment['B']]
        output_text += '<table style="width:20%">'
        if assignment['between_subject']:
            output_text += '<tr><td>' + cap(varnames[1]) + '</td><td>' + cap(varnames[2]) + '</td></tr>'
        else:
            output_text += '<tr><td>Nr</td><td>' + cap(varnames[1]) + '</td><td>' + cap(varnames[2]) + '</td></tr>'
        for i in range(max(len(data[0]),len(data[1]))):
            if i < len(data[0]):
                d1 = str(data[0][i])
            else: d1 = ''
            if i < len(data[1]):
                d2 = str(data[1][i])
            else: d2 = ''
            if assignment['between_subject']:
                output_text += '<tr><td>' + d1 + '</td><td>' + d2 + '</td></tr>'
            else:
                output_text += '<tr><td>' + str(i+1) + '</td><td>' + d1 + '</td><td>' + d2 + '</td></tr>'
        return output_text + '</table>'
    
    def print_anova(self, assignment: Dict) -> str: 
        data: Dict = assignment['data']
        data['varnames'] = [assignment['independent'].get_varnames()]
        if assignment['assignment_type'] == 4:
            data['varnames'].append(assignment['independent2'].get_varnames())
        #Print variables
        output_text = assignment['instruction'] + '<br><table style="width:30%">'
        if not assignment['two_way']:
            output_text += '<tr><td>Gewicht:</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][0] + '</td><td>' + self.mes['A_MEAN'] +'</td><td>'+self.mes['A_STD']+'</td><td>N' + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][1] + '</td><td>' + str(data['means'][0]) + '</td><td>' + str(data['stds'][0]) + '</td><td>' + str(data['ns'][0]) + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][2] + '</td><td>' + str(data['means'][1]) + '</td><td>' + str(data['stds'][1]) + '</td><td>' + str(data['ns'][1]) + '</td></tr>'
        else:
            output_text += '<tr><td>'+self.mes['A_MEANS']+'</td></tr>'
            output_text += '<tr><td>'+self.mes['A_LEVEL']+'</td><td>' + data['varnames'][1][1] + '</td><td>' + data['varnames'][1][2] + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][1] + '</td><td>' + str(data['means'][0]) + '</td><td>' + str(data['means'][1]) + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][2] + '</td><td>' + str(data['means'][2]) + '</td><td>' + str(data['means'][3]) + '</td></tr>'
            output_text += '<tr><td>'+self.mes['A_STDS']+'</td></tr>'
            output_text += '<tr><td>'+self.mes['A_LEVEL']+'</td><td>' + data['varnames'][1][1] + '</td><td>' + data['varnames'][1][2] + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][1] + '</td><td>' + str(data['stds'][0]) + '</td><td>' + str(data['stds'][1]) + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][2] + '</td><td>' + str(data['stds'][2]) + '</td><td>' + str(data['stds'][3]) + '</td></tr>'
            output_text += '<tr><td>N:</td></tr>'
            output_text += '<tr><td>'+self.mes['A_LEVEL']+'</td><td>' + data['varnames'][1][1] + '</td><td>' + data['varnames'][1][2] + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][1] + '</td><td>' + str(data['ns'][0]) + '</td><td>' + str(data['ns'][1]) + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][2] + '</td><td>' + str(data['ns'][2]) + '</td><td>' + str(data['ns'][3]) + '</td></tr>'
        return output_text + '</table>'
    
    def print_rmanova(self, assignment: Dict) -> str:
        data: Dict = assignment['data']
        n_conditions = len(data['means'])
        levels = assignment['independent'].levels
        output_text = assignment['instruction'] + '<br><table style="width:45%">'
        output_text += '<tr><td>'+cap(assignment['independent'].name)+'</td>' + ''.join(['<td>'+cap(x)+'</td>' for x in levels[:n_conditions]]) + '<td>'+self.mes['A_BOOSTED']+'</td></tr>'
        output_text += '<tr><td>'+self.mes['A_MEAN']+'</td>' + ''.join(['<td>'+str(x)+'</td>' for x in data['means'][:n_conditions]]) + '<td>' + str(round(np.mean(data['jackedmeans']),2)) + '</td></tr>'
        output_text += '<tr><td>'+self.mes['A_STD']+'</td>' + ''.join(['<td>'+str(x)+'</td>' for x in data['stds'][:n_conditions]]) + '<td>' + str(round(np.std(data['jackedmeans'], ddof=1),2)) + '</td></tr>'
        output_text += '<tr><td>N</td><td>' + str(data['n_subjects']) + '</td></tr></table>'
        output_text += '<br>'+self.mes['A_OGSCORES']
        output_text += '<br><table style="width:45%">'
        output_text += '<tr><td>Subject</td>' + ''.join(['<td>'+cap(x)+'</td>' for x in levels[:n_conditions]]) + '<td>'+self.mes['A_BOOSTED']+'</td></tr>'
        for i in range(assignment['data']['n_subjects']):
            output_text += '<tr><td>'+str(i+1)+'</td>' + ''.join(['<td>'+str(x)+'</td>' for x in [data['scores'][j][i] for j in range(n_conditions)]]) + '<td>' + str(round(data['jackedmeans'][i],2)) + '</td></tr>'
        return output_text + '</table>'
    
    def print_analysis(self, assignment: Dict):
        return assignment['instruction'] + '<br>'  
    
    def print_report(self, assignment: Dict, answer=False) -> str: 
        output:str = '' #"Answer" is a parameter which triggers when only the mean/ANOVA tables have to be printed
        if assignment['assignment_type'] not in [1,2,11,13,14]:
            data:dict = assignment['data']
        names = ['df','ss','ms','F','p','r2'];names2 = ['df','ss','ms','F','p','eta']
        if assignment['assignment_type'] in [1,2]:
            markers = ['Differential scores'] if self.mes['L_ENGLISH'] else ['Verschilscores']
            if not answer:
                output += self.print_ttest(assignment)
            if assignment['assignment_type'] == 1:
                output += '<p><table style="width:20%">'
                levels = assignment['independent'].levels
                output += '<tr><td>'+cap(assignment['independent'].name)+'</td><td>'+self.mes['A_MEAN']+'</td><td>'+self.mes['A_STD']+'</td><td>N</td></tr>'
                output += '<tr><td>'+cap(levels[0])+'</td><td>'+str(round(assignment['means'][0],2))+'</td><td>'+str(round(assignment['stds'][0],2))+'</td><td>'+str(assignment['ns'][0])+'</td></tr>'
                output += '<tr><td>'+cap(levels[1])+'</td><td>'+str(round(assignment['means'][1],2))+'</td><td>'+str(round(assignment['stds'][1],2))+'</td><td>'+str(assignment['ns'][1])+'</td></tr>'
                output += '</table></p>'
            else:
                output += '<p><table style="width:20%">'
                output += '<tr><td>'+cap(assignment['independent'].name)+'</td><td>Gemiddelde</td><td>Standaarddeviatie</td><td>N</td></tr>'
                output += format_table(markers+[assignment['means'][0],assignment['stds'][0],assignment['ns'][0]])
                output += '</table></p>'
            if not answer:
                output += '<p><table style="width:20%">'
                output += '<tr><td>'+self.mes['A_STATISTIC']+'</td><td>'+self.mes['A_VALUE']+'</td></tr>'
                output += '<tr><td>'+self.mes['A_DF']+' (df)</td><td>'+str(assignment['df'][0])+'</td></tr>'
                output += '<tr><td>'+self.mes['A_RAW']+'</td><td>'+str(round(assignment['raw_effect'][0],2))+'</td></tr>'
                output += '<tr><td>'+self.mes['A_RELATIVE']+'</td><td>'+str(round(assignment['relative_effect'][0],2))+'</td></tr>'
                output += '<tr><td>T</td><td>'+str(round(assignment['T'][0],2))+'</td></tr>'
                output += '<tr><td>p</td><td>'+str(round(assignment['p'][0],2))+'</td></tr>'
                output += '</table></p>'
        if assignment['assignment_type'] == 3:
            if not answer:
                output += self.print_anova(assignment)
            markers = ['Source'] if self.mes['L_ENGLISH'] else ['Bron']
            output += '<p><table style="width:20%">'
            output += '<tr><td>'+markers[0]+'</td><td>df</td><td>SS</td><td>MS</td><td>F</td><td>p</td><td>R<sup>2</sup></td></tr>'
            output += format_table(['Between']+[assignment[x][0] for x in names if len(assignment[x]) > 0])
            output += format_table(['Within']+[assignment[x][1] for x in names if len(assignment[x]) > 1])
            output += format_table(['Totaal']+[assignment[x][2] for x in names if len(assignment[x]) > 2])
            output += '</table></p>'
        if assignment['assignment_type'] == 4:
            if not answer:
                output += self.print_anova(assignment)
            markers = ['Source','Interaction','Total'] if self.mes['L_ENGLISH'] else ['Bron','Interactie','Totaal']
            output += '<p><table style="width:20%">'
            output += '<tr><td>'+markers[0]+'</td><td>df</td><td>SS</td><td>MS</td><td>F</td><td>p</td><td>R<sup>2</sup></td></tr>'
            output += format_table(['Between']+[assignment[x][0] for x in names[:3]])
            output += format_table([cap(assignment['independent'].name)]+[assignment[x][1] for x in names[:3]]+[assignment[x][0] for x in names[3:]])
            output += format_table([cap(assignment['independent2'].name)]+[assignment[x][2] for x in names[:3]]+[assignment[x][1] for x in names[3:]])
            output += format_table([markers[1]]+[assignment[x][3] for x in names[:3]]+[assignment[x][2] for x in names[3:]])
            output += format_table(['Within']+[assignment[x][4] for x in names[:3]])
            output += format_table([markers[2]]+[assignment[x][5] for x in names[:2]])
            output += '</table></p>'
        if assignment['assignment_type'] == 5:
            if not answer:
                output += self.print_rmanova(assignment)
            markers = ['Source','Interaction','Total'] if self.mes['L_ENGLISH'] else ['Bron','Interactie','Totaal']
            output += '<p><table style="width:20%">'
            output += '<tr><td>'+markers[0]+'</td><td>df</td><td>SS</td><td>MS</td><td>F</td><td>p</td><td>R<sup>2</sup></td></tr>'
            output += format_table([cap(assignment['independent'].name)]+[assignment[x][0] for x in names if len(assignment[x]) > 0])
            output += format_table(['Subject']+[assignment[x][1] for x in names if len(assignment[x]) > 1])
            output += format_table([markers[1]]+[assignment[x][2] for x in names if len(assignment[x]) > 2])
            output += format_table([markers[2]]+[assignment[x][3] for x in names if len(assignment[x]) > 3])
            output += '</table></p>'
        if assignment['assignment_type'] == 6:
            output += self.print_analysis(assignment)
            output += '<p><table style="width:20%">'
            output += '<tr><td>Source</td><td>df</td><td>SS</td><td>MS</td><td>F</td><td>p</td><td>R<sup>2</sup></td></tr>'
            output += format_table(['Regression']+[assignment[x][0] for x in names])
            output += format_table(['Residue']+[assignment[x][1] for x in names[:3]])
            output += format_table(['Total']+[assignment[x][2] for x in names[:2]])+'</tr>'
            output += '</table></p>'
            
            output += '<p><table style="width:20%">'
            output += '<tr><td>Predictor</td><td>b</td><td>Beta</td><td>Standard error</td><td>T</td><td>p</td></tr>'
            for i in range(len(data['predictoren'])):
                output += '<tr><td>'+cap(data['predictoren'][i])+'</td><td>'+str(round(assignment['predictor_b'][i],2))+'</td><td>'+str(round(assignment['predictor_beta'][i],2))+'</td><td>'+str(round(assignment['predictor_se'][i],2))+'</td><td>'+str(round(assignment['predictor_t'][i],2))+'</td><td>'+str(round(assignment['predictor_p'][i],3))+'</td></tr>'
            output += '</table></p>'
        if assignment['assignment_type'] == 11:
            nt = sum(assignment['ns']) #Total number of subjects
            nl = assignment['independent'].nlevels #Number of levels factor
            hdf = assignment['hdf']; edf = assignment['edf']
            output += self.print_analysis(assignment)
            output += '<p>'+self.mes['A_MULTIVAR']+'<table style="width:20%">'
            output += format_table(['Effect','','Value','F','Hypothesis df','Error df','p','Partial eta<sup>2</sup>'])
            output += format_table(['Intercept',"Pillai's trace",assignment['value'][0],assignment['F_t1'][0],hdf[0],edf[0],assignment['p_t1'][0],assignment['eta_t1'][0]])
            output += format_table(['',"Wilks' lambda",assignment['value'][1],assignment['F_t1'][1],hdf[1],edf[1],assignment['p_t1'][1],assignment['eta_t1'][1]])
            output += format_table(['',"Hotelling's trace",assignment['value'][2],assignment['F_t1'][2],hdf[2],edf[2],assignment['p_t1'][2],assignment['eta_t1'][2]])
            output += format_table(['',"Roy's largest root",assignment['value'][3],assignment['F_t1'][3],hdf[3],edf[3],assignment['p_t1'][3],assignment['eta_t1'][3]])
            output += format_table([cap(assignment['independent'].name),"Pillai's trace",assignment['value'][4],assignment['F_t1'][4],hdf[4],edf[4],assignment['p_t1'][4],assignment['eta_t1'][4]])
            output += format_table(['',"Wilks' lambda",assignment['value'][5],assignment['F_t1'][5],hdf[5],edf[5],assignment['p_t1'][5],assignment['eta_t1'][5]])
            output += format_table(['',"Hotelling's trace",assignment['value'][6],assignment['F_t1'][6],hdf[6],edf[6],assignment['p_t1'][6],assignment['eta_t1'][6]])
            output += format_table(['',"Roy's largest root",assignment['value'][7],assignment['F_t1'][7],hdf[7],edf[7],assignment['p_t1'][7],assignment['eta_t1'][7]])
            output += '</table></p>'
            
            output += '<p>'+self.mes['A_WITHIN']+'<table style="width:50%">'
            output += format_table(['Bron','Variable','SS','df','MS','F','p','Partial eta<sup>2</sup>'])
            output += format_table(['Corrected model',cap(assignment['dependent'].name),assignment['ss_t2'][0],nl-1,assignment['ms_t2'][0],assignment['F_t2'][0],assignment['p_t2'][0],assignment['eta_t2'][0]])
            output += format_table(['',cap(assignment['dependent2'].name),assignment['ss_t2'][1],nl-1,assignment['ms_t2'][1],assignment['F_t2'][1],assignment['p_t2'][1],assignment['eta_t2'][1]])
            output += format_table(['',cap(assignment['dependent3'].name),assignment['ss_t2'][2],nl-1,assignment['ms_t2'][2],assignment['F_t2'][2],assignment['p_t2'][2],assignment['eta_t2'][2]])
            output += format_table(['Intercept',cap(assignment['dependent'].name),assignment['ss_t2'][3],1,assignment['ms_t2'][3],assignment['F_t2'][3],assignment['p_t2'][3],assignment['eta_t2'][3]])
            output += format_table(['',cap(assignment['dependent2'].name),assignment['ss_t2'][4],1,assignment['ms_t2'][4],assignment['F_t2'][4],assignment['p_t2'][4],assignment['eta_t2'][4]])
            output += format_table(['',cap(assignment['dependent3'].name),assignment['ss_t2'][5],1,assignment['ms_t2'][5],assignment['F_t2'][5],assignment['p_t2'][5],assignment['eta_t2'][5]])
            output += format_table([cap(assignment['independent'].name),cap(assignment['dependent'].name),assignment['ss_t2'][6],nl-1,assignment['ms_t2'][6],assignment['F_t2'][6],assignment['p_t2'][6],assignment['eta_t2'][6]])
            output += format_table(['',cap(assignment['dependent2'].name),assignment['ss_t2'][7],nl-1,assignment['ms_t2'][7],assignment['F_t2'][7],assignment['p_t2'][7],assignment['eta_t2'][7]])
            output += format_table(['',cap(assignment['dependent3'].name),assignment['ss_t2'][8],nl-1,assignment['ms_t2'][8],assignment['F_t2'][8],assignment['p_t2'][8],assignment['eta_t2'][8]])
            output += format_table(['Error',cap(assignment['dependent'].name),assignment['ss_t2'][9],nt-nl,assignment['ms_t2'][9],'','',''])
            output += format_table(['',cap(assignment['dependent2'].name),assignment['ss_t2'][10],nt-nl,assignment['ms_t2'][10],'','',''])
            output += format_table(['',cap(assignment['dependent3'].name),assignment['ss_t2'][11],nt-nl,assignment['ms_t2'][11],'','',''])
            output += format_table(['Total',cap(assignment['dependent'].name),assignment['ss_t2'][12],nt,'','','',''])
            output += format_table(['',cap(assignment['dependent2'].name),assignment['ss_t2'][13],nt,'','','',''])
            output += format_table(['',cap(assignment['dependent3'].name),assignment['ss_t2'][14],nt,'','','',''])
            output += format_table(['Corrected total',cap(assignment['dependent'].name),assignment['ss_t2'][15],nt-1,'','','',''])
            output += format_table(['',cap(assignment['dependent2'].name),assignment['ss_t2'][16],nt-1,'','','',''])
            output += format_table(['',cap(assignment['dependent3'].name),assignment['ss_t2'][17],nt-1,'','','',''])
            output += '</table></p>'
        if assignment['assignment_type'] == 12:
            output += self.print_analysis(assignment)
            output += '<p><table style="width:20%">'
            output += format_table(['Bron', 'df', 'SS', 'MS', 'F', 'p', 'eta<sup>2</sup>'])
            output += format_table(['Corrected model']+[assignment[x][3] for x in names2])
            output += format_table(['Intercept']+[assignment[x][6] for x in names2])
            output += format_table([cap(assignment['predictor_names'][0])]+[assignment[x][0] for x in names2])
            output += format_table([cap(assignment['predictor_names'][1])]+[assignment[x][1] for x in names2])
            output += format_table([cap(assignment['independent'].name)]+[assignment[x][2] for x in names2])
            output += format_table(['Error']+[assignment[x][4] for x in names[:3]])
            output += format_table(['Total']+[assignment[x][7] for x in names[:2]])
            output += format_table(['Corrected total']+[assignment[x][5] for x in names[:2]])
            output += '</table></p>'
        if assignment['assignment_type'] == 13:
            output += self.print_analysis(assignment)
            output += '<p>'+self.mes['A_MULTIVAR']+'<table style="width:60%">'
            output += format_table(['Effect', '', 'Value', 'Hypothesis df','Error df','F','p','eta<sup>2</sup>'])
            for i in range(8):
                i2 = i // 4
                header = cap(assignment['independent'].name) if i == 0 else cap(assignment['independent'].name)+' * '+cap(assignment['independent2'].name) if i == 4 else ''
                measure = ["Pillai's Trace","Wilks' Lambda","Hotelling's Trace","Roy's Largest Root"][i%4]
                output += format_table([header,measure,assignment['value'][i],assignment['hdf'][i2],assignment['edf'][i2],
                                        assignment['F0'][i2],assignment['p0'][i2],assignment['eta0'][i2]])
            output += '</table></p>'
            output += '<p>'+self.mes['A_WITHINCON']+'<table style="width:60%">'
            output += format_table(['Source', cap(assignment['independent'].name), 'SS','df','MS','F','p','eta<sup>2</sup>'])
            levels = assignment['independent'].levels
            for i in range(6):
                header = cap(assignment['independent'].name) if i == 0 else cap(assignment['independent'].name)+' * '+cap(assignment['independent2'].name) if i == 2 else 'Error('+cap(assignment['independent'].name)+')' if i == 4 else ''
                measure = levels[0]+' vs. '+levels[1] if i % 2 == 0 else levels[1]+' vs. '+levels[2]
                if i < 4:
                    output += format_table([header,measure,assignment['ss1'][i],assignment['df1'][i],assignment['ms1'][i],
                                        assignment['F1'][i],assignment['p1'][i],assignment['eta1'][i]])
                else:
                    output += format_table([header,measure,assignment['ss1'][i],assignment['df1'][i],assignment['ms1'][i],
                                        '','',''])
            output += '</table></p>'
            output += '<p>'+self.mes['A_BETWEEN']+'<table style="width:20%">'
            output += format_table(['Source','df','SS','MS','F','p','eta<sup>2</sup>'])
            output += format_table(['Intercept'] + [assignment[x][0] for x in names2])
            output += format_table([cap(assignment['independent2'].name)] + [assignment[x][1] for x in names2])
            output += format_table(['Error'] + [assignment[x][2] for x in names2[:3]])
            output += '</table></p>'
        if assignment['assignment_type'] == 14:
            output += self.print_analysis(assignment)
            output += '</table></p>'
        return output
    
    def print_independent(self, assignment:dict, num:int=1) -> str:
        i_key = 'independent' if num == 1 else 'independent' + str(num)
        levels = assignment[i_key].levels
        if self.mes['L_ENGLISH']:
            if assignment['assignment_type'] < 3:
                return assignment[i_key].name + ', qualitative, with levels ' + levels[0] + ' and ' + levels[1] + '.'
            elif assignment['assignment_type'] < 5:
                return assignment[i_key].name + ', a between-subject factor, with levels ' + levels[0] + ' and ' + levels[1] + '.'
            else:
                i_key:str = 'independent' if num < 2 else 'independent' + str(num)
                return assignment[i_key] + ', a within-subject factor with levels ' + ' and '.join(levels) + '.'
        else:
            if assignment['assignment_type'] < 3:
                return assignment[i_key].name + ', kwalitatief, met niveaus ' + levels[0] + ' en ' + levels[1] + '.'
            elif assignment['assignment_type'] < 5:
                return assignment[i_key].name + ', een between-subject factor, met niveaus ' + levels[0] + ' en ' + levels[1] + '.'
            else:
                i_key:str = 'independent' if num < 2 else 'independent' + str(num)
                return assignment[i_key].name + ', een within-subject factor met niveaus ' + ' en '.join(levels) + '.'
    
    def print_dependent(self, assignment:dict) -> str:
        if self.mes['L_ENGLISH']:
            return assignment['dependent'].name + ', quantitative'
        else:    
            return assignment['dependent'].name + ', kwantitatief'
        
    def print_control(self, assignment:dict, num:int=1) -> str:
        control:bool = assignment['control'] if num == 1 else assignment['control2']
        if self.mes['L_ENGLISH']:
            return 'Experiment' if control else 'Passive-observational'
        else:
            return 'Experiment' if control else 'Passive-observerend'
    
    """
    Functions to generate the standard answers for the short reports
    """
    def answer_name(self,assignment):
        atype = assignment['assignment_type']
        analyses = ['M_BLANK','M_ANALYSIS1','M_ANALYSIS2','M_ANALYSIS3','M_ANALYSIS4','M_ANALYSIS5','M_ANALYSIS6','M_BLANK','M_BLANK','M_BLANK','M_BLANK',\
                    'M_ANALYSIS7','M_ANALYSIS8','M_ANALYSIS9','M_ANALYSIS10','M_BLANK']
        return self.mes[analyses[atype]] + '. '
    
    def answer_design(self,assignment) -> str:
        output = ''
        if self.mes['L_ENGLISH']:
            if assignment['assignment_type'] in [4, 13]:
                output += 'The independent variables are '+assignment['independent'].name+' ('+', '.join(assignment['independent'].levels)+') and '+\
                    assignment['independent2'].name + ' ('+', '.join(assignment['independent2'].levels)+'). '
            elif assignment['assignment_type'] not in [6]:
                output:str = 'The independent variable is '+assignment['independent'].name+' with levels '+' and '.join(assignment['independent'].levels)+'. '
            if not assignment['assignment_type'] in [11]:
                output +='The dependent variable is '+assignment['dependent'].name+'. '
            else:
                output +='The dependent variables are '+assignment['dependent'].name+', '+assignment['dependent2'].name+' and '+assignment['dependent3'].name+'. '
            if assignment['assignment_type'] in [6]:
                output +='The covariates are '+' and '.join(assignment['levels'][1:])+'. '
            if assignment['assignment_type'] in [12]:
                output +='The predictors are '+' and '.join(assignment['predictor_names'])+'. '
        else:
            if assignment['assignment_type'] in [4, 13]:
                output += 'De onafhankelijke variabelen zijn '+assignment['independent'].name+' ('+', '.join(assignment['independent'].levels)+') en '+\
                    assignment['independent2'].name + ' ('+', '.join(assignment['independent2'].levels)+'). '
            elif assignment['assignment_type'] not in [6]:
                output:str = 'De onafhankelijke variabele is '+assignment['independent'].name+' met niveaus '+' en '.join(assignment['independent'].levels)+'. '
            if not assignment['assignment_type'] in [11]:
                output +='De afhankelijke variabele is '+assignment['dependent'].name+'. '
            else:
                output +='De afhankelijke variabelen zijn '+assignment['dependent'].name+', '+assignment['dependent2'].name+' en '+assignment['dependent3'].name+'. '
            if assignment['assignment_type'] in [6]:
                output +='De covariaten zijn '+' en '.join(assignment['levels'][1:])+'. '
            if assignment['assignment_type'] in [12]:
                output +='De predictoren zijn '+' en '.join(assignment['predictor_names'])+'. '
        return output
    
    def answer_decision(self,assignment, variable:str, num:int,FT:float,p:float,eta:float,no_effect:bool=False) -> str:
        output = ''
        
        #Set parameters
        eq_sign = self.mes['S_UNQ'] if p < 0.05 else self.mes['S_EQ']
        n_key = 'independent' if num == 1 else 'independent'+str(num)
        sizes = ['small','moderate','large'] if self.mes['L_ENGLISH'] else ['klein','matig','groot']
        size_ind = 2 if eta > 0.2 else 1 if eta > 0.1 else 0
        
        #Generate text
        if self.mes['L_ENGLISH']:
            output += 'The effect for '+variable+' is significant' if p < 0.05 else 'The effect for '+variable+' is not significant'
            if assignment['assignment_type'] in [1,2,3,4,5]:
                output += ', the population means for '+' and '.join(assignment[n_key].levels)+' are '+eq_sign
            if p < 0.05 and not assignment['assignment_type'] in [1,2] and not no_effect:
                output += ', the effect here is '+sizes[size_ind]#+'.'
            #else:
            #    output += '. '
        else:
            output += 'Het effect van '+variable+' is significant' if p < 0.05 else 'Het effect van '+variable+' is niet significant'
            if assignment['assignment_type'] in [1,2,3,4,5]:
                output += ', de populatiegemiddelden van '+' en '.join(assignment[n_key].levels)+' zijn '+eq_sign
            if p < 0.05 and not assignment['assignment_type'] in [1,2] and not no_effect:
                output += ', dit effect is '+sizes[size_ind]#+'.'
            else:
                output += '. '
        return output
    
    def answer_stats(self,assignment:dict,FT:float,p:float,eta:float,multivar=False):
        output = ''
        if p < 0.05:
            if assignment['assignment_type'] in [11,12,13,14]:
                output += ' (F = '+str(round(FT,2))+', p = '+str(round(p,3))+', eta<sup>2</sup> = '+str(round(eta,2))+'). '
            elif assignment['assignment_type'] in [1,2]:
                output += ' (T = '+str(round(FT,2))+', p = '+str(round(p,3))+'). '
            elif multivar:
                output += ' (F = '+str(round(FT,2))+', p = '+str(round(p,3))+'). '
            else:
                output += ' (F = '+str(round(FT,2))+', p = '+str(round(p,3))+', R<sup>2</sup> = '+str(round(eta,2))+'). '
        return output
    
    def answer_decision_subjects(self,assignment) -> str:
        output:str = ''
        sizes = ['small','moderate','large'] if self.mes['L_ENGLISH'] else ['klein','matig','groot']
        size_ind = 2 if assignment['r2'][1] > 0.2 else 1 if assignment['r2'][1] > 0.1 else 0
        if self.mes['L_ENGLISH']:
            output+= 'The effect of the subjects is significant' if assignment['p'][1] < 0.05 else 'The effect of the subjects is not significant'
            output+= ', the stepped-up means of the subjects are unequal in the population' if assignment['p'][1] < 0.05 else ', the boosted means of the subjects are equal in the population'
            if assignment['p'][1] < 0.05:
                output += ', this effect is '+sizes[size_ind]
            else:
                output += '. '
        else:
            output+= 'Het effect van de subjecten is significant' if assignment['p'][1] < 0.05 else 'Het effect van de subjecten is niet significant'
            output+= ', de opgevoerde gemiddelden van de subjecten zijn ongelijk in de populatie' if assignment['p'][1] < 0.05 else ', de opgevoerde gemiddelden van de subjecten zijn gelijk in de populatie'
            if assignment['p'][1] < 0.05:
                output += ', dit effect is '+sizes[size_ind]
            else:
                output += '. '
        return output
    
    def answer_decision_interaction(self,assignment) -> str:
        sizes = ['small','moderate','large'] if self.mes['L_ENGLISH'] else ['klein','matig','groot']
        size_ind = 2 if assignment['r2'][2] > 0.2 else 1 if assignment['r2'][2] > 0.1 else 0
        output:str = ''
        if self.mes['L_ENGLISH']:
            if assignment['p'][2] < 0.05:
                output += 'There is interaction between '+assignment['independent'].name+' and '+assignment['independent2'].name+' in the population'
            else:
                output += 'There is no interaction between '+assignment['independent'].name+' and '+assignment['independent2'].name+' in the population'
        else:    
            if assignment['p'][2] < 0.05:
                output += 'Er is interactie tussen '+assignment['independent'].name+' en '+assignment['independent2'].name+' in de populatie'
            else:
                output += 'Er is geen interactie tussen '+assignment['independent'].name+' en '+assignment['independent2'].name+' in de populatie'
        if assignment['p'][1] < 0.05:
            if self.mes['L_ENGLISH']:
                output += ', this effect is '+sizes[size_ind]
            else:
                output += ', dit effect is '+sizes[size_ind]
        else:
            output += '. '
        return output
    
    def answer_propvar(self,assignment) -> str:
        sizes = ['small','moderate','large'] if self.mes['L_ENGLISH'] else ['klein','matig','groot']
        size_ind = 2 if assignment['r2'][0] > 0.2 else 1 if assignment['r2'][0] > 0.1 else 0
        if self.mes['L_ENGLISH']:
            if assignment['p'][0] < 0.05:
                output = 'The proportion explained variance is significantly larger than zero, '
            else:
                output = 'The proportion explained variance is not significantly larger than zero. '
        else:
            if assignment['p'][0] < 0.05:
                output = 'De proportie verklaarde variantie is significant groter dan nul, '
            else:
                output = 'De proportie verklaarde variantie is niet significant groter dan nul. '
        if assignment['p'][0] < 0.05:
            if self.mes['L_ENGLISH']:
                output += 'this effect is '+sizes[size_ind]
            else:
                output += 'het effect hiervan is '+sizes[size_ind]
        return output
        
    def answer_predictors(self,assignment) -> str:
        relevants = []
        if assignment['assignment_type'] == 6:
            names = assignment['predictor_names'][1:]
            ps = assignment['predictor_p'][1:]
        else:
            names = assignment['predictor_names']
            ps = assignment['predictor_p'][:len(names)]
        for i, p in enumerate(ps):
            if p < 0.05:
                relevants.append([names[i] + ' ('+str(round(p,2))+')'])
        if self.mes['L_ENGLISH']:
            if relevants == []:
                return 'There are no predictors with significant effects. '
            else:
                return 'Predictors with significant effects are '+' and '.join(relevants)+'. '
        else:
            if relevants == []:
                return 'Er zijn geen predictoren met significante effecten. '
            else:
                return 'Predictoren met significante effecten zijn '+' en '.join(relevants)+'. '
            
    def answer_ancova(self, assignment) -> str:
        output:str = ''
        pnames = assignment['predictor_names']
        sizes = ['small','moderate','large'] if self.mes['L_ENGLISH'] else ['klein','matig','groot']
        size_ind = 2 if assignment['eta'][0] > 0.2 else 1 if assignment['eta'][0] > 0.1 else 0
        if self.mes['L_ENGLISH']:
            if assignment['p'][0] < 0.05:
                output += cap(assignment['independent'].name) + ', ' + pnames[0] + ' and ' + pnames[1] + ' together have a significant predictive effect on ' + assignment['dependent'].name+', '
            else:            
                output += cap(assignment['independent'].name) + ', ' + pnames[0] + ' and ' + pnames[1] + ' together do not have a significant predictive effect on ' + assignment['dependent'].name+'. '
        else:
            if assignment['p'][0] < 0.05:
                output += cap(assignment['independent'].name) + ', ' + pnames[0] + ' en ' + pnames[1] + ' hebben samen een significant voorspellend effect op ' + assignment['dependent'].name+', '
            else:            
                output += cap(assignment['independent'].name) + ', ' + pnames[0] + ' en ' + pnames[1] + ' hebben samen geen significant voorspellend effect op ' + assignment['dependent'].name+'. '
        if assignment['p'][0] < 0.05:
            if self.mes['L_ENGLISH']:
                output += 'this effect is '+sizes[size_ind]
            else:
                output += 'het effect hiervan is '+sizes[size_ind]
        return output
    
    def answer_contrast(self, assignment, v1:str, v2:str, p:str, interact=False) -> str:
        output:str = ''
        if interact:
            effect = 'The interaction' if self.mes['L_ENGLISH'] else 'De interactie'
        else:
            effect = 'The effect' if self.mes['L_ENGLISH'] else 'Het effect'
        if self.mes['L_ENGLISH']:
            if p < 0.05:
                output += effect+' between '+v1+' and '+v2+' is significant(p = '+str(round(p,2))+'). '
            else:
                output += effect+' between '+v1+' and '+v2+' is not significant. '
        else:
            if p < 0.05:
                output += effect+' tussen '+v1+' en '+v2+' is significant(p = '+str(round(p,2))+'). '
            else:
                output += effect+' tussen '+v1+' en '+v2+' is niet significant. '
        return output
    
    def answer_report(self,assignment) -> str:
        output:str = self.answer_name(assignment) + self.answer_design(assignment)+'<br>'
        if assignment['assignment_type'] in [1,2]:
            output += self.answer_decision(assignment, assignment['independent'].name, 1, FT=assignment['T'][0], p=assignment['p'][0],eta=0)
            output += self.answer_stats(assignment, FT=assignment['T'][0], p=assignment['p'][0],eta=0)+'<br>'
        if assignment['assignment_type'] == 3:
            output += self.answer_decision(assignment, assignment['independent'].name, 1, FT=assignment['F'][0], p=assignment['p'][0],eta=assignment['r2'][0])
            output += self.answer_stats(assignment, FT=assignment['F'][0], p=assignment['p'][0],eta=assignment['r2'][0])+'<br>'
        if assignment['assignment_type'] == 4:
            output += self.answer_decision(assignment, assignment['independent'].name, 1, FT=assignment['F'][0], p=assignment['p'][0],eta=assignment['r2'][0])
            output += self.answer_stats(assignment, FT=assignment['F'][0], p=assignment['p'][0],eta=assignment['r2'][0])+'<br>'
            output += self.answer_decision(assignment, assignment['independent2'].name, 2, FT=assignment['F'][1], p=assignment['p'][1],eta=assignment['r2'][1])
            output += self.answer_stats(assignment, FT=assignment['F'][1], p=assignment['p'][1],eta=assignment['r2'][1])+'<br>'
            output += self.answer_decision_interaction(assignment)
            output += self.answer_stats(assignment, FT=assignment['F'][2], p=assignment['p'][2],eta=assignment['r2'][2])+'<br>'
        if assignment['assignment_type'] == 5:
            output += self.answer_decision(assignment, assignment['independent'].name, 1, FT=assignment['F'][0], p=assignment['p'][0],eta=assignment['r2'][0])
            output += self.answer_stats(assignment, FT=assignment['F'][0], p=assignment['p'][0],eta=assignment['r2'][0])+'<br>'
            output += self.answer_decision_subjects(assignment)
            output += self.answer_stats(assignment, FT=assignment['F'][1], p=assignment['p'][1],eta=assignment['r2'][1])+'<br>'
        if assignment['assignment_type'] == 6:
            output += self.answer_propvar(assignment)
            output += self.answer_stats(assignment, FT=assignment['F'][0], p=assignment['p'][0],eta=assignment['r2'][0])+'<br>'
            output += self.answer_predictors(assignment)
        if assignment['assignment_type'] == 11:
            tag = 'het multivariate effect' if not self.mes['L_ENGLISH'] else 'the multivariate effect'
            output += self.answer_decision(assignment, tag, 0, FT=assignment['F_multivar'], p=assignment['p_multivar'],eta=assignment['eta_multivar'])
            output += self.answer_stats(assignment, FT=assignment['F_multivar'], p=assignment['p_multivar'],eta=assignment['eta_multivar'], multivar=True)+'<br>'
            for i in range(3):
                if assignment['p'+str(i)][0] < 0.05:
                    d_key = 'dependent' if i == 0 else 'ddependent'+str(i)
                    output += self.answer_decision(assignment, assignment[d_key].name, 1, FT=assignment['F'+str(i)][0], p=assignment['p'+str(i)][0],eta=assignment['eta'+str(i)][0], no_effect=True)
                    output += self.answer_stats(assignment, FT=assignment['F'+str(i)][0], p=assignment['p'+str(i)][0],eta=assignment['eta'+str(i)][0])+'<br>'
        if assignment['assignment_type'] == 12:
            output += self.answer_ancova(assignment)
            output += self.answer_stats(assignment, FT=assignment['F'][0], p=assignment['p'][0],eta=assignment['eta'][0])+'<br>'
            output += self.answer_predictors(assignment)
            if assignment['p'][4] < 0.05:
                output += self.answer_decision(assignment, assignment['independent'].name, 1, FT=assignment['F'][4], p=assignment['p'][4],eta=assignment['eta'][4])
                output += self.answer_stats(assignment, FT=assignment['F'][4], p=assignment['p'][4],eta=assignment['eta'][4])+'<br>'
        if assignment['assignment_type'] == 13:
            tag = 'the interaction' if self.mes['L_ENGLISH'] else 'de interactie'
            lvls = assignment['independent'].levels
            output += self.answer_decision(assignment, assignment['independent'].name, 1, FT=assignment['F0'][0], p=assignment['p0'][0],eta=assignment['eta0'][0])
            output += self.answer_stats(assignment, FT=assignment['F0'][0], p=assignment['p0'][0],eta=assignment['eta0'][0])+'<br>'
            if assignment['p0'][0] < 0.05:
                output += self.answer_contrast(assignment, lvls[0], lvls[1], assignment['p1'][0])+'<br>'
                output += self.answer_contrast(assignment, lvls[1], lvls[2], assignment['p1'][1])+'<br>'
            output += self.answer_decision(assignment, tag, 1, FT=assignment['F0'][1], p=assignment['p0'][1],eta=assignment['eta0'][1])
            output += self.answer_stats(assignment, FT=assignment['F0'][1], p=assignment['p0'][1],eta=assignment['eta0'][1])+'<br>'
            if assignment['p0'][1] < 0.05:
                output += self.answer_contrast(assignment, lvls[0], lvls[1], assignment['p1'][2], interact=True)+'<br>'
                output += self.answer_contrast(assignment, lvls[1], lvls[2], assignment['p1'][3], interact=True)+'<br>'
            output += self.answer_decision(assignment, assignment['independent2'].name, 2, FT=assignment['F'][1], p=assignment['p'][1],eta=assignment['eta'][1])
            output += self.answer_stats(assignment, FT=assignment['F'][1], p=assignment['p'][1],eta=assignment['eta'][1])+'<br>'
        return output