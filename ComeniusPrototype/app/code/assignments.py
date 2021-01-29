#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:03:09 2020

@author: jelmer
"""
import math
import random
import numpy as np
from scipy import stats
from scipy.stats.distributions import chi2
from typing import Dict, List, Tuple
#import typing
#from typing import *
def format_table(terms:list) -> str:
    return '<tr><td>' + '</td><td>'.join([str(round(x,2)) if type(x) != str else x for x in terms]) + '</td></tr>'

def cap(term:str) -> str:
    return term[0].upper() + term[1:]
    
class Assignments:
    def __init__(self):
        pass
        
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
        var_i:int = random.choice([0,1,2])
        var_j:int = random.choice([0,1,2])
        dependent = ['reactietijd','bloeddruk','concentratieniveau'][var_j]
        if between_subject:
            independent = ['stimulus','nationaliteit','religie'][var_i]
            levels = [['vierkant','cirkel'],['nederlands','duits'],['christelijk','moslim']][var_i]
            ind_syns = [['stimuli'],['nationaliteiten'],['religies']][var_i]
            level_syns = [[['vierkanten'], ['cirkels']],[['nederlandse'],['duitse']],[['christelijke'],['moslims','islamitische']]][var_i]
        else:
            independent = ['tijdstip','examentaal','stimulussmaak'][var_i]
            levels = [['dag','nacht'],['nederlands','frans'],['zoet','zout']][var_i]
            ind_syns = [['tijdstippen'],['examentalen'],['stimulussmaken']][var_i]
            level_syns = [[['dagen'], ['nachten']],[['nederlandse'],['franse']],[['zoete'],['zoute']]][var_i]
        if independent in ['nationaliteit','religie']:
            control = False
        
        #Create the assignment description
        report_type = 'elementair' if elementary else 'beknopt'
        instruction: str = 'Maak een '+report_type+' rapport van onderstaande data voor de hypothese dat '
        if hyp_type == 0:
            instruction += dependent+' bij ' + independent + ' ' + levels[0] + ' gemiddeld ongelijk is aan die bij ' + levels[1] + '.<br><br>'
        if hyp_type == 1:
            instruction += dependent+' bij ' + independent + ' ' + levels[0] + ' gemiddeld groter is dan die bij ' + levels[1] + '.<br><br>'
        if hyp_type == 2:
            instruction += dependent+" bij " + independent + " " + levels[0] + " gemiddeld kleiner is dan die bij " + levels[1] + ".<br><br>"
        if between_subject:
            instruction += 'De proefpersonen doen een experiment, waarin ze worden ingedeeld op hun ' + independent + ' in de groepen '+ levels[0] + ' en ' + levels[1]
        else:
            instruction += 'De proefpersonen doen allemeaal mee aan een experiment, met als ' + independent + ' zowel ' + levels[0] + ' als ' + levels[1]
        if control:
            if between_subject:
                instruction += 'De personen van elk beroep zijn willekeurig geselecteerd. '
            else:
                instruction += 'De volgorde van de toetsen was gerandomiseerd. '
        instruction += 'Voer je antwoorden alsjeblieft tot op 2 decimalen in'\
             ' en gebruik dezelfde vergelijking van de gemiddelden in je antwoord als in de vraagstelling staat (e.g. "groter" of "kleiner"). '
        
        #Generate datapoints: Floats are rounded to 2 decimals
        return{'instruction': instruction,
               'hypothesis': hyp_type,
               'between_subject': between_subject,
               'control': control,
               'dependent': dependent,
               'dep_syns': [['reactietijden'],['bloeddrukken'],['concentratieniveaus','concentratie']][var_j],
               'assignment_type':1 if between_subject else 2,
               'independent':independent,
               'levels':levels,
               'ind_syns':ind_syns,
               'level_syns':level_syns,
               'A': [round(random.gauss(mean1,std1), 2) for i in range(n1)], 
               'B': [round(random.gauss(mean2,std2), 2) for i in range(n2)]
               }
    
    #Calculate internally all of the numbers and string values the student has to present
    def solve_ttest(self, assignment: Dict, solution: Dict) -> Dict:
        numbers: List = [assignment['A'], assignment['B']]
        names: List[str] = assignment['levels']
        between_subject: bool = assignment['between_subject']
        solution['hypothesis'] = assignment['hypothesis']
        solution['assignment_type'] = assignment['assignment_type']
        
        if not between_subject:
            #Differential scores
            diff: List[float] = [numbers[0][i] - numbers[1][i] for i in range(len(numbers[0]))]
        
        #Determine variable names and types
        solution['independent'] = assignment['independent']
        solution['ind_syns'] = assignment['ind_syns']
        solution['levels'] = assignment['levels']
        solution['level_syns'] = assignment['level_syns']
        solution['dependent'] = assignment['dependent']
        solution['dependent_measure']: str = 'kwantitatief'
        solution['independent_measure']: str = 'kwalitatief'
        solution['dep_syns'] = assignment['dep_syns']
        
        #Determine null hypothesis and control measure
        sign: List[str] = ['==','<=','>='][assignment['hypothesis']]
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
        sterkte:str='sterk' if solution['relative_effect'][0] > 0.8 else 'matig' if solution['relative_effect'][0] > 0.5 else 'klein'
        if solution['p'][0] < 0.05:
            decision: Tuple[str] = ('verworpen','', 'Het effect is ' + sterkte + '. ')
        else:
            decision: Tuple[str] = ('behouden','niet ', '')
        comparison: str = ['ongelijk','groter','kleiner'][assignment['hypothesis']]
        solution['decision']: str = 'H0 ' + decision[0] + ', het populatiegemiddelde van ' + names[0] + ' is ' + decision[1] + comparison + ' dan dat van ' + names[1] + '. ' + decision[2]
        
        #Causal interpretation
        if solution['p'][0] < 0.05:
            if assignment['control']:
                solution['interpretation']: str = 'Experiment, dus er is slechts een verklaring mogelijk. Dit is namelijk dat ' + solution['independent'] + ' invloed heeft op ' + solution['dependent'] + '.'
            else:
                solution['interpretation']: str = 'Geen experiment, dus er zijn meerdere verklaringen mogelijk. De primaire verklaring is dat ' + solution['independent'] + ' invloed heeft op ' + solution['dependent'] + '. '+\
                'De alternatieve verklaring is dat ' + solution['dependent'] + ' ' + solution['independent'] + ' beinvloedt. '
        else:
            if assignment['control']: 
                solution['interpretation']: str = 'Experiment, dus er is slechts een verklaring mogelijk. Dit is namelijk dat ' + solution['independent'] + ' geen invloed heeft op ' + solution['dependent'] + '.'
            else:
                solution['interpretation']: str = 'Geen experiment, dus er zijn meerdere verklaringen mogelijk. De primaire verklaring is dat ' + solution['independent'] + ' geen invloed heeft op ' + solution['dependent'] + '. '+\
                'De alternatieve verklaring is dat ' + solution['independent'] + ' ' + solution['dependent'] + ' wel beinvloedt, maar dat dit niet merkbaar is door een storende variabele.'
        return solution
        
    def create_anova(self, two_way: bool, control: bool, control2:bool=False, elementary:bool=True) -> Dict:
        output = {'two_way':two_way, 'control':control}
        output['dependent'] = 'gewicht'
        output['instruction']: str = None
        output['assignment_type']: int = 4 if two_way else 3
        
        var_i = random.choice([0,1,2])
        var_j = random.choice([0,1,2])
        var_k = random.choice([0,1,2])
        output['independent'] = ['stimuluskleur','weerssituatie','muziek'][var_i]
        output['levels'] = [['rood','blauw'],['zon','regen'],['klassiek','pop']][var_i]
        output['ind_syns'] = [['stimuluskleuren'],['weerssituaties'],[]][var_i]
        output['level_syns'] = [[['rode'],['blauwe']],[['zonnig'],['regenachtig']],[['klassieke'],[]]][var_i]
        output['dependent'] = ['gewicht','bloeddruk','geheugenscore'][var_j]
        output['dep_syns'] = [['gewichten'],[],['geheugenscores']][var_j]
        if two_way:
            output['independent2'] = ['stimulusvorm','filmgenre','bloedtype'][var_k]
            output['ind2_syns'] = [['stimulusvormen'],['filmgenres'],['bloedtypen','bloedtypes']][var_k]
            output['levels2'] = [['vierkant','rond'],['drama','horror'],['A','B']][var_k]
            output['level2_syns'] = [[['vierkante'],['ronde']],[[],[]],[[],[]]][var_k]
            output['control2'] = control2
            if output['independent2'] in ['bloedtype']:
                output['control2'] = False
                control2 = False
        
        #Decide the variable names
        report_type = 'elementair' if elementary else 'beknopt'
        if not two_way:
            output['instruction'] = 'Maak een '+report_type+' rapport van de onderstaande data. De variabelen zijn '+output['independent']+', met niveaus '+' en '.join(output['levels'])+', en '+output['dependent']+'. Voer je antwoorden alsjeblieft tot op 2 decimalen in '\
                 'en gebruik dezelfde vergelijking van de gemiddelden in je antwoord als in de vraagstelling staat (e.g. "groter" of "kleiner"). '
            if control:
                output['instruction'] += 'De deelnemers zijn willekeurig verdeeld over de niveaus.'
        else:
            output['instruction'] = 'Maak een '+report_type+' rapport van de onderstaande data. De variabelen zijn '+output['independent']+', met niveaus '+' en '.join(output['levels'])+', '+output['independent2']+' met niveaus '+' en '.join(output['levels2'])+', en '+output['dependent']+'. Voer je antwoorden alsjeblieft tot op 2 decimalen in '\
                 'en gebruik dezelfde vergelijking van de gemiddelden in je antwoord als in de vraagstelling staat (e.g. "groter" of "kleiner"). '
            if control and output['control2']:
                output['instruction'] += 'De deelnemers zijn willekeurig gekozen voor beide factoren. '
            elif control:
                output['instruction'] += 'De deelnemers zijn willekeurig verdeeld bij de factor '+output['independent']+'. '
            elif output['control2']:
                output['instruction'] += 'De deelnemers zijn willekeurig verdeeld bij de factor '+output['independent2']+'. '
        
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
    
    def solve_anova(self, assignment: Dict, solution: Dict) -> Dict:
        data: Dict = assignment['data']
        two_way: bool = assignment['two_way']
        solution['assignment_type'] = assignment['assignment_type']
        solution['independent'] = assignment['independent']
        solution['ind_syns'] = assignment['ind_syns']
        solution['levels'] = assignment['levels']
        solution['level_syns'] = assignment['level_syns']
        solution['dependent'] = assignment['dependent']
        solution['dep_syns'] = assignment['dep_syns']
        solution['dependent_measure']: str = 'kwantitatief'
        solution['dependent_n_measure']: int = 1 #Aantal metingen per persoon
        solution['independent_measure']: str = 'kwalitatief'
        solution['control']: bool = assignment['control']
        if two_way:
            solution['control2']: bool = assignment['control2']
            solution['independent2'] = assignment['independent2']
            solution['ind2_syns'] = assignment['ind2_syns']
            solution['levels2'] = assignment['levels2']
            solution['level2_syns'] = assignment['level2_syns']
        
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
            rejected: Tuple[str] = ('verworpen','ongelijk',' ') if solution['p'][0] < 0.05 else ('behouden', 'gelijk', ' niet ')
            solution['null']: str = 'h0: ' + ' == '.join(['mu(' + l + ')' for l in solution['levels']])
            sterkte:str = 'sterk' if solution['r2'][0] > 0.2 else 'matig' if solution['r2'][0] > 0.1 else 'klein'
            solution['decision']: str = 'h0 ' + rejected[0] + ', de populatiegemiddelden van ' + solution['levels'][0] +' en '+solution['levels'][1]+' zijn gemiddeld ' + rejected[1] + '. '
            if solution['p'][0]:
                solution['decision'] += 'Het effect is '+sterkte+'.'
            if assignment['control']:
                solution['interpretation']: str = 'Experiment, dus er is een verklaring mogelijk. Dit is dat '+solution['independent']+' '+solution['dependent'] +rejected[2] +' veroorzaakt.'
            else:
                solution['interpretation']: str = 'Geen experiment, dus er zijn meerdere verklaringen mogelijk. De primaire verklaring is dat '+solution['independent']+' '+solution['dependent'] + \
                ' veroorzaakt. De alternatieve verklaring is dat ' + solution['dependent'] + ' ' + solution['independent'] + ' veroorzaakt.'
                
        else: #Two-way statistics
            #Intermediary statistics order: Between, A, B, AB, Within, Total
            #Degrees of freedom
            l1: int = len(solution['levels'])
            l2: int = len(solution['levels2'])
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
            rejected: Tuple[str] = ('verworpen','ongelijk') if solution['p'][0] < 0.05 else ('behouden', 'gelijk')
            rejected2: Tuple[str] = ('verworpen','ongelijk') if solution['p'][1] < 0.05 else ('behouden', 'gelijk')
            rejected3: Tuple[str] = ('verworpen','wel') if solution['p'][2] < 0.05 else ('behouden', 'geen')
            sterkte:str = 'sterk' if solution['r2'][0] > 0.2 else 'matig' if solution['r2'][0] > 0.1 else 'klein'
            sterkte2:str = 'sterk' if solution['r2'][1] > 0.2 else 'matig' if solution['r2'][1] > 0.1 else 'klein'
            sterkte3:str = 'sterk' if solution['r2'][2] > 0.2 else 'matig' if solution['r2'][2] > 0.1 else 'klein'
            solution['control2'] = assignment['control2']
            solution['null']: str = 'h0: mu(' + solution['levels'][0] + ') == mu(' + solution['levels'][1] + ')'
            solution['null2']: str =  'h0: mu(' + solution['levels2'][0] + ') == mu(' + solution['levels2'][1] + ')'
            levels = solution['levels']; levels2 = solution['levels2']
            solution['null3']: str = 'h0(' + solution['independent'] + ' x ' + solution['independent2'] + '): mu('+levels[0] + ' & ' + levels2[0]+') = mu('+levels[0]+') + mu('+levels2[0]+') - mu(totaal) [...] en mu('+levels[-1] + ' & ' + levels2[-1]+') = mu('+levels[-1]+') + mu('+levels2[-1]+') - mu(totaal)'
            solution['decision']: str = 'h0 ' + rejected[0] + ', de populatiegemiddelden van ' + solution['levels'][0] +' en '+solution['levels'][1]+' zijn gemiddeld ' + rejected[1] + '. '
            solution['decision2']: str = 'h0 ' + rejected2[0] + ', de populatiegemiddelden van ' + solution['levels2'][0] +' en '+solution['levels2'][1]+' zijn gemiddeld ' + rejected2[1] + '. '
            solution['decision3']: str = 'h0 ' + rejected3[0] + ', er is ' + rejected3[1] + ' interactie tussen ' + solution['independent'] +' en '+solution['independent2'] + ' in de populatie. '
            if solution['p'][0] < 0.05: solution['decision'] += 'Het effect is ' + sterkte + '.'
            if solution['p'][1] < 0.05: solution['decision2'] += 'Het effect is ' + sterkte2 + '.'
            if solution['p'][2] < 0.05: solution['decision3'] += 'Het effect is ' + sterkte3 + '.'
            n1 = '' if solution['p'][0] < 0.05 else 'niet '
            if assignment['control']:
                solution['interpretation']: str = 'Experiment, dus er is een verklaring mogelijk. Dit is dat '+solution['dependent']+' wordt '+n1+'veroorzaakt door '+solution['independent']
            else:
                solution['interpretation']: str = 'Geen experiment, dus er zijn meerdere verklaringen mogelijk. De primaire verklaring is dat '+solution['dependent']+' '+n1+'wordt veroorzaakt door '+solution['independent'] + '. '\
                'De alternatieve is dat ' + solution['independent'] + ' wordt veroorzaakt door ' + solution['dependent']
            n2 = '' if solution['p'][1] < 0.05 else 'niet '
            if assignment['control2']:
                solution['interpretation2']: str = 'Experiment, dus er is een verklaring mogelijk. Dit is dat '+solution['dependent']+' wordt '+n2+'veroorzaakt door '+solution['independent2']
            else:
                solution['interpretation2']: str = 'Geen experiment, dus er zijn meerdere verklaringen mogelijk. De primaire verklaring is dat '+solution['dependent']+'  '+n2+'wordt veroorzaakt door '+solution['independent2'] + '. '\
                'De alternatieve is dat ' + solution['independent2'] + ' wordt veroorzaakt door ' + solution['dependent']
            n3 = 'niet ' if solution['p'][2] < 0.05 else ''
            if assignment['control'] and assignment['control2']:
                solution['interpretation3']: str = 'Experiment, dus er is een verklaring mogelijk. Dit is dat '+solution['independent'] + ' '+n3+'dezelfde invloed heeft op '+solution['dependent']+' zowel bij de niveaus ' + ' en '.join(solution['levels2']) + ' van de factor ' + solution['independent2'] + '.'
            else:
                solution['interpretation3']: str = 'Geen experiment, dus er zijn meerdere verklaringen mogelijk. De primaire is dat '+solution['independent'] + ' '+n3+'dezelfde invloed heeft op '+solution['dependent']+' zowel bij de niveaus ' + ' en '.join(solution['levels2']) + ' van de factor ' + solution['independent2'] + '. '\
                    'Eventuele alternatieve verklaringen zijn storende variabelen of omgekeerde causaliteit. '
        return solution
    
    def create_rmanova(self, control: bool, elementary:bool=True) -> Dict:
        #Determine variable shape and names
        output = {'control': control, 'two_way':False, 'assignment_type':5}
        n_conditions = random.randint(2,4)
        n_subjects = int(random.uniform(8,15))
        
        var_i = random.choice([0,1,2])
        var_j = random.choice([0,1,2])
        output['independent'] = ['seizoen','leeftijd','tijdperk'][var_i]
        output['ind_syns'] = [['seizoenen'],['leeftijden'],['tijdperk']][var_i]
        output['levels'] = [['winter', 'lente', 'zomer', 'herfst'],['kind', 'jong', 'volwassen', 'oud'],['precambrium', 'siluur', 'paleolithicum', 'quartair']][var_i][:n_conditions]
        output['level_syns'] = [[],[],[],[]][:n_conditions]
        output['dependent'] = ['gewicht','bloeddruk','geheugenscore'][var_j]
        output['dep_syns'] = [['gewichten'],[],['geheugenscores']][var_j]
        
        report_type = 'elementair' if elementary else 'beknopt'
        output['instruction']: str = 'Maak een '+report_type+' rapport van de onderstaande data. De variabelen zijn '+output['dependent']+' en '+output['independent']+ ' met niveaus '+' en '.join(output['levels']) + ''\
            '. Voer je antwoorden alsjeblieft tot op 2 decimalen in en gebruik dezelfde vergelijking van de gemiddelden in je antwoord als in de vraagstelling staat (e.g. "groter" of "kleiner"). '
        if control:
            output['instruction'] += 'De subjecten in het experiment zijn willekeurig geselecteerd. '
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
    
    def solve_rmanova(self, assignment: Dict, solution: Dict) -> Dict: 
        data: Dict = assignment['data']
        n_conditions = len(data['means'])
        solution['independent'] = assignment['independent']
        solution['ind_syns'] = assignment['ind_syns']
        solution['levels'] = assignment['levels']
        solution['level_syns'] = assignment['level_syns']
        solution['dependent'] = assignment['dependent']
        solution['dep_syns'] = assignment['dep_syns']
        solution['dependent_measure']: str = 'kwantitatief'
        solution['dependent_n_measure']: int = n_conditions #Aantal metingen per persoon
        solution['independent_measure']: str = 'kwalitatief'
        solution['control']: bool = assignment['control']
        solution['assignment_type'] = assignment['assignment_type']
        
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
        solution['null']: str = 'h0: ' + ' == '.join(['mu(' + x + ')' for x in assignment['levels']])
        solution['null2']: str = 'h0: De personen hebben gelijke ware scores op de opgevoerde meting in de populatie.'
        rejected: Tuple[str] = ('verworpen','ongelijk') if solution['p'][0] < 0.05 else ('behouden', 'gelijk')
        solution['decision']: str = 'h0 ' + rejected[0] + ', de populatiegemiddelden van '+solution['dependent']+' ' + ' en '.join(assignment['levels']) + ' zijn gemiddeld ' + rejected[1] + '. '
        if solution['p'][0]:
            sterkte:str = 'sterk' if solution['r2'][0] > 0.2 else 'matig' if solution['r2'][0] > 0.1 else 'klein'
            solution['decision'] += 'Het effect is '+sterkte+'.'
        rejected2: Tuple[str] = ('verworpen','ongelijk') if solution['p'][1] < 0.05 else ('behouden', 'gelijk')
        solution['decision2']: str = 'h0 ' + rejected2[0] + ', de opgevoerde gemiddelden van de personen in de populatie zijn ' + rejected2[1] + '. '
        sterkte2:str = 'sterk' if solution['r2'][1] > 0.2 else 'matig' if solution['r2'][1] > 0.1 else 'klein'
        if solution['p'][1] < 0.05:
            solution['decision2'] += 'Het effect is ' + sterkte2 + '. '        
        n1 = '' if solution['p'][0] < 0.05 else 'niet '
        if assignment['control']:
            solution['interpretation']: str = 'Experiment, dus er is een verklaring mogelijk. De primaire verklaring is dat '+solution['dependent']+' '+n1+' wordt veroorzaakt door '+solution['independent']
        else:
            solution['interpretation']: str = 'Geen experiment, dus er zijn meerdere verklaringen mogelijk. De primaire verklaring is dat '+solution['dependent']+' wordt '+n1+'veroorzaakt door '+solution['independent'] + '. '\
            'De alternatieve is dat ' + solution['independent'] + ' wordt veroorzaakt door ' + solution['dependent']
        return solution
    
    def create_mregression(self, control: bool, elementary:bool=False):
        report_type = 'elementair' if elementary else 'beknopt'
        output = {'assignment_type':6}
        output['independent'] = 'predictoren'
        output['ind_syns'] = []
        N = 50 + int(150 * random.random())
        output['ns'] = [N]
        n_predictors = random.choice([3,4,5,6])
        output['n_predictors'] = n_predictors
        
        p = random.random()
        s = 3 * random.random()
        output['var_obs'] = (1 + chi2.ppf(p, df=10)) * 10 ** s
        r2 = random.random() ** 2
        output['var_pred'] = output['var_obs'] * r2
        output['data']: dict={'predictoren':['intercept','sociale vaardigheden', 'depressieve gedachten', 'eetlust','intelligentie','assertiviteit','ervaren geluk'][:n_predictors+1]}
        output['levels'] = output['data']['predictoren']
        output['level_syns'] = [[] for x in output['levels']]
        output['dependent'] = 'gewicht'
        output['dep_syns'] = ['gewichten']
        #output['correlations'] = [random.random() for i in range(int(((n_predictors + 1) ** 2 - n_predictors - 1) * 0.5))]
        output['instruction'] = 'Maak een '+report_type+' rapport van de onderstaande data. De variabelen zijn '+' en '.join(output['data']['predictoren'][1:])+' als predictoren en '+output['dependent']+' als criterium. Voer je antwoorden alsjeblieft tot op 2 decimalen in. '
        return output
    
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
        solution['null'] = 'H0: ' + ' == '.join(['beta(' + str(i) + ')' for i in range(1,4)]) + ' == 0'
        return solution
    
    def create_ancova(self, control: bool, elementary:bool=False):
        report_type = 'elementair' if elementary else 'beknopt'
        output = {'assignment_type':12}
        indy_int = random.choice([0,1,2])
        output['independent'] = ['stimuluskleur','weerssituatie','muziek'][indy_int]
        output['levels'] = [['rood','blauw'],['zon','regen'],['klassiek','pop']][indy_int]
        output['ind_syns'] = [['stimuluskleuren'],['weerssituaties'],[]][indy_int]
        output['level_syns'] = [[['rode'],['blauwe']],[['zonnig'],['regenachtig']],[['klassieke'],[]]][indy_int]
        N = N = 50 + int(150 * random.random())
        output['ns'] = [N]
        output['n_predictors'] = 2
        
        p = random.random()
        s = 3 * random.random()
        output['var_obs'] = (1 + chi2.ppf(p, df=10)) * 10 ** s
        r2 = random.random() ** 2
        output['var_pred'] = output['var_obs'] * r2
        output['data']: dict={'predictoren':['sociale vaardigheden', 'depressieve gedachten', 'eetlust',
              'Intelligentie','Assertiviteit','Ervaren geluk'][:2]}
        output['predictor_names'] = output['data']['predictoren']
        output['predictor_syns'] = [[] for x in output['levels']]
        output['dependent'] = 'gewicht'
        output['dep_syns'] = ['gewichten']
        #output['correlations'] = [random.random() for i in range(int(((n_predictors + 1) ** 2 - n_predictors - 1) * 0.5))]
        output['instruction'] = 'Maak een '+report_type+' rapport van de onderstaande data. De variabelen zijn de factor '+output['independent']+' en '+output['dependent']+', met '+' en '.join(output['data']['predictoren'])+' als predictoren. Voer je antwoorden alsjeblieft tot op 2 decimalen in. '
        return output
    
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
        
        #Verbal answers
        solution['null'] = 'H0: ' + ' == '.join(['beta(' + str(i) + ')' for i in range(1,4)]) + ' == 0'
        return solution
    
    def create_manova(self, control: bool, control2:bool=False, elementary:bool=False):
        output = {'assignment_type':12}
        report_type = 'elementair' if elementary else 'beknopt'
        p = random.random() ** 2
        s = 3 * random.random()
        output['var_obs'] = (1 + chi2.ppf(p, df=10)) * 10 ** s
        output['var_pred'] = [output['var_obs'] * random.random() ** 2 for i in range(3)]
        
        indy_int = random.choice([0,1,2])#; var_k = random.choice([0,1,2])
        output['independent'] = ['stimuluskleur','weerssituatie','muziek'][indy_int]
        output['levels'] = [['rood','blauw'],['zon','regen'],['klassiek','pop']][indy_int]
        output['ind_syns'] = [['stimuluskleuren'],['weerssituaties'],[]][indy_int]
        output['level_syns'] = [[['rode'],['blauwe']],[['zonnig'],['regenachtig']],[['klassieke'],[]]][indy_int]
        output['control'] = control
        
        output['ind_syns'] = [['stimuluskleuren'],['weerssituaties'],[]][indy_int]
        output['level_syns'] = [[['rode'],['blauwe']],[['zonnig'],['regenachtig']],[['klassieke'],[]]][indy_int]
        output['sumdependent'] = 'grootte'
        output['dependent'] = 'gewicht'
        output['dependent2'] = 'lengte'
        output['dependent3'] = 'breedte'
        output['dep_syns'] = ['gewichten']; output['dep2_syns'] = ['lengten']; output['dep3_syns'] = ['leeftijden'];
        dependents = [output['dependent'], output['dependent2'], output['dependent3']]
        N = 50 + int(150 * random.random()); output['ns'] = [N]
        
        output['instruction'] = 'Maak een '+report_type+' rapport van de onderstaande data. De onafhankelijke variabele is '+\
            output['independent']+' ('+', '.join(output['levels'])+')'\
            '. De afhankelijke variabelen zijn dimensies van de variabele '+output['sumdependent']+', namelijk '+\
            ' en '.join(dependents)+'. Voer je antwoorden alsjeblieft tot op 2 decimalen in. '
        return output
    
    def solve_manova(self, assignment: Dict, solution:Dict) -> Dict:
        solution = {'df':{}, 'ss':{}, 'ms':{},'F':{},'p':{},'eta':{}}
        N = assignment['ns'][0]
        for key, value in list(assignment.items()):
            solution[key] = value
        for j in range(3):    
            ssreg = (N-1) * assignment['var_pred'][j]
            pred_ss = random.random() * 0.5 * ssreg
            sstotal = (N-1) * assignment['var_obs']
            solution['df'][j]: List[float] = [len(assignment['levels']) - 1, N - len(assignment['levels']), N - 1]
            solution['ss'][j]: List[float] = [pred_ss, sstotal-pred_ss, sstotal]
            solution['ms'][j]: List[float] = [solution['ss'][j][i]/solution['df'][j][i] for i in range(2)]
            solution['F'][j]: List[float] = [solution['ms'][j][0] / solution['ms'][j][1]]
            solution['p'][j]: List[float] = [1 - stats.f.cdf(abs(solution['F'][j][0]),solution['df'][j][0],solution['df'][j][1])]
            #solution['r2'][j]: List[float] = [solution['ss'][0]/solution['ss'][2]]
            solution['eta'][j]: List[float] = [solution['ss'][j][0] / solution['ss'][j][2]] #+solution['ss'][j][2])]
        solution['F_multivar'] = np.mean([solution['F'][i][0] for i in range(3)])
        solution['p_multivar'] = np.mean([solution['p'][i][0] for i in range(3)])
        solution['eta_multivar'] = np.mean([solution['eta'][i][0] for i in range(3)])
        
        #Fill table 1 vars
        #eigenvalues = []
        v0 = random.random()*120; solution['value'] = [random.random(),random.random(),v0,v0]+[random.random(),random.random(),random.random()*2.5,random.random()*2.5]
        f0 = random.random()*500; solution['F1'] = [f0 for i in range(4)] + [solution['F_multivar'] for i in range(4)]
        solution['p1'] = [0.0 for i in range(4)] + [solution['p_multivar'] for i in range(4)]
        intr2 = 0.9*random.random()*0.10; solution['eta1'] = [intr2 for i in range(4)] + [solution['eta_multivar'] for i in range(4)]
        solution['hdf'] = [len(assignment['levels']) - 1 for i in range(8) for i in range(8)]
        solution['edf'] = [N - len(assignment['levels']) for i in range(8) for i in range(8)]
        
        #Fill table 2 vars
        intercepts = [30000 + 2000 * random.random() for i in range(3)]
        solution['ss0'] = [solution['ss'][i][0] for i in range(3)]+intercepts+[solution['ss'][i][0] for i in range(3)]+\
                           [solution['ss'][i][1] for i in range(3)]+[solution['ss'][i][2]+intercepts[i] for i in range(3)]+[solution['ss'][i][2] for i in range(3)]
        nt = sum(assignment['ns']); nl = len(assignment['levels']) #Total subjects #Number of levels per factor
        dfs = [nl-1,nl-1,nl-1,1,1,1,nl-1,nl-1,nl-1,nt-nl,nt-nl,nt-nl,nt,nt,nt,nt-1,nt-1,nt-1]
        solution['ms0'] = [solution['ss0'][i] / dfs[i] for i in range(12)]
        solution['F0'] = [solution['ms0'][i] / solution['ms'][i%3][1] for i in range(9)]
        solution['p0'] = [1-stats.f.cdf(solution['F0'][i],dfs[i],nt-nl-1) for i in range(9)]
        solution['eta0'] = [solution['ss0'][i] / solution['ss'][i%3][2] for i in range(9)]
        return solution
    
    def create_multirm(self, control: bool, control2:bool=False, elementary:bool=False):
        output = {'assignment_type':13}
        report_type = 'elementair' if elementary else 'beknopt'
        p = random.random() ** 2
        s = 3 * random.random()
        output['ns'] = [int(random.random() * 65) + 10, int(random.random() * 65) + 10]
        output['var_obs'] = [(1 + chi2.ppf(p, df=10)) * 10 ** s for i in range(2)]
        output['var_pred'] = [output['var_obs'][i] * random.random() ** 2 for i in range(2)]
        
        indy_int = random.choice([0,1]); var_k = random.choice([0,1,2])
        output['independent'] = ['meting','tijdstip'][indy_int] #WITHIN-SUBJECT FACTOR
        output['levels'] = [['voor','na','followup'],['dag','avond','nacht']][indy_int]
        output['ind_syns'] = [['metingen'],['tijdstippen'],[]][indy_int]
        output['level_syns'] = [[[],[],['follow-up']],[['dagen'],['avonden'],['nachten']]][indy_int]
        output['control'] = control
        output['independent2'] = ['stimulusvorm','filmgenre','bloedtype'][var_k] #BETWEEN-SUBJECT FACTOR
        output['ind2_syns'] = [['stimulusvormen'],['filmgenres'],['bloedtypen','bloedtypes']][var_k]
        output['levels2'] = [['vierkant','rond'],['drama','horror'],['A','B']][var_k]
        output['level2_syns'] = [[['vierkante'],['ronde']],[[],[]],[[],[]]][var_k]
        output['control2'] = control2
        if output['independent2'] in ['bloedtype']:
            output['control2'] = False
        
        output['dependent'] = 'score';output['dependent2'] = 'tevredenheid'
        output['dep_syns'] = ['gewichten'];output['dep2_syns'] = []
        output['instruction'] = 'Maak een '+report_type+' rapport van de onderstaande data. Dit onderzoek bevat de factoren '+\
            output['independent']+' ('+', '.join(output['levels'])+') en ' + output['independent2'] + ' ('+', '.join(output['levels2'])+')'\
            '. De afhankelijke variabelen zijn '+output['dependent']+' en '+output['dependent2']+'. Voer je antwoorden alsjeblieft tot op 2'\
            ' decimalen in. '
        return output
    
    def solve_multirm(self, assignment: Dict, solution:Dict) -> Dict:
        solution = {'df':{}, 'ss':{}, 'ms':{},'F':{},'p':{},'eta':{}}
        N = sum(assignment['ns']); ntimes = len(assignment['levels']); nlevels = len(assignment['levels2'])
        for key, value in list(assignment.items()):
            solution[key] = value
        
        # Tests of Between-Subject Objects
        ssm = [(N-1) * assignment['var_pred'][j] for j in range(2)] #pred_ss = random.random() * 0.5 * ssreg
        sstotal = [(N-1) * assignment['var_obs'][j] for j in range(2)]
        nlev = len(assignment['levels2'])
        print(nlev)
        print(assignment['levels2'])
        solution['ss']: List[float] = [random.random()*1000, random.random()*1000, ssm[0], ssm[1],sstotal[0]-ssm[0],sstotal[1]-ssm[1]]
        solution['df']: List[float] = [1,1,nlev - 1, nlev - 1,N - nlev, N - nlev]
        solution['ms']: List[float] = [solution['ss'][i]/solution['df'][i] for i in range(6)]
        solution['F']: List[float] = [solution['ms'][i] / solution['ms'][(i+4) % 2] for i in range(4)]
        solution['p']: List[float] = [1 - stats.f.cdf(abs(solution['F'][i]),solution['df'][i],solution['df'][(i+4) % 2]) for i in range(4)]
        solution['eta']: List[float] = [solution['ss'][i]/sstotal[i % 2] for i in range(4)]
        
        # Multivariate Tests
        solution['value'] = [random.random() for i in range(16)]
        solution['hdf'] = [(nlevels-1) * 2 for i in range(2)] + [(ntimes-1) * 2 for i in range(2)]
        solution['edf'] = [N - 1 - (nlevels-1) * 2 for i in range(2)] + [N - 1 - (ntimes-1) * 2 for i in range(2)]
        solution['F0'] = [assignment['var_obs'][i%2] * random.random() ** 2 for i in range(4)]
        solution['p0'] = [1-stats.f.cdf(solution['F0'][i], solution['hdf'][i], solution['edf'][i]) for i in range(4)]
        solution['eta0'] = [solution['value'][4*i] for i in range(4)]
        
        # Tests of Within-Subjects Effects
        solution['df1'] = [1 for i in range(8)] + [N-2 for i in range(4)]
        ssprep = [assignment['var_obs'][i%2] * random.random() ** 2 for i in range(8)]
        print(ssprep)
        solution['ss1'] = ssprep + [assignment['var_obs'][i%2] - ssprep[i] for i in range(4)]
        solution['ms1'] = [solution['ss1'][i]/solution['df1'][i] for i in range(12)]
        solution['F1'] = [solution['ss1'][i] / solution['ss1'][8+i%4] for i in range(8)]
        solution['p1'] = [1-stats.f.cdf(solution['F1'][i], solution['df1'][i], solution['df1'][4+i%2]) for i in range(8)]
        solution['eta1'] = [solution['ss1'][i] / (solution['ss1'][i] + solution['ss1'][4+i%2]) for i in range(8)]
        return solution
	
    def create_report(self, control: bool, choice: int=0):
        hyp_type = random.choice([0,1,2])
        if choice == 1:
            assignment = self.create_ttest(True, hyp_type, control, False)
            output = {**assignment, **self.solve_ttest(assignment, {})}
        if choice == 2:
            assignment = self.create_ttest(False, hyp_type, control, False)
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
        output['assignment_type'] = choice
        return output
            
    def print_ttest(self, assignment: Dict) -> str:
        output_text = assignment['instruction'] + '<br>'
        varnames: List[str] = [assignment['independent']] + assignment['levels']
        data: List = [assignment['A'], assignment['B']]
        output_text += '<table style="width:20%">'
        if assignment['between_subject']:
            output_text += '<tr><td>' + varnames[1] + '</td><td>' + varnames[2] + '</td></tr>'
        else:
            output_text += '<tr><td>Nr</td><td>' + varnames[1] + '</td><td>' + varnames[2] + '</td></tr>'
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
        data['varnames'] = [[assignment['independent']] + assignment['levels']]
        if assignment['assignment_type'] == 4:
            data['varnames'].append([assignment['independent2']] + assignment['levels2'])
        #Print variables
        output_text = assignment['instruction'] + '<br><table style="width:30%">'
        if not assignment['two_way']:
            output_text += '<tr><td>Gewicht:</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][0] + '</td><td>' + 'Gemiddelde</td><td>Standaarddeviatie</td><td>N' + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][1] + '</td><td>' + str(data['means'][0]) + '</td><td>' + str(data['stds'][0]) + '</td><td>' + str(data['ns'][0]) + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][2] + '</td><td>' + str(data['means'][1]) + '</td><td>' + str(data['stds'][1]) + '</td><td>' + str(data['ns'][1]) + '</td></tr>'
        else:
            output_text += '<tr><td>Gemiddelden:</td></tr>'
            output_text += '<tr><td>Niveau</td><td>' + data['varnames'][1][1] + '</td><td>' + data['varnames'][1][2] + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][1] + '</td><td>' + str(data['means'][0]) + '</td><td>' + str(data['means'][1]) + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][2] + '</td><td>' + str(data['means'][2]) + '</td><td>' + str(data['means'][3]) + '</td></tr>'
            output_text += '<tr><td>Standaarddeviaties:</td></tr>'
            output_text += '<tr><td>Niveau</td><td>' + data['varnames'][1][1] + '</td><td>' + data['varnames'][1][2] + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][1] + '</td><td>' + str(data['stds'][0]) + '</td><td>' + str(data['stds'][1]) + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][2] + '</td><td>' + str(data['stds'][2]) + '</td><td>' + str(data['stds'][3]) + '</td></tr>'
            output_text += '<tr><td>N:</td></tr>'
            output_text += '<tr><td>Niveau</td><td>' + data['varnames'][1][1] + '</td><td>' + data['varnames'][1][2] + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][1] + '</td><td>' + str(data['ns'][0]) + '</td><td>' + str(data['ns'][1]) + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][2] + '</td><td>' + str(data['ns'][2]) + '</td><td>' + str(data['ns'][3]) + '</td></tr>'
        return output_text + '</table>'
    
    def print_rmanova(self, assignment: Dict) -> str:
        data: Dict = assignment['data']
        n_conditions = len(data['means'])
        output_text = assignment['instruction'] + '<br><table style="width:45%">'
        output_text += '<tr><td>'+assignment['independent']+'</td>' + ''.join(['<td>'+x+'</td>' for x in assignment['levels'][:n_conditions]]) + '<td>Opgevoerde meting</td></tr>'
        output_text += '<tr><td>Gemiddelde</td>' + ''.join(['<td>'+str(x)+'</td>' for x in data['means'][:n_conditions]]) + '<td>' + str(round(np.mean(data['jackedmeans']),2)) + '</td></tr>'
        output_text += '<tr><td>Standaardeviatie</td>' + ''.join(['<td>'+str(x)+'</td>' for x in data['stds'][:n_conditions]]) + '<td>' + str(round(np.std(data['jackedmeans'], ddof=1),2)) + '</td></tr>'
        output_text += '<tr><td>N</td><td>' + str(data['n_subjects']) + '</td></tr></table>'
        output_text += '<br>Originele scores:'
        output_text += '<br><table style="width:45%">'
        output_text += '<tr><td>Subject</td>' + ''.join(['<td>'+x+'</td>' for x in assignment['levels'][:n_conditions]]) + '<td>Opgevoerde meting</td></tr>'
        for i in range(assignment['data']['n_subjects']):
            output_text += '<tr><td>'+str(i+1)+'</td>' + ''.join(['<td>'+str(x)+'</td>' for x in [data['scores'][j][i] for j in range(n_conditions)]]) + '<td>' + str(round(data['jackedmeans'][i],2)) + '</td></tr>'
        return output_text + '</table>'
    
    def print_analysis(self, assignment: Dict):
        return assignment['instruction'] + '<br>'  
    
    def print_report(self, assignment: Dict, answer=False) -> str: 
        output:str = '' #"Answer" is a parameter which triggers when only the mean/ANOVA tables have to be printed
        if assignment['assignment_type'] not in [1,2,11,13]:
            data:dict = assignment['data']
        names = ['df','ss','ms','F','p','r2'];names2 = ['df','ss','ms','F','p','eta']
        if assignment['assignment_type'] == 1:
            if not answer:
                output += self.print_ttest(assignment)
            output += '<p><table style="width:20%">'
            output += '<tr><td>'+str(assignment['independent'])+'</td><td>Gemiddelde</td><td>Standaarddeviatie</td><td>N</td></tr>'
            output += '<tr><td>'+str(assignment['levels'][0])+'</td><td>'+str(round(assignment['means'][0],2))+'</td><td>'+str(round(assignment['stds'][0],2))+'</td><td>'+str(assignment['ns'][0])+'</td></tr>'
            output += '<tr><td>'+str(assignment['levels'][1])+'</td><td>'+str(round(assignment['means'][1],2))+'</td><td>'+str(round(assignment['stds'][1],2))+'</td><td>'+str(assignment['ns'][1])+'</td></tr>'
            output += '</table></p>'
            if not answer:
                output += '<p><table style="width:20%">'
                output += '<tr><td>Statistiek</td><td>Waarde</td></tr>'
                output += '<tr><td>Vrijheidsgraden (df)</td><td>'+str(assignment['df'][0])+'</td></tr>'
                output += '<tr><td>Ruw effect</td><td>'+str(round(assignment['raw_effect'][0],2))+'</td></tr>'
                output += '<tr><td>Relatief effect</td><td>'+str(round(assignment['relative_effect'][0],2))+'</td></tr>'
                output += '<tr><td>T</td><td>'+str(round(assignment['T'][0],2))+'</td></tr>'
                output += '<tr><td>p</td><td>'+str(round(assignment['p'][0],2))+'</td></tr>'
                output += '</table></p>'
        if assignment['assignment_type'] == 2:
            if not answer:
                output += self.print_ttest(assignment)
            output += '<p><table style="width:20%">'
            output += '<tr><td>'+str(assignment['independent'])+'</td><td>Gemiddelde</td><td>Standaarddeviatie</td><td>N</td></tr>'
            output += '<tr><td>Verschilscores</td><td>'+str(round(assignment['means'][0],2))+'</td><td>'+str(round(assignment['stds'][0],2))+'</td><td>'+str(round(assignment['ns'][0],2))+'</td></tr>'
            output += '</table></p>'
            if not answer:
                output += '<p><table style="width:20%">'
                output += '<tr><td>Statistiek</td><td>Waarde</td></tr>'
                output += '<tr><td>Vrijheidsgraden (df)</td><td>'+str(assignment['df'][0])+'</td></tr>'
                output += '<tr><td>Ruw effect</td><td>'+str(round(assignment['raw_effect'][0],2))+'</td></tr>'
                output += '<tr><td>Relatief effect</td><td>'+str(round(assignment['relative_effect'][0],2))+'</td></tr>'
                output += '<tr><td>T</td><td>'+str(round(assignment['T'][0],2))+'</td></tr>'
                output += '<tr><td>p</td><td>'+str(round(assignment['p'][0],2))+'</td></tr>'
                output += '</table></p>'
        if assignment['assignment_type'] == 3:
            if not answer:
                output += self.print_anova(assignment)
            output += '<p><table style="width:20%">'
            output += '<tr><td>Bron</td><td>df</td><td>SS</td><td>MS</td><td>F</td><td>p</td><td>R<sup>2</sup></td></tr>'
            output += '<tr><td>Between</td>'+''.join(['<td>'+str(round(assignment[x][0],2))+'</td>' for x in names if len(assignment[x]) > 0])+'</tr>'
            output += '<tr><td>Within</td>'+''.join(['<td>'+str(round(assignment[x][1],2))+'</td>' for x in names if len(assignment[x]) > 1])+'</tr>'
            output += '<tr><td>Totaal</td>'+''.join(['<td>'+str(round(assignment[x][2],2))+'</td>' for x in names if len(assignment[x]) > 2])+'</tr>'
            output += '</table></p>'
        if assignment['assignment_type'] == 4:
            if not answer:
                output += self.print_anova(assignment)
            output += '<p><table style="width:20%">'
            output += '<tr><td>Bron</td><td>df</td><td>SS</td><td>MS</td><td>F</td><td>p</td><td>R<sup>2</sup></td></tr>'
            output += '<tr><td>Between</td>'+''.join(['<td>'+str(round(assignment[x][0],2))+'</td>' for x in names[:3]])+'</tr>'
            output += '<tr><td>'+assignment['independent']+'</td>'+''.join(['<td>'+str(round(assignment[x][1],2))+'</td>' for x in names[:3]])+''.join(['<td>'+str(round(assignment[x][0],2))+'</td>' for x in names[3:]])+'</tr>'
            output += '<tr><td>'+assignment['independent2']+'</td>'+''.join(['<td>'+str(round(assignment[x][2],2))+'</td>' for x in names[:3]])+''.join(['<td>'+str(round(assignment[x][1],2))+'</td>' for x in names[3:]])+'</tr>'
            output += '<tr><td>Interaction</td>'+''.join(['<td>'+str(round(assignment[x][3],2))+'</td>' for x in names[:3]])+''.join(['<td>'+str(round(assignment[x][2],2))+'</td>' for x in names[3:]])+'</tr>'
            output += '<tr><td>Within</td>'+''.join(['<td>'+str(round(assignment[x][4],2))+'</td>' for x in names[:3]])+'</tr>'
            output += '<tr><td>Totaal</td>'+''.join(['<td>'+str(round(assignment[x][5],2))+'</td>' for x in names[:2]])+'</tr>'
            output += '</table></p>'
        if assignment['assignment_type'] == 5:
            if not answer:
                output += self.print_rmanova(assignment)
            output += '<p><table style="width:20%">'
            output += '<tr><td>Bron</td><td>df</td><td>SS</td><td>MS</td><td>F</td><td>p</td><td>R<sup>2</sup></td></tr>'
            output += '<tr><td>'+assignment['independent']+'</td>'+''.join(['<td>'+str(round(assignment[x][0],2))+'</td>' for x in names if len(assignment[x]) > 0])+'</tr>'
            output += '<tr><td>Persoon</td>'+''.join(['<td>'+str(round(assignment[x][1],2))+'</td>' for x in names if len(assignment[x]) > 1])+'</tr>'
            output += '<tr><td>Interactie</td>'+''.join(['<td>'+str(round(assignment[x][0],2))+'</td>' for x in names if len(assignment[x]) > 2])+'</tr>'
            output += '<tr><td>Totaal</td>'+''.join(['<td>'+str(round(assignment[x][3],2))+'</td>' for x in names if len(assignment[x]) > 3])+'</tr>'
            output += '</table></p>'
        if assignment['assignment_type'] == 6:
            output += self.print_analysis(assignment)
            output += '<p><table style="width:20%">'
            output += '<tr><td>Bron</td><td>df</td><td>SS</td><td>MS</td><td>F</td><td>p</td><td>R<sup>2</sup></td></tr>'
            output += '<tr><td>Regressie</td>'+''.join(['<td>'+str(round(assignment[x][0],2))+'</td>' for x in names])+'</tr>'
            output += '<tr><td>Residu</td>'+''.join(['<td>'+str(round(assignment[x][1],2))+'</td>' for x in names[:3]])+'</tr>'
            output += '<tr><td>Totaal</td>'+''.join(['<td>'+str(round(assignment[x][2],2))+'</td>' for x in names[:2]])+'</tr>'
            output += '</table></p>'
            
            output += '<p><table style="width:20%">'
            output += '<tr><td>Predictor</td><td>b</td><td>Beta</td><td>Standaarderror</td><td>T</td><td>p</td></tr>'
            for i in range(len(data['predictoren'])):
                output += '<tr><td>'+data['predictoren'][i]+'</td><td>'+str(round(assignment['predictor_b'][i],2))+'</td><td>'+str(round(assignment['predictor_beta'][i],2))+'</td><td>'+str(round(assignment['predictor_se'][i],2))+'</td><td>'+str(round(assignment['predictor_t'][i],2))+'</td><td>'+str(round(assignment['predictor_p'][i],3))+'</td></tr>'
            output += '</table></p>'
        if assignment['assignment_type'] == 11:
            nt = sum(assignment['ns']) #Total number of subjects
            nl = len(assignment['levels']) #Number of levels factor
            hdf = assignment['hdf']; edf = assignment['edf']
            output += self.print_analysis(assignment)
            output += '<p>Multivariate tests<table style="width:20%">'
            output += format_table(['Effect','','Value','F','Hypothesis df','Error df','p','Partial eta<sup>2</sup>'])
            output += format_table(['Intercept',"Pillai's trace",assignment['value'][0],assignment['F1'][0],hdf[0],edf[0],assignment['p1'][0],assignment['eta1'][0]])
            output += format_table(['',"Wilks' lambda",assignment['value'][1],assignment['F1'][1],hdf[1],edf[1],assignment['p1'][1],assignment['eta1'][1]])
            output += format_table(['',"Hotelling's trace",assignment['value'][2],assignment['F1'][2],hdf[2],edf[2],assignment['p1'][2],assignment['eta1'][2]])
            output += format_table(['',"Roy's largest root",assignment['value'][3],assignment['F1'][3],hdf[3],edf[3],assignment['p1'][3],assignment['eta1'][3]])
            output += format_table([cap(assignment['independent']),"Pillai's trace",assignment['value'][4],assignment['F1'][4],hdf[4],edf[4],assignment['p1'][4],assignment['eta1'][4]])
            output += format_table(['',"Wilks' lambda",assignment['value'][5],assignment['F1'][5],hdf[5],edf[5],assignment['p1'][5],assignment['eta1'][5]])
            output += format_table(['',"Hotelling's trace",assignment['value'][6],assignment['F1'][6],hdf[6],edf[6],assignment['p1'][6],assignment['eta1'][6]])
            output += format_table(['',"Roy's largest root",assignment['value'][7],assignment['F1'][7],hdf[7],edf[7],assignment['p1'][7],assignment['eta1'][7]])
            output += '</table></p>'
            
            output += '<p>Tests van within-subject effecten<table style="width:50%">'
            output += format_table(['Bron','Variabele','SS','df','MS','F','p','Partial eta<sup>2</sup>'])
            output += format_table(['Corrected model',assignment['dependent'],assignment['ss0'][0],nl-1,assignment['ms0'][0],assignment['F0'][0],assignment['p0'][0],assignment['eta0'][0]])
            output += format_table(['',assignment['dependent2'],assignment['ss0'][1],nl-1,assignment['ms0'][1],assignment['F0'][1],assignment['p0'][1],assignment['eta0'][1]])
            output += format_table(['',assignment['dependent3'],assignment['ss0'][2],nl-1,assignment['ms0'][2],assignment['F0'][2],assignment['p0'][2],assignment['eta0'][2]])
            output += format_table(['Intercept',assignment['dependent'],assignment['ss0'][3],1,assignment['ms0'][3],assignment['F0'][3],assignment['p0'][3],assignment['eta0'][3]])
            output += format_table(['',assignment['dependent2'],assignment['ss0'][4],1,assignment['ms0'][4],assignment['F0'][4],assignment['p0'][4],assignment['eta0'][4]])
            output += format_table(['',assignment['dependent3'],assignment['ss0'][5],1,assignment['ms0'][5],assignment['F0'][5],assignment['p0'][5],assignment['eta0'][5]])
            output += format_table([cap(assignment['independent']),assignment['dependent'],assignment['ss0'][6],nl-1,assignment['ms0'][6],assignment['F0'][6],assignment['p0'][6],assignment['eta0'][6]])
            output += format_table(['',assignment['dependent2'],assignment['ss0'][7],nl-1,assignment['ms0'][7],assignment['F0'][7],assignment['p0'][7],assignment['eta0'][7]])
            output += format_table(['',assignment['dependent3'],assignment['ss0'][8],nl-1,assignment['ms0'][8],assignment['F0'][8],assignment['p0'][8],assignment['eta0'][8]])
            output += format_table(['Error',assignment['dependent'],assignment['ss0'][9],nt-nl,assignment['ms0'][9],'','',''])
            output += format_table(['',assignment['dependent2'],assignment['ss0'][10],nt-nl,assignment['ms0'][10],'','',''])
            output += format_table(['',assignment['dependent3'],assignment['ss0'][11],nt-nl,assignment['ms0'][11],'','',''])
            output += format_table(['Total',assignment['dependent'],assignment['ss0'][12],nt,'','','',''])
            output += format_table(['',assignment['dependent2'],assignment['ss0'][13],nt,'','','',''])
            output += format_table(['',assignment['dependent3'],assignment['ss0'][14],nt,'','','',''])
            output += format_table(['Corrected total',assignment['dependent'],assignment['ss0'][15],nt-1,'','','',''])
            output += format_table(['',assignment['dependent2'],assignment['ss0'][16],nt-1,'','','',''])
            output += format_table(['',assignment['dependent3'],assignment['ss0'][17],nt-1,'','','',''])
            output += '</table></p>'
        if assignment['assignment_type'] == 12:
            output += self.print_analysis(assignment)
            output += '<p><table style="width:20%">'
            output += '<tr><td>Bron</td><td>df</td><td>SS</td><td>MS</td><td>F</td><td>p</td><td>eta<sup>2</sup></td></tr>'
            output += '<tr><td>Corrected model</td>'+''.join(['<td>'+str(round(assignment[x][3],2))+'</td>' for x in names2])+'</tr>'
            output += '<tr><td>Intercept</td>'+''.join(['<td>'+str(round(assignment[x][6],2))+'</td>' for x in names2])+'</tr>'
            output += '<tr><td>'+cap(assignment['predictor_names'][0])+'</td>'+''.join(['<td>'+str(round(assignment[x][0],2))+'</td>' for x in names2])+'</tr>'
            output += '<tr><td>'+cap(assignment['predictor_names'][1])+'</td>'+''.join(['<td>'+str(round(assignment[x][1],2))+'</td>' for x in names2])+'</tr>'
            output += '<tr><td>'+cap(assignment['independent'])+'</td>'+''.join(['<td>'+str(round(assignment[x][2],2))+'</td>' for x in names2])+'</tr>'
            output += '<tr><td>Error</td>'+''.join(['<td>'+str(round(assignment[x][4],2))+'</td>' for x in names[:3]])+'</tr>'
            output += '<tr><td>Total</td>'+''.join(['<td>'+str(round(assignment[x][7],2))+'</td>' for x in names[:2]])+'</tr>'
            output += '<tr><td>Corrected total</td>'+''.join(['<td>'+str(round(assignment[x][5],2))+'</td>' for x in names[:2]])+'</tr>'
            output += '</table></p>'
        if assignment['assignment_type'] == 13:
            output += self.print_analysis(assignment)
            output += '<p>Multivariate Tests<table style="width:60%">'
            output += format_table(['Effect', '', '', 'Value', 'Hypothesis df','Error df','F','p','eta<sup>2</sup>'])
            for i in range(16):
                i2 = i // 4
                header = 'Between Subjects' if i == 0 else 'Within Subjects' if i == 8 else ''
                header2 = 'Intercept' if i == 0 else assignment['independent2'] if i == 4 else assignment['independent'] if i == 8 else \
                        assignment['independent']+' * '+assignment['independent2'] if i == 12 else ''
                measure = ["Pillai's Trace","Wilks' Lambda","Hotelling's Trace","Roy's Largest Root"][i2]
                output += format_table([header,header2,measure,assignment['value'][i],assignment['hdf'][i2],assignment['edf'][i2],
                                        assignment['F0'][i2],assignment['p0'][i2],assignment['eta0'][i2]])
            output += '</table></p>'
            output += '<p>Tests of Within-Subjects Contrasts<table style="width:60%">'
            output += format_table(['Source', 'Measure', assignment['independent'], 'SS','df','MS','F','p','eta<sup>2</sup>'])
            for i in range(12):
                header = assignment['independent'] if i == 0 else assignment['independent']+' * '+assignment['independent2'] if i == 4 else 'Error('+assignment['independent']+')' if i == 8 else ''
                header2 = assignment['dependent'] if (i+2) % 4 == 0 else assignment['dependent2'] if i % 4 == 0 else ''
                measure = assignment['levels'][0]+' vs. '+assignment['levels'][1] if i % 2 == 0 else assignment['levels'][1]+' vs. '+assignment['levels'][2]
                if i < 8:
                    output += format_table([header,header2,measure,assignment['ss1'][i],assignment['df1'][i],assignment['ms1'][i],
                                        assignment['F1'][i],assignment['p1'][i],assignment['eta1'][i]])
                else:
                    output += format_table([header,header2,measure,assignment['ss1'][i],assignment['df1'][i],assignment['ms1'][i],
                                        '','',''])
            output += '</table></p>'
            output += '<p>Tests of Between-Subjects Effects<table style="width:20%">'
            output += format_table(['Source','Measure','df','SS','MS','F','p','eta<sup>2</sup>'])
            output += format_table(['Intercept',assignment['dependent']] + [assignment[x][0] for x in names2])
            output += format_table(['',assignment['dependent2']] + [assignment[x][1] for x in names2])
            output += format_table([assignment['independent2'],assignment['dependent']] + [assignment[x][2] for x in names2])
            output += format_table(['',assignment['dependent2']] + [assignment[x][3] for x in names2])
            output += format_table(['Error',assignment['dependent']] + [assignment[x][4] for x in names2[:3]])
            output += format_table(['',assignment['dependent2']] + [assignment[x][5] for x in names2[:3]])
            output += '</table></p>'
        return output
    
    def print_independent(self, assignment:dict, num:int=1) -> str:
        levels = assignment['levels'] if num < 2 else assignment['levels' + str(num)]
        if assignment['assignment_type'] < 5:
            return assignment['independent'] + ', kwalitatief, met niveaus ' + levels[0] + ' en ' + levels[1] + '.'
        else:    
            i_key:str = 'independent' if num < 2 else 'independent' + str(num)
            return assignment[i_key] + ', een within-subject factor met niveaus ' + ' en '.join(levels) + '.'
    
    def print_dependent(self, assignment:dict) -> str:
        return assignment['dependent'] + ', kwantitatief'
    
    def print_struct(self, d: Dict):
        for key, value in list(d.items()):
            print(key + ': ' + str(value))
    
