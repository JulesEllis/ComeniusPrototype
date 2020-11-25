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

class Assignments:
    def __init__(self):
        self.t_test_vars: List[str] = ['independent','dependent','dependent_measure','independent_measure',
                            'levels','null','control','means','stds','ns','df','raw_effect',
                            'relative_effect','T','p','decision','interpretation'] #Variables in t-test solution  
        self.anova_vars: List[str] = ['independent','independent2','dependent','dependent_measure','independent_measure', 'independent2_measure'
                            'levels','levels2','null','null2','null3','control','ss','df','ms','F','p','r2',
                            'decision','decision2','decision3','interpretation','interpretation2','interpretation3']
        
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
        if between_subject:
            varnames:List = [['Beroep'] + random.sample(['Arts','Leraar','Loodgieter','Bouwvakker'], 2)]
        else:
            varnames:List = [['Tijdstip', 'Dag', 'Nacht']]
            
        #Create the assignment description
        report_type = 'elementair' if elementary else 'beknopt'
        instruction: str = 'Maak een '+report_type+' rapport van onderstaande data voor de hypothese dat '
        if hyp_type == 0:
            instruction += 'de reactiesnelheid bij het ' + varnames[0][0] + ' ' + varnames[0][1] + ' gemiddeld ongelijk is aan die bij ' + varnames[0][2] + '.<br><br>'
        if hyp_type == 1:
            instruction += 'de reactiesnelheid bij het ' + varnames[0][0] + ' ' + varnames[0][1] + ' gemiddeld groter is dan die bij ' + varnames[0][2] + '.<br><br>'
        if hyp_type == 2:
            instruction += "de reactiesnelheid bij het " + varnames[0][0] + " " + varnames[0][1] + " gemiddeld kleiner is dan die bij " + varnames[0][2] + ".<br><br>"
        instruction += "De persoon moet een tijdje achter een beeldscherm zo snel mogelijk op een knop" +" drukken zodra een wit vierkantje veranderd in een zwart vierkantje. De milliseconden tussen het veranderen" + " van het beeld en het indrukken van de knop worden gemeten en opgeteld. "        
        if between_subject:
            instruction += 'De proefpersonen zijn allemaal ofwel ' + varnames[0][1] + ' ofwel '+ varnames[0][2] + '. De niveaus zijn: ' + varnames[0][1] + '/' + varnames[0][2] + '. Voer je antwoorden alsjeblieft tot op 2 decimalen in'\
                 ' en gebruik dezelfde vergelijking van de gemiddelden in je antwoord als in de vraagstelling staat (e.g. "groter" of "kleiner"). '
            if control:
                instruction += 'De personen van elk beroep zijn willekeurig geselecteerd. '
        else:
            instruction += 'De proefpersonen werden tweemaal getoetst op hun reactiesnelheid, een keer overdag en een keer s\'nachts. ' + 'De niveaus zijn dag/nacht. Voer je antwoorden alsjeblieft tot op 2 decimalen in'\
            ' en gebruik dezelfde vergelijking van de gemiddelden in je antwoord als in de vraagstelling staat (e.g. "groter" of "kleiner"). '
            if control:
                instruction += 'De volgorde van de toetsen was gerandomiseerd. '
        
        #Generate datapoints: Floats are rounded to 2 decimals
        return{'instruction': instruction,
               'hypothesis': hyp_type,
               'between_subject': between_subject,
               'control': control,
               'dependent': 'reactiesnelheid',
               'assignment_type':1 if between_subject else 2,
               'data':{'varnames': varnames,
               'A': [round(random.gauss(mean1,std1), 2) for i in range(n1)], 
               'B': [round(random.gauss(mean2,std2), 2) for i in range(n2)]}
               }
        
    def create_anova(self, two_way: bool, control: bool, elementary:bool=True) -> Dict:
        output = {'two_way':two_way, 'control':control}
        output['dependent'] = 'gewicht'
        output['instruction']: str = None
        output['assignment_type']: int = 4 if two_way else 3
        samplevars = [['Nationaliteit','Nederlands','Duits'], #Sample variable names
                      ['Geslacht','Man','Vrouw'],
                      ['Lievelingskleur','Rood','Blauw'],
                      ['Religie','Christelijk','Moslim']]
        if not two_way:
            varnames = random.sample(samplevars, 1)
        else:
            varnames = random.sample(samplevars, 2)
        
        #Decide the variable names
        report_type = 'elementair' if elementary else 'beknopt'
        if not two_way:
            output['instruction'] = 'Maak een '+report_type+' rapport van de onderstaande data. De variabelen zijn '+varnames[0][0]+', met niveaus '+' en '.join(varnames[0][1:])+', en gewicht. Voer je antwoorden alsjeblieft tot op 2 decimalen in '\
                 'en gebruik dezelfde vergelijking van de gemiddelden in je antwoord als in de vraagstelling staat (e.g. "groter" of "kleiner"). '
        else:
            output['instruction'] = 'Maak een '+report_type+' rapport van de onderstaande data. De variabelen zijn '+varnames[0][0]+', met niveaus '+' en '.join(varnames[0][1:])+', '+varnames[1][0]+' met niveaus '+' en '.join(varnames[1][1:])+', en gewicht. Voer je antwoorden alsjeblieft tot op 2 decimalen in '\
                 'en gebruik dezelfde vergelijking van de gemiddelden in je antwoord als in de vraagstelling staat (e.g. "groter" of "kleiner"). '
        if control:
            output['instruction'] += 'De deelnemers zijn willekeurig gekozen. '

        #Generate summary statistics
        n: int = random.randint(9,16)
        if not two_way:
            output['data'] = {
                  'varnames':varnames,
                  'means':[round(random.uniform(50,120),2) for i in range(2)],
                  'stds':[round(random.uniform(5,15),2) for i in range(2)],
                  'ns': [n for i in range(2)]}
        else:
            output['data'] = {
                  'varnames':varnames,
                  'means':[round(random.uniform(50,120),2) for i in range(4)],
                  'stds':[round(random.uniform(5,15),2) for i in range(4)],
                  'ns': [n for i in range(4)]}
        return output
    
    def create_rmanova(self, control: bool, elementary:bool=True) -> Dict:
        #Determine variable shape and names
        output = {'control': control, 'two_way':False, 'assignment_type':5}
        output['dependent'] = 'gewicht'
        n_conditions = random.randint(2,4)
        n_subjects = int(random.uniform(8,15))
        samplevars = [['Kwartaal','Eerste', 'Tweede', 'Derde', 'Vierde'], 
                      ['Seizoen','Winter','Lente','Zomer','Herfst'],
                      ['Maand','Januari','Februari','Maart','April'],
                      ['Dag','Maandag','Dinsdag','Woensdag','Donderdag']]
        varnames = [random.choice(samplevars)[:n_conditions+1]]
        
        report_type = 'elementair' if elementary else 'beknopt'
        output['instruction']: str = 'Maak een '+report_type+' rapport van de onderstaande data. De variabelen zijn '+varnames[0][0]+' en gewicht. Voer je antwoorden alsjeblieft tot op 2 decimalen in '\
                 'en gebruik dezelfde vergelijking van de gemiddelden in je antwoord als in de vraagstelling staat (e.g. "groter" of "kleiner"). '
        if control:
            output['instruction'] += 'De subjecten in het experiment zijn willekeurig geselecteerd. '
        true_means = [int(random.uniform(50,120)) for i in range(n_conditions)]
        true_stds = [int(random.uniform(5,20)) for i in range(n_conditions)]
        output['data'] = {
                  'varnames':varnames,
                  'scores':[[round(random.gauss(true_means[i], true_stds[i]),2) for j in range(n_subjects)] for i in range(n_conditions)]
                  }
        output['data']['means']: List = [round(np.mean(output['data']['scores'][i]),2) for i in range(n_conditions)]
        output['data']['stds']: List = [round(np.std(output['data']['scores'][i]),2) for i in range(n_conditions)]
        output['data']['jackedmeans']: List = [round(np.mean([output['data']['scores'][i][j] for i in range(n_conditions)]),2) for j in range(n_subjects)]
        output['data']['n_subjects'] = n_subjects
        output['data']['n_conditions'] = n_conditions
        return output
    
    def create_mregression(self, control: bool, elementary:bool=False):
        report_type = 'elementair' if elementary else 'beknopt'
        output = {'assignment_type':6}
        N = int(200 * random.random())
        output['ns'] = [N]
        n_predictors = random.choice([3,4,5,6])
        output['n_predictors'] = n_predictors
        
        p = random.random()
        s = 3 * random.random()
        output['var_obs'] = (1 + chi2.ppf(p, df=10)) * 10 ** s
        r2 = random.random() ** 2
        output['var_pred'] = output['var_obs'] * r2
        output['data']: dict={'varnames':['Intercept','Sociale vaardigheden', 'Depressieve gedachten', 'Eetlust','Intelligentie','Assertiviteit','Ervaren geluk'][:n_predictors+1]}
        output['dependent'] = 'gewicht'
        #output['correlations'] = [random.random() for i in range(int(((n_predictors + 1) ** 2 - n_predictors - 1) * 0.5))]
        output['instruction'] = 'Maak een '+report_type+' rapport van de onderstaande data. De variabelen zijn '+' en '.join(output['data']['varnames'][1:])+' als predictoren en '+output['dependent']+' als criterium. Voer je antwoorden alsjeblieft tot op 2 decimalen in. '
        return output
	
    def create_report(self, control: bool, choice: int=0):
        if choice == 0:
            choice = random.choice([1,2,3,4,5])
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
        output['assignment_type'] = choice
        return output
            
    def print_ttest(self, assignment: Dict) -> str:
        output_text = assignment['instruction'] + '<br>'
        varnames: List[str] = assignment['data']['varnames'][0]
        data: List = [assignment['data']['A'], assignment['data']['B']]
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
            output_text += '<tr><td>' + data['varnames'][0][1] + '</td><td>' + str(data['means'][0]) + '</td><td>' + str(data['means'][2]) + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][2] + '</td><td>' + str(data['means'][1]) + '</td><td>' + str(data['means'][3]) + '</td></tr>'
            output_text += '<tr><td>Standaarddeviaties:</td></tr>'
            output_text += '<tr><td>Niveau</td><td>' + data['varnames'][1][1] + '</td><td>' + data['varnames'][1][2] + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][1] + '</td><td>' + str(data['stds'][0]) + '</td><td>' + str(data['stds'][2]) + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][2] + '</td><td>' + str(data['stds'][1]) + '</td><td>' + str(data['stds'][3]) + '</td></tr>'
            output_text += '<tr><td>N:</td></tr>'
            output_text += '<tr><td>Niveau</td><td>' + data['varnames'][1][1] + '</td><td>' + data['varnames'][1][2] + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][1] + '</td><td>' + str(data['ns'][0]) + '</td><td>' + str(data['ns'][2]) + '</td></tr>'
            output_text += '<tr><td>' + data['varnames'][0][2] + '</td><td>' + str(data['ns'][1]) + '</td><td>' + str(data['ns'][3]) + '</td></tr>'
        return output_text + '</table>'
    
    def print_rmanova(self, assignment: Dict) -> str:
        data: Dict = assignment['data']
        n_conditions = len(data['means'])
        output_text = assignment['instruction'] + '<br><table style="width:45%">'
        output_text += '<tr><td>'+assignment['data']['varnames'][0][0]+'</td>' + ''.join(['<td>'+x+'</td>' for x in assignment['data']['varnames'][0][1:n_conditions+1]]) + '<td>Opgevoerde meting</td></tr>'
        output_text += '<tr><td>Gemiddelde</td>' + ''.join(['<td>'+str(x)+'</td>' for x in data['means'][:n_conditions]]) + '<td>' + str(round(np.mean(data['jackedmeans']),2)) + '</td></tr>'
        output_text += '<tr><td>Standaardeviatie</td>' + ''.join(['<td>'+str(x)+'</td>' for x in data['stds'][:n_conditions]]) + '<td>' + str(round(np.std(data['jackedmeans']),2)) + '</td></tr>'
        output_text += '<tr><td>N</td><td>' + str(data['n_subjects']) + '</td></tr></table>'
        output_text += '<br>Originele scores:'
        output_text += '<br><table style="width:45%">'
        output_text += '<tr><td>Subject</td>' + ''.join(['<td>'+x+'</td>' for x in assignment['data']['varnames'][0][1:n_conditions+1]]) + '<td>Opgevoerde meting</td></tr>'
        for i in range(assignment['data']['n_subjects']):
            output_text += '<tr><td>'+str(i+1)+'</td>' + ''.join(['<td>'+str(x)+'</td>' for x in [data['scores'][j][i] for j in range(n_conditions)]]) + '<td>' + str(round(data['jackedmeans'][i],2)) + '</td></tr>'
        return output_text + '</table>'
    
    def print_mregression(self, assignment: Dict):
        output_text = assignment['instruction'] + '<br>'
        #output_text += '<br><table style="width:30%">'
        #output_text += '<tr><td>Statistiek</td><td>N</td><td>Variantie geobserveerde scores</td><td>Variantie voorspelde scores</td></tr>'
        #output_text += '<tr><td>Waarde</td><td>'+str(assignment['ns'][0])+'</td><td>'+str(round(assignment['var_obs'],2))+'</td><td>'+str(round(assignment['var_pred'],2))+'</td></tr>'
        return output_text #+ '</table>'
    
    def print_report(self, assignment: Dict) -> str:
        output:str = ''
        data:dict = assignment['data']
        names = ['df','ss','ms','F','p','r2']
        if assignment['assignment_type'] == 1:
            output += self.print_ttest(assignment)
            output += '<p><table style="width:20%">'
            output += '<tr><td>'+str(data['varnames'][0][0])+'</td><td>Gemiddelde</td><td>Standaarddeviatie</td><td>N</td></tr>'
            output += '<tr><td>'+str(data['varnames'][0][1])+'</td><td>'+str(round(assignment['means'][0],2))+'</td><td>'+str(round(assignment['stds'][0],2))+'</td><td>'+str(assignment['ns'][0])+'</td></tr>'
            output += '<tr><td>'+str(data['varnames'][0][2])+'</td><td>'+str(round(assignment['means'][1],2))+'</td><td>'+str(round(assignment['stds'][1],2))+'</td><td>'+str(assignment['ns'][1])+'</td></tr>'
            output += '</table></p><p><table style="width:20%">'
            output += '<tr><td>Statistiek</td><td>Waarde</td></tr>'
            output += '<tr><td>Vrijheidsgraden (df)</td><td>'+str(assignment['df'][0])+'</td></tr>'
            output += '<tr><td>Ruw effect</td><td>'+str(round(assignment['raw_effect'][0],2))+'</td></tr>'
            output += '<tr><td>Relatief effect</td><td>'+str(round(assignment['relative_effect'][0],2))+'</td></tr>'
            output += '<tr><td>T</td><td>'+str(round(assignment['T'][0],2))+'</td></tr>'
            output += '<tr><td>p</td><td>'+str(round(assignment['p'][0],2))+'</td></tr>'
            output += '</table></p>'
        if assignment['assignment_type'] == 2:
            output += self.print_ttest(assignment)
            output += '<p><table style="width:20%">'
            output += '<tr><td>'+str(data['varnames'][0][0])+'</td><td>Gemiddelde</td><td>Standaarddeviatie</td><td>N</td></tr>'
            output += '<tr><td>Verschilscores</td><td>'+str(round(assignment['means'][0],2))+'</td><td>'+str(round(assignment['stds'][0],2))+'</td><td>'+str(round(assignment['ns'][0],2))+'</td></tr>'
            output += '</table></p><p><table style="width:20%">'
            output += '<tr><td>Statistiek</td><td>Waarde</td></tr>'
            output += '<tr><td>Vrijheidsgraden (df)</td><td>'+str(assignment['df'][0])+'</td></tr>'
            output += '<tr><td>Ruw effect</td><td>'+str(round(assignment['raw_effect'][0],2))+'</td></tr>'
            output += '<tr><td>Relatief effect</td><td>'+str(round(assignment['relative_effect'][0],2))+'</td></tr>'
            output += '<tr><td>T</td><td>'+str(round(assignment['T'][0],2))+'</td></tr>'
            output += '<tr><td>p</td><td>'+str(round(assignment['p'][0],2))+'</td></tr>'
            output += '</table></p>'
        if assignment['assignment_type'] == 3:
            output += self.print_anova(assignment)
            output += '<p><table style="width:20%">'
            output += '<tr><td>Bron</td><td>df</td><td>SS</td><td>MS</td><td>F</td><td>p</td><td>R<sup>2</sup></td></tr>'
            output += '<tr><td>Between</td>'+''.join(['<td>'+str(round(assignment[x][0],2))+'</td>' for x in names if len(assignment[x]) > 0])+'</tr>'
            output += '<tr><td>Within</td>'+''.join(['<td>'+str(round(assignment[x][1],2))+'</td>' for x in names if len(assignment[x]) > 1])+'</tr>'
            output += '<tr><td>Totaal</td>'+''.join(['<td>'+str(round(assignment[x][2],2))+'</td>' for x in names if len(assignment[x]) > 2])+'</tr>'
            output += '</table></p>'
        if assignment['assignment_type'] == 4:
            output += self.print_anova(assignment)
            output += '<p><table style="width:20%">'
            output += '<tr><td>Bron</td><td>df</td><td>SS</td><td>MS</td><td>F</td><td>p</td><td>R<sup>2</sup></td></tr>'
            output += '<tr><td>Between</td>'+''.join(['<td>'+str(round(assignment[x][0],2))+'</td>' for x in names[:3]])+'</tr>'
            output += '<tr><td>'+data['varnames'][0][0]+'</td>'+''.join(['<td>'+str(round(assignment[x][1],2))+'</td>' for x in names[:3]])+''.join(['<td>'+str(round(assignment[x][0],2))+'</td>' for x in names[3:]])+'</tr>'
            output += '<tr><td>'+data['varnames'][1][0]+'</td>'+''.join(['<td>'+str(round(assignment[x][2],2))+'</td>' for x in names[:3]])+''.join(['<td>'+str(round(assignment[x][1],2))+'</td>' for x in names[3:]])+'</tr>'
            output += '<tr><td>Interaction</td>'+''.join(['<td>'+str(round(assignment[x][3],2))+'</td>' for x in names[:3]])+''.join(['<td>'+str(round(assignment[x][2],2))+'</td>' for x in names[3:]])+'</tr>'
            output += '<tr><td>Within</td>'+''.join(['<td>'+str(round(assignment[x][4],2))+'</td>' for x in names[:3]])+'</tr>'
            output += '<tr><td>Totaal</td>'+''.join(['<td>'+str(round(assignment[x][5],2))+'</td>' for x in names[:2]])+'</tr>'
            output += '</table></p>'
        if assignment['assignment_type'] == 5:
            output += self.print_rmanova(assignment)
            output += '<p><table style="width:20%">'
            output += '<tr><td>Bron</td><td>df</td><td>SS</td><td>MS</td><td>F</td><td>p</td><td>R<sup>2</sup></td></tr>'
            output += '<tr><td>'+data['varnames'][0][0]+'</td>'+''.join(['<td>'+str(round(assignment[x][0],2))+'</td>' for x in names if len(assignment[x]) > 0])+'</tr>'
            output += '<tr><td>Persoon</td>'+''.join(['<td>'+str(round(assignment[x][1],2))+'</td>' for x in names if len(assignment[x]) > 1])+'</tr>'
            output += '<tr><td>Interactie</td>'+''.join(['<td>'+str(round(assignment[x][0],2))+'</td>' for x in names if len(assignment[x]) > 2])+'</tr>'
            output += '<tr><td>Totaal</td>'+''.join(['<td>'+str(round(assignment[x][3],2))+'</td>' for x in names if len(assignment[x]) > 3])+'</tr>'
            output += '</table></p>'
        if assignment['assignment_type'] == 6:
            output += self.print_mregression(assignment)
            output += '<p><table style="width:20%">'
            output += '<tr><td>Bron</td><td>df</td><td>SS</td><td>MS</td><td>F</td><td>p</td><td>R<sup>2</sup></td></tr>'
            output += '<tr><td>Regressie</td>'+''.join(['<td>'+str(round(assignment[x][0],2))+'</td>' for x in names])+'</tr>'
            output += '<tr><td>Residu</td>'+''.join(['<td>'+str(round(assignment[x][1],2))+'</td>' for x in names[:3]])+'</tr>'
            output += '<tr><td>Totaal</td>'+''.join(['<td>'+str(round(assignment[x][2],2))+'</td>' for x in names[:2]])+'</tr>'
            output += '</table></p>'
            
            output += '<p><table style="width:20%">'
            output += '<tr><td>Predictor</td><td>b</td><td>Beta</td><td>Standaarderror</td><td>T</td><td>p</td></tr>'
            for i in range(len(data['varnames'])):
                output += '<tr><td>'+data['varnames'][i]+'</td><td>'+str(round(assignment['predictor_b'][i],2))+'</td><td>'+str(round(assignment['predictor_beta'][i],2))+'</td><td>'+str(round(assignment['predictor_se'][i],2))+'</td><td>'+str(round(assignment['predictor_t'][i],2))+'</td><td>'+str(round(assignment['predictor_p'][i],2))+'</td></tr>'
            output += '</table></p>'
            
            #output += '<p><table style="width:20%">'
            #cornames:list = [assignment['dependent']] + data['varnames'][1:]
            #output += '<tr><td>Correlatie</td>' + ''.join(['<td>'+x+'</td>' for x in cornames]) + '</tr>'
            #cors:list = assignment['correlations']
            #n_factors:int = len(data['varnames'])
            #ind:int = 0
            #for i in range(n_factors):
            #    output += '<tr><td>'+cornames[i]+'</td>'+''.join('<td>'+str(round(x,2))+'</td>' for x in cors[ind:ind+i])+''.join(['<td></td>' for x in range(n_factors-i)])+'</tr>'
            #    ind += i
            #output += '</table></p>'
        return output
            
    #Calculate internally all of the numbers and string values the student has to present
    def solve_ttest(self, assignment: Dict, solution: Dict) -> Dict:
        numbers: List = [assignment['data']['A'], assignment['data']['B']]
        names: List[str] = assignment['data']['varnames'][0][1:]
        between_subject: bool = assignment['between_subject']
        solution['hypothesis'] = assignment['hypothesis']
        solution['assignment_type'] = assignment['assignment_type']
        
        if not between_subject:
            #Differential scores
            diff: List[float] = [numbers[0][i] - numbers[1][i] for i in range(len(numbers[0]))]
        
        #Determine variable names and types
        if between_subject:
            solution['independent']: str = 'Beroep'
        else:
            solution['independent']: str = 'Tijdstip'
            
        solution['dependent']: str = assignment['dependent']
        solution['dependent_measure']: str = 'kwantitatief'
        solution['independent_measure']: str = 'kwalitatief'
        solution['levels']: List[str] = assignment['data']['varnames'][0][1:]
        
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
            solution['relative_effect']: List = [solution['raw_effect'][0] / np.std(diff)]
            solution['T']: List = [solution['relative_effect'][0] / math.sqrt(len(diff))]
            
        #Calculate p
        solution['p']: List = None
        if assignment['hypothesis'] == 1 and solution['raw_effect'][0] < 0 or assignment['hypothesis'] == 2 and solution['raw_effect'][0] > 0:
            solution['p'] = [1.0]
        #elif assignment['hypothesis'] == 1 and solution['T'][0] < 0 or assignment['hypothesis'] == 2 and solution['T'][0] > 0:
        #    solution['p'] = [1 - stats.t.sf(np.abs(solution['T'][0]), solution['df'][0])]
        else:
            solution['p']= [stats.t.sf(np.abs(solution['T'][0]), solution['df'][0])]
        if assignment['hypothesis'] == 0:
            solution['p'][0] = min(solution['p'][0] * 2, 1.0)
            
        #Determine textual conclusions
        #Decision
        sterkte: str = 'sterk' if solution['relative_effect'][0] > 0.8 else 'matig' if solution['relative_effect'][0] > 0.5 else 'klein'
        if solution['p'][0] < 0.05:
            decision: Tuple[str] = ('verworpen','', 'Het effect is ' + sterkte + '. ')
        else:
            decision: Tuple[str] = ('behouden','niet ', '')
        comparison: str = ['ongelijk','groter','kleiner'][assignment['hypothesis']]
        solution['decision']: str = 'H0 ' + decision[0] + ', het populatiegemiddelde van ' + names[0] + ' is ' + decision[1] + comparison + ' dan dat van ' + names[1] + '. ' + decision[2]
        
        #Causal interpretation
        if solution['p'][0] < 0.05:
            solution['interpretation']: str = 'Niet van toepassing, er is geen significant effect. '
        else:
            if assignment['control']: 
                solution['interpretation']: str = 'De oorzaak van het effect is het verschil tussen de niveaus van de variabele ' + solution['independent'] + '.'
            else:
                solution['interpretation']: str = 'De oorzaak van het effect is onbekend. '
        return solution
    
    def solve_anova(self, assignment: Dict, solution: Dict) -> Dict:
        data: Dict = assignment['data']
        two_way: bool = assignment['two_way']
        solution['assignment_type'] = assignment['assignment_type']
        
        solution['independent']: str = data['varnames'][0][0]
        solution['dependent']: str = assignment['dependent']
        solution['dependent_measure']: str = 'kwantitatief'
        solution['dependent_n_measure']: int = 1 #Aantal metingen per persoon
        solution['levels']: List[str] = data['varnames'][0][1:]
        solution['control']: bool = assignment['control']
        if two_way:
            solution['independent2'] = data['varnames'][1][0]
            solution['levels2'] = data['varnames'][1][1:]
        
        #One-way statistics
        mean: float = np.mean(data['means'])
        if not two_way:
            #Intermediary statistics order: Between-group, within-group, total
            ssm: float = sum([(data['means'][i] - mean) ** 2 for i in range(len(data['ns']))]) * (sum(data['ns']) - 1)
            sse: float = sum([(data['ns'][i] - 1) * (data['stds'][i]) ** 2 for i in range(len(data['ns']))])
            solution['ss']: List[float] = [ssm, sse, ssm + sse]
            solution['df']: List[float] = [len(data['ns']) - 1, abs(len(data['ns']) - sum(data['ns'])), sum(data['ns']) - 1]
            solution['ms']: List[float] = [solution['ss'][i]/solution['df'][i] for i in range(2)]
            solution['F']: List[float] = [solution['ms'][0] / solution['ms'][1]]
            solution['p']: List[float] = [1 - stats.f.cdf(abs(solution['F'][0]),solution['df'][0],solution['df'][1])]
            solution['r2']: List[float] = [solution['ss'][0]/solution['ss'][2]]
            
            #Verbal parts of the report
            rejected: Tuple[str] = ('verworpen','ongelijk') if solution['p'][0] < 0.05 else ('behouden', 'gelijk')
            solution['null']: str = 'h0: ' + ' == '.join(['mu(' + l + ')' for l in solution['levels']])
            solution['decision']: str = 'h0 ' + rejected[0] + ', de populatiegemiddelden van ' + solution['levels'][0] +' en '+solution['levels'][0]+' zijn gemiddeld ' + rejected[1] + '.'
            if assignment['control']:
                solution['interpretation']: str = 'Het verschil in '+solution['dependent']+' wordt veroorzaakt door de onafhankelijke variabele '+solution['independent']
            else:
                solution['interpretation']: str = 'De echte verklaring is onbekend, de primaire verklaring is dat, de alternatieve is dat het verschil in '+solution['dependent']+' wordt veroorzaakt door de onafhankelijke variabele '+solution['independent']
                
        else: #Two-way statistics
            #Intermediary statistics order: Between, A, B, AB, Within, Total
            #Degrees of freedom
            l1: int = len(solution['levels'])
            l2: int = len(solution['levels2'])
            lt: int = l1 * l2
            N: int = sum(data['ns'])
            solution['df']: List[int] = [lt - 1, l1 - 1, l2 - 1, (l1-1) * (l2-1), sum(data['ns']) - lt, sum(data['ns']) - 1]
            
            #Numerical parts of the report
            ssbetween: float = (N-1) * np.std([val for sublist in [[data['means'][j] for i in range(data['ns'][j])] for j in range(lt)] for val in sublist]) ** 2
            ssa: float = (N-1) * np.std([np.mean([data['means'][0],data['means'][1]]) for i in range(data['ns'][0] + data['ns'][1])]+[np.mean([data['means'][2],data['means'][3]]) for i in range(data['ns'][2] + data['ns'][3])]) ** 2
            ssb: float = (N-1) * np.std([np.mean([data['means'][0],data['means'][2]]) for i in range(data['ns'][0] + data['ns'][2])]+[np.mean([data['means'][1],data['means'][3]]) for i in range(data['ns'][1] + data['ns'][3])]) ** 2
            sswithin: float = sum([data['stds'][i] * (data['ns'][i] - 1) for i in range(lt)])
            solution['ss']: List[float] = [ssbetween, ssa, ssb, ssbetween - ssa - ssb, sswithin, ssbetween + sswithin]
            solution['ms']: List[float] = [solution['ss'][i] / solution['df'][i] for i in range(5)]
            solution['F']: List[float] = [solution['ms'][1] / solution['ms'][4], solution['ms'][2] / solution['ms'][4], solution['ms'][3] / solution['ms'][4]]
            solution['p']: List[float] = [1 - stats.f.cdf(abs(solution['F'][i]),solution['df'][i + 1],solution['df'][4]) for i in range(3)]
            solution['r2']: List[float] = [solution['ss'][1] / solution['ss'][5], solution['ss'][2] / solution['ss'][5], solution['ss'][3] / solution['ss'][5]]
            
            #Verbal parts of the report
            rejected: Tuple[str] = ('verworpen','ongelijk') if solution['p'][0] < 0.05 else ('behouden', 'gelijk')
            rejected2: Tuple[str] = ('verworpen','ongelijk') if solution['p'][1] < 0.05 else ('behouden', 'gelijk')
            rejected3: Tuple[str] = ('verworpen','wel') if solution['p'][2] < 0.05 else ('behouden', 'geen')
            solution['null']: str = 'h0: mu(' + solution['levels'][0] + ') == mu(' + solution['levels'][1] + ')'
            solution['null2']: str =  'h0: mu(' + solution['levels2'][0] + ') == mu(' + solution['levels2'][1] + ')'
            solution['null3']: str = 'h0: Er is geen interactie tussen ' +solution['independent']  + ' en ' + solution['independent2']
            solution['decision']: str = 'h0 ' + rejected[0] + ', de populatiegemiddelden van ' + solution['levels'][0] +' en '+solution['levels'][0]+' zijn gemiddeld ' + rejected[1] + '.'
            solution['decision2']: str = 'h0 ' + rejected2[0] + ', de populatiegemiddelden van ' + solution['levels2'][0] +' en '+solution['levels'][0]+' zijn gemiddeld ' + rejected2[1] + '.'
            solution['decision3']: str = 'h0 ' + rejected3[0] + ', er is ' + rejected3[1] + ' interactie tussen ' + solution['independent'] +' en '+solution['independent2'] + '.'
            if assignment['control']:
                solution['interpretation']: str = 'Het verschil in '+solution['dependent']+' wordt veroorzaakt door de onafhankelijke variabele '+solution['independent']
                solution['interpretation2']: str = 'Het verschil in '+solution['dependent']+' wordt veroorzaakt door de onafhankelijke variabele '+solution['independent']
                solution['interpretation3']: str = 'Het verschil in '+solution['dependent']+' wordt veroorzaakt door de onafhankelijke variabele '+solution['independent']
            else:
                solution['interpretation']: str = 'De echte verklaring is onbekend, de primaire verklaring is dat, de alternatieve is dat het verschil in '+solution['dependent']+' wordt veroorzaakt door de onafhankelijke variabele '+solution['independent']
                solution['interpretation2']: str = 'De echte verklaring is onbekend, de primaire verklaring is dat, de alternatieve is dat het verschil in '+solution['dependent']+' wordt veroorzaakt door de onafhankelijke variabele '+solution['independent2']
                if solution['p'][2] < 0.05:
                    solution['interpretation3']: str = 'De invloed die een verandering in '+solution['independent']+' heeft op de afhankelijke variabele is bij het ene niveau van '+solution['independent2']+' groter of in een andere richting dan bij het andere niveau van '+solution['independent2']+'.'
                else:
                    solution['interpretation3']: str = 'De invloed die een verandering in '+solution['independent']+' heeft op de afhankelijke variabele is bij elk niveau van '+solution['independent2']+' even groot en in dezelfde richting, '
        return solution
    
    def solve_rmanova(self, assignment: Dict, solution: Dict) -> Dict: 
        data: Dict = assignment['data']
        n_conditions = len(data['means'])
        solution['independent']: str = assignment['data']['varnames'][0][0]
        solution['levels'] = assignment['data']['varnames'][0][1:]
        solution['dependent']: str = assignment['dependent']
        solution['dependent_measure']: str = 'kwantitatief'
        solution['dependent_n_measure']: int = n_conditions #Aantal metingen per persoon
        solution['control']: bool = assignment['control']
        solution['assignment_type'] = assignment['assignment_type']
        
        #Numerical parts of the report
        #Order of rows: Kwartaal, Persoon, Interactie, Totaal
        nc: int = data['n_conditions']
        ns: int = data['n_subjects']
        N: int = nc * ns
        solution['df']: List[int] =  [nc - 1, ns - 1, (nc-1) * (ns-1), N - 1]
        ssk: float = (N-1) * np.std(data['means']) ** 2
        ssp: float = (N-1) * np.std(data['jackedmeans']) ** 2
        sstotal: float = (N-1) * np.std([x for y in [data['scores']] for x in y]) ** 2
        ssi = sstotal - ssk - ssp
        solution['ss'] = [ssk, ssp, ssi, sstotal]
        solution['ms']: List[float] = [solution['ss'][i] / solution['df'][i] for i in range(4)]
        solution['F']: List[float] = [solution['ms'][0] / solution['ms'][2], solution['ms'][1] / solution['ms'][2]]
        solution['p']: List[float] = [1 - stats.f.cdf(solution['F'][i],solution['df'][i], solution['df'][2]) for i in range(2)]
        solution['r2']: List[float] = [solution['ss'][0] / solution['ss'][3], solution['ss'][1] / solution['ss'][3]]
        
        #Textual parts of the report
        solution['null']: str = 'h0:' + ' == '.join(['mu(' + x + ')' for x in data['varnames'][0][1:]])
        solution['null2']: str = 'h0: De personen hebben gelijke ware scores op de opgevoerde meting.'
        rejected: Tuple[str] = ('verworpen','ongelijk') if solution['p'][0] < 0.05 else ('behouden', 'gelijk')
        solution['decision']: str = 'h0 ' + rejected[0] + ', de populatiegemiddelden van kwartalen ' + ' en '.join(data['varnames'][0][1:]) + ' zijn gemiddeld ' + rejected[1] + '.'
        if assignment['control']:
            solution['interpretation']: str = 'Het verschil in '+solution['dependent']+' wordt veroorzaakt door de onafhankelijke variabele '+solution['independent']
        else:
            solution['interpretation']: str = 'De echte verklaring is onbekend, de primaire verklaring is dat, de alternatieve is dat het verschil in '+solution['dependent']+' wordt veroorzaakt door de onafhankelijke variabele '+solution['independent']
        return solution
    
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
        solution['predictor_t'] = [stats.t.isf(solution['predictor_p'][i],1,N - 3 - 1) for i in range(n_predictors+1)]
        solution['predictor_beta'] = [np.mean([random.uniform(60,120)])] + [abs(random.gauss(0,0.5)) for i in range(n_predictors)]
        solution['predictor_b'] = [x * np.sqrt(assignment['var_pred']) for x in solution['predictor_beta']]
        solution['predictor_se'] = [solution['predictor_b'][i]/solution['predictor_t'][i] for i in range(n_predictors+1)]
        
        #Verbal answers
        solution['null'] = 'H0: ' + ' == '.join(['beta(' + str(i) + ')' for i in range(1,4)]) + ' == 0'
        return solution
        
    def print_struct(self, d: Dict):
        for key, value in list(d.items()):
            print(key + ': ' + str(value))
    
