#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 15:32:54 2020

@author: jelmer
"""
import math
import random
import numpy as np
from scipy import stats
from app.code.assignments import Assignments
from app.code.scan_functions import *
from typing import Dict, List, Callable, Tuple
"""
VRAGEN VOLGEND SKYPEGESPREK
[1]: Hoe moeten studenten de interactiehypothese invoeren bij two-way ANOVA?

"""
class OuterController:    
    class Controller:
        def __init__(self):
            self.assignments: Assignments = Assignments()
            self.skipable: bool = False
            self.prevable: bool = False
            self.formmode: bool = False
            self.index: int = 0
            self.assignment : Dict = None
            self.solution : Dict = None
            self.protocol : List = self.intro_protocol()
            self.endstate : bool = False
            self.submit_field : int = 6 #0 for one submit line, 1-5 for tables for each of the different reports and 6 for the intro
            self.table_shape = 0
        
        def reset(self):
            self.assignments: Assignments = Assignments()
            self.skipable: bool = False
            self.prevable: bool = False
            self.formmode: bool = False
            self.index: int = 0
            self.assignment : Dict = None
            self.solution : Dict = None
            self.protocol : List = self.intro_protocol()
            self.endstate : bool = False
            self.submit_field : int = 6
            self.table_shape = 0
        
        def update_form_ttest(self, textfields: Dict) -> List[str]:
            output = [[] for i in range(12)]
            instruction = self.assignments.print_ttest(self.assignment)
            output[0].append(scan_indep(textfields['inputtext1'], self.solution)[1])
            output[1].append(scan_dep(textfields['inputtext2'], self.solution)[1])
            output[2].append(scan_control(textfields['inputtext3'], self.solution)[1])
            output[4].append(scan_hypothesis(textfields['inputtext4'], self.solution, num=1)[1])
            
            output[5].append(scan_number(textfields['inputtext5'], 'df', self.solution)[1])
            output[6].append(scan_number(textfields['inputtext6'], 'raw_effect', self.solution)[1])
            output[7].append(scan_number(textfields['inputtext7'], 'relative_effect', self.solution)[1])
            output[8].append(scan_number(textfields['inputtext8'], 'T', self.solution)[1])
            output[9].append(scan_number(textfields['inputtext8'], 'p', self.solution)[1])
            
            output[10].append(scan_decision(textfields['inputtext9'], self.assignment, self.solution)[1])
            output[11].append(scan_interpretation(textfields['inputtext10'], self.solution)[1])
            
            output[3].append(scan_table_ttest(textfields, self.solution)[1])
            return instruction, output
        
        def update_form_anova(self, textfields: Dict) -> List[str]:
            output = [[] for i in range(7)]
            if self.table_shape == 3:
                instruction = self.assignments.print_anova(self.assignment)
            elif self.table_shape == 4:
                instruction = self.assignments.print_anova(self.assignment)
            elif self.table_shape == 5:
                instruction = self.assignments.print_rmanova(self.assignment)
            else:    
                print('ERROR: INVALID TABLE SHAPE')
                
            if not self.assignment['two_way'] and not 'jackedmeans' in list(self.assignment['data'].keys()):
                #One-way ANOVA
                output[0].append(scan_indep_anova(textfields['inputtext1'], self.solution, num=1, between_subject=True)[1])
                output[1].append(scan_dep(textfields['inputtext2'], self.solution)[1])
                output[2].append(scan_control(textfields['inputtext3'], self.solution)[1])
                output[3].append(scan_hypothesis(textfields['inputtext4'], self.solution, num=1)[1])
                output[5].append(scan_decision(textfields['inputtext5'], self.assignment, self.solution, anova=True)[1])
                output[6].append(scan_interpretation(textfields['inputtext6'], self.solution, anova=True, num=1)[1])
                output[4].append(scan_table(textfields, self.solution)[1])
            elif self.assignment['two_way']:
                #Two-way ANOVA
                output[0].append(scan_indep_anova(textfields['inputtext1'], self.solution, num=1, between_subject=True)[1])
                output[0].append(scan_indep_anova(textfields['inputtext12'], self.solution, num=2, between_subject=True)[1])
                output[1].append(scan_dep(textfields['inputtext2'], self.solution)[1])
                output[2].append(scan_control(textfields['inputtext3'], self.solution)[1])
                output[3].append(scan_hypothesis(textfields['inputtext4'], self.solution, num=1)[1])
                output[3].append(scan_hypothesis(textfields['inputtext42'], self.solution, num=2)[1])
                output[3].append(scan_hypothesis_anova(textfields['inputtext43'], self.solution)[1])
                output[5].append(scan_decision(textfields['inputtext5'], self.assignment, self.solution, anova=True, num=1)[1])
                output[5].append(scan_decision(textfields['inputtext52'], self.assignment, self.solution, anova=True, num=2)[1])
                output[5].append(scan_decision_anova(textfields['inputtext53'], self.assignment, self.solution)[1])
                output[6].append(scan_interpretation(textfields['inputtext6'], self.solution, anova=True, num=1)[1])
                output[6].append(scan_interpretation(textfields['inputtext62'], self.solution, anova=True, num=2)[1])
                output[6].append(scan_interpretation_anova(textfields['inputtext63'], self.solution)[1])
                output[4].append(scan_table(textfields, self.solution)[1])
            elif 'jackedmeans' in list(self.assignment['data'].keys()):
                #Within-subject ANOVA
                output[0].append(scan_indep_anova(textfields['inputtext1'], self.solution, num=1, between_subject=True)[1])
                output[1].append(scan_dep(textfields['inputtext2'], self.solution)[1])
                output[2].append(scan_control(textfields['inputtext3'], self.solution)[1])
                output[3].append(scan_hypothesis(textfields['inputtext4'], self.solution, num=1)[1])
                output[3].append(scan_hypothesis_rmanova(textfields['inputtext42'], self.solution)[1])
                output[5].append(scan_decision(textfields['inputtext5'], self.assignment, self.solution, anova=True)[1])
                output[5].append(scan_decision_rmanova(textfields['inputtext52'], self.assignment, self.solution)[1])
                output[6].append(scan_interpretation(textfields['inputtext6'], self.solution, anova=True, num=1)[1])
                output[4].append(scan_table(textfields, self.solution)[1])
            return instruction, output
        
        def update(self, textfields: Dict) -> str:
            #Retrieve values from form text fields
            input_text :str = textfields['inputtext'] if 'inputtext' in list(textfields.keys()) else ''
            
            #Scan the input text fields and and determine the correct response
            x, function, arguments = self.protocol[self.index]
            output_text = ''
            if self.protocol[0][0][:3] == 'Hoi':
                pass
            elif self.protocol[self.index][0][:3] == 'Vul' and input_text != 'skip' and input_text != 'prev':
                again, output_text = function(textfields, *arguments)
            elif (input_text == 'skip' and self.skipable) or (input_text == 'prev' and self.prevable): #Hard-set again to false if the user wants to skip
                again = False
            elif self.protocol[self.index][0][:3] == 'Gef' or self.protocol[self.index][0][:3] == 'Wil':
                again, output = function(input_text, *arguments)
            elif self.protocol[0][0][:3] != 'Vul':
                again, output_text = function(input_text, *arguments)
            
            #Execute the correct response
            if self.endstate: #If end state has been reached
                return 'Tot ziens!'
            if self.protocol[0][0][:3] == 'Hoi': #If intro protocol:
                self.protocol = self.choice_protocol()
                self.submit_field = 0
                return self.protocol[0][0]
            elif self.protocol[self.index][0][:3] == 'Gef':#If return protocol index 0
                if output:
                    self.index += 1
                    return self.protocol[self.index][0]
                else:
                    self.endstate = True
                    return 'Tot ziens!'
            elif self.protocol[self.index][0][:3] == 'Wil': #If choice protocol index 0 or return protocol index 1
                self.formmode = output
                self.index += 1
                return self.protocol[self.index][0]
            elif self.protocol[self.index][0][:3] == 'Wat': #If choice protocol index 1 or return protocol index 2
                if not again:
                    control: bool = random.choice([True,False])
                    hyp_type: int = random.choice([0,1,2])
                    self.skipable = True
                    self.index = 0
                    if input_text == '1':
                        self.assignment = self.assignments.create_ttest(True, hyp_type, control)
                        self.solution = self.assignments.solve_ttest(self.assignment, {})
                        instruction = self.assignments.print_ttest(self.assignment)
                        self.protocol = self.ttest_protocol()
                        self.table_shape = 1
                    if input_text == '2':
                        self.assignment = self.assignments.create_ttest(False, hyp_type, control)
                        self.solution = self.assignments.solve_ttest(self.assignment, {})
                        instruction = self.assignments.print_ttest(self.assignment)
                        self.protocol = self.ttest_protocol()
                        self.table_shape = 2
                    if input_text == '3':
                        self.assignment = self.assignments.create_anova(False, control)
                        self.solution = self.assignments.solve_anova(self.assignment, {})
                        instruction = self.assignments.print_anova(self.assignment)
                        self.protocol = self.anova_protocol()
                        self.table_shape = 3
                    if input_text == '4':
                        self.assignment = self.assignments.create_anova(True, control)
                        self.solution = self.assignments.solve_anova(self.assignment, {})
                        instruction = self.assignments.print_anova(self.assignment)
                        self.protocol = self.anova_protocol()
                        self.table_shape = 4
                    if input_text == '5':
                        self.assignment = self.assignments.create_rmanova(control)
                        self.solution = self.assignments.solve_rmanova(self.assignment, {})
                        instruction = self.assignments.print_rmanova(self.assignment)
                        self.protocol = self.rmanova_protocol()
                        self.table_shape = 5
                    if self.formmode:
                        self.protocol = self.return_protocol()
                    return instruction + '<br>' + self.protocol[self.index][0]
                else:
                    return output_text
            elif self.protocol[self.index][0][:3] == 'Vul': #Main report protocols during table question
                if input_text == 'prev' and self.prevable:
                    self.index -= 1
                    self.submit_field = 0
                    return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0]
                elif input_text == 'skip' and self.skipable:
                    self.index += 1
                    self.submit_field = 0
                    return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0]
                elif again:
                    return self.assignments.print_assignment(self.assignment) + '<br>' + output_text
                else:
                    self.submit_field = 0
                    return output_text + self.protocol[0][0]
            elif len(self.protocol) > 2 and self.index == len(self.protocol) - 1: #Main report protocols during last question
                if input_text == 'prev' and self.prevable:
                    self.index -= 1
                    return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0]
                elif again:
                    return self.assignments.print_assignment(self.assignment) + '<br>' + output_text
                else:
                    self.skipable = False
                    self.prevable = False
                    self.protocol = self.return_protocol()
                    self.index = 0
                    if input_text == 'skip':
                        return self.protocol[self.index][0]
                    else:
                        return output_text + '<br>' + self.protocol[self.index][0]
            elif len(self.protocol) > 2: #Main report protocols before last question
                if input_text == 'prev' and self.prevable and self.index != 0:
                    if self.index == 1:
                        self.prevable = False
                    self.index -= 1
                    if self.protocol[self.index][0][:3] == 'Vul':
                        self.submit_field = self.table_shape
                    return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0]
                elif again:
                    return self.assignments.print_assignment(self.assignment) + '<br>' + output_text
                else:
                    if self.index == 0:
                        self.prevable = True
                    self.index += 1
                    if self.protocol[self.index][0][:3] == 'Vul':
                        self.submit_field = self.table_shape #Signal to the routes class that the next text field has to be a table
                    if input_text == 'skip' and self.skipable:
                        return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0]
                    else:    
                        return self.assignments.print_assignment(self.assignment) + '<br>' + output_text + self.protocol[self.index][0]
            else:
                print('ERROR SWITCHING PROTOCOLS')
                
        def intro_protocol(self) -> List[Tuple]:
            return [('Hoi, je kan in dit programma op twee manieren elementaire rapporten oefenen, namelijke in de standaardmodus en de tentamenmodus. Klik op de submit-knop om verder te gaan', scan_dummy, [])]
            
        def choice_protocol(self) -> List[Tuple]:
            return [('Wil je het rapport in tentamenmodus maken?',scan_yesno,[]),
                    ('Wat voor soort elementair rapport wil je oefenen?<br>(1) T-toets onafhankelijke variabelen<br>(2) T-toets voor gekoppelde paren<br>(3) One-way ANOVA<br>(4) Two-way ANOVA<br>(5) Repeated Measures Anova', scan_protocol_choice, [])]
            
        def return_protocol(self) -> List[Tuple]:
            return [('Gefeliciteerd, je elementair rapport is af! Wil je nog een opgave doen?',scan_yesno,[]),
                    ('Wil je het rapport in tentamenmodus maken?',scan_yesno,[]),
                    ('Wat voor soort elementair rapport wil je oefenen?<br>(1) T-toets onafhankelijke variabelen<br>(2) T-toets voor gekoppelde paren<br>(3) One-way ANOVA<br>(4) Two-way ANOVA<br>(5) Repeated Measures Anova', scan_protocol_choice, [])]
        
        def ttest_protocol(self) -> List[Tuple]:
            output : List[Tuple] = [('Beschrijf de onafhankelijke variabele.', scan_indep, [self.solution]),
               ('Beschrijf de afhankelijke variabele.', scan_dep, [self.solution]),
               ('Beschrijf de mate van controle.', scan_control, [self.solution]),
               ('Voer de gemiddelden van de data in, gescheiden door een spatie.', scan_numbers, ['means', self.solution, 0.01]),
               ('Voer de standaarddeviaties van de data in, gescheiden door een spatie.', scan_numbers, ['stds', self.solution, 0.01]),
               ('Voer de nulhypothese in, geformuleerd met "H0" en "mu".',scan_hypothesis,[self.solution, 1]),
               ('Voer de het aantal vrijheidsgraden in.',scan_number,['df', self.solution, 0.01]),
               ('Voer de het ruwe effect in dat je hebt berekend.',scan_number,['raw_effect', self.solution, 0.01]),
               ('Voer de het relatieve effect in dat je hebt berekend.',scan_number,['relative_effect', self.solution, 0.01]),
               ('Voer de T-waarde in.',scan_number,['T', self.solution, 0.02]),
               ('Voer de p-waarde in.',scan_number,['p', self.solution, 0.02]),
               ('Voer de beslissing in',scan_decision,[self.assignment, self.solution])]
            if self.assignment['between_subject']:
                output.insert(5, ('Voer de waarden van N voor beide niveaus van de onafhankelijke variabele in, gescheiden door een spatie.', scan_numbers, ['ns', self.solution, 0.01]))
            else:
                output.insert(5, ('Voer de waarde van N in.',scan_number,['ns', self.solution, 0.01]))
            if self.solution['p'][0] < 0.05:
                output.append(('Voer de causale interpretatie in.',scan_interpretation,[self.solution]))
            return output
        
        def anova_protocol(self) -> List[Tuple]:
            if not self.assignment['two_way']:
                output :str = [('Beschrijf de onafhankelijke variabele.',scan_indep_anova,[self.solution] ),
                    ('Beschrijf de onafhankelijke variabele.',scan_indep_anova,[self.solution] ),
                    ('Beschrijf de afhankelijke variabele.',scan_dep,[self.solution] ),
                    ('Beschrijf de mate van controle.',scan_control,[self.solution] ),
                    ('Voer de nulhypothese in, geformuleerd met "H0" en "mu".',scan_hypothesis,[self.solution, 1] ),
                    ('Vul de tabel hieronder in.',scan_table,[self.solution, 0.02]),     
                    ('Voer de beslissing in', scan_decision,[self.assignment,self.solution, True, 1])]
                if self.solution['p'][0] < 0.05:
                    output.append(('Voer de causale interpretatie in.',scan_interpretation,[self.solution]))
            else:
                output :str = [('Beschrijf de eerste onafhankelijke variabele.',scan_indep_anova,[self.solution]),
                    ('Beschrijf de tweede onafhankelijke variabele.',scan_indep_anova,[self.solution,2]),
                    ('Beschrijf de afhankelijke variabele.',scan_dep,[self.solution]),
                    ('Beschrijf de mate van controle.',scan_control,[self.solution]),
                    ('Voer de nulhypothese in voor de eerste onafhankelijke variabele, geformuleerd met "H0" en "mu".',scan_hypothesis,[self.solution, 1]),
                    ('Voer de nulhypothese in voor de tweede onafhankelijke variabele, geformuleerd met "H0" en "mu".',scan_hypothesis,[self.solution, 2]),
                    ('Voer de interactienulhypothese in, geformuleerd met "H0" en "mu".',scan_hypothesis_anova,[self.solution]),
                    ('Vul de tabel hieronder in.',scan_table,[self.solution, 0.02]),    
                    ('Voer de beslissing in voor de eerste onafhankelijke variabele', scan_decision,[self.assignment,self.solution, True, 1]),
                    ('Voer de beslissing in voor de tweede onafhankelijke variabele', scan_decision,[self.assignment,self.solution, True, 2]),
                    ('Voer de beslissing in voor de interactie', scan_decision_anova,[self.assignment, self.solution])]
                if self.solution['p'][0] < 0.05:
                    output.append(('Voer de causale interpretatie voor de eerste factor in.',scan_interpretation,[self.solution, True, 1]))
                if self.solution['p'][1] < 0.05:
                    output.append(('Voer de causale interpretatie voor de tweede factor in.',scan_interpretation,[self.solution, True, 2]))
                if self.solution['p'][2] < 0.05:
                    output.append(('Voer de causale interpretatie voor de interactie in.',scan_interpretation_anova,[self.solution]))
            return output
                    
        
        def rmanova_protocol(self) -> List[Tuple]:
            output :str = [('Beschrijf de onafhankelijke variabele.',scan_indep_anova,[self.solution,1,False]),
                ('Beschrijf de afhankelijke variabele (het aantal metingen per persoon hoef je niet te geven).',scan_dep,[self.solution]),
                ('Beschrijf de mate van controle.',scan_control,[self.solution]),
                ('Beschrijf de nulhypothese van de condities',scan_hypothesis,[self.solution,1]),
                ('Beschrijf de nulhypothese van de subjecten',scan_hypothesis_rmanova,[self.solution]),
                ('Vul de tabel hieronder in.',scan_table,[self.solution, 0.02]),
                ('Voer de beslissing in van de condities',scan_decision,[self.assignment,self.solution,True,1]),
                ('Voer de beslissing in van de subjecten',scan_decision_rmanova,[self.assignment,self.solution])]
            if self.solution['p'][0] < 0.05 :
                output.append(('Voer de causale interpretatie voor de condities in.',scan_interpretation,[self.assignment, self.solution, True, 1]))
            return output
        
        def print_assignment(self):
            return self.assignments.print_assignment(self.assignment)
            
    instance = None
    def __new__(cls): # __new__ always a classmethod
        if not OuterController.instance:
            OuterController.instance = OuterController.Controller()
        return OuterController.instance
    def __getattr__(self, name):
        return getattr(self.instance, name)
    def __setattr__(self, name):
        return setattr(self.instance, name)
