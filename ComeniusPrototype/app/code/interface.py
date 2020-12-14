#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 15:32:54 2020

@author: jelmer
"""
import math
import random
import numpy as np
import os
import spacy
from scipy import stats
from app.code.enums import Process, Task
from app.code.assignments import Assignments
from app.code.scan_functions import * #Regular scan functions
from app.code.scan_functions_spacy import * #Scan functions for decision and interpretation
from typing import Dict, List, Callable, Tuple

class OuterController:
    class Controller:
        def __init__(self):
            self.assignments: Assignments = Assignments()
            self.skipable: bool = False
            self.prevable: bool = False
            self.answerable: bool = False
            self.answer_triggered: bool = False
            self.formmode: bool = False
            self.index: int = 0
            self.assignment : Dict = None
            self.solution : Dict = None
            self.protocol : List = self.intro_protocol()
            self.submit_field : int = Task.INTRO
            self.analysis_type = Task.CHOICE
            self.wipetext:bool = False
            self.nl_nlp = spacy.load('nl')
        
        def reset(self):
            self.assignments: Assignments = Assignments()
            self.skipable: bool = False
            self.prevable: bool = False
            self.answerable: bool = False
            self.answer_triggered: bool = False
            self.formmode: bool = False
            self.index: int = 0
            self.assignment : Dict = None
            self.solution : Dict = None
            self.protocol : List = self.intro_protocol()
            self.submit_field : int = Task.INTRO
            self.analysis_type = Task.CHOICE
            self.wipetext:bool = False
        
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
            output[9].append(scan_number(textfields['inputtext9'], 'p', self.solution)[1])
            
            output[10].append(scan_decision(self.nl_nlp(textfields['inputtext10'].lower()), self.solution, anova=False)[1])
            output[11].append(scan_interpretation(self.nl_nlp(textfields['inputtext11'].lower()), self.solution, anova=False)[1])
            
            output[3].append(scan_table_ttest(textfields, self.solution)[1])
            return instruction, output
        
        def update_form_anova(self, textfields: Dict) -> List[str]:
            output = [[] for i in range(7)]
            if self.analysis_type == Task.ONEWAY_ANOVA:
                instruction = self.assignments.print_anova(self.assignment)
            elif self.analysis_type == Task.TWOWAY_ANOVA:
                instruction = self.assignments.print_anova(self.assignment)
            elif self.analysis_type == Task.WITHIN_ANOVA:
                instruction = self.assignments.print_rmanova(self.assignment)
            else:    
                print('ERROR: INVALID TABLE SHAPE')
                
            if not self.assignment['two_way'] and not 'jackedmeans' in list(self.assignment['data'].keys()):
                #One-way ANOVA
                output[0].append(scan_indep_anova(textfields['inputtext1'], self.solution, num=1, between_subject=True)[1])
                output[1].append(scan_dep(textfields['inputtext2'], self.solution)[1])
                output[2].append(scan_control(textfields['inputtext3'], self.solution)[1])
                output[3].append(scan_hypothesis(textfields['inputtext4'], self.solution, num=1)[1])
                output[5].append(scan_decision(self.nl_nlp(textfields['inputtext5'].lower()), self.solution, anova=True)[1])
                output[6].append(scan_interpretation(self.nl_nlp(textfields['inputtext6'].lower()), self.solution, anova=True, num=1)[1])
                output[4].append(scan_table(textfields, self.solution)[1])
            elif self.assignment['two_way']:
                #Two-way ANOVA
                output[0].append(scan_indep_anova(textfields['inputtext1'], self.solution, num=1, between_subject=True)[1])
                output[0].append(scan_indep_anova(textfields['inputtext12'], self.solution, num=2, between_subject=True)[1])
                output[1].append(scan_dep(textfields['inputtext2'], self.solution)[1])
                output[2].append(scan_control(textfields['inputtext3'], self.solution)[1])
                output[2].append(scan_control(textfields['inputtext32'], self.solution, num=2)[1])
                output[3].append(scan_hypothesis(textfields['inputtext4'], self.solution, num=1)[1])
                output[3].append(scan_hypothesis(textfields['inputtext42'], self.solution, num=2)[1])
                output[3].append(scan_hypothesis_anova(textfields['inputtext43'], self.solution)[1])
                output[5].append(scan_decision(self.nl_nlp(textfields['inputtext5'].lower()), self.solution, anova=True, num=1)[1])
                output[5].append(scan_decision(self.nl_nlp(textfields['inputtext52'].lower()), self.solution, anova=True, num=2)[1])
                output[5].append(scan_decision_anova(self.nl_nlp(textfields['inputtext53'].lower()), self.solution)[1])
                output[6].append(scan_interpretation(self.nl_nlp(textfields['inputtext6'].lower()), self.solution, anova=True, num=1)[1])
                output[6].append(scan_interpretation(self.nl_nlp(textfields['inputtext62'].lower()), self.solution, anova=True, num=2)[1])
                output[6].append(scan_interpretation_anova(self.nl_nlp(textfields['inputtext63'].lower()), self.solution)[1])
                output[4].append(scan_table(textfields, self.solution)[1])
            elif 'jackedmeans' in list(self.assignment['data'].keys()):
                #Within-subject ANOVA
                output[0].append(scan_indep_anova(textfields['inputtext1'], self.solution, num=1, between_subject=True)[1])
                output[1].append(scan_dep(textfields['inputtext2'], self.solution)[1])
                output[2].append(scan_control(textfields['inputtext3'], self.solution)[1])
                output[3].append(scan_hypothesis(textfields['inputtext4'], self.solution, num=1)[1])
                output[3].append(scan_hypothesis_rmanova(textfields['inputtext42'], self.solution)[1])
                output[5].append(scan_decision(self.nl_nlp(textfields['inputtext5'].lower()), self.solution, anova=True)[1])
                output[5].append(scan_decision_rmanova(self.nl_nlp(textfields['inputtext52'].lower()), self.solution, num=2)[1])
                output[6].append(scan_interpretation(self.nl_nlp(textfields['inputtext6'].lower()), self.solution, anova=True, num=1)[1])
                output[4].append(scan_table(textfields, self.solution)[1])
            return instruction, output
        
        def update_form_report(self, textfields: Dict) -> List[str]:
            instruction = self.assignments.print_report(self.assignment)
            text = textfields['inputtext']
            feedback = None
            if self.assignment['assignment_type'] == 1:
                feedback = split_grade_ttest(text, self.solution, between_subject=True)
            if self.assignment['assignment_type'] == 2:
                feedback = split_grade_ttest(text, self.solution, between_subject=False)
            if self.assignment['assignment_type'] == 3:
                feedback = split_grade_anova(text, self.solution, two_way=False)
            if self.assignment['assignment_type'] == 4:
                feedback = split_grade_anova(text, self.solution, two_way=True)
            if self.assignment['assignment_type'] == 5:
                feedback = split_grade_rmanova(text, self.solution)
            if self.assignment['assignment_type'] == 6:
                feedback = split_grade_mregression(text, self.solution)
            return instruction, feedback
        
        def update(self, textfields: Dict) -> str:
            #Retrieve values from form text fields
            if 'inputtext' in list(textfields.keys()):
                input_text:str = textfields['inputtext']
                if 'inputtextlarge' in list(textfields.keys()):
                    input_text += textfields['inputtextlarge']
            else:
                input_text:str = ''
            
            #Scan the input text fields and and determine the correct response
            _, function, arguments, process, answer = self.protocol[self.index]
            output_text = ''
            if process == Process.TABLE and input_text != 'skip' and input_text != 'prev':                
                again, output_text = function(textfields, *arguments)
            elif process == Process.CHOOSE_ANALYSIS:
                analysis = textfields['selectanalysis']
                report = textfields['selectreport']
            elif (input_text == 'skip' and self.skipable) or (input_text == 'prev' and self.prevable): #Hard-set again to false if the user wants to skip
                again = False
            elif process != Process.TABLE:
                if function in [scan_decision, scan_decision_anova, scan_decision_rmanova, scan_interpretation, scan_interpretation_anova]:
                    again, output_text = function(self.nl_nlp(input_text.lower()), *arguments)
                else:
                    again, output_text = function(input_text.lower(), *arguments)
                self.wipetext = not again
            
            #Execute the correct response
            if process == Process.INTRO: #If intro protocol:
                self.protocol = self.choice_protocol()
                self.submit_field = Task.CHOICE
                self.formmode = False
                self.analysis_type = Task.INTRO
                return self.protocol[0][0]
            elif process == Process.CHOOSE_ANALYSIS: #If choice protocol index 1 or return protocol index 2
                control: bool = random.choice([True,False])
                hyp_type: int = random.choice([0,1,2])
                instruction: str = ''
                #Select analysis
                if analysis == 'T-toets onafhankelijke variabelen':
                    self.assignment = self.assignments.create_ttest(True, hyp_type, control)
                    self.solution = self.assignments.solve_ttest(self.assignment, {})
                    instruction = self.assignments.print_ttest(self.assignment)
                    self.analysis_type = Task.TTEST_BETWEEN
                if analysis == 'T-toets voor gekoppelde paren':
                    self.assignment = self.assignments.create_ttest(False, hyp_type, control)
                    self.solution = self.assignments.solve_ttest(self.assignment, {})
                    instruction = self.assignments.print_ttest(self.assignment)
                    self.analysis_type = Task.TTEST_WITHIN
                if analysis == 'One-way ANOVA':
                    self.assignment = self.assignments.create_anova(False, control)
                    self.solution = self.assignments.solve_anova(self.assignment, {})
                    instruction = self.assignments.print_anova(self.assignment)
                    self.analysis_type = Task.ONEWAY_ANOVA
                if analysis == 'Two-way ANOVA':
                    self.assignment = self.assignments.create_anova(True, control)
                    self.solution = self.assignments.solve_anova(self.assignment, {})
                    instruction = self.assignments.print_anova(self.assignment)
                    self.analysis_type = Task.TWOWAY_ANOVA
                if analysis == 'Repeated Measures Anova':
                    self.assignment = self.assignments.create_rmanova(control)
                    self.solution = self.assignments.solve_rmanova(self.assignment, {})
                    instruction = self.assignments.print_rmanova(self.assignment)
                    self.analysis_type = Task.WITHIN_ANOVA
                if analysis == 'Multiple-regressieanalyse':
                    self.analysis_type = Task.MREGRESSION
                    if report != 'Beknopt rapport':
                        return self.protocol[self.index][0] + '<span style="color: blue;">Sorry, bij multiple regressie kan je alleen een beknopt rapport maken.</span>'
                
                #Select report type
                self.index = 0
                if report == 'Elementair rapport (oefenmodus)':
                    self.skipable = True
                    self.answerable = True
                    self.submit_field = Task.TEXT_FIELD
                    if analysis == 'T-toets onafhankelijke variabelen':
                        self.protocol = self.ttest_protocol()
                    if analysis == 'T-toets voor gekoppelde paren':
                        self.protocol = self.ttest_protocol()
                    if analysis == 'One-way ANOVA':
                        self.protocol = self.anova_protocol()
                    if analysis == 'Two-way ANOVA':
                        self.protocol = self.anova_protocol()
                    if analysis == 'Repeated Measures Anova':
                        self.protocol = self.rmanova_protocol()
                if report == 'Elementair rapport (tentamenmodus)':
                    self.submit_field = Task.INTRO
                    self.skipable = False
                    self.formmode = True
                    self.protocol = self.completion_protocol()
                if report == 'Beknopt rapport':
                    self.submit_field = Task.INTRO
                    self.skipable = False
                    self.formmode= True
                    self.assignment = self.assignments.create_report(control, self.analysis_type.value)
                    self.analysis_type = Task.REPORT
                    self.solution = self.assignment
                    self.protocol = self.completion_protocol()
                    instruction = self.assignments.print_report(self.assignment)
                    return instruction
                return instruction + '<br>' + self.protocol[self.index][0]
            elif process == Process.TABLE: #Main report protocols during table question
                if input_text == 'prev' and self.prevable:
                    self.index -= 1
                    self.submit_field = Task.TEXT_FIELD
                    self.answer_triggered = False
                    return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0]
                elif input_text == 'skip' and self.skipable:
                    self.index += 1
                    self.submit_field = Task.TEXT_FIELD
                    self.answer_triggered = False
                    return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0]
                elif again:
                    return self.assignments.print_assignment(self.assignment) + '<br>' + output_text
                else:
                    self.index += 1
                    self.submit_field = Task.TEXT_FIELD
                    self.answer_triggered = False
                    return self.assignments.print_assignment(self.assignment) + '<br>' + output_text + self.protocol[self.index][0]
            elif process == Process.LAST_QUESTION: #Main report protocols during last question
                if input_text == 'prev' and self.prevable:
                    self.index -= 1
                    self.answer_triggered = False
                    return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0]
                elif again:
                    return self.assignments.print_assignment(self.assignment) + '<br>' + output_text
                else:
                    self.skipable = False
                    self.prevable = False
                    self.answerable = False
                    self.answer_triggered = False
                    self.protocol = self.completion_protocol()
                    self.index = 0
                    self.submit_field = Task.INTRO
                    if input_text == 'skip':
                        return self.protocol[self.index][0]
                    else:
                        return output_text + '<br>' + self.protocol[self.index][0]
            elif process == Process.QUESTION: #Main report protocols before last question
                if input_text == 'prev' and self.prevable and self.index != 0:
                    if self.index == 1:
                        self.prevable = False
                    self.index -= 1
                    self.answer_triggered = False
                    if self.protocol[self.index][3] == Process.TABLE:
                        self.submit_field = self.analysis_type
                    return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0]
                elif again:
                    return self.assignments.print_assignment(self.assignment) + '<br>' + output_text
                else:
                    if self.index == 0:
                        self.prevable = True
                    self.index += 1
                    self.answer_triggered = False
                    if self.protocol[self.index][3] == Process.TABLE:
                        self.submit_field = self.analysis_type #Signal to the routes class that the next text field has to be a table
                    if input_text == 'skip' and self.skipable:
                        return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0]
                    else:    
                        return self.assignments.print_assignment(self.assignment) + '<br>' + output_text + self.protocol[self.index][0]
            else:
                print('ERROR SWITCHING PROTOCOLS')
                
        def intro_protocol(self) -> List[Tuple]:
            return [('Hoi, met dit programma kan je elementaire en beknopte rapporten oefenen. Klik op de enter-knop om verder te gaan.', 
                     scan_dummy, [], Process.INTRO, None)]
    
        def completion_protocol(self) -> List[Tuple]:
            return [('Gefeliciteerd, je rapport is af! Klik op de knop hieronder om verder te gaan.', 
                     scan_dummy, [], Process.INTRO, None)]
            
        def choice_protocol(self) -> List[Tuple]:
            return [('Voer hieronder het soort opgave in dat je wil oefenen.<br>',None,[], Process.CHOOSE_ANALYSIS, None)]
   
        def ttest_protocol(self) -> List[Tuple]:
            output : List[Tuple] = [('Beschrijf de onafhankelijke variabele.', scan_indep, [self.solution], Process.QUESTION,self.assignments.print_independent(self.assignment)),
               ('Beschrijf de afhankelijke variabele.', scan_dep, [self.solution], Process.QUESTION,self.assignments.print_dependent(self.assignment)),
               ('Beschrijf de mate van controle.', scan_control, [self.solution], Process.QUESTION,['Passief-observerend','Experiment'][int(self.solution['control'])]),
               ('Voer de ondestaande tabel in.',scan_table_ttest,[self.solution],Process.TABLE,self.assignments.print_report({**self.assignment, **self.solution}, answer=True)),
               ('Voer de nulhypothese in, geformuleerd met "H0" en "mu".',scan_hypothesis,[self.solution, 1], Process.QUESTION,self.solution['null']),
               ('Voer de het aantal vrijheidsgraden in.',scan_number,['df', self.solution, 0.01], Process.QUESTION,str(self.solution['df'][0])),
               ('Voer de het ruwe effect in dat je hebt berekend.',scan_number,['raw_effect', self.solution, 0.01], Process.QUESTION,str(self.solution['raw_effect'][0])),
               ('Voer de het relatieve effect in dat je hebt berekend.',scan_number,['relative_effect', self.solution, 0.01], Process.QUESTION,str(self.solution['relative_effect'][0])),
               ('Voer de T-waarde in.',scan_number,['T', self.solution, 0.02], Process.QUESTION,str(self.solution['T'][0])),
               ('Voer de p-waarde in.',scan_p,[self.solution, 0.02], Process.QUESTION,str(self.solution['p'][0])),
               ('Voer de beslissing in',scan_decision,[self.solution, False], Process.QUESTION,self.solution['decision']),
               ('Voer de causale interpretatie in.',scan_interpretation,[self.solution, False], Process.LAST_QUESTION,self.solution['interpretation'])]
            return output
        
        def anova_protocol(self) -> List[Tuple]:
            if not self.assignment['two_way']:
                output :str = [('Beschrijf de onafhankelijke variabele.',scan_indep_anova,[self.solution], Process.QUESTION, self.assignments.print_independent(self.assignment)),
                    ('Beschrijf de afhankelijke variabele.',scan_dep,[self.solution], Process.QUESTION, self.assignments.print_dependent(self.assignment)),
                    ('Beschrijf de mate van controle.',scan_control,[self.solution], Process.QUESTION,['Passief-observerend','Experiment'][int(self.solution['control'])]),
                    ('Voer de nulhypothese in, geformuleerd met "H0" en "mu".',scan_hypothesis,[self.solution, 1], Process.QUESTION, self.solution['null']),
                    ('Vul de tabel hieronder in.',scan_table,[self.solution, 0.02], Process.TABLE, self.assignments.print_report({**self.assignment, **self.solution}, answer=True)),     
                    ('Voer de beslissing in', scan_decision,[self.solution, True, 1], Process.QUESTION, self.solution['decision']),
                    ('Voer de causale interpretatie in.',scan_interpretation,[self.solution, True], Process.LAST_QUESTION, self.solution['interpretation'])]
            else:
                output :str = [('Beschrijf de eerste onafhankelijke variabele.',scan_indep_anova,[self.solution], Process.QUESTION, self.assignments.print_independent(self.assignment)),
                    ('Beschrijf de tweede onafhankelijke variabele.',scan_indep_anova,[self.solution,2], Process.QUESTION, self.assignments.print_independent(self.assignment, num=2)),
                    ('Beschrijf de afhankelijke variabele.',scan_dep,[self.solution], Process.QUESTION, self.assignments.print_dependent(self.assignment)),
                    ('Beschrijf de mate van controle voor factor 1.',scan_control,[self.solution], Process.QUESTION,['Passief-observerend','Experiment'][int(self.solution['control'])]),
                    ('Beschrijf de mate van controle voor factor 2.',scan_control,[self.solution, 2], Process.QUESTION,['Passief-observerend','Experiment'][int(self.solution['control2'])]),
                    ('Voer de nulhypothese in voor de eerste onafhankelijke variabele, geformuleerd met "H0" en "mu".',scan_hypothesis,[self.solution, 1], Process.QUESTION,self.solution['null']),
                    ('Voer de nulhypothese in voor de tweede onafhankelijke variabele, geformuleerd met "H0" en "mu".',scan_hypothesis,[self.solution, 2], Process.QUESTION,self.solution['null2']),
                    ('Voer de interactienulhypothese in. Je mag deze inkorten door alleen de eerste en laatste vergelijking van de hypothese te geven.',scan_hypothesis_anova,[self.solution], Process.QUESTION,self.solution['null3']),
                    ('Vul de tabel hieronder in.',scan_table,[self.solution, 0.02], Process.TABLE, self.assignments.print_report({**self.assignment, **self.solution}, answer=True)),    
                    ('Voer de beslissing in voor de eerste onafhankelijke variabele', scan_decision,[self.solution, True, 1], Process.QUESTION,self.solution['decision']),
                    ('Voer de beslissing in voor de tweede onafhankelijke variabele', scan_decision,[self.solution, True, 2], Process.QUESTION,self.solution['decision2']),
                    ('Voer de beslissing in voor de interactie', scan_decision_anova,[self.solution], Process.QUESTION, self.solution['decision3']),
                    ('Voer de causale interpretatie voor de eerste factor in.',scan_interpretation,[self.solution, True, 1], Process.QUESTION,self.solution['interpretation']),
                    ('Voer de causale interpretatie voor de tweede factor in.',scan_interpretation,[self.solution, True, 2], Process.QUESTION,self.solution['interpretation2']),
                    ('Voer de causale interpretatie voor de interactie in.',scan_interpretation_anova,[self.solution], Process.LAST_QUESTION,self.solution['interpretation3'])]
            return output
        
        def rmanova_protocol(self) -> List[Tuple]:
            output :str = [('Beschrijf de onafhankelijke variabele.',scan_indep_anova,[self.solution,1,False], Process.QUESTION,self.assignments.print_independent(self.assignment)),
                ('Beschrijf de afhankelijke variabele (het aantal metingen per persoon hoef je niet te geven).',scan_dep,[self.solution], Process.QUESTION,self.assignments.print_dependent(self.assignment)),
                ('Beschrijf de mate van controle.',scan_control,[self.solution], Process.QUESTION,['Passief-observerend','Experiment'][int(self.solution['control'])]),
                ('Beschrijf de nulhypothese van de condities',scan_hypothesis,[self.solution,1], Process.QUESTION,self.solution['null']),
                ('Beschrijf de nulhypothese van de subjecten',scan_hypothesis_rmanova,[self.solution], Process.QUESTION,self.solution['null2']),
                ('Vul de tabel hieronder in.',scan_table,[self.solution, 0.02], Process.TABLE, self.assignments.print_report({**self.assignment, **self.solution}, answer=True)),
                ('Voer de beslissing in van de condities',scan_decision,[self.solution,True,1], Process.QUESTION,self.solution['decision']),
                ('Voer de beslissing in van de subjecten',scan_decision_rmanova,[self.solution], Process.QUESTION,self.solution['decision2']),
                ('Voer de causale interpretatie voor de condities in.',scan_interpretation,[self.solution, True, 1], Process.LAST_QUESTION,self.solution['interpretation'])]
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
