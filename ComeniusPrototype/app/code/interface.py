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
import logging
import json
from scipy import stats
from app.code.enums import Process, Task
from app.code.assignments import Assignments
from app.code.language import LanguageInterface #Regular scan functions
from app.code.scan_functions import * #Scan functions for decision and interpretation
from typing import Dict, List, Callable, Tuple

#class OuterController:
class Controller:
    def __init__(self, jsondict:dict=None):
        #Create a new controller object with standard values
        if jsondict == None:
            self.reset()
        #Create a controller object from a JSON file
        else:
            self.mes = jsondict['mes']
            self.sfs = ScanFunctions(self.mes) #Scan functions
            fdict:dict = {0:self.sfs.scan_indep,1:self.sfs.scan_indep_anova,2:self.sfs.scan_dep,3:self.sfs.scan_control,4:self.sfs.scan_hypothesis,5:self.sfs.scan_hypothesis_anova,6:self.sfs.scan_number,
                        7:self.sfs.scan_p,8:self.sfs.scan_table_ttest,9:self.sfs.scan_table,10:self.sfs.scan_decision,11:self.sfs.scan_decision_anova,12:self.sfs.scan_interpretation,
                        13:self.sfs.scan_interpretation_anova,14:self.sfs.scan_decision_rmanova,15:self.sfs.scan_dummy, 16:self.sfs.scan_hypothesis_rmanova}
            self.assignments = Assignments(self.mes)
            self.skipable: bool = jsondict['skipable']
            self.prevable: bool = jsondict['prevable']
            self.answerable: bool = jsondict['answerable']
            self.answer_triggered: bool = jsondict['answer_triggered']
            self.formmode: bool = jsondict['formmode']
            self.index: int = jsondict['index']
            self.assignment : Dict = self.assignments.deserialize(jsondict['assignment'])
            self.solution : Dict = self.assignments.deserialize(jsondict['solution'])
            self.protocol : List = []
            for question, function, args, phase, answer in jsondict['protocol']:
                args = [x if not type(x) == dict else self.solution if 'p' in x.keys() else self.assignment 
                                        if 'assignment_type' in x.keys() else x for x in args]
                self.protocol.append((question, fdict[function],args,Process(phase),answer))
            [(x[0], fdict[x[1]],x[2],Process(x[3]),x[4]) for x in jsondict['protocol']]
            self.submit_field : int = Task(jsondict['submit_field'])
            self.analysis_type = Task(jsondict['analysis_type'])
            self.wipetext:bool = jsondict['wipetext']
    
    #Convert the current controller object to a dictionary with only string and integer values so it can be saved as JSON
    def serialize(self) -> dict:
        fdict={self.sfs.scan_indep:0,self.sfs.scan_indep_anova:1,self.sfs.scan_dep:2,self.sfs.scan_control:3,self.sfs.scan_hypothesis:4,self.sfs.scan_hypothesis_anova:5,
                self.sfs.scan_number:6,self.sfs.scan_p:7,self.sfs.scan_table_ttest:8,self.sfs.scan_table:9,self.sfs.scan_decision:10,self.sfs.scan_decision_anova:11,self.sfs.scan_interpretation:12,
                self.sfs.scan_interpretation_anova:13,self.sfs.scan_decision_rmanova:14,self.sfs.scan_dummy:15, self.sfs.scan_hypothesis_rmanova:16}
        assignment = self.assignments.serialize(self.assignment)
        solution = self.assignments.serialize(self.solution)
        protocol : List = []
        for question, function, args, phase, answer in self.protocol:
            args = [x if not type(x) == dict else solution if 'p' in x.keys() else assignment 
                                    if 'assignment_type' in x.keys() else x for x in args]
            protocol.append((question, fdict[function],args,phase.value,answer))
        output = {"mes":self.mes,
                  "skipable":self.skipable,
                  "prevable":self.prevable,
                  "answerable":self.answerable,
                  "answer_triggered":self.answer_triggered,
                  "formmode":self.formmode,
                  "index":self.index,
                  "assignment":assignment,
                  "solution":solution,
                  "protocol":protocol,
                  "submit_field":self.submit_field.value,
                  "analysis_type":self.analysis_type.value,
                  "wipetext":self.wipetext}
        return output
    
    #Reset controller attributes to their initial values (called when user reloads URL)
    def reset(self):
        self.assignments: Assignments = Assignments()
        self.sfs = ScanFunctions()
        self.mes = None
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
    
    #Print controller attributes
    def print_internal_state(self):
        print('skipable = ' + str(self.skipable))
        print('prevable = ' + str(self.prevable))
        print('answerable = ' + str(self.answerable))
        print('self.answer_triggered = ' + str(self.answer_triggered))
        print('formmode = ' + str(self.formmode))
        print('INDEX = ' + str(self.index))
        print('ASSIGNMENT = ' + self.assignments.print_assignment(self.assignment) if self.assignment != None else 'None')
        print('PROTOCOL = ' + str(self.protocol))
        print('submit_field = ' + str(self.submit_field))
        print('analysis_type = ' + self.analysis_type.name)
        print('wipetext = ' + str(self.wipetext))
    
    #Update function for the introductory screens and "practice mode"
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
            nl_nlp = spacy.load('nl')
            if function in [self.sfs.scan_decision, self.sfs.scan_decision_anova, self.sfs.scan_decision_rmanova, self.sfs.scan_interpretation, 
                            self.sfs.scan_interpretation_anova]:
                again, output_text = function(nl_nlp(input_text.lower()), *arguments)
            else:
                again, output_text = function(input_text.lower(), *arguments)
            self.wipetext = not again
        
        #Execute the correct response
        #Intro protocol:
        if process == Process.INTRO: 
            li = LanguageInterface()
            textfields['selectlanguage']
            self.mes = li.get_messages(textfields['selectlanguage'] == 'English') #Retrieve the dictionary of messages in the right language
            self.assignments.set_messages(self.mes)
            self.sfs.set_messages(self.mes)
            self.protocol = self.choice_protocol()
            self.submit_field = Task.CHOICE
            self.formmode = False
            self.analysis_type = Task.INTRO
            return self.protocol[0][0]
        #After completing assignment:
        elif process == Process.FINISH:
            self.protocol = self.choice_protocol()
            self.submit_field = Task.CHOICE
            self.formmode = False
            self.analysis_type = Task.FINISHED
        #Choosing an analysis:
        elif process == Process.CHOOSE_ANALYSIS: #If choice protocol index 1 or return protocol index 2
            control: bool = random.choice([True,False])
            hyp_type: int = random.choice([0,1,2])
            instruction: str = ''
            #Select analysis
            if analysis in ['T-toets onafhankelijke variabelen','T-test for independent samples']:
                self.assignment = self.assignments.create_ttest(True, hyp_type, control)
                self.solution = self.assignments.solve_ttest(self.assignment, {})
                instruction = self.assignments.print_ttest(self.assignment)
                self.analysis_type = Task.TTEST_BETWEEN
            if analysis in ['T-toets voor gekoppelde paren','T-test for paired samples']:
                self.assignment = self.assignments.create_ttest(False, hyp_type, True) #Override random control
                self.solution = self.assignments.solve_ttest(self.assignment, {})
                instruction = self.assignments.print_ttest(self.assignment)
                self.analysis_type = Task.TTEST_WITHIN
            if analysis == 'One-way ANOVA':
                self.assignment = self.assignments.create_anova(False, control)
                self.solution = self.assignments.solve_anova(self.assignment, {})
                instruction = self.assignments.print_anova(self.assignment)
                self.analysis_type = Task.ONEWAY_ANOVA
            if analysis == 'Two-way ANOVA':
                control2 = random.choice([True,False])
                self.assignment = self.assignments.create_anova(True, control, control2=control2, elementary=True)
                self.solution = self.assignments.solve_anova(self.assignment, {})
                instruction = self.assignments.print_anova(self.assignment)
                self.analysis_type = Task.TWOWAY_ANOVA
            if analysis == 'Repeated Measures Anova':
                self.assignment = self.assignments.create_rmanova(control)
                self.solution = self.assignments.solve_rmanova(self.assignment, {})
                instruction = self.assignments.print_rmanova(self.assignment)
                self.analysis_type = Task.WITHIN_ANOVA
            if analysis in ['Multipele-regressieanalyse', 'Multiple-regression analysis']:
                self.analysis_type = Task.MREGRESSION
                if report not in ['Beknopt rapport', 'Summary report']:
                    return self.protocol[self.index][0] + '<br><span style="color: blue;">'+self.mes['M_APOLOGIES']+analysis+self.mes['S_SHORTONLY']+'</span>'
            if analysis == 'MANOVA':
                self.analysis_type = Task.MANOVA
                if report not in ['Beknopt rapport', 'Summary report']:
                    return self.protocol[self.index][0] + '<br><span style="color: blue;">'+self.mes['M_APOLOGIES']+analysis+self.mes['S_SHORTONLY']+'</span>'
            if analysis == 'ANCOVA':
                self.analysis_type = Task.ANCOVA
                if report not in ['Beknopt rapport', 'Summary report']:
                    return self.protocol[self.index][0] + '<br><span style="color: blue;">'+self.mes['M_APOLOGIES']+analysis+self.mes['S_SHORTONLY']+'</span>'
            if analysis == 'Multivariate RMANOVA':
                self.analysis_type = Task.MULTIRM
                if report not in ['Beknopt rapport', 'Summary report']:
                    return self.protocol[self.index][0] + '<br><span style="color: blue;">'+self.mes['M_APOLOGIES']+analysis+self.mes['S_SHORTONLY']+'</span>'
            if analysis in ['Dubbel Multivariate-RMANOVA','Dubble multivariate RMANOVA']:
                self.analysis_type = Task.MULTIRM2
                if report not in ['Beknopt rapport', 'Summary report']:
                    return self.protocol[self.index][0] + '<br><span style="color: blue;">'+self.mes['M_APOLOGIES']+analysis+self.mes['S_SHORTONLY']+'</span>'
            
            #Select report type
            self.index = 0
            if report in ['Elementair rapport (oefenmodus)','Elementary report (practice mode)']:
                self.skipable = True
                self.answerable = True
                self.submit_field = Task.TEXT_FIELD
                if self.analysis_type == Task.TTEST_BETWEEN: #'T-toets onafhankelijke variabelen':
                    self.protocol = self.ttest_protocol()
                if self.analysis_type == Task.TTEST_WITHIN:
                    self.protocol = self.ttest_protocol()
                if self.analysis_type == Task.ONEWAY_ANOVA:
                    self.protocol = self.anova_protocol()
                if self.analysis_type == Task.TWOWAY_ANOVA:
                    self.protocol = self.anova_protocol()
                if self.analysis_type == Task.WITHIN_ANOVA:
                    self.protocol = self.rmanova_protocol()
            if report in ['Elementair rapport (tentamenmodus)','Elementary report (examination mode)']:
                self.submit_field = Task.FINISHED #Task.INTRO
                self.skipable = False
                self.formmode = True
                self.protocol = self.completion_protocol()
            if report in ['Beknopt rapport', 'Summary report']:
                self.submit_field = Task.FINISHED
                self.skipable = False
                self.formmode= True
                self.assignment = self.assignments.create_report(control, self.analysis_type.value)
                self.analysis_type = Task.REPORT
                self.solution = self.assignment
                self.protocol = self.completion_protocol()
                instruction = self.assignments.print_report(self.assignment)
                return instruction
            return instruction + '<br>' + self.protocol[self.index][0]
        #Main report protocols (practice mode) during table question
        elif process == Process.TABLE: 
            if input_text == 'prev' and self.prevable:
                self.index -= 1
                self.submit_field = Task.TEXT_FIELD
                if self.protocol[self.index][1] in [self.sfs.scan_decision, self.sfs.scan_decision_anova, self.sfs.scan_decision_rmanova, 
                                    self.sfs.scan_interpretation, self.sfs.scan_interpretation_anova, self.sfs.scan_hypothesis_anova]: 
                    self.submit_field = Task.TEXT_FIELD_LARGE #Set text field to large if appropriate
                else:
                    self.submit_field = Task.TEXT_FIELD
                self.answer_triggered = False
                return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0]
            elif input_text == 'skip' and self.skipable:
                self.index += 1
                self.submit_field = Task.TEXT_FIELD
                self.answer_triggered = False
                if self.protocol[self.index][1] in [self.sfs.scan_decision, self.sfs.scan_decision_anova, self.sfs.scan_decision_rmanova, 
                                    self.sfs.scan_interpretation, self.sfs.scan_interpretation_anova, self.sfs.scan_hypothesis_anova]: 
                    self.submit_field = Task.TEXT_FIELD_LARGE #Set text field to large if appropriate
                return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0]
            elif again:
                return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0] + '<br><br>' + output_text
            else:
                self.index += 1 #TODO: COMMENT IF FEEDBACK MEANS NO PROGRESS
                self.submit_field = Task.TEXT_FIELD
                self.answer_triggered = False
                if self.protocol[self.index][1] in [self.sfs.scan_decision, self.sfs.scan_decision_anova, self.sfs.scan_decision_rmanova, 
                                    self.sfs.scan_interpretation, self.sfs.scan_interpretation_anova, self.sfs.scan_hypothesis_anova]: 
                    self.submit_field = Task.TEXT_FIELD_LARGE #Set text field to large if appropriate
                return self.assignments.print_assignment(self.assignment) + '<br>' + output_text + self.protocol[self.index][0]
        #Main report protocols (practice mode) during last question
        elif process == Process.LAST_QUESTION: 
            if input_text == 'prev' and self.prevable:
                self.index -= 1
                self.answer_triggered = False
                if self.protocol[self.index][1] in [self.sfs.scan_decision, self.sfs.scan_decision_anova, self.sfs.scan_decision_rmanova, 
                                    self.sfs.scan_interpretation, self.sfs.scan_interpretation_anova, self.sfs.scan_hypothesis_anova]: 
                    self.submit_field = Task.TEXT_FIELD_LARGE #Set text field to large if appropriate
                return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0]
            elif again:
                return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0] + '<br><br>' + output_text
            else:
                self.skipable = False
                self.prevable = False
                self.answerable = False
                self.answer_triggered = False
                self.protocol = self.completion_protocol()
                self.index = 0
                self.submit_field = Task.FINISHED #Task.INTRO
                if input_text == 'skip':
                    return self.protocol[self.index][0]
                else:
                    return output_text + '<br>' + self.protocol[self.index][0]
        #Main report protocols (practice mode) before last question
        elif process == Process.QUESTION: 
            if input_text == 'prev' and self.prevable and self.index != 0:
                if self.index == 1:
                    self.prevable = False
                self.index -= 1
                self.answer_triggered = False
                if self.protocol[self.index][1] in [self.sfs.scan_decision, self.sfs.scan_decision_anova, self.sfs.scan_decision_rmanova, 
                                    self.sfs.scan_interpretation, self.sfs.scan_interpretation_anova, self.sfs.scan_hypothesis_anova]: 
                    self.submit_field = Task.TEXT_FIELD_LARGE #Set text field to large if appropriate
                else:
                    self.submit_field = Task.TEXT_FIELD
                if self.protocol[self.index][3] == Process.TABLE:
                    self.submit_field = self.analysis_type
                return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0]
            elif again:
                return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0] + '<br><br>' + output_text
            else:
                if self.index == 0:
                    self.prevable = True
                self.index += 1 #TODO: COMMENT IF FEEDBACK MEANS NO PROGRESS
                self.answer_triggered = False
                if self.protocol[self.index][1] in [self.sfs.scan_decision, self.sfs.scan_decision_anova, self.sfs.scan_decision_rmanova, 
                                    self.sfs.scan_interpretation, self.sfs.scan_interpretation_anova, self.sfs.scan_hypothesis_anova]: 
                    self.submit_field = Task.TEXT_FIELD_LARGE #Set text field to large if appropriate
                if self.protocol[self.index][3] == Process.TABLE:
                    self.submit_field = self.analysis_type #Signal to the routes class that the next text field has to be a table
                if input_text == 'skip' and self.skipable:
                    return self.assignments.print_assignment(self.assignment) + '<br>' + self.protocol[self.index][0]
                else:    
                    return self.assignments.print_assignment(self.assignment) + '<br>' + output_text + self.protocol[self.index][0]
        else:
            print('ERROR SWITCHING PROTOCOLS')
    
    #Apply scan functions to the input fields of the T-test form and return a list of feedback
    def update_form_ttest(self, textfields: Dict) -> [str, List[str]]:
        output:list = [[] for i in range(12)]
        instruction:str = self.assignments.print_ttest(self.assignment)
        
        output[0].append(self.sfs.scan_indep(textfields['inputtext1'].lower(), self.solution)[1])
        output[1].append(self.sfs.scan_dep(textfields['inputtext2'].lower(), self.solution)[1])
        output[2].append(self.sfs.scan_control(textfields['inputtext3'].lower(), self.solution)[1])
        output[4].append(self.sfs.scan_hypothesis(textfields['inputtext4'].lower(), self.solution, num=1)[1])
        
        output[5].append(self.sfs.scan_number(textfields['inputtext5'], 'df', self.solution)[1])
        output[6].append(self.sfs.scan_number(textfields['inputtext6'], 'raw_effect', self.solution)[1])
        output[7].append(self.sfs.scan_number(textfields['inputtext7'], 'relative_effect', self.solution)[1])
        output[8].append(self.sfs.scan_number(textfields['inputtext8'], 'T', self.solution)[1])
        output[9].append(self.sfs.scan_number(textfields['inputtext9'], 'p', self.solution)[1])
        nl_nlp = spacy.load('nl')
        output[10].append(self.sfs.scan_decision(nl_nlp(textfields['inputtext10'].lower()), self.solution, anova=False)[1])
        output[11].append(self.sfs.scan_interpretation(nl_nlp(textfields['inputtext11'].lower()), self.solution, anova=False)[1])
        
        output[3].append(self.sfs.scan_table_ttest(textfields, self.solution)[1])
        return instruction, output
    
    #Return the standard answers for the T-test assignments in a list
    def form_answers(self) -> [str, List[str]]:
        output:list = [[] for i in range(12)]
        instruction:str = self.assignments.print_ttest(self.assignment)
        output[0].append('Antwoord: '+self.assignments.print_independent(self.assignment))
        output[1].append('Antwoord: '+self.assignments.print_dependent(self.assignment))
        output[2].append('Antwoord: '+self.assignments.print_control(self.assignment))
        output[3].append('Antwoord: '+self.assignments.print_report({**self.assignment, **self.solution}, answer=True))
        output[4].append('Antwoord: '+self.solution['null'])
        output[5].append('Antwoord: '+str(self.solution['df'][0]))
        output[6].append('Antwoord: '+str(round(self.solution['raw_effect'][0],2)))
        output[7].append('Antwoord: '+str(round(self.solution['relative_effect'][0],2)))
        output[8].append('Antwoord: '+str(round(self.solution['T'][0],2)))
        output[9].append('Antwoord: '+str(round(self.solution['p'][0],2)))
        output[10].append('Antwoord: '+self.solution['decision'])
        output[11].append('Antwoord: '+self.solution['interpretation'])
        return instruction, output
    
    #Apply scan functions to the input fields of the ANOVA form and return a list of feedback points
    def update_form_anova(self, textfields: Dict) -> [str, list]:
        output:list = [[] for i in range(7)] #Empty list for every type of input field (independent variable, dependent variable, etc.)
        nl_nlp = spacy.load('nl')
        if self.analysis_type == Task.ONEWAY_ANOVA:
            instruction = self.assignments.print_anova(self.assignment)
        elif self.analysis_type == Task.TWOWAY_ANOVA:
            instruction = self.assignments.print_anova(self.assignment)
        elif self.analysis_type == Task.WITHIN_ANOVA:
            instruction = self.assignments.print_rmanova(self.assignment)
        else:    
            print('ERROR: INVALID TABLE SHAPE')
            
        if self.assignment['assignment_type'] == 3:
            #One-way ANOVA
            output[0].append(self.sfs.scan_indep_anova(textfields['inputtext1'].lower(), self.solution, num=1, between_subject=True)[1])
            output[1].append(self.sfs.scan_dep(textfields['inputtext2'].lower(), self.solution)[1])
            output[2].append(self.sfs.scan_control(textfields['inputtext3'].lower(), self.solution)[1])
            output[3].append(self.sfs.scan_hypothesis(textfields['inputtext4'].lower(), self.solution, num=1)[1])
            output[5].append(self.sfs.scan_decision(nl_nlp(textfields['inputtext5'].lower()), self.solution, anova=True)[1])
            output[6].append(self.sfs.scan_interpretation(nl_nlp(textfields['inputtext6'].lower()), self.solution, anova=True, num=1)[1])
            output[4].append(self.sfs.scan_table(textfields, self.solution)[1])
        elif self.assignment['assignment_type'] == 4:
            #Two-way ANOVA
            output[0].append(self.sfs.scan_indep_anova(textfields['inputtext1'].lower(), self.solution, num=1, between_subject=True)[1])
            output[0].append(self.sfs.scan_indep_anova(textfields['inputtext12'].lower(), self.solution, num=2, between_subject=True)[1])
            output[1].append(self.sfs.scan_dep(textfields['inputtext2'].lower(), self.solution)[1])
            output[2].append(self.sfs.scan_control(textfields['inputtext3'].lower(), self.solution)[1])
            output[2].append(self.sfs.scan_control(textfields['inputtext32'].lower(), self.solution, num=2)[1])
            output[3].append(self.sfs.scan_hypothesis(textfields['inputtext4'].lower(), self.solution, num=1)[1])
            output[3].append(self.sfs.scan_hypothesis(textfields['inputtext42'].lower(), self.solution, num=2)[1])
            output[3].append(self.sfs.scan_hypothesis_anova(textfields['inputtext43'].lower(), self.solution)[1])
            output[5].append(self.sfs.scan_decision(nl_nlp(textfields['inputtext5'].lower()), self.solution, anova=True, num=1)[1])
            output[5].append(self.sfs.scan_decision(nl_nlp(textfields['inputtext52'].lower()), self.solution, anova=True, num=2)[1])
            output[5].append(self.sfs.scan_decision_anova(nl_nlp(textfields['inputtext53'].lower()), self.solution)[1])
            output[6].append(self.sfs.scan_interpretation(nl_nlp(textfields['inputtext6'].lower()), self.solution, anova=True, num=1)[1])
            output[6].append(self.sfs.scan_interpretation(nl_nlp(textfields['inputtext62'].lower()), self.solution, anova=True, num=2)[1])
            output[6].append(self.sfs.scan_interpretation_anova(nl_nlp(textfields['inputtext63'].lower()), self.solution)[1])
            output[4].append(self.sfs.scan_table(textfields, self.solution)[1])
        elif self.assignment['assignment_type'] == 5:
            #Within-subject ANOVA
            output[0].append(self.sfs.scan_indep_anova(textfields['inputtext1'].lower(), self.solution, num=1, between_subject=True)[1])
            output[1].append(self.sfs.scan_dep(textfields['inputtext2'].lower(), self.solution)[1])
            output[2].append(self.sfs.scan_control(textfields['inputtext3'].lower(), self.solution)[1])
            output[3].append(self.sfs.scan_hypothesis(textfields['inputtext4'].lower(), self.solution, num=1)[1])
            output[3].append(self.sfs.scan_hypothesis_rmanova(textfields['inputtext42'].lower(), self.solution)[1])
            output[5].append(self.sfs.scan_decision(nl_nlp(textfields['inputtext5'].lower()), self.solution, anova=True)[1])
            output[5].append(self.sfs.scan_decision_rmanova(nl_nlp(textfields['inputtext52'].lower()), self.solution, num=2)[1])
            output[6].append(self.sfs.scan_interpretation(nl_nlp(textfields['inputtext6'].lower()), self.solution, anova=True, num=1)[1])
            output[4].append(self.sfs.scan_table(textfields, self.solution)[1])
        return instruction, output
    
    #Return the standard answers for the ANOVA assignments in a list
    def form_answers_anova(self) -> [str, list]:
        output = [[] for i in range(7)]
        
        #Determine instruction
        if self.analysis_type == Task.ONEWAY_ANOVA:
            instruction = self.assignments.print_anova(self.assignment)
        elif self.analysis_type == Task.TWOWAY_ANOVA:
            instruction = self.assignments.print_anova(self.assignment)
        elif self.analysis_type == Task.WITHIN_ANOVA:
            instruction = self.assignments.print_rmanova(self.assignment)
        else:    
            print('ERROR: INVALID TABLE SHAPE')
            
        #Get answers for assignment type
        if self.assignment['assignment_type'] == 3:
            #One-way ANOVA
            output[0].append(self.mes['A_ANSWER']+self.assignments.print_independent(self.assignment))
            output[1].append(self.mes['A_ANSWER']+self.assignments.print_dependent(self.assignment))
            output[2].append(self.mes['A_ANSWER']+self.assignment[''])
            output[3].append(self.mes['A_ANSWER']+self.solution['null'])
            output[4].append(self.mes['A_ANSWER']+self.assignments.print_report({**self.assignment, **self.solution}, answer=True))
            output[5].append(self.mes['A_ANSWER']+self.solution['decision'])
            output[6].append(self.mes['A_ANSWER']+self.solution['interpretation'])
        elif self.assignment['assignment_type'] == 4:
            output[0].append(self.mes['A_ANSWER']+self.assignments.print_independent(self.assignment))
            output[0].append(self.mes['A_ANSWER']+self.assignments.print_independent(self.assignment, num=2))
            output[1].append(self.mes['A_ANSWER']+self.assignments.print_dependent(self.assignment))
            output[2].append(self.mes['A_ANSWER']+self.assignments.print_control(self.assignment))
            output[2].append(self.mes['A_ANSWER']+self.assignments.print_control(self.assignment, num=2))
            output[3].append(self.mes['A_ANSWER']+self.solution['null'])
            output[3].append(self.mes['A_ANSWER']+self.solution['null2'])
            output[3].append(self.mes['A_ANSWER']+self.solution['null3'])
            output[4].append(self.mes['A_ANSWER']+self.assignments.print_report({**self.assignment, **self.solution}, answer=True))
            output[5].append(self.mes['A_ANSWER']+self.solution['decision'])
            output[5].append(self.mes['A_ANSWER']+self.solution['decision2'])
            output[5].append(self.mes['A_ANSWER']+self.solution['decision3'])
            output[6].append(self.mes['A_ANSWER']+self.solution['interpretation'])
            output[6].append(self.mes['A_ANSWER']+self.solution['interpretation2'])
            output[6].append(self.mes['A_ANSWER']+self.solution['interpretation3'])
        elif self.assignment['assignment_type'] == 5:
            output[0].append(self.mes['A_ANSWER']+self.assignments.print_independent(self.assignment))
            output[1].append(self.mes['A_ANSWER']+self.assignments.print_dependent(self.assignment))
            output[2].append(self.mes['A_ANSWER']+self.assignments.print_control(self.assignment))
            output[3].append(self.mes['A_ANSWER']+self.solution['null'])
            output[3].append(self.mes['A_ANSWER']+self.solution['null2'])
            output[4].append(self.mes['A_ANSWER']+self.assignments.print_report({**self.assignment, **self.solution}, answer=True))
            output[5].append(self.mes['A_ANSWER']+self.solution['decision'])
            output[5].append(self.mes['A_ANSWER']+self.solution['decision2'])
            output[6].append(self.mes['A_ANSWER']+self.solution['interpretation'])
        return instruction, output
    
    #Update function for the report screen, returns a list of feedback for each field
    def update_form_report(self, textfields: Dict) -> List[str]:
        instruction = self.assignments.print_report(self.assignment)
        text = textfields['inputtext']
        feedback = None
        if self.assignment['assignment_type'] == 1:
            feedback = self.sfs.split_grade_ttest(text, self.solution, between_subject=True)
        if self.assignment['assignment_type'] == 2:
            feedback = self.sfs.split_grade_ttest(text, self.solution, between_subject=False)
        if self.assignment['assignment_type'] == 3:
            feedback = self.sfs.split_grade_anova(text, self.solution, two_way=False)
        if self.assignment['assignment_type'] == 4:
            feedback = self.sfs.split_grade_anova(text, self.solution, two_way=True)
        if self.assignment['assignment_type'] == 5:
            feedback = self.sfs.split_grade_rmanova(text, self.solution)
        if self.assignment['assignment_type'] == 6:
            feedback = self.sfs.split_grade_mregression(text, self.solution)
        if self.assignment['assignment_type'] == 11:
            feedback = self.sfs.split_grade_manova(text, self.solution)
        if self.assignment['assignment_type'] == 12:
            feedback = self.sfs.split_grade_ancova(text, self.solution)
        if self.assignment['assignment_type'] == 13:
            feedback = self.sfs.split_grade_multirm(text, self.solution)
        if self.assignment['assignment_type'] == 14: #Will be scrapped
            feedback = self.sfs.split_grade_multirm2(text, self.solution)
        return instruction, feedback
    
    """
    Protocols for practice mode: Every state within the protocol is written below as a tuple with five values. These represent respectivel:
    1. The question to be presented to the user, as a string. 
    2. The function which will be used to process the answer to that question
    3. The arguments for that function, apart from the input text itself
    4. An enum value representing the type of state and how the program should respond (INTRO, FINISH, QUESTION, TABLE or LAST_QUESTION)
    5. The standard answer for the question
    """
    def intro_protocol(self) -> List[Tuple]:
        return [('Hoi, met dit programma kan je elementaire en beknopte rapporten oefenen. Klik op de knop hieronder om verder te gaan.', 
                 self.sfs.scan_dummy, [], Process.INTRO, '')]

    def completion_protocol(self) -> List[Tuple]:
        return [(self.mes['Q_COMPLETION'], 
                 self.sfs.scan_dummy, [], Process.FINISH, '')]
        
    def choice_protocol(self) -> List[Tuple]:
        return [(self.mes['Q_CHOICE'],self.sfs.scan_dummy, [], Process.CHOOSE_ANALYSIS, '')]
   
    def ttest_protocol(self) -> List[Tuple]:
        output : List[Tuple] = [(self.mes['Q_IND'], self.sfs.scan_indep, [self.solution], Process.QUESTION,self.assignments.print_independent(self.assignment)),
           (self.mes['Q_DEP'], self.sfs.scan_dep, [self.solution], Process.QUESTION,self.assignments.print_dependent(self.assignment)),
           (self.mes['Q_MEASURE'], self.sfs.scan_control, [self.solution], Process.QUESTION,self.assignments.print_control(self.assignment)),
           (self.mes['Q_TABLE'],self.sfs.scan_table_ttest,[self.solution],Process.TABLE,self.assignments.print_report({**self.assignment, **self.solution}, answer=True)),
           (self.mes['Q_HYP'],self.sfs.scan_hypothesis,[self.solution, 1], Process.QUESTION,self.solution['null']),
           (self.mes['Q_DF'],self.sfs.scan_number,['df', self.solution, 0.01], Process.QUESTION,str(self.solution['df'][0])),
           (self.mes['Q_RAW'],self.sfs.scan_number,['raw_effect', self.solution, 0.01], Process.QUESTION,str(self.solution['raw_effect'][0])),
           (self.mes['Q_RELATIVE'],self.sfs.scan_number,['relative_effect', self.solution, 0.01], Process.QUESTION,str(self.solution['relative_effect'][0])),
           (self.mes['Q_T'],self.sfs.scan_number,['T', self.solution, 0.02], Process.QUESTION,str(self.solution['T'][0])),
           (self.mes['Q_P'],self.sfs.scan_p,[self.solution, 0.02], Process.QUESTION,str(self.solution['p'][0])),
           (self.mes['Q_DECISION'],self.sfs.scan_decision,[self.solution, False], Process.QUESTION,self.solution['decision']),
           (self.mes['Q_INTERPRET'],self.sfs.scan_interpretation,[self.solution, False], Process.LAST_QUESTION,self.solution['interpretation'])]
        return output
    
    def anova_protocol(self) -> List[Tuple]:
        if not self.assignment['two_way']:
            output :str = [(self.mes['Q_IND'],self.sfs.scan_indep_anova,[self.solution], Process.QUESTION, self.assignments.print_independent(self.assignment)),
                (self.mes['Q_DEP'],self.sfs.scan_dep,[self.solution], Process.QUESTION, self.assignments.print_dependent(self.assignment)),
                (self.mes['Q_MEASURE'],self.sfs.scan_control,[self.solution], Process.QUESTION,self.assignments.print_control(self.assignment)),
                (self.mes['Q_HYP'],self.sfs.scan_hypothesis,[self.solution, 1], Process.QUESTION, self.solution['null']),
                (self.mes['Q_TABLE'],self.sfs.scan_table,[self.solution, 0.02], Process.TABLE, self.assignments.print_report({**self.assignment, **self.solution}, answer=True)),     
                (self.mes['Q_DECISION'], self.sfs.scan_decision,[self.solution, True, 1], Process.QUESTION, self.solution['decision']),
                (self.mes['Q_INTERPRET'],self.sfs.scan_interpretation,[self.solution, True], Process.LAST_QUESTION, self.solution['interpretation'])]
        else:
            output :str = [(self.mes['Q_IND1'],self.sfs.scan_indep_anova,[self.solution], Process.QUESTION, self.assignments.print_independent(self.assignment)),
                (self.mes['Q_IND2'],self.sfs.scan_indep_anova,[self.solution,2], Process.QUESTION, self.assignments.print_independent(self.assignment, num=2)),
                (self.mes['Q_DEP'],self.sfs.scan_dep,[self.solution], Process.QUESTION, self.assignments.print_dependent(self.assignment)),
                (self.mes['Q_MEASURE1'],self.sfs.scan_control,[self.solution], Process.QUESTION,self.assignments.print_control(self.assignment)),
                (self.mes['Q_MEASURE2'],self.sfs.scan_control,[self.solution, 2], Process.QUESTION,self.assignments.print_control(self.assignment, num=2)),
                (self.mes['Q_HYP1'],self.sfs.scan_hypothesis,[self.solution, 1], Process.QUESTION,self.solution['null']),
                (self.mes['Q_HYP2'],self.sfs.scan_hypothesis,[self.solution, 2], Process.QUESTION,self.solution['null2']),
                (self.mes['Q_HYPINT'],self.sfs.scan_hypothesis_anova,[self.solution], Process.QUESTION,self.solution['null3']),
                (self.mes['Q_TABLE'],self.sfs.scan_table,[self.solution, 0.02], Process.TABLE, self.assignments.print_report({**self.assignment, **self.solution}, answer=True)),    
                (self.mes['Q_DECISION1'], self.sfs.scan_decision,[self.solution, True, 1], Process.QUESTION,self.solution['decision']),
                (self.mes['Q_DECISION2'], self.sfs.scan_decision,[self.solution, True, 2], Process.QUESTION,self.solution['decision2']),
                (self.mes['Q_DECISIONINT'], self.sfs.scan_decision_anova,[self.solution], Process.QUESTION, self.solution['decision3']),
                (self.mes['Q_INTERPRET1'],self.sfs.scan_interpretation,[self.solution, True, 1], Process.QUESTION,self.solution['interpretation']),
                (self.mes['Q_INTERPRET2'],self.sfs.scan_interpretation,[self.solution, True, 2], Process.QUESTION,self.solution['interpretation2']),
                (self.mes['Q_INTERPRETINT'],self.sfs.scan_interpretation_anova,[self.solution], Process.LAST_QUESTION,self.solution['interpretation3'])]
        return output
    
    def rmanova_protocol(self) -> List[Tuple]:
        output :str = [(self.mes['Q_IND'],self.sfs.scan_indep_anova,[self.solution,1,False], Process.QUESTION,self.assignments.print_independent(self.assignment)),
            (self.mes['Q_DEPRM'],self.sfs.scan_dep,[self.solution], Process.QUESTION,self.assignments.print_dependent(self.assignment)),
            (self.mes['Q_MEASURE'],self.sfs.scan_control,[self.solution], Process.QUESTION,self.assignments.print_control(self.assignment)),
            (self.mes['Q_HYPCON'],self.sfs.scan_hypothesis,[self.solution,1], Process.QUESTION,self.solution['null']),
            (self.mes['Q_HYPSUB'],self.sfs.scan_hypothesis_rmanova,[self.solution], Process.QUESTION,self.solution['null2']),
            (self.mes['Q_TABLE'],self.sfs.scan_table,[self.solution, 0.02], Process.TABLE, self.assignments.print_report({**self.assignment, **self.solution}, answer=True)),
            (self.mes['Q_DECCON'],self.sfs.scan_decision,[self.solution,True,1], Process.QUESTION,self.solution['decision']),
            (self.mes['Q_DECSUB'],self.sfs.scan_decision_rmanova,[self.solution,2], Process.QUESTION,self.solution['decision2']),
            (self.mes['Q_INTERCON'],self.sfs.scan_interpretation,[self.solution, True, 1], Process.LAST_QUESTION,self.solution['interpretation'])]
        return output
    
    def print_assignment(self):
        return self.assignments.print_assignment(self.assignment)
            
