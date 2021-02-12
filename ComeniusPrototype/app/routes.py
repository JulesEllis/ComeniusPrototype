from flask import render_template, flash, redirect, url_for, request
from app import app
from app.forms import BaseForm, BigForm, SmallForm, ReportForm
from app.code.interface import Controller #OuterController
from app.code.enums import Task, Process
from app.code.scan_functions_spacy import *
from app.code.scan_functions import scan_hypothesis_anova
import flask
import pickle
import os

@app.route('/')
@app.route('/index', methods=['GET','POST'])
def index():
    #Get controller
    path = 'app/controller.pickle' if 'Github' in os.getcwd() else '/var/www/ComeniusPrototype/ComeniusPrototype/app/controller.pickle'
    with open(path, 'rb') as f:
        mc:dict = pickle.load(f)
        ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        if not ip in list(mc.keys()):
            mc[ip] = Controller()
        controller = mc[ip]
        
    #Assign local variables
    varnames = [['dummy1'],['dummy2']]
    title:str = 'Oefeningsmodule voor statistische rapporten'
    form = BaseForm()
    if flask.request.method == 'GET':
        controller.reset()
        #Save controller
        with open(path, 'wb') as f:
            pickle.dump(mc, f, protocol=pickle.HIGHEST_PROTOCOL)
        instruction = controller.protocol[0][0]
        return render_template('index.html', display=instruction, 
                               form=form, skip=False, submit_field=8, varnames=varnames, title=title)
    elif flask.request.method == 'POST':        
        #Isolate text fields
        textfields:list = [x for x in dir(form) if str(type(form.__getattribute__(x))) in ["<class 'wtforms.fields.core.StringField'>","<class 'wtforms.fields.core.SelectField'>","<class 'wtforms.fields.simple.TextAreaField'>"]]
        textdict:dict = dict([(x, form.__getattribute__(x).data) for x in textfields])        
        
        #Prepare webpage parameters
        if form.submit.data:
            output_text : str = controller.update(textdict)
            if not controller.mes == None:
                title:str = controller.mes['M_TITLE']
            else:
                title:str = 'Oefeningsmodule voor statistische rapporten'
            if controller.assignment != None: #Retrieve variable names
                a = controller.assignment
                varnames:list = [[a['independent']] + a['levels']] if a['assignment_type'] != 4 else [[a['independent']] + a['levels'],[a['independent2']] + a['levels2']]
            form_shape = controller.analysis_type.value
            if controller.formmode and form_shape > 0 and form_shape < 3:
                form = SmallForm()
                controller.formmode = False
                instruction = controller.print_afssignment()
                #Store controller
                with open(path, 'wb') as f:
                    pickle.dump(mc, f, protocol=pickle.HIGHEST_PROTOCOL)    
                return render_template('smallform.html', form=form, instruction=instruction, displays=[[''] for i in range(12)], shape=form_shape, varnames=varnames, title=title)
            elif controller.formmode and form_shape > 2 and form_shape < 6:
                form = BigForm()
                controller.formmode = False
                instruction = controller.print_assignment()
                #Store controller
                with open(path, 'wb') as f:
                    pickle.dump(mc, f, protocol=pickle.HIGHEST_PROTOCOL)    
                return render_template('bigform.html', form=form, instruction=instruction, displays=[[''] for i in range(7)], shape=form_shape, varnames=varnames, title=title)
            elif form_shape == 7:
                form = ReportForm()
                controller.formmode = False
                instruction = output_text
                #Store controller
                with open(path, 'wb') as f:
                    pickle.dump(mc, f, protocol=pickle.HIGHEST_PROTOCOL)    
                return render_template('reportform.html', form=form, instruction=instruction, display='', title=title)
        if form.skip.data:
            output_text : str = controller.update({'inputtext': 'skip'})
        if form.prev.data:
            output_text : str = controller.update({'inputtext': 'prev'})
        if form.answer.data:
            controller.answer_triggered = not controller.answer_triggered
            output_text : str = controller.assignments.print_assignment(controller.assignment) + '<br>' + controller.protocol[controller.index][0]
        if controller.assignment != None: #Retrieve variable names
            a = controller.assignment
            varnames:list = [[a['independent']] + a['levels']] if a['assignment_type'] != 4 else [[a['independent']] + a['levels'],[a['independent2']] + a['levels2']]
        if controller.wipetext:
            form.inputtext.data = ""
            form.inputtextlarge.data = ""
        if controller.submit_field == Task.INTRO or controller.submit_field == Task.CHOICE or controller.submit_field == Task.FINISHED: #Determine enter button text
            form.submit.label.text = 'Doorgaan'
        else:
            form.submit.label.text = 'Feedback'
        if controller.submit_field == Task.CHOICE: #Determine dropdown language options
            mes:dict = controller.mes
            form.__getattribute__('selectanalysis').choices = [mes['M_ANALYSIS' + str(i+1)] for i in range(9)]
            form.__getattribute__('selectanalysis').label = mes['M_CHOOSEANALYSIS']
            form.__getattribute__('selectreport').choices = [mes['M_REPORT' + str(i+1)] for i in range(3)]
            form.__getattribute__('selectreport').label = mes['M_CHOOSEREPORT']
            
        skip :bool = controller.skipable
        prev :bool = controller.prevable
        answer :bool = controller.answerable
        answer_text :str = controller.protocol[controller.index][4] if controller.answer_triggered else ""
        if answer_text != '': #Capitalize the first letter of each answer
            answer_text = answer_text[0].upper() + answer_text[1:]
        submit_field :int = controller.submit_field.value
        if controller.protocol[controller.index][1] in [scan_decision, scan_decision_anova, scan_decision_rmanova, scan_interpretation, scan_interpretation_anova, scan_hypothesis_anova]: 
            #Convert textbox to large textbox if appropriate
            submit_field = 10
        
        #Store controller
        with open(path, 'wb') as f:
            pickle.dump(mc, f, protocol=pickle.HIGHEST_PROTOCOL)    
        
        #Render page
        return render_template('index.html', display=output_text, answer_text=answer_text, form=form, skip=skip, prev=prev, answer=answer, submit_field=submit_field, varnames=varnames, title=title)
    else:
        print('ERROR: INVALID METHOD')

@app.route('/bigform', methods=['POST'])
def bigform():
    #Get controller
    path = 'app/controller.pickle' if 'Github' in os.getcwd() else '/var/www/ComeniusPrototype/ComeniusPrototype/app/controller.pickle'
    with open(path, 'rb') as f:
        mc:dict = pickle.load(f)
        ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        controller = mc[ip]
    a = controller.assignment
    form = BigForm()
    title:str = controller.mes['M_TITLE']
    varnames:list = [[a['independent']] + a['levels']] if a['assignment_type'] != 4 else [[a['independent']] + a['levels'],[a['independent2']] + a['levels2']]
        
    if flask.request.method == 'POST':
        if form.submit.data:
            form_shape = controller.analysis_type.value
            textfields = [x for x in dir(form) if str(type(form.__getattribute__(x))) in ["<class 'wtforms.fields.core.StringField'>","<class 'wtforms.fields.simple.TextAreaField'>"]]
            textdict = dict([(x, form.__getattribute__(x).data) for x in textfields])
            instruction, outputfields = controller.update_form_anova(textdict)
            with open(path, 'wb') as f:
                pickle.dump(mc, f, protocol=pickle.HIGHEST_PROTOCOL) 
            return render_template('bigform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape, varnames=varnames, title=title)
        elif form.nextt.data:
            skip:bool = controller.skipable
            prev:bool = controller.prevable
            display = controller.protocol[0][0]
            form = BaseForm()
            field = controller.submit_field.value
            return render_template('index.html', display=display, form=form, skip=skip, prev=prev, submit_field=field, varnames=varnames)
        elif form.answer.data:
            form_shape = controller.analysis_type.value
            instruction, outputfields = controller.form_answers_anova()
            return render_template('bigform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape, varnames=varnames, title=title)
        else:
            print('ERROR: INVALID METHOD')
    #elif flask.request.method == 'GET':
    #    form_shape = controller.analysis_type
    #    return render_template('bigform.html', form=form, instruction='', displays=[[''] for i in range(7)], shape=form_shape)
    else:
        print('ERROR: INVALID METHOD')
        
@app.route('/smallform', methods=['POST'])
def smallform():
    path = 'app/controller.pickle' if 'Github' in os.getcwd() else '/var/www/ComeniusPrototype/ComeniusPrototype/app/controller.pickle'
    with open(path, 'rb') as f:
        mc:dict = pickle.load(f)
        ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        controller = mc[ip]
    a = controller.assignment
    form = SmallForm()
    title:str = controller.mes['M_TITLE']
    varnames:list = [[a['independent']] + a['levels']] if a['assignment_type'] != 4 else [[a['independent']] + a['levels'],[a['independent2']] + a['levels2']]
    if flask.request.method == 'POST':
        if form.submit.data:
            form_shape = controller.analysis_type.value
            textfields = [x for x in dir(form) if str(type(form.__getattribute__(x))) in ["<class 'wtforms.fields.core.StringField'>","<class 'wtforms.fields.simple.TextAreaField'>"]]
            textdict = dict([(x, form.__getattribute__(x).data) for x in textfields])
            instruction, outputfields = controller.update_form_ttest(textdict)
            with open(path, 'wb') as f:
                pickle.dump(mc, f, protocol=pickle.HIGHEST_PROTOCOL) 
            return render_template('smallform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape, varnames=varnames, title=title)
        elif form.nextt.data:
            skip :bool = controller.skipable
            prev :bool = controller.prevable
            display = controller.protocol[0][0]
            form = BaseForm()
            field = controller.submit_field.value
            return render_template('index.html', display=display, form=form, skip=skip, prev=prev, submit_field=field, varnames=varnames)
        elif form.answer.data:
            form_shape = controller.analysis_type.value
            instruction, outputfields = controller.form_answers()
            return render_template('smallform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape, varnames=varnames, title=title)
        else:
            print('ERROR: INVALID METHOD')
    #elif flask.request.method == 'GET':
    #    form_shape = controller.analysis_type
    #    return render_template('smallform.html', form=form, instruction='', displays=[[''] for i in range(7)], shape=form_shape)
    else:
        print('ERROR: INVALID METHOD')
        
@app.route('/reportform', methods=['POST'])
def reportform():
    path = 'app/controller.pickle' if 'Github' in os.getcwd() else '/var/www/ComeniusPrototype/ComeniusPrototype/app/controller.pickle'
    with open(path, 'rb') as f:
        mc:dict = pickle.load(f)
        ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        controller = mc[ip]
    varnames:list = []#[[a['independent']] + a['levels']] if a['assignment_type'] != 4 else [[a['independent']] + a['levels'],[a['independent2']] + a['levels2']]
    form = ReportForm()
    title:str = controller.mes['M_TITLE']
    if flask.request.method == 'POST':
        if form.submit.data:
            textfields = [x for x in dir(form) if str(type(form.__getattribute__(x))) == "<class 'wtforms.fields.simple.TextAreaField'>"]
            textdict = dict([(x, form.__getattribute__(x).data) for x in textfields])
            instruction, output = controller.update_form_report(textdict)
            with open(path, 'wb') as f:
                pickle.dump(mc, f, protocol=pickle.HIGHEST_PROTOCOL) 
            return render_template('reportform.html', form=form, instruction=instruction, display=output, title=title)
        elif form.nextt.data:
            skip :bool = controller.skipable
            prev :bool = controller.prevable
            display = controller.protocol[0][0]
            form = BaseForm()
            form.inputtext.data = ""
            field = controller.submit_field.value
            return render_template('index.html', display=display, form=form, skip=skip, prev=prev, submit_field=field, varnames=varnames, title=title)
        else:
            print('ERROR: INVALID METHOD')
    #elif flask.request.method == 'GET':
    #    instruction = controller.print_assignment()
    #    return render_template('reportform.html', form=form, instruction=instruction, display='')
    else:
        print('ERROR: INVALID METHOD')