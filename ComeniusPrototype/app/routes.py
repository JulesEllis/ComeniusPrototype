from flask import render_template, flash, redirect, url_for, request
from app import app
from app.forms import BaseForm, BigForm, SmallForm, ReportForm
from app.code.interface import Controller #OuterController
from app.code.enums import Task, Process
from app.code.scan_functions_spacy import *
from app.code.scan_functions import scan_hypothesis_anova
from app.code.assignments import cap
import flask
import os
import json

@app.route('/')
@app.route('/index', methods=['GET','POST'])
def index():
    #Get controller
    path = 'app/controller.json' if 'Github' in os.getcwd() else '/var/www/ComeniusPrototype/ComeniusPrototype/app/controller.json'
    with open(path, 'r') as f:
        mc:dict = json.load(f)
        ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        if not ip in list(mc.keys()):
            mc[ip] = Controller()
            controller = mc[ip]
        else:
            controller = Controller(jsondict=mc[ip])
        
    #Assign local variables
    varnames = [['dummy1'],['dummy2']]
    title:str = 'Oefeningsmodule voor statistische rapporten'
    form = BaseForm()
    if flask.request.method == 'GET':
        controller.reset()
        #Save controller
        with open(path, 'w') as f:
            mc[ip] = controller.serialize()
            json.dump(mc, f)
        instruction = controller.protocol[0][0]
        if controller.mes != None:
            title:str = controller.mes['M_TITLE']
        return render_template('index.html', display=instruction, form=form, skip=False, submit_field=8, varnames=varnames, title=title)
    elif flask.request.method == 'POST':        
        #Isolate text fields
        mes = controller.mes
        if not controller.mes == None:
            title:str = controller.mes['M_TITLE']
        textfields:list = [x for x in dir(form) if str(type(form.__getattribute__(x))) in ["<class 'wtforms.fields.core.StringField'>","<class 'wtforms.fields.core.SelectField'>","<class 'wtforms.fields.simple.TextAreaField'>"]]
        textdict:dict = dict([(x, form.__getattribute__(x).data) for x in textfields])        
        
        #Prepare webpage parameters
        if form.submit.data:
            output_text : str = controller.update(textdict)
            if controller.assignment != None: #Retrieve variable names
                a = controller.assignment
                varnames:list = [[cap(a['independent'])] + [cap(x) for x in a['levels']]] if a['assignment_type'] != 4 else [[cap(a['independent'])] + [cap(x) for x in a['levels']],[cap(a['independent2'])] + [cap(x) for x in a['levels2']]]
            form_shape = controller.analysis_type.value
            if controller.formmode and form_shape > 0 and form_shape < 3:
                form = SmallForm()
                form.__getattribute__('inputtext1').label = mes['Q_IND']
                form.__getattribute__('inputtext2').label = mes['Q_DEP']
                form.__getattribute__('inputtext3').label = mes['Q_MEASURE']
                form.__getattribute__('inputtext4').label = mes['Q_HYP']
                form.__getattribute__('inputtext5').label = mes['Q_DF']
                form.__getattribute__('inputtext6').label = mes['Q_RAW']
                form.__getattribute__('inputtext7').label = mes['Q_RELATIVE']
                form.__getattribute__('inputtext8').label = mes['Q_T']
                form.__getattribute__('inputtext9').label = mes['Q_P']
                form.__getattribute__('inputtext10').label = mes['Q_DECISION']
                form.__getattribute__('inputtext11').label = mes['Q_INTERPRET']
                form.__getattribute__('mean1').label = mes['Q_TABLE']
                form.__getattribute__('mean2').label = mes['A_STATISTIC']
                form.__getattribute__('std1').label = mes['A_MEAN']
                form.__getattribute__('std2').label = mes['A_STD']
                form.__getattribute__('nextt').label.text = mes['B_NEXT']
                form.__getattribute__('answer').label.text = mes['B_ANSWER']
                controller.formmode = False
                instruction = controller.print_assignment()
                #Store controller
                with open(path, 'w') as f:        
                    mc[ip] = controller.serialize()
                    json.dump(mc, f)        
                
                return render_template('smallform.html', form=form, instruction=instruction, displays=[[''] for i in range(12)], shape=form_shape, varnames=varnames, title=title)
            elif controller.formmode and form_shape > 2 and form_shape < 6:
                form = BigForm()
                form.__getattribute__('inputtext1').label = mes['Q_IND']
                form.__getattribute__('inputtext12').label = mes['Q_IND2']
                form.__getattribute__('inputtext2').label = mes['Q_DEP']
                form.__getattribute__('inputtext3').label = mes['Q_MEASURE']
                form.__getattribute__('inputtext32').label = mes['Q_MEASURE2']
                form.__getattribute__('inputtext4').label = mes['Q_HYP']
                form.__getattribute__('inputtext42').label = mes['Q_HYP2']
                form.__getattribute__('inputtext43').label = mes['Q_HYPINT']
                form.__getattribute__('inputtext5').label = mes['Q_DECISION']
                form.__getattribute__('inputtext52').label = mes['Q_DECISION2']
                form.__getattribute__('inputtext53').label = mes['Q_DECISIONINT']
                form.__getattribute__('inputtext6').label = mes['Q_INTERPRET']
                form.__getattribute__('inputtext62').label = mes['Q_INTERPRET2']
                form.__getattribute__('inputtext63').label = mes['Q_INTERPRETINT']
                form.__getattribute__('df1').label = mes['Q_TABLE']
                form.__getattribute__('df2').label = mes['A_SOURCE']
                form.__getattribute__('df3').label = mes['A_PERSON']
                form.__getattribute__('df4').label = mes['A_INTERACT']
                form.__getattribute__('df5').label = mes['A_TOTAL']
                form.__getattribute__('nextt').label.text = mes['B_NEXT']
                form.__getattribute__('answer').label.text = mes['B_ANSWER']
                controller.formmode = False
                instruction = controller.print_assignment()
                #Store controller
                with open(path, 'w') as f:
                    mc[ip] = controller.serialize()
                    json.dump(mc, f)            
                
                return render_template('bigform.html', form=form, instruction=instruction, displays=[[''] for i in range(7)], shape=form_shape, varnames=varnames, title=title)
            elif form_shape == 7:
                form = ReportForm()
                form.__getattribute__('nextt').label.text = mes['B_NEXT']
                form.__getattribute__('inputtext').label = mes['Q_SHORTREPORT']
                controller.formmode = False
                instruction = output_text
                #Store controller
                with open(path, 'w') as f:
                    mc[ip] = controller.serialize()
                    json.dump(mc, f)    
                
                return render_template('reportform.html', form=form, instruction=instruction, display='', title=title)
        
        #Detect which button triggered the current screen
        if form.skip.data:
            output_text : str = controller.update({'inputtext': 'skip'})
        if form.prev.data:
            output_text : str = controller.update({'inputtext': 'prev'})
        if form.answer.data:
            controller.answer_triggered = not controller.answer_triggered
            output_text : str = controller.assignments.print_assignment(controller.assignment) + '<br>' + controller.protocol[controller.index][0]
        
        #Retrieve variable names
        if controller.assignment != None: 
            a = controller.assignment
            varnames:list = [[cap(a['independent'])] + [cap(x) for x in a['levels']]] if a['assignment_type'] != 4 else [[cap(a['independent'])] + [cap(x) for x in a['levels']],[cap(a['independent2'])] + [cap(x) for x in a['levels2']]]
        
        #Remove text from field after new question
        if controller.wipetext:
            form.inputtext.data = ""
            form.inputtextlarge.data = ""
            
        #Set Dutch or English text in fields
        if controller.submit_field == Task.INTRO or controller.submit_field == Task.CHOICE or controller.submit_field == Task.FINISHED: #Determine enter button text
            if not controller.mes['L_ENGLISH']:
                form.submit.label.text = 'Doorgaan'
            else:
                form.submit.label.text = 'Continue'
        else:
            form.submit.label.text = 'Feedback'
        if not controller.submit_field == Task.INTRO:
            mes:dict = controller.mes
            form.__getattribute__('skip').label.text = mes['B_NEXT']
            form.__getattribute__('prev').label.text = mes['B_PREV']
            form.__getattribute__('answer').label.text = mes['B_ANSWER']
        if controller.submit_field == Task.CHOICE: #Determine dropdown language options
            mes:dict = controller.mes
            form.__getattribute__('selectanalysis').choices = [mes['M_ANALYSIS' + str(i+1)] for i in range(9)]
            form.__getattribute__('selectanalysis').label = mes['M_CHOOSEANALYSIS']
            form.__getattribute__('selectreport').choices = [mes['M_REPORT' + str(i+1)] for i in range(3)]
            form.__getattribute__('selectreport').label = mes['M_CHOOSEREPORT']
        if controller.submit_field.value > 0 and controller.submit_field.value < 6:
            form.__getattribute__('mean2').label = mes['A_STATISTIC']
            form.__getattribute__('std1').label = mes['A_MEAN']
            form.__getattribute__('std2').label = mes['A_STD']
            form.__getattribute__('df2').label = mes['A_SOURCE']
            form.__getattribute__('df3').label = mes['A_PERSON']
            form.__getattribute__('df4').label = mes['A_INTERACT']
            form.__getattribute__('df5').label = mes['A_TOTAL']
            form.__getattribute__('n1').label = mes['A_DIFF']
        
        #Prepare parameters for rendering
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
        with open(path, 'w') as f:
            mc[ip] = controller.serialize()
            json.dump(mc, f)    
        
        #Render page
        return render_template('index.html', display=output_text, answer_text=answer_text, form=form, skip=skip, prev=prev, answer=answer, submit_field=submit_field, varnames=varnames, title=title)
    else:
        print('ERROR: INVALID METHOD')

@app.route('/bigform', methods=['POST'])
def bigform():
    #Get controller
    path = 'app/controller.json' if 'Github' in os.getcwd() else '/var/www/ComeniusPrototype/ComeniusPrototype/app/controller.json'
    with open(path, 'r') as f:
        mc:dict = json.load(f)
        ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        controller = Controller(jsondict=mc[ip])
    mes:dict = controller.mes
    a = controller.assignment
    form = BigForm()
    title:str = mes['M_TITLE']
    form.__getattribute__('nextt').label.text = mes['B_NEXT']
    form.__getattribute__('answer').label.text = mes['B_ANSWER']
    
    #Fill text positions that will be shown in the form
    varnames:list = [[cap(a['independent'])] + [cap(x) for x in a['levels']]] if a['assignment_type'] != 4 else [[cap(a['independent'])] + [cap(x) for x in a['levels']],[cap(a['independent2'])] + [cap(x) for x in a['levels2']]]
    form.__getattribute__('inputtext1').label = mes['Q_IND']
    form.__getattribute__('inputtext12').label = mes['Q_IND2']
    form.__getattribute__('inputtext2').label = mes['Q_DEP']
    form.__getattribute__('inputtext3').label = mes['Q_MEASURE']
    form.__getattribute__('inputtext32').label = mes['Q_MEASURE2']
    form.__getattribute__('inputtext4').label = mes['Q_HYP']
    form.__getattribute__('inputtext42').label = mes['Q_HYP2']
    form.__getattribute__('inputtext43').label = mes['Q_HYPINT']
    form.__getattribute__('inputtext5').label = mes['Q_DECISION']
    form.__getattribute__('inputtext52').label = mes['Q_DECISION2']
    form.__getattribute__('inputtext53').label = mes['Q_DECISIONINT']
    form.__getattribute__('inputtext6').label = mes['Q_INTERPRET']
    form.__getattribute__('inputtext62').label = mes['Q_INTERPRET2']
    form.__getattribute__('inputtext63').label = mes['Q_INTERPRETINT']
    form.__getattribute__('df1').label = mes['Q_TABLE']
    form.__getattribute__('df2').label = mes['A_SOURCE']
    form.__getattribute__('df3').label = mes['A_PERSON']
    form.__getattribute__('df4').label = mes['A_INTERACT']
    form.__getattribute__('df5').label = mes['A_TOTAL']
    
    #Determine rendering parameters    
    if flask.request.method == 'POST':
        if form.submit.data:
            form_shape = controller.analysis_type.value
            textfields = [x for x in dir(form) if str(type(form.__getattribute__(x))) in ["<class 'wtforms.fields.core.StringField'>","<class 'wtforms.fields.simple.TextAreaField'>"]]
            textdict = dict([(x, form.__getattribute__(x).data) for x in textfields])
            instruction, outputfields = controller.update_form_anova(textdict)
            with open(path, 'w') as f:
                mc[ip] = controller.serialize()
                json.dump(mc, f) 
            return render_template('bigform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape, varnames=varnames, title=title)
        elif form.nextt.data:
            skip:bool = controller.skipable
            prev:bool = controller.prevable
            display = controller.protocol[0][0]
            form = BaseForm()
            field = controller.submit_field.value
            return render_template('index.html', display=display, form=form, skip=skip, prev=prev, submit_field=field, varnames=varnames, title=title)
        elif form.answer.data:
            form_shape = controller.analysis_type.value
            instruction, outputfields = controller.form_answers_anova()
            return render_template('bigform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape, varnames=varnames, title=title)
        else:
            print('ERROR: INVALID METHOD')
    else:
        print('ERROR: INVALID METHOD')
        
@app.route('/smallform', methods=['POST'])
def smallform():
    path = 'app/controller.json' if 'Github' in os.getcwd() else '/var/www/ComeniusPrototype/ComeniusPrototype/app/controller.json'
    with open(path, 'r') as f:
        mc:dict = json.load(f)
        ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        controller = Controller(jsondict=mc[ip])
    mes:dict = controller.mes
    a = controller.assignment
    form = SmallForm()
    title:str = controller.mes['M_TITLE']
    print(mes['B_NEXT'])
    form.__getattribute__('nextt').label.text = mes['B_NEXT']
    form.__getattribute__('answer').label.text = mes['B_ANSWER']
    
    #Enter text labels
    varnames:list = [[cap(a['independent'])] + [cap(x) for x in a['levels']]] if a['assignment_type'] != 4 else [[cap(a['independent'])] + [cap(x) for x in a['levels']],[cap(a['independent2'])] + [cap(x) for x in a['levels2']]]
    form.__getattribute__('inputtext1').label = mes['Q_IND']
    form.__getattribute__('inputtext2').label = mes['Q_DEP']
    form.__getattribute__('inputtext3').label = mes['Q_MEASURE']
    form.__getattribute__('inputtext4').label = mes['Q_HYP']
    form.__getattribute__('inputtext5').label = mes['Q_DF']
    form.__getattribute__('inputtext6').label = mes['Q_RAW']
    form.__getattribute__('inputtext7').label = mes['Q_RELATIVE']
    form.__getattribute__('inputtext8').label = mes['Q_T']
    form.__getattribute__('inputtext9').label = mes['Q_P']
    form.__getattribute__('inputtext10').label = mes['Q_DECISION']
    form.__getattribute__('inputtext11').label = mes['Q_INTERPRET']
    form.__getattribute__('mean1').label = mes['Q_TABLE']
    form.__getattribute__('mean2').label = mes['A_STATISTIC']
    form.__getattribute__('std1').label = mes['A_MEAN']
    form.__getattribute__('std2').label = mes['A_STD']
    
    #Deterimine rendering parameters
    if flask.request.method == 'POST':
        if form.submit.data:
            form_shape = controller.analysis_type.value
            textfields = [x for x in dir(form) if str(type(form.__getattribute__(x))) in ["<class 'wtforms.fields.core.StringField'>","<class 'wtforms.fields.simple.TextAreaField'>"]]
            textdict = dict([(x, form.__getattribute__(x).data) for x in textfields])
            instruction, outputfields = controller.update_form_ttest(textdict)
            with open(path, 'w') as f:
                mc[ip] = controller.serialize()
                json.dump(mc, f) 
            return render_template('smallform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape, varnames=varnames, title=title)
        elif form.nextt.data:
            skip :bool = controller.skipable
            prev :bool = controller.prevable
            display = controller.protocol[0][0]
            form = BaseForm()
            field = controller.submit_field.value
            return render_template('index.html', display=display, form=form, skip=skip, prev=prev, submit_field=field, varnames=varnames, title=title)
        elif form.answer.data:
            form_shape = controller.analysis_type.value
            instruction, outputfields = controller.form_answers()
            return render_template('smallform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape, varnames=varnames, title=title)
        else:
            print('ERROR: INVALID METHOD')
    else:
        print('ERROR: INVALID METHOD')
        
@app.route('/reportform', methods=['POST'])
def reportform():
    path = 'app/controller.json' if 'Github' in os.getcwd() else '/var/www/ComeniusPrototype/ComeniusPrototype/app/controller.json'
    with open(path, 'r') as f:
        mc:dict = json.load(f)
        ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        controller = Controller(jsondict=mc[ip])
    mes:dict = controller.mes
    varnames:list = []#[[a['independent']] + a['levels']] if a['assignment_type'] != 4 else [[a['independent']] + a['levels'],[a['independent2']] + a['levels2']]
    form = ReportForm()
    
    #Fill text fields
    title:str = controller.mes['M_TITLE']
    form.__getattribute__('nextt').label.text = mes['B_NEXT']
    form.__getattribute__('inputtext').label = mes['Q_SHORTREPORT']
    
    #Determine rendering parameters
    if flask.request.method == 'POST':
        if form.submit.data:
            textfields = [x for x in dir(form) if str(type(form.__getattribute__(x))) == "<class 'wtforms.fields.simple.TextAreaField'>"]
            textdict = dict([(x, form.__getattribute__(x).data) for x in textfields])
            instruction, output = controller.update_form_report(textdict)
            with open(path, 'w') as f:
                mc[ip] = controller.serialize()
                json.dump(mc, f) 
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
    else:
        print('ERROR: INVALID METHOD')