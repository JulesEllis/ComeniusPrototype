from flask import render_template, flash, redirect, url_for, request, make_response
from app import app
from app.forms import BaseForm, BigForm, SmallForm, ReportForm
from app.code.interface import Controller #OuterController
from app.code.enums import Task, Process
from app.code.assignments import cap
import flask
import os
import json
import hashlib

@app.route('/')
@app.route('/index', methods=['GET','POST'])
def index():
    #Get controller
    path = 'app/controller.json' if 'Github' in os.getcwd() else '/var/www/ComeniusPrototype/ComeniusPrototype/app/controller.json'
    with open(path, 'r') as f:
        mc:dict = json.load(f)
        session_id = request.cookies.get('sessionID')
        ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        ipcode = hashlib.md5(ip.encode('utf-8')).hexdigest()
        if session_id == None: #Create new session ID if session is not present
            session_id = ipcode + '-' + str(len([x for x in list(mc.keys) if ipcode in x]) + 1)
        if not session_id in list(mc.keys()):
            mc[session_id] = Controller()
            controller = mc[session_id]
        else:
            controller = Controller(jsondict=mc[session_id])
        
    #Assign local variables
    varnames = [['dummy1'],['dummy2']]
    title:str = 'Oefeningsmodule voor statistische rapporten'
    form = BaseForm()
    if flask.request.method == 'GET':
        controller.reset()
        instruction = controller.protocol[0][0]
        if controller.mes != None:
            title:str = controller.mes['M_TITLE']
        
        #Save controller
        with open(path, 'w') as f:
            mc[ipcode] = controller.serialize()
            json.dump(mc, f)
            
        #Render page
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
                if a['assignment_type'] == 4:
                    varnames:list = [a['independent'].get_varnames(),a['independent2'].get_varnames()]
                elif a['assignment_type'] == 6:
                    varnames:list=[]
                else:
                    varnames:list = [a['independent'].get_varnames()]
            form_shape = controller.analysis_type.value
            
            #If the user chose a T-test in exam mode
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
                form.__getattribute__('explain').label.text = mes['B_EXPLAIN']
                form.__getattribute__('b1').label.text = mes['B_INFO1']
                form.__getattribute__('b2').label.text = mes['B_INFO2']
                form.__getattribute__('b3').label.text = mes['B_INFO3']
                controller.formmode = False
                instruction = controller.print_assignment()
                #Save controller
                with open(path, 'w') as f:        
                    mc[ipcode] = controller.serialize()
                    json.dump(mc, f)        
                
                #Render page
                return render_template('smallform.html', form=form, instruction=instruction, displays=[[''] for i in range(12)], shape=form_shape, varnames=varnames, title=title)
            
            #If the user chose an ANOVA or RM-ANOVA in exam mode
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
                form.__getattribute__('explain').label.text = mes['B_EXPLAIN']
                form.__getattribute__('b1').label.text = mes['B_INFO1']
                form.__getattribute__('b2').label.text = mes['B_INFO2']
                form.__getattribute__('b3').label.text = mes['B_INFO3']
                controller.formmode = False
                instruction = controller.print_assignment()
                
                #Store controller
                with open(path, 'w') as f:
                    mc[ipcode] = controller.serialize()
                    json.dump(mc, f)            
                
                #Render page
                return render_template('bigform.html', form=form, instruction=instruction, displays=[[''] for i in range(7)], shape=form_shape, varnames=varnames, title=title)
            
            #If the user chose to make a short report
            elif form_shape == 7:
                form = ReportForm()
                form.__getattribute__('nextt').label.text = mes['B_NEXT']
                form.__getattribute__('answer').label.text = mes['B_ANSWER']    
                form.__getattribute__('explain').label.text = mes['B_EXPLAIN']
                form.__getattribute__('b1').label.text = mes['B_INFO1']
                form.__getattribute__('b2').label.text = mes['B_INFO2']
                form.__getattribute__('b3').label.text = mes['B_INFO3']
                form.__getattribute__('inputtext').label = mes['Q_SHORTREPORT']
                controller.formmode = False
                instruction = output_text
                
                #Store controller
                with open(path, 'w') as f:
                    mc[ipcode] = controller.serialize()
                    json.dump(mc, f)    
                
                #Render page
                return render_template('reportform.html', form=form, instruction=instruction, display='', title=title)
        
        ## Other HTML pages ruled out: The user has chosen to do an elementary report in practice mode, or is at one of the starting pages
        #Detect which button triggered the current screen
        if form.skip.data:
            output_text : str = controller.update({'inputtext': 'skip'})
        if form.prev.data:
            output_text : str = controller.update({'inputtext': 'prev'})
        answer_text:str = ''
        if form.answer.data:
            output_text : str = controller.assignments.print_assignment(controller.assignment) + '<br>' + controller.protocol[controller.index][0]
            answer_text : str = controller.mes['A_ANSWER']+'<br>'+cap(controller.protocol[controller.index][4])
        elif form.explain.data or form.b1.data or form.b2.data or form.b3.data:
            button_id = 0 if form.explain.data else 1 if form.b1.data else 2 if form.b2.data else 3 #form.b3.data
            output_text : str = controller.assignments.print_assignment(controller.assignment) + '<br>' + controller.protocol[controller.index][0]
            answer_text : str = controller.mes['A_EXPLANATION']+'<br>'+controller.explain_elementary(anslist=False, button_id=button_id)
        
        #Retrieve variable names
        if controller.assignment != None: 
            a = controller.assignment
            if a['assignment_type'] == 4:
                varnames:list = [a['independent'].get_varnames(),a['independent2'].get_varnames()]
            elif a['assignment_type'] == 6:
                varnames:list=[]
            else:
                varnames:list = [a['independent'].get_varnames()]
        
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
            
        #Determine which objects to render and what text to give them
        if not controller.submit_field == Task.INTRO:
            mes:dict = controller.mes
            form.__getattribute__('skip').label.text = mes['B_NEXT']
            form.__getattribute__('prev').label.text = mes['B_PREV']
            form.__getattribute__('answer').label.text = mes['B_ANSWER']
            form.__getattribute__('explain').label.text = mes['B_EXPLAIN']
            form.__getattribute__('b1').label.text = mes['B_INFO1']
            form.__getattribute__('b2').label.text = mes['B_INFO2']
            form.__getattribute__('b3').label.text = mes['B_INFO3']
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
        
        #Convert textbox to large textbox if appropriate
        submit_field :int = controller.submit_field.value
        
        #Store controller
        with open(path, 'w') as f:
            mc[ipcode] = controller.serialize()
            json.dump(mc, f)    
        
        #Render page
        return render_template('index.html', display=output_text, answer_text=answer_text, form=form, skip=skip, prev=prev, answer=answer, submit_field=submit_field, varnames=varnames, title=title)
        resp = make_response(render_template('readcookie.html'))
        resp.set_cookie('sessionID', user)
    else:
        print('ERROR: INVALID METHOD')

@app.route('/bigform', methods=['POST'])
def bigform():
    #Get controller
    path = 'app/controller.json' if 'Github' in os.getcwd() else '/var/www/ComeniusPrototype/ComeniusPrototype/app/controller.json'
    with open(path, 'r') as f:
        mc:dict = json.load(f)
        ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        ipcode = hashlib.md5(ip.encode('utf-8')).hexdigest()
        controller = Controller(jsondict=mc[ipcode])
    mes:dict = controller.mes
    a = controller.assignment
    form = BigForm()
    title:str = mes['M_TITLE']
    form.__getattribute__('nextt').label.text = mes['B_NEXT']
    form.__getattribute__('answer').label.text = mes['B_ANSWER']
    form.__getattribute__('explain').label.text = mes['B_EXPLAIN']
    form.__getattribute__('b1').label.text = mes['B_INFO1']
    form.__getattribute__('b2').label.text = mes['B_INFO2']
    form.__getattribute__('b3').label.text = mes['B_INFO3']
    
    #Fill text positions that will be shown in the form
    if a['assignment_type'] == 4:
        varnames:list = [a['independent'].get_varnames(),a['independent2'].get_varnames()]
    elif a['assignment_type'] == 6:
        varnames:list=[]
    else:
        varnames:list = [a['independent'].get_varnames()]
    form.__getattribute__('inputtext1').label = mes['Q_IND']
    form.__getattribute__('inputtext12').label = mes['Q_IND2']
    form.__getattribute__('inputtext2').label = mes['Q_DEP']
    form.__getattribute__('inputtext3').label = mes['Q_MEASURE']
    form.__getattribute__('inputtext32').label = mes['Q_MEASURE2']
    form.__getattribute__('inputtext4').label = mes['Q_HYP']
    form.__getattribute__('inputtext42').label = mes['Q_HYP2'] if controller.assignment['assignment_type'] == 4 else mes['Q_HYPSUB']
    form.__getattribute__('inputtext43').label = mes['Q_HYPINT']
    form.__getattribute__('inputtext5').label = mes['Q_DECISION']
    form.__getattribute__('inputtext52').label = mes['Q_DECISION2'] if controller.assignment['assignment_type'] == 4 else mes['Q_DECSUB']
    form.__getattribute__('inputtext53').label = mes['Q_DECISIONINT']
    form.__getattribute__('inputtext6').label = mes['Q_INTERPRET']
    form.__getattribute__('inputtext62').label = mes['Q_INTERPRET2']
    form.__getattribute__('inputtext63').label = mes['Q_INTERPRETINT']
    form.__getattribute__('df1').label = mes['Q_TABLE']
    form.__getattribute__('df2').label = mes['A_SOURCE']
    form.__getattribute__('df3').label = mes['A_PERSON']
    form.__getattribute__('df4').label = mes['A_INTERACT']
    form.__getattribute__('df5').label = mes['A_TOTAL']
    
    #Determine rendering parameters based on button input
    if flask.request.method == 'POST':
        if form.submit.data:
            form_shape = controller.analysis_type.value
            textfields = [x for x in dir(form) if str(type(form.__getattribute__(x))) in ["<class 'wtforms.fields.core.StringField'>","<class 'wtforms.fields.simple.TextAreaField'>"]]
            textdict = dict([(x, form.__getattribute__(x).data) for x in textfields])
            instruction, outputfields = controller.update_form_anova(textdict)
            with open(path, 'w') as f:
                mc[ipcode] = controller.serialize()
                json.dump(mc, f) 
            return render_template('bigform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape, varnames=varnames, title=title)
        elif form.nextt.data:
            skip:bool = controller.skipable
            prev:bool = controller.prevable
            display = controller.protocol[0][0]
            form = BaseForm()
            if not controller.mes['L_ENGLISH']:
                form.submit.label.text = 'Doorgaan'
            else:
                form.submit.label.text = 'Continue'
            field = controller.submit_field.value
            return render_template('index.html', display=display, form=form, skip=skip, prev=prev, submit_field=field, varnames=varnames, title=title)
        elif form.answer.data:
            form_shape = controller.analysis_type.value
            instruction, outputfields = controller.form_answers_anova()
            with open(path, 'w') as f:
                mc[ipcode] = controller.serialize()
                json.dump(mc, f) 
            return render_template('bigform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape, varnames=varnames, title=title)
        elif form.explain.data or form.b1.data or form.b2.data or form.b3.data:
            button_id = 0 if form.explain.data else 1 if form.b1.data else 2 if form.b2.data else 3 #form.b3.data
            form_shape = controller.analysis_type.value
            instruction = controller.assignments.print_assignment(controller.assignment)
            outputfields = controller.explain_elementary(anslist=True, button_id=button_id)
            return render_template('bigform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape, varnames=varnames, title=title)
        else:
            print('ERROR: INVALID METHOD')
    else:
        print('ERROR: INVALID METHOD')
        
@app.route('/smallform', methods=['POST'])
def smallform():
    #Get controller and associated objects
    path = 'app/controller.json' if 'Github' in os.getcwd() else '/var/www/ComeniusPrototype/ComeniusPrototype/app/controller.json'
    with open(path, 'r') as f:
        mc:dict = json.load(f)
        ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        ipcode = hashlib.md5(ip.encode('utf-8')).hexdigest()
        controller = Controller(jsondict=mc[ipcode])
    mes:dict = controller.mes
    a = controller.assignment
    form = SmallForm()
    title:str = controller.mes['M_TITLE']
    form.__getattribute__('nextt').label.text = mes['B_NEXT']
    form.__getattribute__('answer').label.text = mes['B_ANSWER']
    form.__getattribute__('explain').label.text = mes['B_EXPLAIN']
    form.__getattribute__('b1').label.text = mes['B_INFO1']
    form.__getattribute__('b2').label.text = mes['B_INFO2']
    form.__getattribute__('b3').label.text = mes['B_INFO3']
    
    #Enter text labels
    if a['assignment_type'] == 4:
        varnames:list = [a['independent'].get_varnames(),a['independent2'].get_varnames()]
    elif a['assignment_type'] == 6:
        varnames:list=[]
    else:
        varnames:list = [a['independent'].get_varnames()]
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
    
    #Deterimine rendering parameters based on input button
    if flask.request.method == 'POST':
        if form.submit.data:
            form_shape = controller.analysis_type.value
            textfields = [x for x in dir(form) if str(type(form.__getattribute__(x))) in ["<class 'wtforms.fields.core.StringField'>","<class 'wtforms.fields.simple.TextAreaField'>"]]
            textdict = dict([(x, form.__getattribute__(x).data) for x in textfields])
            instruction, outputfields = controller.update_form_ttest(textdict)
            with open(path, 'w') as f:
                mc[ipcode] = controller.serialize()
                json.dump(mc, f) 
            return render_template('smallform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape, varnames=varnames, title=title)
        elif form.nextt.data:
            skip :bool = controller.skipable
            prev :bool = controller.prevable
            display = controller.protocol[0][0]
            form = BaseForm()
            if not controller.mes['L_ENGLISH']:
                form.submit.label.text = 'Doorgaan'
            else:
                form.submit.label.text = 'Continue'
            field = controller.submit_field.value
            return render_template('index.html', display=display, form=form, skip=skip, prev=prev, submit_field=field, varnames=varnames, title=title)
        elif form.answer.data:
            form_shape = controller.analysis_type.value
            instruction, outputfields = controller.form_answers()
            with open(path, 'w') as f:
                mc[ipcode] = controller.serialize()
                json.dump(mc, f) 
            return render_template('smallform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape, varnames=varnames, title=title)
        elif form.explain.data or form.b1.data or form.b2.data or form.b3.data:
            button_id = 0 if form.explain.data else 1 if form.b1.data else 2 if form.b2.data else 3 #form.b3.data
            form_shape = controller.analysis_type.value
            instruction = controller.assignments.print_assignment(controller.assignment)
            outputfields = controller.explain_elementary(anslist=True, button_id=button_id)
            return render_template('smallform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape, varnames=varnames, title=title)
        else:
            print('ERROR: INVALID METHOD')
    else:
        print('ERROR: INVALID METHOD')
        
@app.route('/reportform', methods=['POST'])
def reportform():
    #Get controller
    path = 'app/controller.json' if 'Github' in os.getcwd() else '/var/www/ComeniusPrototype/ComeniusPrototype/app/controller.json'
    with open(path, 'r') as f:
        mc:dict = json.load(f)
        ip = request.environ.get('HTTP_X_REAL_IP', request.remote_addr)
        ipcode = hashlib.md5(ip.encode('utf-8')).hexdigest()
        controller = Controller(jsondict=mc[ipcode])
    mes:dict = controller.mes
    varnames:list = []
    form = ReportForm()
    
    #Fill text fields
    title:str = controller.mes['M_TITLE']
    form.__getattribute__('nextt').label.text = mes['B_NEXT']
    form.__getattribute__('inputtext').label = mes['Q_SHORTREPORT']
    form.__getattribute__('answer').label.text = mes['B_ANSWER']
    form.__getattribute__('explain').label.text = mes['B_EXPLAIN']
    form.__getattribute__('b1').label.text = mes['B_INFO1']
    form.__getattribute__('b2').label.text = mes['B_INFO2']
    form.__getattribute__('b3').label.text = mes['B_INFO3']
    
    #Determine rendering parameters based on input button
    if flask.request.method == 'POST':
        if form.submit.data:
            textfields = [x for x in dir(form) if str(type(form.__getattribute__(x))) == "<class 'wtforms.fields.simple.TextAreaField'>"]
            textdict = dict([(x, form.__getattribute__(x).data) for x in textfields])
            instruction, output = controller.update_form_report(textdict)
            with open(path, 'w') as f:
                mc[ipcode] = controller.serialize()
                json.dump(mc, f) 
            return render_template('reportform.html', form=form, instruction=instruction, display=output, title=title)
        elif form.nextt.data:
            if controller.assignment['feedback_requests'] > 0:
                controller.save_assignment()
            skip :bool = controller.skipable
            prev :bool = controller.prevable
            display = controller.protocol[0][0]
            form = BaseForm()
            if not controller.mes['L_ENGLISH']:
                form.submit.label.text = 'Doorgaan'
            else:
                form.submit.label.text = 'Continue'
            form.inputtext.data = ""
            field = controller.submit_field.value
            return render_template('index.html', display=display, form=form, skip=skip, prev=prev, submit_field=field, varnames=varnames, title=title)
        elif form.answer.data:
            controller.assignment['feedback_requests'] += 1
            instruction = controller.assignments.print_report(controller.assignment)
            output = controller.assignments.answer_report(controller.assignment) #controller.assignment['answer']
            with open(path, 'w') as f:
                mc[ipcode] = controller.serialize()
                json.dump(mc, f) 
            return render_template('reportform.html', form=form, instruction=instruction, display=output, title=title)
        elif form.explain.data or form.b1.data or form.b2.data or form.b3.data:
            button_id = 0 if form.explain.data else 1 if form.b1.data else 2 if form.b2.data else 3 # form.b3.data
            instruction = controller.assignments.print_report(controller.assignment)
            output = controller.explain_short(button_id) #controller.asssignment['answer']
            return render_template('reportform.html', form=form, instruction=instruction, display=output, title=title)
        else:
            print('ERROR: INVALID METHOD')
    else:
        print('ERROR: INVALID METHOD')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
