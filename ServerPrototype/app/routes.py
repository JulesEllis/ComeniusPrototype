from flask import render_template, flash, redirect, url_for
from app import app
from app.forms import BaseForm, BigForm, SmallForm, ReportForm
from app.code.interface import OuterController
from app.code.enums import Task, Process
import flask

@app.route('/')
@app.route('/index', methods=['GET','POST'])
def index():
    form = BaseForm()
    controller :OuterController = OuterController()
    varnames = [['dummy1'],['dummy2']]
    if flask.request.method == 'GET':
        controller.reset()
        return render_template('index.html', display="Hoi, je kan in dit programma op twee manieren elementaire rapporten oefenen, namelijke in de standaardmodus en de tentamenmodus. Klik op de submit-knop om verder te gaan", 
                               form=form, skip=False, submit_field=6, varnames=varnames)
    elif flask.request.method == 'POST':        
        #Isolate text fields
        textfields:list = [x for x in dir(form) if str(type(form.__getattribute__(x))) == "<class 'wtforms.fields.core.StringField'>"]
        textdict:dict = dict([(x, form.__getattribute__(x).data) for x in textfields])        
        
        if form.submit.data:
            output_text : str = controller.update(textdict)
            if controller.assignment != None: #Retrieve variable names
                varnames = controller.assignment['data']['varnames']
            form_shape = controller.report_type.value
            if controller.formmode and form_shape > 0 and form_shape < 3:
                form = SmallForm()
                controller.formmode = False
                instruction = controller.print_assignment()
                return render_template('smallform.html', form=form, instruction=instruction, displays=[[''] for i in range(12)], shape=form_shape, varnames=varnames)
            elif controller.formmode and form_shape > 2 and form_shape < 6:
                form = BigForm()
                controller.formmode = False
                instruction = controller.print_assignment()
                return render_template('bigform.html', form=form, instruction=instruction, displays=[[''] for i in range(7)], shape=form_shape, varnames=varnames)
            elif form_shape == 6:
                form = ReportForm()
                controller.formmode = False
                instruction = controller.print_assignment()
                return render_template('reportform.html', form=form, instruction=instruction, display='')
        if form.skip.data:
            output_text : str = controller.update({'inputtext': 'skip'})
        if form.prev.data:
            output_text : str = controller.update({'inputtext': 'prev'})
        if controller.assignment != None: #Retrieve variable names
                varnames = controller.assignment['data']['varnames']
        form.inputtext.data = ""
        skip :bool = controller.skipable
        prev :bool = controller.prevable
        submit_field :int = controller.submit_field.value
        return render_template('index.html', display=output_text, form=form, skip=skip, prev=prev, submit_field=submit_field, varnames=varnames)
    else:
        print('ERROR: INVALID METHOD')

@app.route('/bigform', methods=['POST'])
def bigform():
    form = BigForm()
    controller : OuterController = OuterController()
    varnames = controller.assignment['data']['varnames']
        
    if flask.request.method == 'POST':
        if form.submit.data:
            form_shape = controller.report_type.value
            textfields = [x for x in dir(form) if str(type(form.__getattribute__(x))) == "<class 'wtforms.fields.core.StringField'>"]
            textdict = dict([(x, form.__getattribute__(x).data) for x in textfields])
            instruction, outputfields = controller.update_form_anova(textdict)
            return render_template('bigform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape, varnames=varnames)
        elif form.nextt.data:
            skip :bool = controller.skipable
            prev :bool = controller.prevable
            submit_field :int = controller.submit_field.value
            display = controller.protocol[0][0]
            form = BaseForm()
            controller.report_type = Task.TEXT_FIELD
            return render_template('index.html', display=display, form=form, skip=skip, prev=prev, submit_field=submit_field, varnames=varnames)
        else:
            print('ERROR: INVALID METHOD')
    #elif flask.request.method == 'GET':
    #    form_shape = controller.report_type
    #    return render_template('bigform.html', form=form, instruction='', displays=[[''] for i in range(7)], shape=form_shape)
    else:
        print('ERROR: INVALID METHOD')
        
@app.route('/smallform', methods=['POST'])
def smallform():
    form = SmallForm()
    controller : OuterController = OuterController()
    varnames = controller.assignment['data']['varnames']
        
    if flask.request.method == 'POST':
        if form.submit.data:
            form_shape = controller.report_type.value
            textfields = [x for x in dir(form) if str(type(form.__getattribute__(x))) == "<class 'wtforms.fields.core.StringField'>"]
            textdict = dict([(x, form.__getattribute__(x).data) for x in textfields])
            instruction, outputfields = controller.update_form_ttest(textdict)
            return render_template('smallform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape, varnames=varnames)
        elif form.nextt.data:
            skip :bool = controller.skipable
            prev :bool = controller.prevable
            submit_field :int = controller.submit_field
            display = controller.protocol[0][0]
            form = BaseForm()
            controller.report_type = Task.TEXT_FIELD
            return render_template('index.html', display=display, form=form, skip=skip, prev=prev, submit_field=submit_field, varnames=varnames)
        else:
            print('ERROR: INVALID METHOD')
    #elif flask.request.method == 'GET':
    #    form_shape = controller.report_type
    #    return render_template('smallform.html', form=form, instruction='', displays=[[''] for i in range(7)], shape=form_shape)
    else:
        print('ERROR: INVALID METHOD')
        
@app.route('/reportform', methods=['GET','POST'])
def reportform():
    form = ReportForm()
    controller : OuterController = OuterController()
    varnames = controller.assignment['data']['varnames']
        
    if flask.request.method == 'POST':
        if form.submit.data:
            textfields = [x for x in dir(form) if str(type(form.__getattribute__(x))) == "<class 'wtforms.fields.core.StringField'>"]
            textdict = dict([(x, form.__getattribute__(x).data) for x in textfields])
            instruction, output = controller.update_form_report(textdict)
            return render_template('reportform.html', form=form, instruction=instruction, display=output)
        elif form.nextt.data:
            skip :bool = controller.skipable
            prev :bool = controller.prevable
            submit_field :int = controller.submit_field
            display = controller.protocol[0][0]
            form = BaseForm()
            controller.report_type = Task.TEXT_FIELD
            return render_template('index.html', display=display, form=form, skip=skip, prev=prev, submit_field=submit_field, varnames=varnames)
        else:
            print('ERROR: INVALID METHOD')
    #elif flask.request.method == 'GET':
    #    instruction = controller.print_assignment()
    #    return render_template('reportform.html', form=form, instruction=instruction, display='')
    else:
        print('ERROR: INVALID METHOD')