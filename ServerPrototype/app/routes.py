from flask import render_template, flash, redirect, url_for
from app import app
from app.forms import BaseForm, BigForm, SmallForm
from app.code.interface import OuterController
import flask

@app.route('/')
@app.route('/index', methods=['GET','POST'])
def index():
    form = BaseForm()
    controller :OuterController = OuterController()
    if flask.request.method == 'GET':
        controller.reset()
        return render_template('index.html', display="Hoi wil je dit programma gebruiken in de testmodus of niet?", 
                               form=form, skip=False, submit_field=0)
    elif flask.request.method == 'POST':        
        #Isolate text fields
        textfields:list = [x for x in dir(form) if str(type(form.__getattribute__(x))) == "<class 'wtforms.fields.core.StringField'>"]
        textdict:dict = dict([(x, form.__getattribute__(x).data) for x in textfields])        
        
        if form.submit.data:
            output_text : str = controller.update(textdict)
            form_shape = controller.table_shape
            if controller.formmode and form_shape > 0 and form_shape < 3:
                form = SmallForm()
                controller.formmode = False
                instruction = controller.print_assignment()
                return render_template('smallform.html', form=form, instruction=instruction, displays=[[''] for i in range(12)], shape=form_shape)
            elif controller.formmode and form_shape > 2:
                form = BigForm()
                controller.formmode = False
                instruction = controller.print_assignment()
                return render_template('bigform.html', form=form, instruction=instruction, displays=[[''] for i in range(7)], shape=form_shape)
        if form.skip.data:
            output_text : str = controller.update({'inputtext': 'skip'})
        if form.prev.data:
            output_text : str = controller.update({'inputtext': 'prev'})
        form.inputtext.data = ""
        skip :bool = controller.skipable and controller.testmode
        prev :bool = controller.prevable
        submit_field :int = controller.submit_field
        return render_template('index.html', display=output_text, form=form, skip=skip, prev=prev, submit_field=submit_field)
    else:
        print('ERROR: INVALID METHOD')

@app.route('/bigform', methods=['POST'])
def bigform():
    form = BigForm()
    controller : OuterController = OuterController()
    if flask.request.method == 'POST':
        if form.submit.data:
            form_shape = controller.table_shape
            textfields = [x for x in dir(form) if str(type(form.__getattribute__(x))) == "<class 'wtforms.fields.core.StringField'>"]
            textdict = dict([(x, form.__getattribute__(x).data) for x in textfields])
            instruction, outputfields = controller.update_form_anova(textdict)
            return render_template('bigform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape)
        elif form.nextt.data:
            skip :bool = controller.skipable and controller.testmode
            prev :bool = controller.prevable
            submit_field :int = controller.submit_field
            display = controller.protocol[0][0]
            form = BaseForm()
            controller.table_shape = 0
            return render_template('index.html', display=display, form=form, skip=skip, prev=prev, submit_field=submit_field)
        else:
            print('ERROR: INVALID METHOD')
    #elif flask.request.method == 'GET':
    #    form_shape = controller.table_shape
    #    return render_template('bigform.html', form=form, instruction='', displays=[[''] for i in range(7)], shape=form_shape)
    else:
        print('ERROR: INVALID METHOD')
        
@app.route('/smallform', methods=['POST'])
def smallform():
    form = SmallForm()
    controller : OuterController = OuterController()
    if flask.request.method == 'POST':
        if form.submit.data:
            form_shape = controller.table_shape
            textfields = [x for x in dir(form) if str(type(form.__getattribute__(x))) == "<class 'wtforms.fields.core.StringField'>"]
            textdict = dict([(x, form.__getattribute__(x).data) for x in textfields])
            instruction, outputfields = controller.update_form_ttest(textdict)
            return render_template('smallform.html', form=form, instruction=instruction, displays=outputfields, shape=form_shape)
        elif form.nextt.data:
            skip :bool = controller.skipable and controller.testmode
            prev :bool = controller.prevable
            submit_field :int = controller.submit_field
            display = controller.protocol[0][0]
            form = BaseForm()
            controller.table_shape = 0
            return render_template('index.html', display=display, form=form, skip=skip, prev=prev, submit_field=submit_field)
        else:
            print('ERROR: INVALID METHOD')
    #elif flask.request.method == 'GET':
    #    form_shape = controller.table_shape
    #    return render_template('smallform.html', form=form, instruction='', displays=[[''] for i in range(7)], shape=form_shape)
    else:
        print('ERROR: INVALID METHOD')