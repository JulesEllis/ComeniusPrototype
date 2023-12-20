#    This project contains CASWIC: Coaching App for Statistical Writing in Introductory Course.
#    Copyright (C) 2023 Jules Ellis and Jelmer Jansen
#
#    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
#
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms import TextAreaField, SelectField
from wtforms.validators import Required, Length
from wtforms.validators import DataRequired

class BaseForm(FlaskForm):
    inputtext = StringField('', validators=[DataRequired()])
    inputtextlarge = TextAreaField('', validators=[DataRequired()])
    selectanalysis = SelectField()
    selectreport = SelectField()
    selectlanguage = SelectField(label='Kies taal - Choose language', choices = ['Nederlands','English'])
    submit = SubmitField('Doorgaan')
    skip = SubmitField('Volgende')
    prev = SubmitField('Vorige')
    answer = SubmitField('Antwoord')
    explain = SubmitField('Leg uit')
    b1 = SubmitField('Docent')
    b2 = SubmitField('Werkgroepbegeleider')
    b3 = SubmitField('Student')
    
    mean1 = StringField('')
    std1 = StringField('')
    n1 = StringField('')
    mean2 = StringField('')
    std2 = StringField('')
    n2 = StringField('')
    df1 = StringField('')
    df2 = StringField('')
    df3 = StringField('')
    df4 = StringField('')
    df5 = StringField('')
    df6 = StringField('')
    ss1 = StringField('')
    ss2 = StringField('')
    ss3 = StringField('')
    ss4 = StringField('')
    ss5 = StringField('')
    ss6 = StringField('')
    ms1 = StringField('')
    ms2 = StringField('')
    ms3 = StringField('')
    ms4 = StringField('')
    ms5 = StringField('')
    f1 = StringField('')
    f2 = StringField('')
    f3 = StringField('')
    p1 = StringField('')
    p2 = StringField('')
    p3 = StringField('')
    r21 = StringField('')
    r22 = StringField('')
    r23 = StringField('')
    
class SmallForm(FlaskForm):
    inputtext1 = StringField('', validators=[DataRequired()])
    inputtext2 = StringField('', validators=[DataRequired()])
    inputtext3 = StringField('', validators=[DataRequired()])
    mean1 = StringField('', validators=[DataRequired()])
    std1 = StringField('', validators=[DataRequired()])
    n1 = StringField('', validators=[DataRequired()])
    mean2 = StringField('', validators=[DataRequired()])
    std2 = StringField('', validators=[DataRequired()])
    n2 = StringField('', validators=[DataRequired()])
    inputtext4 = StringField('', validators=[DataRequired()])
    inputtext5 = StringField('', validators=[DataRequired()])
    inputtext6 = StringField('', validators=[DataRequired()])
    inputtext7 = StringField('', validators=[DataRequired()])
    inputtext8 = StringField('', validators=[DataRequired()])
    inputtext9 = StringField('', validators=[DataRequired()])
    inputtext10 = TextAreaField('', validators=[DataRequired()]) #StringField('', validators=[DataRequired()])
    inputtext11 = TextAreaField('', validators=[DataRequired()]) #StringField('', validators=[DataRequired()])
    
    submit = SubmitField('Feedback')
    answer = SubmitField('Antwoord')
    nextt = SubmitField('Volgende')
    explain = SubmitField('Leg uit')
    b1 = SubmitField('Docent')
    b2 = SubmitField('Werkgroepbegeleider')
    b3 = SubmitField('Student')

class BigForm(FlaskForm):
    inputtext1 = StringField('', validators=[DataRequired()])
    inputtext2 = StringField('', validators=[DataRequired()])
    inputtext3 = StringField('', validators=[DataRequired()])
    inputtext4 = StringField('', validators=[DataRequired()])
    inputtext5 = TextAreaField('', validators=[DataRequired()]) #StringField('', validators=[DataRequired()])
    inputtext6 = TextAreaField('', validators=[DataRequired()]) #StringField('', validators=[DataRequired()])
    
    #Fields for 2-way ANOVA only
    inputtext12 = StringField('', validators=[DataRequired()])
    inputtext32= StringField('', validators=[DataRequired()])
    inputtext42 = StringField('', validators=[DataRequired()])
    inputtext43 = TextAreaField('', validators=[DataRequired()])
    inputtext52 = TextAreaField('', validators=[DataRequired()]) #StringField('', validators=[DataRequired()])
    inputtext53 = TextAreaField('', validators=[DataRequired()]) #StringField('', validators=[DataRequired()])
    inputtext62 = TextAreaField('', validators=[DataRequired()]) #StringField('', validators=[DataRequired()])
    inputtext63 = TextAreaField('', validators=[DataRequired()]) #StringField('', validators=[DataRequired()])
    
    submit = SubmitField('Feedback')
    answer = SubmitField('Antwoord')
    nextt = SubmitField('Volgende')
    explain = SubmitField('Leg uit')
    b1 = SubmitField('Docent')
    b2 = SubmitField('Werkgroepbegeleider')
    b3 = SubmitField('Student')
    
    df1 = StringField('')
    df2 = StringField('')
    df3 = StringField('')
    df4 = StringField('')
    df5 = StringField('')
    df6 = StringField('')
    ss1 = StringField('')
    ss2 = StringField('')
    ss3 = StringField('')
    ss4 = StringField('')
    ss5 = StringField('')
    ss6 = StringField('')
    ms1 = StringField('')
    ms2 = StringField('')
    ms3 = StringField('')
    ms4 = StringField('')
    ms5 = StringField('')
    f1 = StringField('')
    f2 = StringField('')
    f3 = StringField('')
    p1 = StringField('')
    p2 = StringField('')
    p3 = StringField('')
    r21 = StringField('')
    r22 = StringField('')
    r23 = StringField('')
    
class ReportForm(FlaskForm):
    inputtext = TextAreaField('', validators=[DataRequired()])
    submit = SubmitField('Feedback')
    nextt = SubmitField('Volgende')
    answer = SubmitField('Antwoord')
    explain = SubmitField('Leg uit')
    b1 = SubmitField('Docent')
    b2 = SubmitField('Werkgroepbegeleider')
    b3 = SubmitField('Student')
    
    
    
    
    
