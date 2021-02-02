from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms import TextAreaField, SelectField
from wtforms.validators import Required, Length
from wtforms.validators import DataRequired

class BaseForm(FlaskForm):
    inputtext = StringField('', validators=[DataRequired()])
    inputtextlarge = TextAreaField('', validators=[DataRequired()])
    a_choices = ['T-toets onafhankelijke variabelen','T-toets voor gekoppelde paren','One-way ANOVA',
                 'Two-way ANOVA','Repeated Measures Anova','Multiple-regressieanalyse','MANOVA','ANCOVA', 'Multivariate-RMANOVA']#,'Dubbel Multivariate-RMANOVA']
    r_choices = ['Elementair rapport (oefenmodus)','Elementair rapport (tentamenmodus)','Beknopt rapport']
    selectanalysis = SelectField(label='Kies je analyse', choices=a_choices)
    selectreport = SelectField(label='Kies wat voor rapport je wil oefenen', choices=r_choices)
    submit = SubmitField('Doorgaan')
    skip = SubmitField('Volgende')
    prev = SubmitField('Vorige')
    answer = SubmitField('Antwoord')
    
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
    nextt = SubmitField('Next')

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
    nextt = SubmitField('Next')
    
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
    nextt = SubmitField('Next')
    
    
    
    
    