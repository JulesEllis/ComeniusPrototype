#    This project contains CASWIC: Coaching App for Statistical Writing in Introductory Course.
#    Copyright (C) 2023 Jules Ellis and Jelmer Jansen
#
#    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FUNCTIONS FOR SCANNING CAUSAL INTERPRETATION AND DECISION

THESE TAKE THE INPUT TEXT IN DOC INSTEAD OF STRING
"""


def negation_counter(tokens: List[str]) -> int:
    count: int = 0
    for token in tokens:
        if token in ['geen', 'niet']:   # or token[:2] == 'on':
            count += 1
    return count

def descendants(node) -> List[Token]:
    output:list = []
    for child in node.children:
        output.append(child)
        output += descendants(child)
    return output

def check_causality(independent:Doc, dependent:Doc, alternative:bool=False) -> bool:
    #print(independent.dep_ + '-' + dependent.dep_)
    if not alternative:
        tuples = [('nsubj', 'obj'),('obj', 'ROOT'),('nsubj', 'nmod'),('obl', 'obj'),('ROOT', 'obj'),
              ('obj', 'nmod'), ('amod', 'obj'), ('obl','obl'),('nsubj','obl'),('obj','obj'),('nsubj','amod'),
              ('obj','obl'),('nmod','obj'),('obl','ROOT'),('obl','nsubj'),('obl','csubj'),('advmod','obj'),
              ('advmod','nmod'),('advmod','obj')]
    else: #Add reverse causality and disturbing variable options
        tuples = [('obj','obj'),('obj','nsubj'), ('ROOT','obj'),('nmod','nsubj'),('obj','obl'),('obj','ROOT'),('amod','nsubj'),
                       ('nmod','obj'),('obj','amod'),('obl','nsubj'),('obl','obj'),('obj','nmod'),('ROOT','obl'),('nsubj','obl'),
                       ('obl','obl'), ('csubj','obl')]
    for t in tuples:
        if independent.dep_ == t[0] and dependent.dep_ == t[1]:
            return True
    return False

def scan_decision(doc:Doc, solution:dict, anova:bool, num:int=1, prefix=True, elementair=True) -> [bool, List[str]]:
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:'] if prefix else []
    if elementair:
        output.extend(detect_h0(doc, solution, num))
    else:
        output.extend(detect_significance(doc, solution, num))
    output.extend(detect_comparison(doc, solution, anova, num))
    if solution['p'][num - 1] < 0.05:
        output.extend(detect_strength(doc, solution, anova, num))
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return False, 'Mooi, deze interpretatie klopt. ' if prefix else ''
    else:
        return True, '<br>'.join(output)
    
def scan_decision_anova(doc:Doc, solution:dict, num:int=3, prefix=True, elementair=True) -> [bool, List[str]]:
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:'] if prefix else []
    if elementair:
        output.extend(detect_h0(doc, solution, num))
    else:
        output.extend(detect_significance(doc, solution, num))
    output.extend(detect_interaction(doc, solution, True))
    output.extend(detect_strength(doc, solution, True, num))
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return False, 'Mooi, deze interpretatie klopt. ' if prefix else ''
    else:
        return True, '<br>'.join(output)
    
def scan_decision_rmanova(doc:Doc, solution:dict, num:int=1, prefix=True, elementair=True) -> [bool, List[str]]:
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:'] if prefix else []
    if elementair:
        output.extend(detect_h0(doc, solution, num))
    else:
        output.extend(detect_significance(doc, solution, num))
    output.extend(detect_true_scores(doc, solution, 2))
    if solution['p'][1] < 0.05:
        output.extend(detect_strength(doc, solution, True, num))
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return False, 'Mooi, deze interpretatie klopt. ' if prefix else ''
    else:
        return True, '<br>'.join(output)
    
def scan_interpretation(doc:Doc, solution:dict, anova:bool, num:int=1, prefix=True):
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:'] if prefix else []
    control: bool = solution['control'] if num < 2 else solution['control'+str(num)]
    primary_checks:list = ['primaire','eerste'] if not control else [solution['dependent']]
    unk_sents = [x for x in doc.sents if any([y in [z.text for z in x] for y in ['mogelijk','mogelijke','verklaring','verklaringen']])]
    if unk_sents != []:
        output.extend(detect_unk(unk_sents[0], solution, num))
    else:
        output.append(' -niet genoemd hoeveel mogelijke verklaringen er zijn')
    primair_sents = [x for x in doc.sents if any([z in [y.text for y in x] for z in primary_checks])]
    if primair_sents != []:
        output.extend(detect_primary(primair_sents[0], solution, num))
    else:
        output.append(' -de primaire verklaring wordt niet genoemd')
    if not control:
        alt_sents = [x for x in doc.sents if 'alternatieve' in [y.text for y in x]]
        #displacy.serve(alt_sents[0])
        if alt_sents != []:
            output.extend(detect_alternative(alt_sents[0], solution, num))
        else:
            output.append(' -de alternatieve verklaring wordt niet genoemd')
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return False, 'Mooi, deze interpretatie klopt. ' if prefix else ''
    else:
        return True, '<br>'.join(output)
    
def scan_interpretation_anova(doc:Doc, solution:dict, num:int=3, prefix=True):
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:'] if prefix else []
    control:bool = solution['control'] or solution['control2']
    primary_checks:list = ['primaire','eerste'] if not control else [solution['dependent']]
    unk_sents = [x for x in doc.sents if 'mogelijk' in [y.text for y in x] or 'mogelijke' in [y.text for y in x]]
    if unk_sents != []:
        output.extend(detect_unk(unk_sents[0], solution))
    else:
        output.append(' -niet genoemd hoeveel mogelijke interpretaties er zijn')
    primair_sents = [x for x in doc.sents if any([z in [y.text for y in x] for z in primary_checks])]
    if primair_sents != []:
        output.extend(detect_primary_interaction(primair_sents[0], solution))
    else:
        output.append(' -de primaire verklaring wordt niet genoemd')
    # EXPLICIETE ALTERNATIEVE VERKLARINGEN HOEVEN NIET BIJ INTERACTIE, STATISMogelijke alternatieve verklaringen zijn storende variabelen en omgekeerde causaliteitTIEK VOOR DE PSYCHOLOGIE 3 PAGINA 80
    if not control:
        alt_sents = [x for x in doc.sents if 'alternatieve' in [y.text for y in x]]
        if alt_sents != []:
            output.extend(detect_alternative_interaction(alt_sents[0], solution))
        else:
            output.append(' -de mogelijkheid van alternatieve verklaringen wordt niet genoemd')
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return False, 'Mooi, deze interpretatie klopt. ' if prefix else ''
    else:
        return True, '<br>'.join(output)

def scan_predictors(doc:Doc, solution:dict, prefix:bool=True):
    tokens = [x.text for x in doc]
    output = ['Er ontbreekt nog wat aan je antwoord, namelijk:'] if prefix else []
    varnames = [x.lower() for x in solution['data']['predictoren'][1:]] if solution['assignment_type'] == 6 \
                    else [x.lower() for x in solution['data']['predictoren']]
    for x in varnames:
        if ' ' in x:
            names = x.split()
            if not all([y in tokens for y in names]):
                output.append(' -predictor ' + x + ' niet genoemd.')
        else:
            if not x in tokens:
                output.append(' -predictor ' + x + ' niet genoemd.')
    for i in range(len(varnames)):
        if solution['predictor_p'][i] < 0.05:
            output.extend(detect_p(doc, solution['predictor_p'][i], label=varnames[i]))
    correct:bool = len(output) == 1 if prefix else output == []
    if correct:
        return False, 'Mooi, deze interpretatie klopt. ' if prefix else ''
    else:
        return True, '<br>'.join(output)

def scan_design(doc:Doc, solution:dict, prefix:bool=True) -> [bool, List[str]]:
    criteria = ['ind', 'indcorrect','ind2','ind2correct','dep','depcorrect','factor1','factor2']
    scorepoints = dict([(x,False) for x in criteria])
    if solution['assignment_type'] != 13:
        scorepoints['factor1'] = True;scorepoints['factor2'] = True
    output:List[str] = []
    indeps = [x for x in doc.sents if solution['independent'] in x.text or any([y in x.text for y in solution['ind_syns']])]#if x.text == solution['independent']]
    if indeps != []:
        scorepoints['ind'] = True
        indep_span = indeps[0]
        scorepoints['indcorrect'] = 'onafhankelijke' in indep_span.text or 'factor' in indep_span.text
        if solution['assignment_type'] == 13:
            scorepoints['factor1'] = 'within-subject' in indep_span.text or 'within' in indep_span.text
    if solution['assignment_type'] == 2 or solution['assignment_type'] == 13:    
        indeps2 = [x for x in doc.sents if solution['independent2'] in x.text or any([y in x.text for y in solution['ind2_syns']])]
        if indeps2 != []:
            scorepoints['ind2'] = True
            indep2_span = indeps2[0]
            scorepoints['ind2correct'] = 'onafhankelijke' in indep2_span.text or 'factor' in indep2_span.text 
            if solution['assignment_type'] == 13:
                scorepoints['factor2'] = 'between-subject' in indep2_span.text or 'between' in indep2_span.text
    else:
        scorepoints['ind2'] = True;scorepoints['ind2correct'] = True
    deps = [x for x in doc.sents if solution['dependent'] in x.text or any([y in x.text for y in solution['dep_syns']])]
    if deps != []:
        scorepoints['dep'] = True
        dep_span = deps[0]
        scorepoints['depcorrect'] = 'afhankelijke' in dep_span.text and not 'onafhankelijke' in dep_span.text 
    
    #Add feedback text
    if not scorepoints['ind']:
        output.append(' -de onafhankelijke variabele wordt niet genoemd in het design')
    if not scorepoints['indcorrect'] and scorepoints['ind']:
        output.append(' -de rol van de onafhankelijke variabele wordt niet juist genoemd in het design')
    if not scorepoints['ind2']:
        output.append(' -de tweede onafhankelijke variabele wordt niet genoemd in het design')
    if not scorepoints['ind2correct'] and scorepoints['ind2']:
        output.append(' -de rol van de tweede onafhankelijke variabele wordt niet juist genoemd in het design')
    if not scorepoints['dep']:
        output.append(' -de afhankelijke variabele wordt niet genoemd in het design')
    if not scorepoints['factor1']:
        output.append(' -niet genoemd wat voor factor de eerste onafhankelijke variabele is')
    if not scorepoints['factor2']:
        output.append(' -niet genoemd wat voor factor de tweede onafhankelijke variabele is')
    if not scorepoints['depcorrect'] and scorepoints['dep']:
        output.append(' -de rol van de afhankelijke variabele wordt niet juist genoemd in het design')
    if not False in list(scorepoints.values()):        
        return False, 'Mooi, dit design klopt.' if prefix else ''
    else:
        return True, '<br>'.join(output)
    
def scan_design_manova(doc:Doc, solution:dict, prefix:bool=True):
    text = doc.text
    scorepoints = {'indcorrect':False,
                   #'levels1':all([x in text for x in solution['levels']]),
                   'mes':all([solution[x] in text for x in ['dependent','dependent2','dependent3']]),
                   'dep1':False,
                   'dep2':False,
                   'dep3':False
                   }
    scorepoints['indcorrect'] = any([True if solution['independent'] in sent.text and ('factor' in sent.text \
                                    or 'onafhankelijke' in sent.text) else False for sent in doc.sents])
    deps = [x for x in doc if x.text == solution['dependent'].lower()]
    if deps != []:
        dep_span = descendants(deps[0].head)
        scorepoints['dep1'] = 'afhankelijke' in [x.text for x in dep_span] #and not 'onafhankelijke' in [x.text for x in dep_span]
    deps2 = [x for x in doc if x.text == solution['dependent'].lower()]
    if deps2 != []:
        dep2_span = descendants(deps2[0].head)
        scorepoints['dep2'] = 'afhankelijke' in [x.text for x in dep2_span] #and not 'onafhankelijke' in [x.text for x in dep2_span] 
    deps3 = [x for x in doc if x.text == solution['dependent'].lower()]
    if deps3 != []:
        dep3_span = descendants(deps3[0].head)
        scorepoints['dep3'] = 'afhankelijke' in [x.text for x in dep3_span] #and not 'onafhankelijke' in [x.text for x in dep3_span] 
    
    output:List[str] = []
    if not scorepoints['dep1']:
        output.append(' -eerste afhankelijke variabele niet juist genoemd')
    if not scorepoints['dep2']:
        output.append(' -tweede afhankelijke variabele niet juist genoemd')
    if not scorepoints['dep3']:
        output.append(' -derde afhankelijke variabele niet juist genoemd')
    if not scorepoints['indcorrect']:
        output.append(' -de onafhankelijke variabele wordt niet juist genoemd')
    elif not scorepoints['mes']:
        output.append(' -niet alle aparte afhankelijke variabelen juist genoemd')
    if not False in list(scorepoints.values()):
        return False, 'Mooi, dit design klopt.' if prefix else ''
    else:
        return True, '<br>'.join(output)

def split_grade_ttest(text: str, solution:dict, between_subject:bool) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc,solution))
    output += '<br>' + scan_design(doc, solution, prefix=False)[1]
    #if solution['p'][0] < 0.05:
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'T', solution['T'][0], aliases=['T(' + solution['independent'] + ')']))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'p', solution['p'][0]))
    output += '<br>' + scan_decision(doc, solution, anova=False, prefix=False, elementair=False)[1]
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)
    
    
def split_grade_anova(text: str, solution:dict, two_way:bool) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc,solution))
    output += '<br>' + scan_design(doc, solution, prefix=False)[1]
    if not two_way:
        if solution['p'][0] < 0.05:
            output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F'][0], aliases=['F(' + solution['independent'] + ')']))
            output += '<br>'+'<br>'.join(detect_p(doc, solution['p'][0]))
            output += '<br>'+'<br>'.join(detect_report_stat(doc, 'R<sup>2</sup>', solution['r2'][0], aliases=['r2','r','kwadraat']))
        output += '<br>' + scan_decision(doc, solution, anova=True, prefix=False, elementair=False)[1]
    else:
        for i in range(3):
            if solution['p'][i] < 0.05:
                f_aliases = ['F(' + solution['independent' + str(i+1)] + ')'] if i > 0 and i < 2 else []
                output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F'][i], aliases=f_aliases, num=i+1))
                output += '<br>'+'<br>'.join(detect_p(doc, solution['p'][i], num=i+1))
                output += '<br>'+'<br>'.join(detect_report_stat(doc, 'R<sup>2</sup>', solution['r2'][i], aliases=['r2','r','kwadraat'], num=i+1))
            #Find right decision
            varss = [solution['independent'],solution['independent2'],'interactie']
            levels = [solution['levels'],solution['levels2'],['interactie']]
            comparisons = ['ongelijk','gelijk','anders','verschillend'] if i < 2 else ['']
            decision_sent = [x for x in doc.sents if any([y in x.text for y in comparisons]) \
                             and all([y.lower() in x.text for y in levels[i]])]
            if decision_sent != []: 
                if i < 2:
                    output += '<br>' + scan_decision(decision_sent[0], solution, anova=True, num=i+1, prefix=False, elementair=False)[1]
                else:
                    output += '<br>' + scan_decision_anova(decision_sent[0], solution, num=i+1, prefix=False, elementair=False)[1]
            else:
                varss = [solution['independent'],solution['independent2'],'interactie']
                output += '<br> -de beslissing van ' + varss[i] + ' niet genoemd'
            
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)
        
def split_grade_rmanova(text: str, solution:dict) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc,solution))
    output += '<br>' + scan_design(doc, solution, prefix=False)[1]
    if solution['p'][0] < 0.05:
        output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F'][0], aliases=['F(' + solution['independent'] + ')']))
        output += '<br>'+'<br>'.join(detect_p(doc, solution['p'][0]))
        output += '<br>'+'<br>'.join(detect_report_stat(doc, 'R<sup>2</sup>', solution['r2'][0], aliases=['r2','r','kwadraat']))
    output += '<br>' + scan_decision(doc, solution, anova=True, num=1, prefix=False, elementair=False)[1]
    output += '<br>' + scan_decision_rmanova(doc, solution, num=2, prefix=False, elementair=False)[1]
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)
        
def split_grade_mregression(text:str, solution:dict) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc,solution))
    output += '<br>'+'<br>'.join(detect_comparison_mreg(doc, solution))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F'][0]))
    output += '<br>'+'<br>'.join(detect_p(doc, solution['p'][0]))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'R<sup>2</sup>', solution['r2'][0], aliases=['r2','r','kwadraat']))
    output += '<br>' + scan_predictors(doc, solution, prefix=False)[1]
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)

def split_grade_ancova(text:str, solution:dict) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc,solution))
    output += '<br>' + scan_design(doc, solution, prefix=False)[1]
    output += '<br>' + scan_predictors(doc, solution, prefix=False)[1]
    
    multivar_sent = [x for x in doc.sents if 'voorspellende waarde' in x.text]
    if multivar_sent != []:
        output += '<br>'+'<br>'.join(detect_decision_ancova(multivar_sent[0], solution))
        if(solution['p'][3] < 0.05):
            output += '<br>'+'<br>'.join(detect_effect(multivar_sent[0],solution, variable='multivariate', p=solution['p'][3], eta=solution['eta'][3]))
    else:
        output += '<br> -niet genoemd of het model een significant voorspellende waarde heeft'
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F'][3]))
    output += '<br>'+'<br>'.join(detect_p(doc, solution['p'][3]))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'eta<sup>2</sup>', solution['eta'][3], aliases=['eta2','eta']))
    
    #print(str(solution['F'][3]) + ' - '+ str(solution['p'][3]) + ' - '+ str(solution['eta'][3]))
    between_sent = [x for x in doc.sents if (solution['independent'] in x.text or 'between-subject' in x.text) and ('significant' in x.text or 'effect' in x.text) and not 'voorspellend' in x.text]
    if between_sent != []:
        output += '<br>'+'<br>'.join(detect_decision_multirm(between_sent[0], solution, solution['independent'], ['between-subject'], solution['p'][2],solution['eta'][2]))
        if(solution['p'][2] < 0.05):
            output += '<br>'+'<br>'.join(detect_effect(between_sent[0],solution, variable=solution['independent'], p=solution['p'][2], eta=solution['eta'][2]))
            output += '<br>'+'<br>'.join(detect_p(doc, solution['p'][2], label=solution['independent']))
    else:
        output += '<br> -de beslissing van de between-subjectfactor wordt niet genoemd'
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)
 
def split_grade_manova(text:str, solution:dict) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc,solution))
    output += '<br>' + scan_design_manova(doc, solution, prefix=False)[1]
    for i in range(3):
        var_key = 'dependent' if i < 1 else 'dependent' + str(i+1)
        if solution['p_multivar'] < 0.05 and solution['p_multivar'] < 0.05:
            decision_sent = [x for x in doc.sents if solution[var_key] in x.text and ('significant' in x.text or 'effect' in x.text)]
            if decision_sent != []:
                output += '<br>'+'<br>'.join(detect_decision_manova(decision_sent[0],solution, variable=solution[var_key], synonyms=[], p=solution['p'+str(i)][0], eta=solution['eta'+str(i)][0], num=1))
                if solution['p'+str(i)][0] < 0.05:
                    output += '<br>'+'<br>'.join(detect_effect(decision_sent[0],solution, variable=solution[var_key], p=solution['p'+str(i)][0], eta=solution['eta'+str(i)][0]))
            else:
                output += '<br> -de beslissing van '+solution[var_key]+' wordt niet genoemd'
        if solution['p'+str(i)][0] < 0.05 and solution['p_multivar'] < 0.05:
            output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F'+str(i)][0], appendix='bij '+solution[var_key]+' '))
            output += '<br>'+'<br>'.join(detect_p(doc, solution['p'+str(i)][0], label=solution[var_key]))
            output += '<br>'+'<br>'.join(detect_report_stat(doc, 'eta<sup>2</sup>', solution['eta'+str(i)][0], aliases=['eta','eta2','eta-kwadraat'],appendix='bij '+solution[var_key]+' '))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F_multivar'], appendix='bij de multivariate beslissing '))
    output += '<br>'+'<br>'.join(detect_p(doc, solution['p_multivar'], label='de multivariate beslissing '))
    output += '<br>'+'<br>'.join(detect_report_stat(doc, 'eta<sup>2</sup>', solution['eta_multivar'], aliases=['eta','eta2','eta-kwadraat'], appendix='bij de multivariate beslissing '))
    decision_sent = [x for x in doc.sents if ('multivariate' in x.text or 'multivariaat' in x.text) \
                         and ('significant' in x.text or 'effect' in x.text)]
    if decision_sent != []:
        output += '<br>'+'<br>'.join(detect_decision_manova(decision_sent[0],solution,variable='multivariaat',synonyms=['multivariate'], p=solution['p_multivar'], eta=solution['eta_multivar'], num=0))
        output += '<br>'+'<br>'.join(detect_effect(decision_sent[0],solution, variable='multivariaat', p=solution['p_multivar'], eta=solution['eta_multivar']))
    else:
        output += '<br> -de multivariate beslissing wordt niet genoemd'
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)

def split_grade_multirm(text:str, solution:dict) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    output:str = '';num:int = 0
    output += '<br>'+'<br>'.join(detect_name(doc,solution))
    output += '<br>'+scan_design(doc,solution,prefix=False)[1]
    
    #Multivar within subject
    decision_sent = [x for x in doc.sents if (solution['independent'] in x.text or 'within-subject' in x.text) \
                         and ('significant' in x.text or 'effect' in x.text) and 'interactie' not in x.text]
    if decision_sent != []: 
        num += 1
        user_given_name:str = solution['independent'] if solution['independent'] in decision_sent[0].text else 'within-subject'
        output += '<br>'+'<br>'.join(detect_decision_multirm(decision_sent[0],solution,variable=user_given_name,synonyms=['multivariate within-subject'], p=solution['p0'][0], eta=solution['eta0'][0]))
        output += '<br>'+'<br>'.join(detect_effect(decision_sent[0],solution, variable=solution['independent'], p=solution['p0'][0], eta=solution['eta0'][0]))
    else:
        output += '<br> -de multivariate within-subject beslissing wordt niet genoemd'
    if solution['p0'][0] < 0.05:
        output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F0'][0], appendix='bij de within-subject factor '))
        output += '<br>'+'<br>'.join(detect_p(doc, solution['p0'][0], label='bij de within-subject factor '))
        output += '<br>'+'<br>'.join(detect_report_stat(doc, 'eta<sup>2</sup>', solution['eta0'][0], aliases=['eta','eta2','eta-kwadraat'],appendix='bij de within-subject factor '))
        if solution['p1'][0] < 0.05:
            output += '<br>'+'<br>'.join(detect_p(doc, solution['p1'][0], label='van het contrast tussen '+solution['levels'][0]+' en '+solution['levels'][1]+' '))
        if solution['p1'][1] < 0.05:
            output += '<br>'+'<br>'.join(detect_p(doc, solution['p1'][1], label='van het contrast tussen '+solution['levels'][1]+' en '+solution['levels'][2]+' '))
    
    #Multivar interaction
    decision_sent2 = [x for x in doc.sents if ('interactie' in x.text) and ('significant' in x.text or 'effect' in x.text)]
    if decision_sent2 != []:
        num += 1
        output += '<br>'+'<br>'.join(detect_decision_multirm(decision_sent2[0],solution,variable='interactie',synonyms=[], p=solution['p0'][1], eta=solution['eta0'][1]))
        output += '<br>'+'<br>'.join(detect_effect(decision_sent2[0],solution, variable='interactie', p=solution['p0'][1], eta=solution['eta0'][1]))
    else:
        output += '<br> -de multivariate interactiebeslissing wordt niet genoemd'
    if solution['p0'][1] < 0.05:
        output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F0'][1], appendix='bij de interactie '))
        output += '<br>'+'<br>'.join(detect_p(doc, solution['p0'][1], label='bij de interactie '))
        output += '<br>'+'<br>'.join(detect_report_stat(doc, 'eta<sup>2</sup>', solution['eta0'][1], aliases=['eta','eta2','eta-kwadraat'],appendix='bij de interactie '))
        if solution['p1'][2] < 0.05:
            output += '<br>'+'<br>'.join(detect_p(doc, solution['p1'][2], label='van het contrast tussen '+solution['levels'][0]+' en '+solution['levels'][1]+' bij de interactie '))
        if solution['p1'][3] < 0.05:
            output += '<br>'+'<br>'.join(detect_p(doc, solution['p1'][3], label='van het contrast tussen '+solution['levels'][1]+' en '+solution['levels'][2]+' bij de interactie '))
    
    #Between-subject
    decision_sent3 = [x for x in doc.sents if (solution['independent2'] in x.text or 'between-subject' in x.text) and ('significant' in x.text or 'effect' in x.text) and 'interactie' not in x.text]
    if decision_sent3 != []:
        num += 1
        user_given_name:str = solution['independent2'] if solution['independent2'] in decision_sent3[0].text else 'between-subject'
        output += '<br>'+'<br>'.join(detect_decision_multirm(decision_sent3[0],solution,variable=user_given_name,synonyms=['multivariate between-subject'], p=solution['p'][1], eta=solution['eta'][1]))
        output += '<br>'+'<br>'.join(detect_effect(decision_sent3[0],solution, variable=solution['independent2'], p=solution['p'][1], eta=solution['eta'][1]))
    else:
        output += '<br> -de between-subject beslissing wordt niet genoemd'
    if solution['p'][1] < 0.05:
        output += '<br>'+'<br>'.join(detect_report_stat(doc, 'F', solution['F'][1], appendix='bij de between-subject factor '))
        output += '<br>'+'<br>'.join(detect_p(doc, solution['p'][1], label='bij de between-subject factor '))
        output += '<br>'+'<br>'.join(detect_report_stat(doc, 'eta<sup>2</sup>', solution['eta'][1], aliases=['eta','eta2','eta-kwadraat'],appendix='bij de between-subject factor '))
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)

def split_grade_multirm2(text:str, solution:dict) -> str:
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text.lower())
    output:str = ''
    output += '<br>'+'<br>'.join(detect_name(doc,solution))
    if output.replace('<br>','') == '':
        return 'Mooi, dit beknopt rapport bevat alle juiste details!'
    else:
        return 'Er ontbreekt nog wat aan je antwoord, namelijk:' + re.sub(r'<br>(<br>)+', '<br>', output)
    
"""
FUNCTIONS FOR TESTING
"""
def print_dissection(text:str):
    import spacy
    nl_nlp = spacy.load('nl')
    doc = nl_nlp(text)
    print([(x.text, x.dep_) for x in doc])
    
def load_dissection(text:str):
    import spacy
    from spacy import displacy
    nlp = spacy.load('nl')
    doc = nlp(text)
    displacy.serve(doc, style='dep')
        
