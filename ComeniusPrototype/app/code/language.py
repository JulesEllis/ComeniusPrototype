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
Created on Tue Feb  9 18:41:13 2021

@author: jelmer
"""
import os

class LanguageInterface:
    def __init__(self, mes:dict=None):
        path = '/home/jelmer/Github/ComeniusPrototype/ComeniusPrototype/app/code/messages/' if 'Github' in os.getcwd() \
                        else '/var/www/ComeniusPrototype/ComeniusPrototype/app/code/messages/'
        dutch_pairs = [('L_ENGLISH',False)]
        english_pairs = [('L_ENGLISH',True)]
        for t in ['texts.csv']:
            with open(path+t, encoding='utf-8', errors='ignore') as file:
                for line in file.readlines():
                    parts = line.split(';')
                    dutch_pairs.append((parts[0], parts[1]))
                    english_pairs.append(((parts[0], parts[2][:-1])))
        for key, t in [('EXPLAIN_SHORT','sjabloon_beknopt.csv'),('B1_SHORT','Sjabloon_beknopt_Docent.csv'),
                       ('B2_SHORT','Sjabloon_beknopt_Student.csv'),('B3_SHORT','Sjabloon_beknopt_Werkgroepbegeleider.csv')]:
            with open(path+t, encoding='utf-8', errors='ignore') as file:
                engparts = []
                dutchparts = []
                for line in file.readlines():
                    parts = line.split(';')
                    dutchparts.append((parts[0], parts[1]))
                    engparts.append(((parts[0], parts[2][:-1])))
                dutch_pairs.append((key+'_NL', dict(dutchparts)))
                english_pairs.append((key+'_EN', dict(engparts)))
        for key, t in [('EXPLAIN_ELEM','sjabloon_elementair.csv'),('B1_ELEM','Sjabloon_elementair_Docent.csv'),
                       ('B2_ELEM','Sjabloon_elementair_Student.csv'),('B3_ELEM','Sjabloon_elementair_Werkgroepbegeleider.csv')]:
            #print(t)
            with open(path+t, encoding='utf-8', errors='ignore') as file:
                analyses = ['TBETWEEN','TWITHIN','1ANOVA','2ANOVA','RMANOVA']
                elab_nl = dict([(an,{}) for an in analyses])
                elab_en = dict([(an,{}) for an in analyses])
                for line in file.readlines():
                    parts = line.split(';')
                    tag = parts[0]
                    #print(line)
                    for i in range(1,11):
                        #print(i)
                        if i % 2 == 1:
                            elab_nl[analyses[(i-1) // 2]][tag] = parts[i]
                        else:
                            elab_en[analyses[(i-1) // 2]][tag] = parts[i]
                dutch_pairs.append((key+'_NL', elab_nl))
                english_pairs.append((key+'_EN', elab_en))
        self.dutch = dict(dutch_pairs)
        self.english = dict(english_pairs)
            
    def get_messages(self, english:str):
        if english:
            return self.english
        else:
            return self.dutch
LanguageInterface()
