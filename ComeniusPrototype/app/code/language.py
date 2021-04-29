#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 18:41:13 2021

@author: jelmer
"""
import os

class LanguageInterface:
    def __init__(self, mes:dict=None):
        path = '/home/jelmer/Github/ComeniusPrototype/ComeniusPrototype/app/code/' if 'Github' in os.getcwd() else '/var/www/ComeniusPrototype/ComeniusPrototype/app/code/'
        dutch_pairs = [('L_ENGLISH',False)]
        english_pairs = [('L_ENGLISH',True)]
        with open(path+'texts.csv', encoding='utf-8', errors='ignore') as file:
            for line in file.readlines():
                parts = line.split(';')
                dutch_pairs.append((parts[0], parts[1]))
                english_pairs.append(((parts[0], parts[2][:-1])))
        with open(path+'tabel_uitlegcodes.csv', encoding='utf-8', errors='ignore') as file:
            for line in file.readlines():
                parts = line.split(';')
                dutch_pairs.append((parts[0], parts[1]))
                english_pairs.append(((parts[0], parts[2][:-1])))
        self.dutch = dict(dutch_pairs)
        self.english = dict(english_pairs)
            
    def get_messages(self, english:str):
        if english:
            return self.english
        else:
            return self.dutch
