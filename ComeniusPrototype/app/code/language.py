#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 18:41:13 2021

@author: jelmer
"""

class LanguageInterface:
    def __init__(self):
        path = '/home/jelmer/Github/ComeniusPrototype/ComeniusPrototype/app/code/texts.csv' if 'Github' in os.getcwd() else '/var/www/ComeniusPrototype/ComeniusPrototype/app/code/texts.csv'
        dutch_pairs = []
        english_pairs = []
        with open(path) as file:
            for line in file.readlines():
                parts = line.split('","')
                dutch_pairs.append((parts[0][1:], parts[1]))
                english_pairs.append(((parts[0][1:], parts[2][:-2])))
        self.dutch = dict(dutch_pairs)
        self.english = dict(english_pairs)
            
    def get_messages(self, english:str):
        if english:
            return self.english
        else:
            return self.dutch
