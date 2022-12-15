#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 17:16:58 2022

@author: jelmer
"""
from datetime import datetime, timedelta
import secrets

class EncryptedCodeSaver:
    def __init__(self):
        self.key = "simpelesleutelvoortestenvanpythoncode349ms49"
        self.offset = 44
        self.analysis_name_dict = {1:'TTEST_BETWEEN',
                            2:' TTEST_WITHIN', 
                            3:'ONEWAY_ANOVA',
                            4:'TWOWAY_ANOVA',
                            5:'WITHIN_ANOVA', 
                            6:'MREGRESSION', 
                            11:'MANOVA',
                            12:'ANCOVA',
                            13:'MULTIRM',
                            14:'MULTIRM2'}
    
    def dec2hex(self,dec) -> str:
        hexstr: str = hex(dec)
        returnstr:str = hexstr[2:len(hexstr)]
        if len(returnstr)<2:
            returnstr = "0"+returnstr
        return returnstr

    def encrypt(self, encdata) -> str:
        dataarr = bytes(encdata, 'utf-8')
        keyarr = bytes(self.key, 'utf-8')
        encoffset = secrets.randbits(8)
        returnstr = self.dec2hex(encoffset)
        keylen = len(self.key)
        for i in range(len(encdata)):
            keypos = (i + encoffset) % keylen
            newchar = (dataarr[i])^(keyarr[keypos])
            returnstr = returnstr + self.dec2hex(newchar)
        return returnstr
    
    def decrypt(self,encdata) -> str:
        dataarr = bytes(encdata[2::], 'utf-8')
        keyarr = bytes(self.key, 'utf-8')
        encoffset = int("0x"+encdata[0:2],16)
        keylen = len(self.key)
        returnstr = []
        for i in range(len(encdata)):
            datahex=encdata[2+2*i:2+2*i+2]
            databyte=int("0x"+datahex,16)
            keypos = (i + encoffset) % keylen
            newchar = databyte^(keyarr[keypos])
            returnstr.append(newchar)
        return str(bytes(returnstr), 'utf-8')
    
    def encrypt_assignment(self,assignment:dict):
        to_encrypt = ';'.join([self.analysis_name_dict[assignment['assignment_type']],str(assignment['n_mistakes']),str(assignment['feedback_requests']),str(datetime.now() + timedelta(hours=1))])
        return self.encrypt(to_encrypt)
        



#data = "2869597;9;8-12-2022;13:23:58"
#x = EncryptedCodeSaver()
#print(x.encrypt('2869597;9;8-12-2022;13:23:58'))