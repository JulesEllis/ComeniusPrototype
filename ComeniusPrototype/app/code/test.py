#    This project contains CASWIC: Coaching App for Statistical Writing in Introductory Course.
#    Copyright (C) 2023 Jules Ellis and Jelmer Jansen
#
#    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
#
import math
import numpy as np
import os
import spacy
import logging
import json
import math
import random
import numpy as np
import nltk
import spacy
import re
import nltk
import inflect
from copy import copy
from spacy.tokens.token import Token
from spacy.tokens.doc import Doc
from spacy import displacy
from nltk import CFG, Tree, edit_distance
from scipy import stats
from typing import Dict, List, Tuple
from scipy import stats
from typing import Dict, List, Callable, Tuple
from nltk.corpus import wordnet as wn

    def scan_indep_anova(self, text: str, solution: Dict, num: int=1, between_subject:bool=True, coach:bool=False) -> [bool, str]:
        #Determine which of the necessary elements are present in the answer
        texts: List[str] = nltk.word_tokenize(text.lower())
        n_key: str = 'independent' if num == 1 else 'independent' + str(num)
        scorepoints: Dict = {'factor': any(x in text for x in ['factor', 'between-subjectfactor','within-subjectfactor']), 
                       'domain': any(x in text for x in ['between','between-subject', 'between-subjectfactor']) if between_subject 
                       else any(x in text for x in ['within','within-subject', 'within-subjectfactor']), 
                       'name': lef(solution[n_key].get_all_syns(),texts), 
                       'levels': [any([x in text for x in y]) for y in solution[n_key].get_all_level_syns()]
                       }
        #Determine the response of the chatbot
        if False in list(scorepoints.values()):
            if not coach:
                output: str = self.mes['F_INCOMPLETE']+'<br>' #'Er ontbreekt nog wat aan je antwoord, namelijk:<br>'
                if not scorepoints['factor']:
                    output += self.mes['F_ISFACTOR']+'<br>' #' -de uitspraak dat deze variabele een factor is<br>'
                if not scorepoints['domain']:
                    output += self.mes['F_DOMAIN']+'<br>' #' -het domein van de variabele<br>'
                if not scorepoints['name']:
                    output += self.mes['F_VARNAME']+'<br>' #' -de naam van de variabele<br>'
                if True not in scorepoints['levels']:
                    output += self.mes['F_INDEPLEVELS']+'<br>' #' -alle niveaus van de onafhankelijke variabele<br>'
                elif False in scorepoints['levels']:
                    output += self.mes['F_INDEPLEVEL']+'<br>' #' -enkele niveaus van de onafhankelijke variabele<br>'
                return True, output
            else:
                scorekeys = {'factor':'dat het een factor is','domain':'het domein van de variabele','name':' de naam ','levels':' de niveaus van de variabele'}
                output: str = 'Mooi, je hebt ' + ' en '.join([if y for x, y in scorepoints.items]) + ' al opgenomen in je antwoord. '
                return True, output
        else:
            return False, self.mes['F_CORRECT'] #'Mooi, deze beschrijving klopt. '


