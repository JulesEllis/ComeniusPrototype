#    This project contains CASWIC: Coaching App for Statistical Writing in Introductory Course.
#    Copyright (C) 2023 Jules Ellis and Jelmer Jansen
#
#    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.
#

from enum import Enum

#Enum of how representing the five report types, plus two numbers representing the generic input field and the
#introductory page where there is no input field. These are used for the attributes "report_type" and "submit_field"
#With slightly different meanings. In "submit_field", the represent the shape of the page input, as a text field or table
#In report_type, they represent the report types in general.
class Task(Enum):
    TEXT_FIELD = 0
    TTEST_BETWEEN = 1
    TTEST_WITHIN = 2
    ONEWAY_ANOVA = 3
    TWOWAY_ANOVA = 4
    WITHIN_ANOVA = 5
    MREGRESSION = 6
    REPORT = 7
    INTRO = 8
    CHOICE = 9
    TEXT_FIELD_LARGE = 10
    MANOVA = 11
    ANCOVA = 12
    MULTIRM = 13
    FINISHED = 15
    
#Variable indicating whether and how the given function from the protocol index (e.g. scan_dependent) should be called
#e.g. as bool, bool, or bool, str, or not at all
class Process(Enum):
    INTRO = 0
    QUESTION = 1
    LAST_QUESTION = 2
    CHOOSE_ANALYSIS = 3
    TABLE = 4
    FINISH = 5
