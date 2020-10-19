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
    REPORT = 6
    INTRO = 7
    
#Variable indicating whether and how the given function from the protocol index should be called
#e.g. as bool, bool, or bool, str, or not at all
class Process(Enum):
    INTRO = 0
    QUESTION = 1
    LAST_QUESTION = 2
    YES_NO = 3
    CHOOSE_ANALYSIS = 4
    CHOOSE_REPORT = 5
    TABLE = 6
    ANOTHER = 7
