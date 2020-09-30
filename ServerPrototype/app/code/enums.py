from enum import Enum

#Enum of how representing the five report types, plus two numbers representing the generic input field and the
#introductory page where there is no input field. These are used for the attributes "table_shape" and "submit_field"
#With slightly different meanings. In "submit_field", the represent the shape of the page input, as a text field or table
#In table_shape, they represent the report types in general.
class Task(Enum):
    TEXT_FIELD = 0
    TTEST_BETWEEN = 1
    TTEST_WITHIN = 2
    ONEWAY_ANOVA = 3
    TWOWAY_ANOVA = 4
    WITHIN_ANOVA = 5
    INTRO = 6
    
#Variable indicating whether and how the given function from the protocol index should be called
#e.g. as bool, bool, or bool, str, or not at all
class Process(Enum):
    INTRO = 0
    QUESTION = 1
    LAST_QUESTION = 2
    YES_NO = 3
    CHOOSE_REPORT = 4
    TABLE = 5
    ANOTHER = 6
