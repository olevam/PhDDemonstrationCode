import DatabaseConnect  
import pandas as pd
from matplotlib import pyplot as plt 

dc = DatabaseConnect.connect('Gamburigan.147' ,'womeninmilanphd')
def get_monasteires (type):
    query_male = '''
    SELECT monasteryname FROM monastery
    WHERE type_instiution IN ('2')
   
    '''
    query_female = '''
    SELECT monasteryname FROM monastery
    WHERE type_instiution IN ('1')
    '''
    query_all = '''
    SELECT monasteryname FROM monastery
    WHERE type_instiution IN ('1', '2')
    '''
    if type == 'm':
        dc.execute(query_male)
        list = []
        for x in dc.fetchall():
            list.append(x[0])
    if type == 'f':
        dc.execute(query_female)
        list = []
        for x in dc.fetchall():
            list.append(x[0])
    if type == 'a':
        dc.execute(query_all)
        list = []
        for x in dc.fetchall():
            list.append(x[0])
    return tuple(list)


def humiliati (type):
    query_male = '''
    SELECT monasteryname FROM monastery
    WHERE type_instiution IN ('2')
    AND monasteryname LIKE '%miliat%' 
   
    '''
    query_female = '''
    SELECT monasteryname FROM monastery
    WHERE type_instiution IN ('1') AND monasteryname LIKE '%miliat%' 
    '''
    query_all = '''
    SELECT monasteryname FROM monastery
    WHERE monasteryname LIKE '%miliat%' 
    '''
    if type == 'm':
        dc.execute(query_male)
        list = []
        for x in dc.fetchall():
            list.append(x[0])
    if type == 'f':
        dc.execute(query_female)
        list = []
        for x in dc.fetchall():
            list.append(x[0])
    if type == 'a':
        dc.execute(query_all)
        list = []
        for x in dc.fetchall():
            list.append(x[0])
    return tuple(list)

#this two lists of institutions form six case studies in chapter three
maleMon = ['Ambrogio', 'Giorgio_Palazzo', 'MariaDelMonte']
femaleMon = ['Monastero Maggiore', 'Apollinare', 'Lentasio']

