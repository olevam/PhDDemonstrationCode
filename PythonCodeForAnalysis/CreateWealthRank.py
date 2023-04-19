#This script creates wealth rank for each monastery in the database and stores it in an xlsx file.
#The xlsx file forms the basis for some calculations in AllQueriesForPhD.py script and in the InformationOnInstitutions.py script.
#Run this script to recreate an overall wealth frame, one with only male institutions, one with only female institutions, one with only the top 20 male and female institutions, and one with only the membership for male and female institutions. 


import InformationOnInstitutions as im
import DatabaseConnect  
dc = DatabaseConnect.connect('Gamburigan.147' ,'womeninmilanphd')

import pandas as pd
import networkx as nx

male = im.get_monasteires('m')
female = im.get_monasteires('f')
all = im.get_monasteires('a')
directoryToSaveGraphs = 'Data'

def price_total (monastery):
    price_query = '''
    SELECT SUM (price) FROM price 
    JOIN alldocuments ON price.docid = alldocuments.docid 
    WHERE alldocuments.docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments 
    JOIN actor ON alldocuments.docid = actor.docid
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monasteryname = %s)
    AND doctype = '6'
    '''
    dc.execute(price_query, [monastery])
    price = dc.fetchall()[0][0]
    if price == None:
        price = 0
    return int(price)

def totalPopulation(monatery, classification):
    query = '''
        SELECT COUNT (*) FROM alldocuments
        JOIN actor ON alldocuments.docid = actor.docid
        JOIN classification ON classification.classid = actor.classification
        JOIN monastery ON monastery.monasteryid = actor.monastery
        WHERE   classification.classification = %s AND monastery.monasteryname = %s
        GROUP BY year, alldocuments.docid
        ORDER BY COUNT desc
    '''
    dc.execute(query, [classification, monatery])
    try:
        total = dc.fetchall()[0][0]
    except IndexError:
        total = 0
    return total
    
def Obtain_network_data (mon = ''):
    all_monasteries = '''
    SELECT docid, monasteryname FROM actor 
    JOIN monastery ON monastery = monasteryid
    WHERE type_instiution IN ('1','2')
    GROUP BY monasteryname, docid
    ORDER BY docid
    '''
    specific_mon = '''
       SELECT docid, monasteryname FROM actor 
    JOIN monastery ON monastery = monasteryid
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments 
    JOIN actor ON alldocuments.docid = actor.docid
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monasteryname IN %s)
    AND type_instiution IN ('1','2')
    GROUP BY monasteryname, docid
    ORDER BY docid
    '''
    if mon == '':    
        dc.execute(all_monasteries)
        all_mon = dc.fetchall()
    else:
        dc.execute(specific_mon, [mon])
        all_mon = dc.fetchall()

    connected_monasteries = []
    
    for docid, monastery in all_mon:
        for docid2, monastery2 in all_mon:
            if docid == docid2 and monastery != monastery2:
                    connected_monasteries.append((monastery, monastery2))    
    return connected_monasteries


def create_network (mon = ''):
    mon_network_graph = nx.Graph()
    if mon =='':
        mon_network_graph.add_edges_from(Obtain_network_data())
    else:
        mon_network_graph.add_edges_from(Obtain_network_data(mon))
    return mon_network_graph


def neighbours_network (monastery):
    important_neighbors = 0
    tied_neighbors = 0
    mon_network_graph = create_network()
    if monastery in mon_network_graph:
        for neigh in mon_network_graph.neighbors(monastery):
            important_neighbors += 1
        return important_neighbors
    else:
        return (0)

def location_size (monastery):
    # different localities
    total = '''
    SELECT COUNT (DISTINCT coordid) FROM land_loc 
    JOIN alldocuments ON land_loc.docid = alldocuments.docid 
    WHERE alldocuments.docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments 
    JOIN actor ON alldocuments.docid = actor.docid
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monasteryname = %s)
    AND doctype = '11'
   
    '''
    dc.execute(total, [monastery])
    total = dc.fetchall()[0][0]

    #total localities
    location = '''
    SELECT COUNT (*) FROM land_loc 
    JOIN alldocuments ON land_loc.docid = alldocuments.docid 
    WHERE alldocuments.docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments 
    JOIN actor ON alldocuments.docid = actor.docid
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monasteryname = %s)
    AND doctype = '11'
    '''
    dc.execute(location, [monastery])
    location = dc.fetchall()[0][0]
    if total != 0:
        average = location/total
    else:
        average = 0 

    return location, average, total

def concentration (monastery):
    def average_mentions (monastery):
            query = '''
            SELECT COUNT (*), coordid  FROM land_loc 
            JOIN alldocuments ON land_loc.docid = alldocuments.docid 
            WHERE alldocuments.docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments 
            JOIN actor ON alldocuments.docid = actor.docid
            JOIN monastery ON monastery.monasteryid = actor.monastery
            WHERE monasteryname = %s)
            GROUP BY coordid
            ORDER BY count desc
            '''
            dc.execute(query, [monastery])
            count_list = []
            for count, location in dc.fetchall():
                count_list.append(count)
            return count_list

    def get_top (monastery):
        list = average_mentions(monastery)
        if len(list) >= 3:
            top = list[0] + list[1] + list [2]
            return top/sum(list, 0)
        else:
            return 0

    return get_top(monastery)

def doc_number (monastery):
    doc_number = '''
    SELECT COUNT (*) FROM alldocuments
    WHERE alldocuments.docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments 
    JOIN actor ON alldocuments.docid = actor.docid
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monasteryname = %s)
    '''
    dc.execute(doc_number, [monastery])
    docnumber = dc.fetchall()[0][0]
    return docnumber


def normalize(variable):
   return [float(i)/max(variable) for i in variable]

def ranking_frame (monastery_list):

    price_list = []
    mon_list = []
    location_list = []
    concetration_list =[]
    doc_number_list = []
    indepent_list = []
    adjusted_list = []
    memberList = []
    mon_network_graph = create_network()
    for monastery in monastery_list:   
        Neigh = 0  
        if monastery in mon_network_graph:
            for x in mon_network_graph.neighbors(monastery):
                Neigh += 1
            indepent_list.append(Neigh)
        else:
            indepent_list.append (0)
        if monastery in female:
            memberList.append(totalPopulation(monastery, 'Nun'))
        elif monastery in male:
            memberList.append(totalPopulation(monastery, 'Clergymen'))
        num_locations, average, tot_individual = location_size(monastery)
        price_list.append(price_total(monastery))
        mon_list.append(monastery)
        doc_number_list.append(doc_number(monastery)) 
        location_list.append(num_locations)
        concetration_list.append(concentration(monastery))
        adjusted_list.append(average)
 
    ranking_frame = pd.DataFrame()
    ranking_frame['Monastery'] = mon_list
    ranking_frame['Denari_spent'] = price_list
    ranking_frame['Denari_spent_standard'] = normalize(price_list)
    ranking_frame['Loc_number'] = location_list
    ranking_frame['Loc_number_standard'] = normalize(location_list)
    ranking_frame['Concentration'] = concetration_list
    ranking_frame['Adjusted_location_number'] = adjusted_list
    ranking_frame['Adjusted_location_number_standard'] = (normalize(adjusted_list))
    ranking_frame['Number_documents'] = doc_number_list
    ranking_frame['Independent_list'] = indepent_list
    ranking_frame['Independent_list_standard'] = normalize(indepent_list)
    ranking_frame['Membership'] = memberList
    ranking_frame['Overall_wealth'] = ranking_frame['Independent_list_standard']+ ranking_frame['Denari_spent_standard'] + ranking_frame['Loc_number_standard']
    ranking_frame.sort_values(by='Overall_wealth', ascending= False, inplace=True)
    ranking_frame.reset_index(inplace= True, drop=True)
    return(ranking_frame)



def anl_frame ():
    frame = ranking_frame(all)
    male_frame = frame[frame['Monastery'].isin(male)]
    female_frame = frame[frame['Monastery'].isin(female)]
    return frame, male_frame, female_frame



def top20Membership ():
    all, male, female = anl_frame()
    male = male.iloc[0:10]
    male = male[['Monastery', 'Membership']]
    female = female.iloc[0:10]
    female = female[['Monastery', 'Membership']]
    top20 = pd.concat([male, female])
    top20.sort_values(by='Membership', ascending= False, inplace=True)
    top20.reset_index(inplace= True, drop=True)
    top20.index +=1
    return top20

def top20FullData():
    all, male, female = anl_frame()
    male = male.iloc[0:10]
    female = female.iloc[0:10]
    top20 = pd.concat([male, female])
    return top20

mainFrame, maleFrame, femaleFrame = anl_frame()
mainFrame.to_excel(  f'{directoryToSaveGraphs}/MainFrame.xlsx')
femaleFrame.to_excel(f'{directoryToSaveGraphs}/FemaleFrame.xlsx')
maleFrame.to_excel(  f'{directoryToSaveGraphs}/MaleFrame.xlsx')
top20Membership().to_excel(f'{directoryToSaveGraphs}/Top20Membership.xlsx')
top20FullData().to_excel(f'{directoryToSaveGraphs}/Top20FullData.xlsx')