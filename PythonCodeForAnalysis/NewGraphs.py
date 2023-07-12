import InfoMon as im
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from Database_connecting import connecting as database 
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
from shapely.geometry import Point
from shapely.wkt import loads
import datetime as yr
maleMonasteries = ("Ambrogio", "CanonicaAmbrogio", "MilanArchdioces", "Chiaravalle", "Lorenzo", "Giorgio_Palazzo", "Simpliciano", "MariaDelMonte", "DecumaniMilanesi", "OspedaleBroletoMilano")
femaleMonasteries = ("Monastero Maggiore", "Apollinare", "Radegonda", "Lentasio", "MariaAurona", "Margherita", "Agnese", "SanFelice", "AmbrogioRivalta", "CappucineConcorezzo")
directory = 'C:/Users/Olevam/OneDrive - University of Glasgow/PhD_WomenInMilan/Python/AllGraphsFinalThesis/CreatedGraphs'
sidedirectory = 'C:/Users/Olevam/OneDrive - University of Glasgow/PhD_WomenInMilan/Writing/Drafts/Male_Leadership/Results/Regressions'
dc = database.connect('womeninmilanphd')
top20Frame = pd.read_excel('C:/Users/Olevam/OneDrive - University of Glasgow/PhD_WomenInMilan/Results/WealthData/Top20All.xlsx')
top20 = (maleMonasteries+femaleMonasteries)
top10Female = top20Frame[top20Frame['Monastery'].isin(femaleMonasteries)] 
top10Male = top20Frame[top20Frame  ['Monastery'].isin(maleMonasteries)]  

class typeOfGraph ():
    def __init__(self,column, label, title ) -> None:
        self.column = column
        self.label = label
        self.title = title

class frameSet ():
    def __init__(self, frame1, frame2, label1, label2, type) -> None:
        self.frame1 = frame1
        self.frame2 = frame2
        self.label1 =label1
        self.label2 =label2
        self.type = type

topTenFrame = frameSet (top10Male, top10Female, 'Male institutions', 'Female institutions', 'by top ten male and female institutions')     
removeOutliersFrame = frameSet (top10Male.iloc[2:10], top10Female.iloc[2:10], 'Male institutions', 'Female institutions', 'after trimming outliers')          
locationFrame = frameSet(top10Male.iloc[1:10], top10Female.iloc[1:10], 'Male institutions', 'Female institutions', 'after trimming outliers')

distanceDistribution = typeOfGraph('Coordinates', 'Distance', 'Distance Travelled')
locationDistribution = typeOfGraph('Loc_number', 'Number of investitures', 'locations invested')
denariDistribution = typeOfGraph('Denari_spent', 'Denari spent', 'denari spent')
networkDistribution = typeOfGraph('Independent_list', 'network size', 'Network')
overallDistribution = typeOfGraph('Overall_wealth', 'overall wealth', 'Overallwealth')
MembershipDistribution = typeOfGraph('Membership', 'Membership', 'membership')     
lombardia = gpd.read_file("F:/Python/Geopanda/REGIONE_LOMBARDIA/Regione_2020.shp").to_crs('EPSG:4326')

male_leaders = ('Abbott', 'Preposto', 'Archpriest', 'Archbishop')
female_leaders = ('Abbess',)
male_clergy = ('Clergymen',)
female_clergy = ('Nun',)
male_intermediaries = ('Intermediary', 'Laybrother')
maleagent = ('Intermediary', 'Laybrother', 'Clergymen')
femaleagent = ('Intermediary', 'Laybrother', 'Nun', 'Clergymen')
male_actors =   [[male_leaders, 'Male Leaders'], [maleagent, 'Agents']]
female_actors = [[female_leaders, 'Abbesses']  , [femaleagent, 'Agents for nunneries']]
def sort_frame(col, frame):
    if col !='Concentration':
        frame.sort_values(by=col, ascending= False, inplace=True)
    else:  
        frame.sort_values(by='Overall_wealth', ascending= False, inplace=True)
    return frame
def mean_std (sample):
    average = sample.mean()
    standard = np.std(sample)
    return average, standard

def ttest_ind (graphType, frames, type = 'Normal'):
    frame1 = sort_frame(graphType.column, frames.frame1)
    frame2 = sort_frame(graphType.column, frames.frame2)
    if type != 'Normal':
        frame1 = frame1.iloc[2:10]
        frame2 = frame2.iloc[2:10]
    return stats.ttest_ind(frame1[graphType.column],frame2[graphType.column], equal_var=False)

def standard_plot (ax, column, frame, label):
    avg, std = mean_std(frame[column])
    print (avg)
    ordered_list = frame[column]
    distribution = stats.norm.pdf(ordered_list, avg, std)
    fig1 = ax.scatter(ordered_list, distribution, label=label )
    fig2 = ax.axvline(avg, color = fig1.get_facecolors(), label= f'Mean = {round(avg,2)}')
    return (fig1, fig2)

def create_distributions(frameSet, graphType,numbertitle,dir=directory, type = 'Normal'):
    frame1 = sort_frame(graphType.column, frameSet.frame1)
    frame2 = sort_frame(graphType.column, frameSet.frame2)
    stat, pvalue = ttest_ind(graphType, frameSet)
    if type != 'Normal':
        frameSet.type = 'after trimming outliers'
    fig, ax = plt.subplots(figsize = (7,7))
    plt.title(f'Distribution curve of {graphType.title} {frameSet.type}', pad = 15)
    standard_plot(ax,graphType.column, frame1, frameSet.label1)
    standard_plot(ax,graphType.column, frame2, frameSet.label2)
    ax.set_xlabel(f'{graphType.label}')
    ax.set_ylabel('Frequency')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label=f'P-Value {pvalue.round(4)}'))
    ax.legend(handles=handles)
    plt.savefig(f'{dir}/{numbertitle}.jpeg')
    plt.close(fig)



def SonAndLoneMen(numbertitle):
    #FourthGraph Number of sons and number of lone man 
    genders = [['y', 'Female'], ['n', 'Male']]
    def overtime (classification):
        query = '''
            SELECT COUNT (*),EXTRACT (DECADE FROM year) as decade FROM actor 
            JOIN alldocuments ON actor.docid = alldocuments.docid
            JOIN classification ON actor.classification = classification.classid
            WHERE classification.classification = %s
            AND year BETWEEN '1100-01-01' AND '1299-01-01'
            GROUP BY decade
            ORDER BY decade
        '''
        dc.execute(query, [classification])
        count = []
        decade = []
        for x, y in dc.fetchall():
            count.append(x)
            decade.append(y*10)
        return count, decade

    def plotGraph (classification, numbertitle):
        count, decade = overtime(classification[0])
        count2, decade2 = overtime(classification[1])
        plt.plot(decade,count, label = classification[0] )
        plt.plot(decade2,count2, label = classification[1])
        plt.title (f'Number of {classification[0]}  and {classification[1]}')
        plt.ylabel('Number of documents')
        plt.xlabel('Years')
        plt.legend (fontsize = 13)
        plt.savefig (f'{directory}/{numbertitle}.jpeg')
        plt.close()
    plotGraph(['Son', 'Lone Man'], numbertitle)

def PercentageWomenOvertime(numbertitle):
    ## first graph -- percentage overtime
    def Total_number_actor ():
        time_query = '''
                SELECT COUNT(*), EXTRACT  (DECADE FROM year) as decade FROM actor 
                JOIN alldocuments ON actor.docid = alldocuments.docid
                JOIN monastery ON monastery.monasteryid = actor.monastery
                WHERE type_instiution = '3'
                AND year IS NOT NULL
            AND year BETWEEN '1100-01-01' AND '1299-01-01'
                GROUP BY decade
        '''
        year_number = []
        dc.execute(time_query)
        for count, year in dc.fetchall():
            year_number.append([count, int(year*10)])
            year_number = sorted(year_number, key=lambda x:x[1])
        return year_number

    def Number_overall (gender):
        time_query = '''
                SELECT COUNT(*), EXTRACT  (DECADE FROM year) as decade FROM actor 
                JOIN alldocuments ON actor.docid = alldocuments.docid
                JOIN monastery ON monastery.monasteryid = actor.monastery
                WHERE female = %s
                AND type_instiution = '3'
                AND year IS NOT NULL
            AND year BETWEEN '1100-01-01' AND '1299-01-01'
                GROUP BY decade
        '''
        year_number = []
        dc.execute(time_query, [gender])
        for count, year in dc.fetchall():
            year_number.append([count, int(year*10)])
            year_number = sorted(year_number, key=lambda x:x[1])
        return year_number

    def percentage_overtimne (total, female, title, numbertitle):
        data_female = []
        overall = total()   
        for num, year in female('y'):
            data_female.append(num)
        frame = pd.DataFrame(overall, columns=['Number_male', 'Year'])
        frame['Number_female'] = data_female
        frame['Percentage'] = frame['Number_female']/frame['Number_male']
        fig, ax = plt.subplots (figsize = (10, 7))
        ax.plot(frame['Year'], frame['Percentage'], label= 'Women in documents')
        ax.set_ylabel('Percentage', size=14)
        ax.set_xlabel('Years',size=12)
        plt.legend(fontsize=13)
        decades = np.arange(1100, 1300, 10)
        plt.xticks(decades)
        plt.title(title, fontsize = 15)
        plt.savefig(f'{directory}/{numbertitle}.jpeg')
        plt.close(fig)
    percentage_overtimne(Total_number_actor, Number_overall, 'Percentage of documents with women over time', numbertitle)

def WomneinSalesOvertime(numbertitle):
    def GetSalesOvertime ():
        querySalesOverTime = '''
            SELECT COUNT (*) FROM alldocuments 
            WHERE doctype = '6'
            AND EXTRACT(decade FROM year) = %s
            and docid IN  (SELECT DISTINCT docid FROM actor 
            JOIN monastery ON monastery.monasteryid = actor.monastery
            WHERE female IN %s
            AND type_instiution = '3')
        '''
        decadeList = list(range(110, 130))
        femaleList = [(True, False), (False,),  (True,)]
        salesPerGender = []
        for gender in femaleList:
            salesPerDecade = []
            for decade in decadeList:
                dc.execute(querySalesOverTime, [decade, gender])
                salesPerDecade.append(dc.fetchall()[0][0])
            salesPerGender.append([salesPerDecade, gender])
        return salesPerGender

    def createDataFrameFromSaleList(listOfSales):
        dictionaryForFrame = {}
        for numberOfSales, gender in listOfSales:
            if gender == (True,):
                gender_label = 'Female'
            elif gender == (False,):
                gender_label = 'Male'
            else:
                gender_label = 'Both'
            dictionaryForFrame[gender_label] = numberOfSales
        dataFrame = pd.DataFrame(dictionaryForFrame)
        dataFrame['Decade'] = list(range(110, 130))
        dataFrame['Decade'] = dataFrame['Decade']*10
        return dataFrame

    def createPlot(dataFrame):
        fig, ax = plt.subplots(figsize=(10,6))
        #ax.plot(dataFrame['Decade'], dataFrame['Male']/dataFrame['Both'] * 100, color = 'blue', label = 'Male')
        ax.plot(dataFrame['Decade'], dataFrame['Female']/dataFrame['Both'] *100, color = 'red', label = 'Women in documents of sale')
        plt.title('Percentage of women in sales over time',fontsize = 14 )
        plt.ylabel('Percentage of sales', size = 14)
        plt.xlabel('Years', size = 14)
        plt.legend(fontsize=13)
        plt.xticks(dataFrame['Decade'])
        plt.savefig(f'{directory}/{numbertitle}.jpg')
    createPlot(createDataFrameFromSaleList(GetSalesOvertime()))
#WomneinSalesOvertime(3)

def percentageAndSumOfDenariOvertime(numbertitle):
    def totalDenariOvertime ():
        denari = '''
            SELECT SUM (price.price) FROM alldocuments 
            JOIN price ON alldocuments.docid = price.docid
            WHERE alldocuments.docid IN  (SELECT DISTINCT docid FROM actor 
            JOIN monastery ON monastery.monasteryid = actor.monastery
            WHERE female IN %s AND doctype = '6' AND activity = '35' AND type_instiution = '3' 
            AND type_instiution = '3')
            AND EXTRACT (DECADE from year) BETWEEN 110 AND 130
            GROUP BY EXTRACT (DECADE from year)
            ORDER BY EXTRACT (DECADE from year)
        '''
        femaleList = [[(True, False), 'All'], [(False,), 'Male'], [(True,), 'Female']]
        moneyList = {}
        for gender, mask in femaleList:
            singleList = []
            dc.execute(denari,[gender])
            for money in dc.fetchall():
                decadeTotal = money[0]
                singleList.append(decadeTotal)

            moneyList[mask] = singleList
        decadeList = list(range(110, 130))
        frame = pd.DataFrame(moneyList, columns=['All', 'Male', 'Female'])
        frame['PercentageFemale'] = frame['Female']/frame['All']
        frame['Decade'] = decadeList
        frame['Decade'] = frame['Decade']*10
        return(frame)


    def createPlot(dataFrame):
        fig, ax = plt.subplots(figsize=(10,6))
        line1, = ax.plot(dataFrame['Decade'], dataFrame['PercentageFemale'] *100, color = 'red', label = 'Percentage of denari transacted')
        ax2 = ax.twinx()
        line2, = ax2.plot(dataFrame['Decade'], dataFrame['Female'], color = 'blue', label = 'Sum of denari transacted')
        plt.title('Percentage of denari transacted by women', fontsize = 14)
        ax.set_ylabel('Percentage of denari', size =15)
        ax2.set_ylabel('Denari Transacted', size =15)
        ax.legend(handles=[line1], loc='right', fontsize = 12)
        ax2.legend(handles=[line2], loc='upper right', fontsize = 12)
        ax.set_xlabel('Years', size = 14)
        plt.xticks(dataFrame['Decade'])
        plt.savefig(f'{directory}/{numbertitle}.jpg')
    createPlot(totalDenariOvertime())

listnumber = np.arange(1100, 1300, 1)
listdecaden = np.arange(1100, 1300, 10)
listdecade = np.arange(110, 130, 1)
decades = []
for y in listdecaden:
        year = f'{y}-01-01'
        decades.append(year)
def get_numeber_years (gender, institutions):
    number_query = '''
        SELECT COUNT(*) FROM alldocuments 
        WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        JOIN monastery ON monastery.monasteryid = actor.monastery
        JOIN classification ON classification.classid = actor.classification
        WHERE  classification.classification IN %s AND monastery.monasteryname IN %s)
        AND year = %s
    '''
    query_without_abbot = '''
        SELECT COUNT(*) FROM alldocuments 
        WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        JOIN monastery ON monastery.monasteryid = actor.monastery
        JOIN classification ON classification.classid = actor.classification
        WHERE  classification.classification IN %s AND monastery.monasteryname IN %s )
        AND docid NOT IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        JOIN classification ON classification.classid = actor.classification
        JOIN monastery ON monastery.monasteryid = actor.monastery
        WHERE classification.classification IN %s AND monastery.monasteryname IN %s)
        AND year = %s
    '''
    agent_list = []
    leader_list = []
    for y in listnumber:
        year = f'{y}-01-01'
        if gender == 'm':
            dc.execute(query_without_abbot, [maleagent, institutions, male_leaders, institutions, year])
            agent_list.append(float(dc.fetchall()[0][0]))
            dc.execute(number_query, [male_leaders, institutions, year])
            leader_list.append(float(dc.fetchall()[0][0]))

        if gender == 'f':
            dc.execute(query_without_abbot, [femaleagent, institutions, female_leaders, institutions, year])
            agent_list.append(float(dc.fetchall()[0][0]))
            dc.execute(number_query, [female_leaders, institutions, year])
            leader_list.append(float(dc.fetchall()[0][0]))

    return agent_list, leader_list 

def regression (first_list, second_list):
    res = stats.linregress(first_list, second_list)
    print (res)
    return res, res.rvalue **2

def regression_plot (first_list,second_list, title, numbertitle):
    res, rSquare = regression(first_list,second_list)
    fig, ax = plt.subplots(figsize = (8,8))
    ax.plot (first_list, second_list,'o', label = f'Number of documents of leaders/agents')
    ax.plot (first_list, res.intercept + res.slope*np.array(first_list), 'r', label =f'r-squared {round(rSquare, 4)}')
    ax.set_ylabel('Agent documents', size = 14)
    ax.set_xlabel('Leader documents', size = 14)
    plt.title(title, fontsize = 14)
    plt.legend(fontsize='14')
    plt.savefig(f'{directory}/{numbertitle}.jpeg')
    plt.close(fig)


def getPercentDecade (gender, institutions, listOfDecades):
    number_query = '''
        SELECT COUNT(*) FROM alldocuments 
        WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        JOIN monastery ON monastery.monasteryid = actor.monastery
        JOIN classification ON classification.classid = actor.classification
        WHERE  classification.classification IN %s AND monastery.monasteryname IN %s)
        AND EXTRACT (decade from year) = %s
    '''
    query_without_abbot = '''
        SELECT COUNT(*) FROM alldocuments 
        WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        JOIN monastery ON monastery.monasteryid = actor.monastery
        JOIN classification ON classification.classid = actor.classification
        WHERE  classification.classification IN %s AND monastery.monasteryname IN %s )
        AND docid NOT IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        JOIN classification ON classification.classid = actor.classification
        JOIN monastery ON monastery.monasteryid = actor.monastery
        WHERE classification.classification IN %s AND monastery.monasteryname IN %s)
        AND EXTRACT (decade from year) = %s
    '''
    agent_list = []
    leader_list = []
    actualdecates = []
    for year in listOfDecades:
        year = int(year)
        if getDecadeTotal(institutions, year) == 0:
            #agent_list.append(0)
            #leader_list.append(0)
            continue
        else:
            total = getDecadeTotal(institutions, year)
            actualdecates.append(year)
            
        if gender == 'm':
            dc.execute(query_without_abbot, [maleagent, institutions, male_leaders, institutions, year])
            agent_list.append(dc.fetchall()[0][0]/total)
            dc.execute(number_query, [male_leaders, institutions, year])
            leader_list.append(dc.fetchall()[0][0]/total)
        if gender == 'f':
            dc.execute(query_without_abbot, [femaleagent, institutions, female_leaders, institutions, year])
            agent_list.append(dc.fetchall()[0][0]/total)
            dc.execute(number_query, [female_leaders, institutions, year])
            leader_list.append(dc.fetchall()[0][0]/total)
    return agent_list, leader_list, np.array(actualdecates) 

def getDecadeTotal (inst, year):
    number_query = '''
        SELECT COUNT(*) FROM alldocuments 
        WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        JOIN monastery ON monastery.monasteryid = actor.monastery
        JOIN classification ON classification.classid = actor.classification
        WHERE monastery.monasteryname IN %s)
        AND EXTRACT (decade from year) = %s
    '''

    dc.execute(number_query,[ inst, year])
    return dc.fetchall()[0][0]

def busynessMaleLeaders (numbertitle):
    agent, leader = get_numeber_years('m', maleMonasteries)
    title = f'Documents conducted by male leaders and agents in the same year'
    regression_plot(leader, agent, title,numbertitle)


def busynessFemaleLeaders(numbertitle):
    agent, leader = get_numeber_years('f', femaleMonasteries)
    title = f'Documents conducted by abbesses and agents in the same year'
    regression_plot(leader, agent, title, numbertitle)


def multiRegressionTime (years, first_list, second_list, title, numbertitle):
    res_leader, squaredLeader = regression(years, first_list)
    res_agent, squaredAgent = regression(years, second_list)
    fig, ax = plt.subplots(figsize = (8,8))
    ax.plot (years*10, first_list,'ro', label = f'Leader over time \n r squared ={round(squaredLeader, 4)}')
    ax.plot (years*10, res_leader.intercept + res_leader.slope*np.array(years), 'r')
    ax.plot (years*10, second_list,'bo', label = f'Agent over time\n r squared ={round(squaredAgent, 4)}')
    ax.plot (years*10, res_agent.intercept + res_agent.slope*np.array(years), 'b')
    ax.set_xlabel('Decades', fontsize=14)
    ax.set_ylabel('Percentage of documents', fontsize=14)
    plt.title(title, fontsize = 14)
    plt.legend(fontsize=12)
    plt.savefig(f'{directory}/{numbertitle}.jpeg')
    plt.close()

def abbessesAndAgentsOvertime(numbertitle):
    title = 'Document by abbesses and agents over time'
    yearsList = np.arange(110, 130, 1)
    agent, abbess, yearsf = getPercentDecade('f', tuple(femaleMonasteries), yearsList)
    multiRegressionTime(yearsf, abbess, agent, title, numbertitle)

def maleLeadersAndAgentsOvertime(numbertitle):
    title = 'Document by male leaders and agents over time'
    yearsList = np.arange(110, 130, 1)
    agent, abbess, yearsf = getPercentDecade('m', tuple(maleMonasteries), yearsList)
    multiRegressionTime(yearsf, abbess, agent, title, numbertitle)

def SantaMariaDelMonteleaderovertime (numbertitle):
    title = 'Document by male leaders and agents over time of Santa Maria del Monte'
    yearsList = np.arange(119, 130, 1)
    agent, abbess, yearsf = getPercentDecade('m', ('MariaDelMonte',), yearsList)
    multiRegressionTime(yearsf, abbess, agent, title, numbertitle)

def multipleMultiGraphs (mon, year, gen):
    title = f'Document by male leaders and agents over time of {mon} '
    yearsList = year
    agent, abbess, yearsf = getPercentDecade(gen, (f'{mon}',), yearsList)
    multiRegressionTime(yearsf, abbess, agent, title, title)
sixClearInst =[['Monastero Maggiore', np.arange(110, 130, 1), 'f'], ['Apollinare', np.arange(122, 131, 1), 'f'], ['Lentasio', np.arange(120, 130, 1), 'f'], ['Ambrogio', np.arange(110, 130, 1), 'm'], ['Giorgio_Palazzo', np.arange(120, 130, 1), 'm'], ['MariaDelMonte', np.arange(119, 131, 1), 'm']]

milan =Point(9.19, 45.4642)
Velate = Point (8.4735, 45.5136)
rivolta = Point (9.5130, 45.4705)

def Distance (frame, mon, comparison = 'Null'):
    geoseries=gpd.GeoSeries.from_wkt(frame ['Coordinates'], crs='EPSG:32633')
    if comparison == 'Null' :
        try:
            comparison =  compareCoordinate(mon)
        except KeyError:
            frame['Distance'] = []
            return frame
    distances = geoseries.distance(comparison)
    frame['Distance'] = distances *111
    return frame

def compareCoordinate (monastery):
    mainLocationQuery = '''
        SELECT ST_AsText(geocoordinates::geometry), coordinates.locations, COUNT (*) FROM alldocuments
        JOIN coordinates ON coordinates.coordid = alldocuments.redaction
        WHERE alldocuments.docid IN (
        SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        JOIN classification ON classification.classid = actor.classification
        JOIN monastery ON monastery.monasteryid = actor.monastery
        WHERE monastery.monasteryname = %s)
        GROUP BY ST_AsText(geocoordinates::geometry), coordinates.locations
        ORDER BY COUNT DESC
        LIMIT 1
    '''
    dc.execute(mainLocationQuery, [monastery])
    try:
        result = dc.fetchall()[0][0]
    except IndexError:
        return milan
    if result == None:
        return milan
    point = loads(result)
    return point


def coordinatesCounts (monastery, classification, mask, docytpe = 'Null', comparison = 'Null'):


    query_redaction = '''
        SELECT ST_AsText(geocoordinates::geometry), coordinates.locations FROM alldocuments
        JOIN coordinates ON coordinates.coordid = alldocuments.redaction
        WHERE alldocuments.docid IN (
        SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        JOIN classification ON classification.classid = actor.classification
        JOIN monastery ON monastery.monasteryid = actor.monastery
        WHERE monastery.monasteryname IN %s AND classification.classification IN %s)
        
        
    '''
    agentQuery = '''
        SELECT ST_AsText(geocoordinates::geometry), coordinates.locations FROM alldocuments
        JOIN coordinates ON coordinates.coordid = alldocuments.redaction
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE  classification.classification IN %s AND monastery.monasteryname IN %s)
    AND docid NOT IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE classification.classification IN %s AND monastery.monasteryname IN %s)
    
    
    '''
    query_doctype = '''
        SELECT ST_AsText(geocoordinates::geometry),coordinates.locations FROM alldocuments
        JOIN coordinates ON coordinates.coordid = alldocuments.redaction
        JOIN doctype ON doctype.id = alldocuments.doctype 
        WHERE alldocuments.docid IN (
        SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        JOIN classification ON classification.classid = actor.classification
        JOIN monastery ON monastery.monasteryid = actor.monastery
        WHERE monastery.monasteryname IN %s AND classification.classification IN %s)
        AND doctype.translation = %s
        
        
    '''
    agentDoctype = '''
        SELECT ST_AsText(geocoordinates::geometry),coordinates.locations FROM alldocuments
        JOIN coordinates ON coordinates.coordid = alldocuments.redaction
        JOIN doctype ON doctype.id = alldocuments.doctype 
        WHERE alldocuments.docid IN (
        SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        JOIN classification ON classification.classid = actor.classification
        JOIN monastery ON monastery.monasteryid = actor.monastery
        WHERE classification.classification IN %s AND monastery.monasteryname IN %s )
        AND doctype.translation = %s
          AND docid NOT IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        JOIN classification ON classification.classid = actor.classification
        JOIN monastery ON monastery.monasteryid = actor.monastery
        WHERE classification.classification IN %s AND monastery.monasteryname IN %s)
        
        
    '''
    if  docytpe == 'Null' and mask in  ('Male Leaders', 'Abbesses'):
        dc.execute(query_redaction, [monastery, classification])
    elif mask == 'Agents' and docytpe == 'Null': 
        dc.execute(agentQuery, [classification, monastery, male_leaders, monastery])
    elif mask == 'Agents for nunneries' and docytpe == 'Null':  
        dc.execute(agentQuery, [classification, monastery, female_leaders, monastery])
    elif mask == 'Male Leaders' or mask == 'Abbesses':
        dc.execute(query_doctype, [monastery, classification, docytpe])
    elif mask == 'Agents': 
        dc.execute(agentDoctype, [classification, monastery, docytpe,  male_leaders, monastery])
    elif mask == 'Agents for nunneries':  
        dc.execute(agentDoctype, [classification, monastery,docytpe, female_leaders, monastery])
    coord = []
    countList = []
    location =[]
    for co, loc in dc.fetchall():
        if co is None:
            pass
        elif loc == 'Laterano':
            pass
        else:
            coord.append(co)
            location.append(loc)        
    data = {'Coordinates': coord}
    frame = pd.DataFrame(data)
    distanceFrame = Distance(frame, monastery, comparison)
    return distanceFrame['Distance']

def bringTogetherAllDistanceFrames (monastery, classification, mask, docytpe = 'Null', comparison = 'Null'):
    allDistances= []
    for mon in monastery:
        allDistances.append(coordinatesCounts((mon,), classification, mask, docytpe, comparison))
    flatList = [item for sublist in allDistances for item in sublist]
    return pd.DataFrame(flatList, columns= ['Distance'])

sixInstitutions = [['Monastero Maggiore', 'f'],['Apollinare', 'f'],['Margherita', 'f'],['Ambrogio', 'm'],['Giorgio_Palazzo', 'm'],['MariaDelMonte', 'm']]
   
def framesForTtest (monlist):
    resultDict = {}
    for monastery, gender in monlist:
        if gender == 'f':
            frameLeader = coordinatesCounts((monastery,), female_leaders, 'Abbesses')
            frameAgent = coordinatesCounts((monastery,), femaleagent, 'Agents for nunneries')
            ttest, pvalue = stats.ttest_ind(frameLeader, frameAgent, equal_var=False)
            resultDict[monastery]= [ttest, pvalue]
        if gender == 'm':
            frameLeader = coordinatesCounts((monastery,), male_leaders, 'Male Leaders')
            frameAgent = coordinatesCounts((monastery,), maleagent, 'Agents')
            ttest, pvalue = stats.ttest_ind(frameLeader, frameAgent, equal_var=False)
            resultDict[monastery]= [ttest, pvalue]
    frame = pd.DataFrame.from_dict(resultDict, orient='index')
    return frame 
#framesForTtest(sixInstitutions).to_excel(f'{directory}/TtestsDistanceTravelled.xlsx')


def createDistanceDistributions(column, title, frames, labels, numbertitle):  
    ttest, pvalue = stats.ttest_ind(frames[0][column],frames[1][column], equal_var=False)
    fig, ax = plt.subplots(figsize = (7,7))
    standard_plot(ax,column, frames[0], labels[0])
    standard_plot(ax,column, frames[1], labels [1])
    ax.set_xlabel('Distance (Km)', size=13)
    ax.set_ylabel('Frequency', size=13)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label=f'P-Value {pvalue.round(4)}'))
    ax.legend(handles=handles)
    plt.title(title)
    plt.savefig(f'{directory}/{numbertitle}.jpeg')
    plt.close(fig)

def executePlots (institution, classification, title, labels, numbertitle, comparison = 'Null',doctype = ['Null', 'Null'] ):
    ledaerFrame = bringTogetherAllDistanceFrames(institution[0], classification[0][0], classification[0][1],doctype[0], comparison)
    AgentFrame  = bringTogetherAllDistanceFrames(institution[1], classification[1][0], classification[1][1],doctype[1], comparison)
    createDistanceDistributions('Distance', title, [ledaerFrame, AgentFrame], labels, numbertitle)

def abbessesAndAgentsTravel (numbertitle):
    classificationSet = [[female_leaders, 'Abbesses'], [femaleagent, 'Agents for nunneries']]
    title = 'Distribution of distance travelled by abbesses and agents'
    labels = ['Abbess', 'Agent']
    frames = [femaleMonasteries, femaleMonasteries]
    executePlots(frames, classificationSet, title, labels, numbertitle=numbertitle)


def maleleadersAndAgentsTravel(numbertitle):
    classificationSet = [[male_leaders, 'Male Leaders'], [maleagent, 'Agents']]
    title = 'Distribution of distance travelled by male leaders and agents'
    labels = ['Leader', 'Agent']
    frames = [maleMonasteries, maleMonasteries]
    executePlots(frames, classificationSet, title, labels, numbertitle=numbertitle)

def abbessesAndAbbotsTravel(numbertitle):
    classificationSet = [[female_leaders, 'Abbesses'], [male_leaders, 'Male Leaders']]
    title = 'Distribution of distance travelled by abbesses and male leaders'
    labels = ['Abbess', 'Male leader']
    frames = [femaleMonasteries, maleMonasteries]
    executePlots(frames, classificationSet, title, labels, numbertitle=numbertitle)



def overall_redaction (monastery, fication, mask):
    query = '''
    SELECT ST_AsText(geocoordinates::geometry),coordinates.locations, count(*) FROM alldocuments
    JOIN coordinates ON coordinates.coordid = alldocuments.redaction
    AND alldocuments.docid IN (
    SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.monasteryname IN %s AND classification.classification IN %s)
    GROUP BY ST_AsText(geocoordinates::geometry), coordinates.locations
    ORDER BY count DESC
    '''
    q_clergy = '''
     SELECT ST_AsText(geocoordinates::geometry),coordinates.locations, count(*) FROM alldocuments
    JOIN coordinates ON coordinates.coordid = alldocuments.redaction
    AND alldocuments.docid IN (
    SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.monasteryname IN %s AND classification.classification IN %s)
    AND docid NOT IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        JOIN classification ON classification.classid = actor.classification
        JOIN monastery ON monastery.monasteryid = actor.monastery
        WHERE classification.classification IN %s AND monastery.monasteryname IN %s)
    GROUP BY ST_AsText(geocoordinates::geometry), coordinates.locations
    ORDER BY count DESC
     
    '''
    
    if mask != 'Clergy':
        dc.execute(query, [monastery, fication])
        frame_landloc = pd.DataFrame(dc.fetchall(), columns=('Geometry', 'location', 'Count'))
        geoframe_landloc = gpd.GeoDataFrame(frame_landloc, geometry=gpd.GeoSeries.from_wkt(frame_landloc ['Geometry'], crs='EPSG:4326'))
    else: 
        dc.execute(q_clergy, [monastery, fication, male_leaders, monastery])
        frame_landloc = pd.DataFrame(dc.fetchall(), columns=('Geometry', 'location', 'Count'))
        geoframe_landloc = gpd.GeoDataFrame(frame_landloc, geometry=gpd.GeoSeries.from_wkt(frame_landloc ['Geometry'], crs='EPSG:4326'))
    geoframe_landloc.sort_values('Count', ascending= False, inplace= True)
    return geoframe_landloc

def createMultiMap (dataframes, titles, dir, numbertitle):
    leaderFrame, agentFrame = dataframes
    fig, ax = plt.subplots(1,2,figsize=(12,12))
    size = leaderFrame['Count'].apply(lambda x: x*20)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('bottom', size='7%', pad=2)
    lombardia.plot(ax=ax[0], color= 'none', edgecolor='black')
    leaderFrame.plot(ax=ax[0], markersize=size, column='Count', legend=True, legend_kwds={'orientation': "horizontal"}, cax=cax)
    ax[0].axis('off')
    ax[0].annotate('Milan', xy=(9.19, 45.4642), size=20)
    ax[0].set_title(titles[0], size=14)
    cax.tick_params('x', labelsize = 15)
    plt.xlabel('Number of documents', fontsize=15)
    size = agentFrame['Count'].apply(lambda x: x*20)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('bottom', size='7%', pad=2) 
    lombardia.plot(ax=ax[1], color= 'none', edgecolor='black')
    agentFrame.plot(ax=ax[1], markersize=size, column='Count', legend=True, legend_kwds={'orientation': "horizontal"}, cax=cax)
    ax[1].axis('off')
    ax[1].annotate('Milan', xy=(9.19, 45.4642), size=20)
    ax[1].set_title(titles[1], size =14)
    cax.tick_params('x', labelsize = 15)
    plt.xlabel('Number of documents',fontsize=15)
    plt.suptitle (titles[2], fontsize=15)
    plt.savefig(f'{dir}/{numbertitle}.jpeg')

def MonasteroMaggioreMap (numbertitle):
    titles = ['By the abbess','By the agents', 'Location of business of the Monastero Maggiore'] 
    leaderFrame = overall_redaction(('Monastero Maggiore',),female_leaders, 'Abbess')
    agentFrame = overall_redaction(('Monastero Maggiore',),femaleagent , 'Agents')
    createMultiMap([leaderFrame, agentFrame], titles, directory, numbertitle=numbertitle)
MonasteroMaggioreMap(50)

def AmbrogioMap (numbertitle):
    titles = ['By the abbot','By the agents', 'Location of business of the monastery of Sant\'Ambrogio '] 
    leaderFrame = overall_redaction(('Ambrogio',),male_leaders, 'Male Leaders')
    agentFrame = overall_redaction(('Ambrogio',),maleagent , 'Agents')
    createMultiMap([leaderFrame, agentFrame], titles, directory, numbertitle)

doctypes = ['Investiture', 'Dispute', 'Exchange', 'Promise', 'Payment', 'Renounciation', 'Sale']
doctypes.sort()

def get_numbers (fication, mask, monastery, doctype):
    query_doctype_count = '''
    SELECT COUNT(*) FROM alldocuments
    JOIN doctype ON alldocuments.doctype = doctype.id
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE  classification.classification IN %s AND monastery.monasteryname IN %s)
    AND doctype.translation = %s
    '''
    clergy_query = '''
    SELECT COUNT(*) FROM alldocuments
    JOIN doctype ON alldocuments.doctype = doctype.id
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE  classification.classification IN %s AND monastery.monasteryname IN %s)
    AND doctype.translation = %s
    AND docid NOT IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE classification.classification IN %s AND monastery.monasteryname IN %s)
    '''
    if mask == 'Agents':
        dc.execute(clergy_query, [fication, monastery, doctype, male_leaders, monastery])
        return dc.fetchall()[0][0]
    elif mask == 'Agents for nunneries':
        dc.execute(clergy_query, [fication, monastery, doctype, female_leaders, monastery])
        return dc.fetchall()[0][0]
    else:
        dc.execute(query_doctype_count, [fication, monastery, doctype])
        return dc.fetchall()[0][0]
    
def create_list (fication, mask, monastery):
    list = []
    for type in doctypes:
        list.append(get_numbers(fication, mask, monastery, type))
    return list

def create_frame (list, monastery, function):
    frame = pd.DataFrame()
    frame['Document Type'] = doctypes
    for fication, mask in list:
        frame[mask] = function(fication, mask, monastery)
    return frame

def BarGraphPlot (classMask, monastery, func, title, numbertitle, label= ''):
    frame = create_frame(classMask, monastery, func)
    leader_x_point = np.arange(1, 4*len(doctypes), 4)
    agent_x_point = np.arange(2, 4*len(doctypes) +1, 4)
    middle_x_point = np.arange(1.5, 4*len(doctypes) +1, 4)
    points = [leader_x_point, agent_x_point]
    x= 0
    fig, ax = plt.subplots (figsize=(8,8))
    if label == '':
        for fication, mask in classMask:
            ax.bar(points[x], frame[mask],label=mask)
            x += 1   
    else:
        for fication, mask, legend in label:
            ax.bar(points[x], frame[mask],label=legend)
            x += 1
    ax.legend(loc='upper center', prop={'size': 15})
    ax.set_xlabel('Parchment  Type',   size=15)
    ax.set_ylabel ('Parchment  Count', size=15)
    plt.title(title, size=15)
    plt.xticks(middle_x_point, doctypes, size=10)
    plt.savefig(f'{directory}/{numbertitle}.jpeg')
    plt.close(fig)

def AbbessesBarGraph (numbertitle):
    title = 'Types of documents led by abbesses and agents'
    BarGraphPlot (female_actors, femaleMonasteries, create_list, title, numbertitle)


def MaleLeadersBarGraph(numbertitle):
    title = 'Types of documents led by male leaders and agents'
    BarGraphPlot (male_actors, maleMonasteries, create_list, title, numbertitle)


def MonasteroMaggioreBarGraph(numbertitle):
    title = 'Types of documents led by abbess and agents of Monastero Maggiore'
    BarGraphPlot (female_actors, ('Monastero Maggiore',), create_list, title, numbertitle)


def MargheritaBarGraph(numbertitle):
    title = 'Types of documents led by abbess and agents of Santa Margherita'
    BarGraphPlot (female_actors, ('Margherita',), create_list, title,numbertitle)


def ApollinareBarGraph(numbertitle):
    title = 'Types of documents led by abbess and agents of Sant\'Apollinare'
    BarGraphPlot (female_actors, ('Apollinare',), create_list, title, numbertitle)

def SantAmbrogioBarGraph(numbertitle):
    title = 'Types of documents led by abbots and agents of Sant\'Ambrogio'
    BarGraphPlot (male_actors, ('Ambrogio',), create_list, title, numbertitle)

def GiorgioBarGraph(numbertitle):
    title = 'Types of documents led by leaders and agents of San Giorgio al Palazzo'
    BarGraphPlot (male_actors, ('Giorgio_Palazzo',), create_list, title, numbertitle)


def MariaDelMonteBarGraph(numbertitle):
    title = 'Types of documents led by leaders and agents of Santa Maria del Monte'
    BarGraphPlot (male_actors, ('MariaDelMonte',), create_list, title, numbertitle)

def denariTopTen (numbertitle):
    create_distributions(topTenFrame, denariDistribution,numbertitle)

def denariWithoutOutliers (numbertitle):
    create_distributions(removeOutliersFrame, denariDistribution,numbertitle = numbertitle, type= '')

def returnFourListsFromFrames (frameSets, graphType):
    maleFrame = sort_frame(graphType.column, frameSets[0].frame1)
    maleMainList = maleFrame[graphType.column]
    femaleFrame = sort_frame(graphType.column, frameSets[0].frame2)
    femaleMainList = femaleFrame[graphType.column]
    ttest, pvalue1 = ttest_ind(graphType, frameSets[0])
    maleLight = frameSets[1].frame1
    maleSecondList = maleLight[graphType.column]
    femaleLight = frameSets[1].frame2
    femaleSecondList = femaleLight[graphType.column]
    ttest, pvalue2 = ttest_ind(graphType, frameSets[1])
    return [maleMainList, femaleMainList, maleSecondList, femaleSecondList], [pvalue1, pvalue2]

def fourAvgStdDistr (data):
    results = []
    for x in data:
        avg, std = mean_std(x)
        distribution = stats.norm.pdf(x, avg, std)
        results.append([round(avg,2), std, distribution])
    return results

def sideBySidePlot (lists, stats, graphType,ttest, numbertitle, dir = directory):
    colors = ['r', 'b', 'r', 'b']
    fig, axs = plt.subplots(1,2, figsize = (12,12))
    axs[0].scatter(lists[0], stats[0][2], label = 'Male institutions', color = colors[0])
    axs[0].axvline(stats[0][0], color = colors[0], label=f'male average {stats[0][0]}')

    axs[0].scatter(lists[1], stats[1][2], label = 'Female institutions', color = colors[1])
    axs[0].axvline(stats[1][0], color = colors[1], label=f'female average {stats[1][0]}')

    axs[1].scatter(lists[2], stats[2][2], label = 'Male institutions',color = colors[2])
    axs[1].axvline(stats[2][0], color = colors[2], label=f'male average {stats[2][0]}')

    axs[1].scatter(lists[3], stats[3][2], label = 'Female institutions', color = colors[3])
    axs[1].axvline(stats[3][0], color = colors[3], label=f'female average {stats[3][0]}')
    
    axs[0].set_xlabel(f'{graphType.label}', size = 15)
    axs[0].set_ylabel('Frequency', size = 15)
    axs[0].set_title(f'Distribution of {graphType.title} by top twenty institutions')
    axs[0].tick_params(labelsize =15)
    axs[1].set_xlabel(f'{graphType.label}', size = 15)
    axs[1].set_title(f'Distribution of {graphType.title} after trimming outliers')


    handles, labels = axs[0].get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label=f'P-Value {ttest[0].round(4)}'))
    axs[0].legend(handles=handles, fontsize = 15)
    handles, labels = axs[1].get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label=f'P-Value {ttest[1].round(4)}'))
    axs[1].legend(handles=handles, fontsize = 15)
    plt.savefig(f'{dir}/{numbertitle}.jpeg', bbox_inches='tight')
    plt.close(fig)
    

def locationsidebyside (numbertitle):
    list, ttest = returnFourListsFromFrames([topTenFrame,locationFrame], locationDistribution)
    sideBySidePlot(list, fourAvgStdDistr(list), locationDistribution,ttest,numbertitle)

managementDoc = ('11', '26', '1','7','15')
buyingDoc = ('6', '9', '17', '18')
disputeDoc = ('3','16','5')
doctypeList = [(managementDoc, 'Documents regarding management'),(buyingDoc,'Documents regarding expansion'), (disputeDoc, 'Documents regarding litigation')]
nameList = ['M','E', 'L' ]
def getPercentageDoctype (listmon):
    queryDoctype = '''
        SELECT COUNT(*) FROM alldocuments
    JOIN doctype ON alldocuments.doctype = doctype.id
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.monasteryname = %s)
    AND doctype.id IN %s
    '''
    numberDoctypes = {}
    for doc, name in doctypeList:
        typenumber = 0
        for mon in listmon:
            dc.execute(queryDoctype, [mon, doc])
            typenumber += dc.fetchall()[0][0]
        numberDoctypes[name] = typenumber
    frame =pd.DataFrame.from_dict(numberDoctypes, 'index',  columns= ['Count'])
    frame['Percentage'] = frame['Count']/frame['Count'].sum()
    return frame

def singleMonData (mon):
    queryDoctype = '''
        SELECT COUNT(*) FROM alldocuments
    JOIN doctype ON alldocuments.doctype = doctype.id
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.monasteryname = %s)
    AND doctype.id IN %s
    '''
    numberDoctypes = {}
    for doc, name in doctypeList:
        dc.execute(queryDoctype, [mon, doc])
        typenumber = dc.fetchall()[0][0]
        numberDoctypes[name] = typenumber
    return pd.DataFrame.from_dict(numberDoctypes, 'index', columns= ['Percentage'])

def multipleMonData (listMon):
    queryDoctype = '''
        SELECT COUNT(*) FROM alldocuments
    JOIN doctype ON alldocuments.doctype = doctype.id
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.monasteryname = %s)
    AND doctype.id IN %s
    '''
    numberDoctypes = {}
    for doc, name in doctypeList:
        typenumber = 0
        for mon in listMon:
            dc.execute(queryDoctype, [mon, doc])
            typenumber += dc.fetchall()[0][0]
        numberDoctypes[name] = typenumber
    return pd.DataFrame.from_dict(numberDoctypes, 'index', columns= ['Percentage'])


def chiSquare (mainData, compareData):
    excepted = mainData['Percentage']*compareData['Percentage'].sum()
    return stats.chisquare(compareData['Percentage'], excepted)


def barGraph (mainData, compareData, title, numbertitle):
    chivalue, pvalue = chiSquare(mainData, compareData)
    excepted = mainData['Percentage']*compareData['Percentage'].sum()
    actual = compareData['Percentage']
    observedPoints = [1, 4, 7]
    expectedPoints = [2, 5, 8]
    fig, ax = plt.subplots(figsize = (12,12))
    ax.bar(observedPoints,excepted, label = 'Expected Results')
    ax.bar(expectedPoints,actual, label = 'Actual Results')
    ax.set_ylabel('Count', size = 20)
    ax.set_xlabel('Type of document', size =20)
    ax.tick_params('y', labelsize =20)
    plt.xticks(observedPoints, nameList,fontsize=20)
    plt.title (f'Spread of documents of {title}', size =20)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label=f'P-value {pvalue.round(6)}'))
    ax.legend(handles=handles, fontsize=15)
    plt.savefig(f'{directory}/{numbertitle}.jpeg')
    plt.close(fig)


def graphForSideBySide (mainData, compareData):
    chivalue, pvalue = chiSquare(mainData, compareData)
    excepted = mainData['Percentage']*compareData['Percentage'].sum()
    actual = compareData['Percentage']
    return excepted, actual, pvalue

def barSideBySide (dataSets, numbertitle):
    
    numberOfGraphs = len(dataSets)
    even = False
    if (numberOfGraphs%2)== 0:
        numberOfColumns= 2
        numberOfRows = int(numberOfGraphs/2)
        even = True
    else:
        numberOfColumns = numberOfGraphs
        numberOfRows = 1

    fig, axs = plt.subplots(numberOfRows,numberOfColumns, figsize = (14,14))
    numberOfSubplotColumn = 0
    numberOfSubplotRow = 0
    observedPoints = [1, 4, 7]
    expectedPoints = [2, 5, 8]
    
    for i, (data, title) in enumerate(dataSets):
        if even==False or numberOfGraphs == 2:
            locationMatrix = numberOfSubplotColumn
        else:
            locationMatrix = numberOfSubplotRow, numberOfSubplotColumn
        expected, actual, pvalue = data
        if numberOfSubplotColumn == 0:
            axs[locationMatrix].set_ylabel('Number of documents', size=20)
        
        axs[locationMatrix].bar(observedPoints,expected,  label = 'Expected Results')
        axs[locationMatrix].bar(expectedPoints,actual, label = 'Actual Results')
        axs[locationMatrix].set_title(title, fontsize = 20)

        if i== 0:
            handles, labels = axs[locationMatrix].get_legend_handles_labels()
            handles.append(mpatches.Patch(color='none', label=f'P-Value {pvalue.round(4)}'))
            axs[locationMatrix].legend(handles=handles, fontsize=15)
        else: 
            axs[locationMatrix].legend([mpatches.Patch(color='none')], [f'P-Value {pvalue.round(4)}'], fontsize=12)
        axs[locationMatrix].tick_params(labelsize=17)
        axs[locationMatrix].set_xticks(observedPoints, nameList)
        numberOfSubplotColumn += 1
        axs[locationMatrix].set_xlabel('Type of document', size = 20)
        if numberOfSubplotColumn == 2 and even == True:
            numberOfSubplotColumn -=2
            numberOfSubplotRow +=1
    fig.tight_layout(pad=5.0)
    fig.suptitle('Frequency of documents of managment, expansion, and litigation', fontsize = 20)
    plt.savefig(f'{directory}/{numbertitle}.jpeg')
    plt.close(fig)
    
def singleMonChiGraph(monastery):
    title = monastery
    barGraph(getPercentageDoctype(top20), singleMonData(monastery), title, title)


def ManagementAllMaleFemale (numbertitle):
    groupSets = []
    for group, name in [[femaleMonasteries, 'Female institutions'], [maleMonasteries, 'Male institutions']]:
        groupSets.append([graphForSideBySide(getPercentageDoctype(top20), multipleMonData(group)), name])
    barSideBySide(groupSets, numbertitle)

def MonasteroMaggioreAmbrogio (numbertitle):
    mainSet = []
    for institutions, titles in [['Monastero Maggiore', 'Monastero Maggiore'], ['Ambrogio', 'Monastery of Sant\'Ambrogio']]:
        mainSet.append([graphForSideBySide(getPercentageDoctype(top20), singleMonData(institutions)), titles])
    barSideBySide(mainSet, numbertitle)


def fourAncientNunneries (numbertitle):
    mainSet = []
    for institutions, titles in [['MariaAurona','Santa Maria d\'Aurona'], ['Margherita','Santa Margherita'], ['Lentasio', 'Santa Maria al Lentasio'], ['Radegonda', 'Santa Radegonda']]:
        mainSet.append([graphForSideBySide(getPercentageDoctype(top20), singleMonData(institutions)), titles])
    barSideBySide(mainSet, numbertitle)

def fourAncientChurches (numbertitle):
    mainSet = []
    for institutions, titles in [['Giorgio_Palazzo','San Giorgio al Palazzo'], ['CanonicaAmbrogio','Canonica of Sant\'Ambrogio'],['Lorenzo','San Lorenzo'],['Simpliciano','San Simpliciano']]:
        mainSet.append([graphForSideBySide(getPercentageDoctype(top20), singleMonData(institutions)), titles])
    barSideBySide(mainSet, numbertitle)


def ApollinareManagement (numbertitle):
    title = 'Sant\'Apollinare'
    barGraph(getPercentageDoctype(top20), singleMonData('Apollinare'),title ,numbertitle =numbertitle)


def fourNewNunneries (numbertitle):
    mainSet = []
    for institutions, title in [['CappucineConcorezzo','Cappucine di San Pietro'],['Agnese','Sant\'Agnese'], ['SanFelice','San Felice'],['AmbrogioRivalta','Sant\'Ambrogio di Rivolta']]:
        mainSet.append([graphForSideBySide(getPercentageDoctype(top20), singleMonData(institutions)), title])
    barSideBySide(mainSet, numbertitle)


def ChiravalleBrolo (numbertitle):
    mainSet = []
    for institutions, title in [['Chiaravalle','Chiaravalle'], ['OspedaleBroletoMilano', 'Ospedale del Broletto']]:
        mainSet.append([graphForSideBySide(getPercentageDoctype(top20), singleMonData(institutions)), title])
    barSideBySide(mainSet, numbertitle)


def MariaDelMonte (numbertitle):
    title = 'Santa Maria del Monte'
    barGraph(getPercentageDoctype(top20), singleMonData('MariaDelMonte'),title ,numbertitle =numbertitle)

def MinorMajorChapter (numbertitle):
    mainSet = []
    for institutions, title in [['DecumaniMilanesi','Minor Cathedral Chapter'], ['MilanArchdioces','Major Cathedral Chapter']]:
        mainSet.append([graphForSideBySide(getPercentageDoctype(top20), singleMonData(institutions)), title])
    barSideBySide(mainSet, numbertitle)

def distributionMembers (numbertitle):
    create_distributions(topTenFrame, MembershipDistribution,numbertitle)


def RivoltaDaddaNunuse(numbertitle):
    title = 'Documents led by abbess and nuns of Sant\'Ambrogio of Rivolta'
    actors =[[female_leaders, 'Abbesses']  , [('Nun',), 'Agents for nunneries']]
    legend = ['Abbesses', 'Nuns']
    newList = [[actors[i][0], actors[i][1], legend[i]] for i in range(2)]
    BarGraphPlot (actors, ('AmbrogioRivalta',), create_list, title, numbertitle, newList)



