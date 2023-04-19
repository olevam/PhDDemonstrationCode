import InformationOnInstitutions as im #this allows to find specific types of institutions
import DatabaseConnect  
dc = DatabaseConnect.connect('Gamburigan.147' ,'womeninmilanphd') #this allows to connect to the database, and select the password, and database name

#various imports for the graphs and relevant calculations
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
from shapely.geometry import Point
from shapely.wkt import loads
import datetime as yr


#where the output of each function will go
directoryToSaveGraphs = 'Data/CreatedGraphs'

#complete list of institutions by gender 
male = im.get_monasteires('m')
female = im.get_monasteires('f')
all = im.get_monasteires('a')

#complete list of humiliati institutions
humiliatiMen = im.humiliati('m')
humiliatiWomen = im.humiliati('f')
humiliatiAll = im.humiliati('a')

#map of lombardy 
lombardia = gpd.read_file("Data/MapsForCode/Regione_2020.shp").to_crs('EPSG:4326')

#calls the frames with various data on various groups of institutions
#this has all the institutions
mainFrame = pd.read_excel('Data/MainFrame.xlsx')
#only male institutions
maleFrame = pd.read_excel('Data/MaleFrame.xlsx')
#only female institutions
femaleFrame = pd.read_excel('Data/FemaleFrame.xlsx')
#top 10 male institutions, and top 10 female institutions, according to overall wealth. 
top20Frame = pd.read_excel('Data/Top20FullData.xlsx')

#these are lists of institutional names for the upcoming analysis
allMale = maleFrame['Monastery']
allFemale = femaleFrame['Monastery']
top20 = top20Frame['Monastery']   
top10Female = top20[top20.isin(female)] 
top10Male = top20[top20.isin(male)]  

#these find the top 10 and top 30 male and female institutions, according to the overall wealth
maleprime = maleFrame.iloc[0:10]
femaleprime =femaleFrame.iloc[0:10]
top30male = maleFrame.iloc[0:30]
top30female = femaleFrame.iloc[0:30]


# this graph shows the percentage of women acting in document for each decade between 1100 and 1299
def firstGraph(numbertitle):
    # this function returns the number of overall actors for each relevant decade if gender is none
    # otherwise it returns the number of female actors for each relevant decade 
    def Total_number_actor (gender = None): 
        time_query = '''
                SELECT COUNT(*), EXTRACT  (DECADE FROM year) as decade FROM actor 
                JOIN alldocuments ON actor.docid = alldocuments.docid
                JOIN monastery ON monastery.monasteryid = actor.monastery
                WHERE type_instiution = '3'
                AND year IS NOT NULL
            AND year BETWEEN '1100-01-01' AND '1299-01-01'
        '''
        year_number = []
        if gender == None:
            time_query += 'GROUP BY decade ORDER BY decade'
            dc.execute(time_query)
            for count, year in dc.fetchall():
                year_number.append([count, int(year*10)])
                year_number = sorted(year_number, key=lambda x:x[1])
            return year_number
        else:
            time_query += '''AND female = 'y' GROUP BY decade ORDER BY decade'''
            dc.execute(time_query)
            for count, year in dc.fetchall():
                year_number.append(count)
            return year_number
       
    #this function calculates the percentage of women per decade, and plots it.
    def percentage_overtimne (title, numbertitle):
        allactors = Total_number_actor()
        femaleactors = Total_number_actor('f')
        frame = pd.DataFrame(allactors, columns=['Number_male', 'Year'])
        frame['Number_female'] = femaleactors
        frame['Percentage'] = frame['Number_female']/frame['Number_male']
        fig, ax = plt.subplots (figsize = (7, 7))
        ax.plot(frame['Year'], frame['Percentage'], label= 'Women')
        ax.set_ylabel('Percentage', size=12)
        ax.set_xlabel('Years',size=12)
        plt.legend(fontsize=13)
        plt.title(title)
        plt.savefig(f'{directoryToSaveGraphs}/{numbertitle}.jpeg')
        plt.close(fig)
    percentage_overtimne('Percentage of female actors overtime', numbertitle)


firstGraph(1)
#This function plots the number of documents of men and women over each decade
def secondGraph (numbertitle):
    #this function creates the figure
    def create_plot (ax, data, gender, label):
        year = []
        number = []
        for num, ye in data(gender):
            year.append(ye)
            number.append(num)
        figure = ax.plot(year, number, label = label)
        return(figure)
    #this function counts the number of documents with men or women in each decade
    def nm_in_docs (gender):
        n_of_docs_q = ''' 
            SELECT COUNT(*), EXTRACT  (DECADE FROM year) as decade FROM alldocuments 
            WHERE docid IN  (SELECT DISTINCT docid FROM actor 
            JOIN monastery ON monastery.monasteryid = actor.monastery
            WHERE female = %s
            AND type_instiution = '3')
            AND year IS NOT NULL
            AND year BETWEEN '1100-01-01' AND '1299-01-01'
            GROUP BY decade
        '''
        year_number = []
        dc.execute(n_of_docs_q, [gender])
        for count, year in dc.fetchall():
            year_number.append([count, int(year*10)])
            year_number = sorted(year_number, key=lambda x:x[1])
        return year_number
    #this function creates a figure more men and for women, then sets various details. It takes the function for the data, the title and the number of the graph in relation to the thesis.
    def save_figs (data, title, numbertitle):
        fig, ax = plt.subplots(1,2,figsize= (7,7))
        create_plot(ax[0], data, 'n', label= 'Men')
        create_plot(ax[1], data, 'y',label= 'Women')
        ax[0].title.set_text(title + ' men')
        ax[1].title.set_text(title + ' women')
        ax[0].set_ylabel('Number', size=12)
        ax[0].set_xlabel('Years',size=12)
        ax[1].set_ylabel('Number', size=12)
        ax[1].set_xlabel('Years',size=12)
        ax[0].legend(fontsize=13)
        ax[1].legend(fontsize=13)
        plt.savefig(f'{directoryToSaveGraphs}/{numbertitle}.jpeg')
        plt.close(fig)
    save_figs(nm_in_docs, 'Number of documents with',numbertitle)


#This function plots the average denari obtain in a sale  for men and women over time
def thirdGraph (numbertitle):
    # the function counts the number of sales for male and female per decade
    def number_overtime (gen):
        number_overtime = '''
        SELECT COUNT (*), EXTRACT (DECADE from year) as decade FROM alldocuments 
        WHERE alldocuments.docid IN  (SELECT DISTINCT docid FROM actor 
        JOIN monastery ON monastery.monasteryid = actor.monastery
        WHERE female = %s AND doctype = '6' AND activity = '35' AND type_instiution = '3')
         AND doctype = '6'
        GROUP BY decade
        ORDER BY decade
    '''
        dc.execute(number_overtime,[gen])
        year_list = []
        for count, year in dc.fetchall():
            year_list.append([int(count), int(year)*10])
        return year_list
    
    # this function counts the denari spent in each document with a men and a woman for each decade. 
    #It does so by obtaining the number of documents with men or women in a given decade, and then dividing the total denari transacted in a decade with men or women by the number of documents with men or women in that decade.
    def denari_overtime (gen):
        denari = '''
            SELECT SUM (price.price) FROM alldocuments 
            JOIN price ON alldocuments.docid = price.docid
            WHERE alldocuments.docid IN  (SELECT DISTINCT docid FROM actor 
            JOIN monastery ON monastery.monasteryid = actor.monastery
            WHERE female = %s AND doctype = '6' AND activity = '35' AND type_instiution = '3' 
            AND type_instiution = '3')
            AND EXTRACT (DECADE from year) = %s
        '''
        year_list = []
        for count, year in number_overtime(gen):
            dc.execute(denari,[gen, year/10])
            money = dc.fetchall()[0][0]
            if money:
                year_list.append([count, int(money), year])
        frame = pd.DataFrame(year_list, columns=['Number', 'Denari', 'Year'])
        frame['PerCapita']= round(frame['Denari']/frame['Number'], 2)
        return frame

    #this plots the figure calling the previous functuion for men and for women. It takes the number of the figure in relation to the thesis.
    def get_figure (numbertitle):
        male_frame = denari_overtime('n')
        female_frame = denari_overtime('y')
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(female_frame['Year'], female_frame['PerCapita'], label = 'Female')
        ax.plot(male_frame['Year'], male_frame['PerCapita'], label = 'Male')
        ax.legend(fontsize=13)
        ax.set_ylabel('Denari', size=12)
        ax.set_xlabel('Years',size=12)
        title = 'Average denari transacted per sale per decade'
        plt.title(title)
        plt.savefig(f'{directoryToSaveGraphs}/{numbertitle}.jpeg')
        plt.close(fig)
    get_figure(numbertitle)

#this function plots the number of men identified as "sons" and those without identification over the period
def fourthGraph(numbertitle):
    #this function counts the number of documents with each classification
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
    #this function plots the figure, it takes the classifications, and the number of the figure in relation to the thesis.
    def plotGraph (classification, numbertitle):
        count, decade = overtime(classification[0])
        count2, decade2 = overtime(classification[1])
        plt.plot(decade,count, label = classification[0] )
        plt.plot(decade2,count2, label = classification[1])
        plt.title (f'Number of {classification[0]}  and {classification[1]}')
        plt.ylabel('Number of documents')
        plt.xlabel('Years')
        plt.legend (fontsize = 13)
        plt.savefig (f'{directoryToSaveGraphs}/{numbertitle}.jpeg')

    plotGraph(['Son', 'Lone Man'], numbertitle)

### here is general code for all the standard deviation stuff
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
    ordered_list = frame[column]
    distribution = stats.norm.pdf(ordered_list, avg, std)
    fig1 = ax.scatter(ordered_list, distribution, label=label )
    fig2 = ax.axvline(avg, color = fig1.get_facecolors(), label= f'Mean = {avg.round(2)}')
    return (fig1, fig2)

def create_distributions(frameSet, graphType,numbertitle,dir=directoryToSaveGraphs, type = 'Normal'):
    frame1 = sort_frame(graphType.column, frameSet.frame1)
    frame2 = sort_frame(graphType.column, frameSet.frame2)
    stat, pvalue = ttest_ind(graphType, frameSet)
    if type != 'Normal':
        frameSet.type = 'of 3rd to 10th'
    fig, ax = plt.subplots(figsize = (7,7))
    plt.title(f'Distribution curve of {graphType.label} {frameSet.type}')
    standard_plot(ax,graphType.column, frame1, frameSet.label1)
    standard_plot(ax,graphType.column, frame2, frameSet.label2)
    ax.set_xlabel(f'{graphType.label}')
    ax.set_ylabel('Frequency')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label=f'P-Value {pvalue.round(2)}'))
    ax.legend(handles=handles)
    plt.savefig(f'{dir}/{numbertitle}.jpeg', bbox_inches='tight')
    plt.close(fig)

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

topTenFrame = frameSet (maleprime, femaleprime, 'Male institutions', 'Female institutions', 'Top ten')     
removeOutliersFrame = frameSet (maleFrame.iloc[2:10], femaleFrame.iloc[2:10], 'Male institutions', 'Female institutions', 'Removing Outliers')          
allFrame = frameSet (maleFrame, femaleFrame, 'Male institutions', 'Female institutions', 'All institutions')          

distanceDistribution = typeOfGraph('Coordinates', 'Distance', 'Distance Travelled')
locationDistribution = typeOfGraph('Loc_number', 'land invested', 'Locations Invested')
denariDistribution = typeOfGraph('Denari_spent', 'denari spent', 'DenariSpent')
networkDistribution = typeOfGraph('Independent_list', 'network size', 'Network')
overallDistribution = typeOfGraph('Overall_wealth', 'overall wealth', 'Overallwealth')
## single distributiongraphs
def fifthGraph (numbertitle):
    create_distributions(allFrame, overallDistribution, numbertitle)
def sixthGraph (numbertitle):
    create_distributions(topTenFrame, overallDistribution,numbertitle)
def seventhGraph (numbertitle):
    create_distributions(removeOutliersFrame, overallDistribution, numbertitle = numbertitle, type= '')
def eigthGraph (numbertitle):
    create_distributions(topTenFrame, denariDistribution,numbertitle)
def ninthGraph (numbertitle):
    create_distributions(removeOutliersFrame, denariDistribution,numbertitle = numbertitle, type= '')

##side by side graphs
def returnFourListsFromFrames (frameSet, graphType):
    maleFrame = sort_frame(graphType.column, frameSet.frame1)
    maleMainList = maleFrame[graphType.column]
    femaleFrame = sort_frame(graphType.column, frameSet.frame2)
    femaleMainList = femaleFrame[graphType.column]
    ttest, pvalue1 = ttest_ind(graphType, frameSet)
    maleLight = removeOutliersFrame.frame1
    maleSecondList = maleLight[graphType.column]
    femaleLight = removeOutliersFrame.frame2
    femaleSecondList = femaleLight[graphType.column]
    ttest, pvalue2 = ttest_ind(graphType, removeOutliersFrame)
    return [maleMainList, femaleMainList, maleSecondList, femaleSecondList], [pvalue1, pvalue2]

def fourAvgStdDistr (data):
    results = []
    for x in data:
        avg, std = mean_std(x)
        distribution = stats.norm.pdf(x, avg, std)
        results.append([avg.round(2), std, distribution])
    return results

def sideBySidePlot (lists, stats, graphType,ttest, numbertitle, dir = directoryToSaveGraphs):
    colors = ['r', 'b', 'r', 'b']
    fig, axs = plt.subplots(1,2, figsize = (10,10))
    axs[0].scatter(lists[0], stats[0][2], label = 'Male institutions', color = colors[0])
    axs[0].axvline(stats[0][0], color = colors[0], label=f'male average {stats[0][0]}')

    axs[0].scatter(lists[1], stats[1][2], label = 'Female institutions', color = colors[1])
    axs[0].axvline(stats[1][0], color = colors[1], label=f'female average {stats[1][0]}')

    axs[1].scatter(lists[2], stats[2][2], label = 'Male institutions',color = colors[2])
    axs[1].axvline(stats[2][0], color = colors[2], label=f'male average {stats[2][0]}')

    axs[1].scatter(lists[3], stats[3][2], label = 'Female institutions', color = colors[3])
    axs[1].axvline(stats[3][0], color = colors[3], label=f'female average {stats[3][0]}')
    
    axs[0].set_xlabel(f'{graphType.title}')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title(f'Distribution of {graphType.label} top ten')
    axs[1].set_xlabel(f'{graphType.title}')
    axs[1].set_xlabel('Frequency')
    axs[1].set_title(f'Distribution of {graphType.label} without largest')


    handles, labels = axs[0].get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label=f'P-Value {ttest[0].round(2)}'))
    axs[0].legend(handles=handles)
    handles, labels = axs[1].get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label=f'P-Value {ttest[1].round(2)}'))
    axs[1].legend(handles=handles)
    plt.savefig(f'{dir}/{numbertitle}.jpeg', bbox_inches='tight')
    plt.close(fig)
    

def tenthGraph (numbertitle):
    list, ttest = returnFourListsFromFrames(topTenFrame, locationDistribution)
    sideBySidePlot(list, fourAvgStdDistr(list), locationDistribution,ttest,numbertitle)
def eleventhGraph(numbertitle):
    list, ttest = returnFourListsFromFrames(topTenFrame, networkDistribution)
    sideBySidePlot(list, fourAvgStdDistr(list), networkDistribution,ttest, numbertitle)
#economicFrequency
doctypeList = ['Investiture', 'Sale', 'Sentence']
def getPercentageDoctype (listmon):
    queryDoctype = '''
        SELECT COUNT(*) FROM alldocuments
    JOIN doctype ON alldocuments.doctype = doctype.id
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.monasteryname = %s)
    AND doctype.translation = %s
    '''
    alldocs = '''
        SELECT COUNT(*) FROM alldocuments
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.monasteryname = %s)
    '''

    numberDoctypes = {}
    totaldocs = 0

    for doc in doctypeList:
        typenumber = 0
        for mon in listmon:
            dc.execute(queryDoctype, [mon, doc])
            typenumber += dc.fetchall()[0][0]
        numberDoctypes[doc] = typenumber
    frame =pd.DataFrame.from_dict(numberDoctypes, 'index',  columns= ['Percentage'])
    frame['Percentage'] = frame['Percentage']/frame['Percentage'].sum()
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
    AND doctype.translation = %s
    '''
    alldocs = '''
        SELECT COUNT(*) FROM alldocuments
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.monasteryname = %s)
    '''
    numberDoctypes = {}
    for doc in doctypeList:
        dc.execute(queryDoctype, [mon, doc])
        typenumber = dc.fetchall()[0][0]
        numberDoctypes[doc] = typenumber
    return pd.DataFrame.from_dict(numberDoctypes, 'index', columns= ['Percentage'])

def multipleMonData (list):
    queryDoctype = '''
        SELECT COUNT(*) FROM alldocuments
    JOIN doctype ON alldocuments.doctype = doctype.id
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.monasteryname IN %s)
    AND doctype.translation = %s
    '''
    alldocs = '''
        SELECT COUNT(*) FROM alldocuments
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.monasteryname IN %s)
    '''
    numberDoctypes = {}
    for doc in doctypeList:
        dc.execute(queryDoctype, [tuple(list), doc])
        typenumber = dc.fetchall()[0][0]
        numberDoctypes[doc] = typenumber
    
    return pd.DataFrame.from_dict(numberDoctypes, 'index', columns= ['Percentage'])


def chiSquare (mainData, compareData):
    excepted = mainData['Percentage']*compareData['Percentage'].sum()
    return stats.chisquare(compareData['Percentage'], excepted)


def chiSquareFrame (mainData, list):
    dictOfChi = {}
    for mon in list: 
        stat, pValue = chiSquare(mainData, singleMonData(mon))
        dictOfChi[mon] = [stat, round(pValue, 4)]
    frame = pd.DataFrame.from_dict(dictOfChi, 'Index', columns= [['xSquareStat', 'PValue']])
    return frame

def barGraph (mainData, compareData, title, numbertitle):
    chivalue, pvalue = chiSquare(mainData, compareData)
    excepted = mainData['Percentage']*compareData['Percentage'].sum()
    actual = compareData['Percentage']
    fig, ax = plt.subplots(figsize = (12,12))
    ax.plot(doctypeList,excepted, marker = 'o', label = 'Expected Results')
    ax.plot(doctypeList,actual, marker = 'v', label = 'Actual Results')
    ax.set_ylabel('Count', size = 15)
    plt.xticks(fontsize=15)
    plt.title (f'Spread of documents of {title}', size =14)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label=f'P-value {pvalue.round(2)}'))
    ax.legend(handles=handles, fontsize=12)
    plt.savefig(f'{directoryToSaveGraphs}/{numbertitle}.jpeg')
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

    fig, axs = plt.subplots(numberOfRows,numberOfColumns, figsize = (12,12))
    numberOfSubplotColumn = 0
    numberOfSubplotRow = 0
    
    
    for data, title in dataSets:
        if even==False or numberOfGraphs == 2:
            locationMatrix = numberOfSubplotColumn
        else:
            locationMatrix = numberOfSubplotRow, numberOfSubplotColumn
        expected, actual, pvalue = data
        
        axs[locationMatrix].plot(doctypeList,expected,  marker = 'o', label = 'Expected Results')
        axs[locationMatrix].plot(doctypeList,actual,  marker = '*', label = 'Actual Results')
        axs[locationMatrix].set_title(title, fontsize = 14)
        axs[locationMatrix].set_ylabel('Number of documents', size=15)
        handles, labels = axs[locationMatrix].get_legend_handles_labels()
        handles.append(mpatches.Patch(color='none', label=f'P-Value {pvalue.round(2)}'))
        axs[locationMatrix].legend(handles=handles, fontsize=12)
        axs[locationMatrix].tick_params(labelsize=15)
        numberOfSubplotColumn += 1
        if numberOfSubplotColumn == 2 and even == True:
            numberOfSubplotColumn -=2
            numberOfSubplotRow +=1

    fig.suptitle('Frequency of investitures, sales, disputes', fontsize = 15)
    plt.savefig(f'{directoryToSaveGraphs}/{numbertitle}.jpeg')
    plt.close(fig)
    


def twelfthGraph(numbertitle):
    title = 'Ospedale del Broletto Milano'
    barGraph(getPercentageDoctype(top20), singleMonData('OspedaleBroletoMilano'), title, numbertitle)

def thirteenthGraph (numbertitle):
    mainSet = []
    for institutions in ['Monastero Maggiore', 'CanonicaAmbrogio']:
        mainSet.append([graphForSideBySide(getPercentageDoctype(top20), singleMonData(institutions)), institutions])
    barSideBySide(mainSet, numbertitle)

def fourteenthGraph (numbertitle):
    mainSet = []
    for institutions in ['MariaAurona', 'Margherita']:
        mainSet.append([graphForSideBySide(getPercentageDoctype(top20), singleMonData(institutions)), institutions])
    barSideBySide(mainSet, numbertitle)


def fifteenthGraph (numbertitle):
    mainSet = []
    for institutions in ['Agnese', 'SanFelice', 'AmbrogioRivalta', 'CappucineConcorezzo']:
        mainSet.append([graphForSideBySide(getPercentageDoctype(top20), singleMonData(institutions)), institutions])
    barSideBySide(mainSet, numbertitle)

def sixteenthGraph(numbertitle):
    title = 'Santa Apollinare'
    barGraph(getPercentageDoctype(top20), singleMonData('Apollinare'),title ,numbertitle =numbertitle)

def seventeenthGraph(numbertitle):
    title = 'Santa Radegonda'
    barGraph(getPercentageDoctype(top20), singleMonData('Radegonda'),title ,numbertitle=numbertitle)

def eighteenthGraph (numbertitle):
    mainSet = []
    for institutions in ['Ambrogio', 'Lorenzo', 'Simpliciano', 'DecumaniMilanesi']:
        mainSet.append([graphForSideBySide(getPercentageDoctype(top20), singleMonData(institutions)), institutions])
    barSideBySide(mainSet, numbertitle)

def nineteenthGraph (numbertitle):
    mainSet = []
    for institutions in ['Giorgio_Palazzo', 'OspedaleBroletoMilano', 'MilanArchdioces']:
        mainSet.append([graphForSideBySide(getPercentageDoctype(top20), singleMonData(institutions)), institutions])
    barSideBySide(mainSet, numbertitle)

def twentiethGraph (numbertitle):
    mainSet = []
    for institutions in ['Chiaravalle', 'MariaDelMonte']:
        mainSet.append([graphForSideBySide(getPercentageDoctype(top20), singleMonData(institutions)), institutions])
    barSideBySide(mainSet, numbertitle)

def twentyFirtstGraph (numbertitle):
    groupSets = []
    for group, name in [[top10Female, 'Female institutions'], [top10Male, 'Male institutions']]:
        groupSets.append([graphForSideBySide(getPercentageDoctype(top20), multipleMonData(group)), name])
    barSideBySide(groupSets, numbertitle)
    
def twentySecondGraph(numbertitle):
    groupSets = []
    for group, name in [[humiliatiMen, 'Humiliati Men'], [humiliatiWomen, 'Humiliati Women']]:
        groupSets.append([graphForSideBySide(getPercentageDoctype(humiliatiAll), multipleMonData(group)), name])
    barSideBySide(groupSets, numbertitle)

#MembershipGraphs
def countDate (classification, insti):
    query = '''
        SELECT COUNT (*), EXTRACT (year from year) as year, alldocuments.docid, alldocuments.docnumber FROM alldocuments
        JOIN actor ON alldocuments.docid = actor.docid
        JOIN classification ON classification.classid = actor.classification
        JOIN monastery ON monastery.monasteryid = actor.monastery
        WHERE   classification.classification = %s AND monastery.monasteryname IN %s
        GROUP BY year, alldocuments.docid
        ORDER BY COUNT desc
    '''
    count = []
    date = []
    docid = []
    docnumber = []
    dc.execute(query, [classification, insti])
    for c, d, doc, number in dc.fetchall():
        count.append(c)
        date.append(d)
        docid.append(doc)
        docnumber.append(number)
    return count, date, docid, docnumber
 
def createFrame (monasteries, actor):
    list = []
    for mon in monasteries:
        count, date, docid, number = countDate(actor, (mon,))
        try:
            list.append([count[0], date[0],docid[0], number[0], mon])
        except IndexError:
            pass
    frame = pd.DataFrame(list, columns= ['Count', 'Date', 'Docid', 'Docnumber','Name'])
    return frame.sort_values('Count', ascending=False)

def createMembershiDistribution (dir, frames, label, numbertitle , column = 'Membership'):  
    ttest, pvalue = stats.ttest_ind(frames[0][column],frames[1][column], equal_var=False)
    fig, ax = plt.subplots(figsize = (7,7))
    plt.title(f'Distribution of members of {label}')
    standard_plot(ax,column, frames[0], 'Clergy')
    standard_plot(ax,column, frames[1], 'Nuns')
    ax.set_xlabel('Number')
    ax.set_ylabel('Frequency')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label=f'P-Value {pvalue.round(2)}'))
    ax.legend(handles=handles)
    plt.savefig(f'{dir}/{numbertitle}.jpeg', bbox_inches='tight')
    plt.close(fig)

def twentyThirdGraph(numbertitle):
    createMembershiDistribution(directoryToSaveGraphs, [maleFrame, femaleFrame], 'All', numbertitle=numbertitle)
def twentyfourthGraph(numbertitle):
    createMembershiDistribution(directoryToSaveGraphs,[top30male, top30female] , 'Top Thirty',numbertitle=numbertitle)
def twentyfiveGraph(numbertitle):
    createMembershiDistribution(directoryToSaveGraphs, [maleprime, femaleprime], 'Largest Wealth',numbertitle=numbertitle)

#RegressionMembership
def twentysixthGraph(numbertitle):
    togetherFrame = pd.concat([maleprime, femaleprime])
    togetherData = stats.linregress(togetherFrame['Membership'], togetherFrame['Overall_wealth'])
    fig, ax = plt.subplots(figsize = (7,7))
    ax.scatter (togetherFrame['Membership'], togetherFrame['Overall_wealth'], label = 'All institutions')
    ax.plot(togetherFrame['Membership'], togetherData.intercept + togetherData.slope*np.array(togetherFrame['Membership']), 'r', label = f'R_value {round(togetherData.rvalue, 4)}')
    ax.set_xlabel('Membership')
    ax.set_ylabel('Overall Wealth')
    plt.legend(fontsize='14')
    plt.title ('Comparison of wealth and membership of institutions')
    plt.savefig(f'{directoryToSaveGraphs}/{numbertitle}.jpeg')
    plt.close(fig)

#RegressionAgentLeader
male_leaders = ('Abbott', 'Preposto', 'Archpriest', 'Archbishop')
female_leaders = ('Abbess',)
male_clergy = ('Clergymen',)
female_clergy = ('Nun',)
male_intermediaries = ('Intermediary', 'Laybrother')
maleagent = ('Intermediary', 'Laybrother', 'Clergymen')
femaleagent = ('Intermediary', 'Laybrother', 'Nun')
male_actors =   [[male_leaders, 'Male Leaders'], [maleagent, 'Agents']]
female_actors = [[female_leaders, 'Abbesses']  , [femaleagent, 'Agents for nunneries']]

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
            dc.execute(query_without_abbot, [femaleagent, institutions, male_leaders, institutions, year])
            agent_list.append(float(dc.fetchall()[0][0]))
            dc.execute(number_query, [female_leaders, institutions, year])
            leader_list.append(float(dc.fetchall()[0][0]))

    return agent_list, leader_list 

def regression (first_list, second_list):
    return stats.linregress(first_list, second_list)

def regression_plot (first_list,second_list, title, numbertitle):
    res = regression(first_list,second_list)
    fig, ax = plt.subplots(figsize = (8,8))
    ax.plot (first_list, second_list,'o', label = 'Number of documents of leaders/agents')
    ax.plot (first_list, res.intercept + res.slope*np.array(first_list), 'r', label =f'R_value {round(res.rvalue, 4)}')
    ax.set_ylabel('Agent documents')
    ax.set_xlabel('Leader documents')
    plt.title(title)
    plt.legend(fontsize='14')
    plt.savefig(f'{directoryToSaveGraphs}/{numbertitle}.jpeg')
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
            dc.execute(query_without_abbot, [femaleagent, institutions, male_leaders, institutions, year])
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


def twentyseventhGraph(numbertitle):
    agent, leader = get_numeber_years('m', male)
    title = f'Regression of male leaders and agents'
    regression_plot(leader, agent, title,numbertitle)

def twentyeightGraph(numbertitle):
    agent, leader = get_numeber_years('f', female)
    title = f'Regression of abbesses and agents'
    regression_plot(leader, agent, title, numbertitle)

def twentyninthGraph(numbertitle):
    agent, leader = get_numeber_years('f', ('Monastero Maggiore',))
    title = f'Regression of abbesses and agents of Monastero Maggiore'
    regression_plot(leader, agent, title, numbertitle)

def thirtiethGraph(numbertitle):
    agent, leader = get_numeber_years('m', ('Ambrogio',))
    title = f'Regression of leader and agents of Monastery Ambrogio'
    regression_plot(leader, agent, title,numbertitle)

def thirtyfirstGraph(numbertitle):
    agent, leader = get_numeber_years('m', ('MariaDelMonte',))
    title = f'Regression of leader and agents of Santa Maria del Monte'
    regression_plot(leader, agent, title,numbertitle)

def thirtysecondGraph(numbertitle):
    agent, leader = get_numeber_years('f', ('Lentasio',))
    title = f'Regression of abbesses and agents of Santa Maria al Lentasio'
    regression_plot(leader, agent, title,numbertitle)

def thirtythirdGraph(numbertitle):
    agent, leader = get_numeber_years('f', ('Apollinare',))
    title = f'Regression of abbesses and agents of Santa Apollinare'
    regression_plot(leader, agent, title,numbertitle)

def thirtyfourthGraph(numbertitle):
    agent, leader = get_numeber_years('m', ('Giorgio_Palazzo',))
    title = f'Regression of leader and agents of San Giorgio al Palazzo'
    regression_plot(leader, agent, title,numbertitle)


def multiRegressionTime (years, first_list, second_list, title, numbertitle ):
    res_leader = regression(years, first_list)
    res_agent = regression(years, second_list)
    fig, ax = plt.subplots(figsize = (8,8))
    ax.plot (years*10, first_list,'ro', label = f'Leader over time \n rValue ={round(res_leader.rvalue, 4)}')
    ax.plot (years*10, res_leader.intercept + res_leader.slope*np.array(years), 'r')
    ax.plot (years*10, second_list,'bo', label = f'Agent over time\n rValue ={round(res_agent.rvalue, 4)}')
    ax.plot (years*10, res_agent.intercept + res_agent.slope*np.array(years), 'b')
    ax.set_xlabel('Decades', fontsize=13)
    ax.set_ylabel('Percentage of documents', fontsize=13)
    plt.title(title)
    plt.legend()
    plt.savefig(f'{directoryToSaveGraphs}/{numbertitle}.jpeg')
    plt.close()

def seventyfirstGraph(numbertitle):
    title = 'Document by abbesses and agents over time'
    yearsList = np.arange(110, 130, 1)
    agent, abbess, yearsf = getPercentDecade('f', tuple(allFemale), yearsList)
    multiRegressionTime(yearsf, abbess, agent, title, numbertitle)

def seventysecondGraph(numbertitle):
    title = 'Document by male leaders and agents over time'
    yearsList = np.arange(110, 130, 1)
    agent, abbess, yearsf = getPercentDecade('m', tuple(allMale), yearsList)
    multiRegressionTime(yearsf, abbess, agent, title, numbertitle)

def seventythirdGraph (numbertitle):
    title = 'Document by male leaders and agents over time of Santa Maria del Monte'
    yearsList = np.arange(119, 130, 1)
    agent, abbess, yearsf = getPercentDecade('m', ('MariaDelMonte',), yearsList)
    multiRegressionTime(yearsf, abbess, agent, title, numbertitle)

def seventyfourthGraph (numbertitle):
    title = 'Document by male leaders and agents over time of Sant Apollinare '
    yearsList = np.arange(120, 130, 1)
    agent, abbess, yearsf = getPercentDecade('f', ('Apollinare',), yearsList)
    multiRegressionTime(yearsf, abbess, agent, title, numbertitle)

#Graphs on distance travel
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




def createDistanceDistributions(column, title, frames, labels, numbertitle):  
    ttest, pvalue = stats.ttest_ind(frames[0][column],frames[1][column], equal_var=False)
    fig, ax = plt.subplots(figsize = (7,7))
    standard_plot(ax,column, frames[0], labels[0])
    standard_plot(ax,column, frames[1], labels [1])
    ax.set_xlabel('Distance (Km)', size=13)
    ax.set_ylabel('Frequency', size=13)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label=f'P-Value {pvalue.round(2)}'))
    ax.legend(handles=handles)
    plt.title(title)
    plt.savefig(f'{directoryToSaveGraphs}/{numbertitle}.jpeg')
    plt.close(fig)


# using 32633 for best italian projection 

def executePlots (institution, classification, title, labels, numbertitle, comparison = 'Null',doctype = ['Null', 'Null'] ):
    ledaerFrame = bringTogetherAllDistanceFrames(institution[0], classification[0][0], classification[0][1],doctype[0], comparison)
    AgentFrame  = bringTogetherAllDistanceFrames(institution[1], classification[1][0], classification[1][1],doctype[1], comparison)
    createDistanceDistributions('Distance', title, [ledaerFrame, AgentFrame], labels, numbertitle)

def thirtyfifthGraph (numbertitle):
    classificationSet = [[female_leaders, 'Abbesses'], [femaleagent, 'Agents for nunneries']]
    title = 'Distribution of distance travelled by abbesses and agents'
    labels = ['Abbess', 'Agent']
    frames = [female, female]
    executePlots(frames, classificationSet, title, labels, numbertitle=numbertitle)

def thirtysixthGraph(numbertitle):
    classificationSet = [[male_leaders, 'Male Leaders'], [maleagent, 'Agents']]
    title = 'Distribution of distance travelled by male leaders and agents'
    labels = ['Leader', 'Agent']
    frames = [male, male]
    executePlots(frames, classificationSet, title, labels, numbertitle=numbertitle)

def thirtyseventhGraph(numbertitle):
    classificationSet = [[female_leaders, 'Abbesses'], [male_leaders, 'Male Leaders']]
    title = 'Distribution of distance travelled by abbesses and male leaders'
    labels = ['Abbess', 'Male leader']
    frames = [female, male]
    executePlots(frames, classificationSet, title, labels, numbertitle=numbertitle)

# maps 

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
    ax[0].set_title(titles[0])
    plt.xlabel('Number of documents', fontsize=13)
    size = agentFrame['Count'].apply(lambda x: x*20)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('bottom', size='7%', pad=2) 
    lombardia.plot(ax=ax[1], color= 'none', edgecolor='black')
    agentFrame.plot(ax=ax[1], markersize=size, column='Count', legend=True, legend_kwds={'orientation': "horizontal"}, cax=cax)
    ax[1].axis('off')
    ax[1].annotate('Milan', xy=(9.19, 45.4642), size=20)
    ax[1].set_title(titles[1])
    plt.xlabel('Number of documents',fontsize=13)
    plt.suptitle (titles[2], fontsize=15)
    plt.savefig(f'{dir}/{numbertitle}.jpeg')
    plt.close(fig)

def thirtyeigthGraph (numbertitle):
    titles = ['By the abbess','By the agents', 'Redaction of Monastero Maggiore parchments'] 
    leaderFrame = overall_redaction(('Monastero Maggiore',),female_leaders, 'Abbess')
    agentFrame = overall_redaction(('Monastero Maggiore',),femaleagent , 'Agents')
    createMultiMap([leaderFrame, agentFrame], titles, directoryToSaveGraphs, numbertitle=numbertitle)


def thirtynineGraph(numbertitle):
    classificationSet = [[female_leaders, 'Abbesses'], [femaleagent, 'Agents for nunneries']]
    title = 'Distance travelled by abbesses and agents of Monastero Maggiore'
    labels = ['Leader', 'Agent']
    frames = [('Monastero Maggiore',),('Monastero Maggiore',)]
    executePlots(frames, classificationSet, title, labels, numbertitle=numbertitle)

#mapsOvertTime
def redaction_overtime (monastery, classfication, numbertitle):
    query_redaction_time = '''
        SELECT ST_AsText(geocoordinates::geometry), count(*) FROM alldocuments
        JOIN coordinates ON coordinates.coordid = alldocuments.redaction
        WHERE alldocuments.docid IN (
        SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        JOIN classification ON classification.classid = actor.classification
        JOIN monastery ON monastery.monasteryid = actor.monastery
        WHERE monastery.monasteryname = %s AND classification.classification  = %s)
        AND year between %s AND %s
        GROUP BY ST_AsText(geocoordinates::geometry)
        ORDER BY count DESC
    '''
    beginning = []
    end = []
    loc = [0,1]
    for quarters in range(1200,1300, 25):
        beginning.append(yr.date(quarters, 1, 1))
    for quarters in range(1225,1325, 25):
        end.append(yr.date(quarters, 1, 1))
    g_coor = [(0,0), (0,1),(1,0), (1,1)]
   

    for mon in monastery:
        fig, ax = plt.subplots(2,2, figsize=(25,25))
        fig.suptitle('Location of parchment redacted by the leader of ' + mon + ' over the 13th century', size=25)
        for yeara, yearb, l in zip (beginning, end, g_coor):
            
            dc.execute(query_redaction_time, [mon, classfication, yeara, yearb])
            frame_landloc = pd.DataFrame(dc.fetchall(), columns=('Geometry', 'Count'))
            if len(frame_landloc. index) == 0:
                pass
            else:
                geoframe_landloc = gpd.GeoDataFrame(frame_landloc['Count'], geometry=gpd.GeoSeries.from_wkt(frame_landloc ['Geometry'], crs='EPSG:4326'))
                size = geoframe_landloc['Count'].apply(lambda x: x*50)
            
            divider = make_axes_locatable(ax[l])
            cax = divider.append_axes('bottom', size='7%', pad=1.5)

            ax[l].set_title(str(yeara) + ' - ' + str(yearb), size=25)
            lombardia.plot(ax=ax[l], color= 'none', edgecolor='black')
            geoframe_landloc.plot(ax=ax[l], markersize=size, column='Count', legend=True, legend_kwds={'orientation': "horizontal"}, cax=cax)
            cax.set_xlabel('Document Number',fontsize=20)
            cax.tick_params(labelsize=15)
            ax[l].axis('off')
            ax[l].annotate('Milan', xy=(9.19, 45.4642), size=25)

        plt.savefig(f'{directoryToSaveGraphs}/{numbertitle}.jpeg',bbox_inches='tight')
        plt.close(fig)
def fortyGraph(numbertitle):
    redaction_overtime(('Monastero Maggiore',), female_leaders, numbertitle)


def fortyoneGraph (numbertitle):
    titles = ['By the abbot','By the agents', 'Redaction of monastery Ambrogio parchments'] 
    leaderFrame = overall_redaction(('Ambrogio',),male_leaders, 'Male Leaders')
    agentFrame = overall_redaction(('Ambrogio',),maleagent , 'Agents')
    createMultiMap([leaderFrame, agentFrame], titles, directoryToSaveGraphs, numbertitle)

def fortytwoGraph(numbertitle):
    classificationSet = [[male_leaders, 'Male Leaders'], [maleagent, 'Agents']]
    title = 'Distance travelled by abbots and agents of monastery Ambrogio'
    labels = ['Leader', 'Agent']
    frames = [('Ambrogio',),('Ambrogio',)]
    executePlots(frames, classificationSet, title, labels, numbertitle)


def fortythreeGraph(numbertitle):
    classificationSet = [[male_leaders, 'Male Leaders'], [maleagent, 'Agents']]
    title = 'Distance travelled by abbots and agents of Giorgio al Palazzo'
    labels = ['Leader', 'Agent']
    frames = [('Giorgio_Palazzo',),('Giorgio_Palazzo',)]
    executePlots(frames, classificationSet, title, labels, numbertitle)

def fortyfourGraph(numbertitle):
    classificationSet = [[male_leaders, 'Male Leaders'], [maleagent, 'Agents']]
    title = 'Distance travelled by abbots and agents of Maria del Monte'
    labels = ['Leader', 'Agent']
    frames = [('MariaDelMonte',),('MariaDelMonte',)]
    executePlots(frames, classificationSet, title, labels, numbertitle)

#doctypes 
doctypes = ['Investiture', 'Sentence', 'Exchange', 'Promise', 'Payment', 'Renounciation', 'Sale']
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

def BarGraphPlot (list, monastery, func, title, numbertitle):
    frame = create_frame(list, monastery, func)
    leader_x_point = np.arange(1, 4*len(doctypes), 4)
    agent_x_point = np.arange(2, 4*len(doctypes) +1, 4)
    middle_x_point = np.arange(1.5, 4*len(doctypes) +1, 4)
    points = [leader_x_point, agent_x_point]
    x= 0
    fig, ax = plt.subplots (figsize=(8,8))
    for fication, mask in list:
        ax.bar(points[x], frame[mask],label=mask)
        x += 1     
    ax.legend(loc='upper center', prop={'size': 15})
    ax.set_xlabel('Parchment  Type',   size=15)
    ax.set_ylabel ('Parchment  Count', size=15)
    plt.title(title, size=15)
    plt.xticks(middle_x_point, doctypes, size=10)
    plt.savefig(f'{directoryToSaveGraphs}/{numbertitle}.jpeg')
    plt.close(fig)

def fortyfiveGraph(numbertitle):
    title = 'Documents led by abbesses and agents'
    BarGraphPlot (female_actors, female, create_list, title, numbertitle)

def fortysixGraph(numbertitle):
    title = 'Documents led by male leaders and agents'
    BarGraphPlot (male_actors, male, create_list, title, numbertitle)

def fortysevenGraph(numbertitle):
    title = 'Documents led by abbess and agents of Monastero Maggiore'
    BarGraphPlot (female_actors, ('Monastero Maggiore',), create_list, title, numbertitle)

def fortyeightGraph(numbertitle):
    title = 'Documents led by abbess and agents of Maria al Lentasio'
    BarGraphPlot (female_actors, ('Lentasio',), create_list, title,numbertitle)

def fortynineGraph(numbertitle):
    title = 'Documents led by abbess and agents of Apollinare'
    BarGraphPlot (female_actors, ('Apollinare',), create_list, title, numbertitle)

def fiftyGraph(numbertitle):
    title = 'Documents led by abbess and agents of monastery Sant Ambrogio'
    BarGraphPlot (male_actors, ('Ambrogio',), create_list, title, numbertitle)

def fiftyfirstGraph(numbertitle):
    title = 'Documents led by abbess and agents of San Giorgio al Palazzo'
    BarGraphPlot (male_actors, ('Giorgio_Palazzo',), create_list, title, numbertitle)

def fiftysecondGraph(numbertitle):
    title = 'Documents led by abbess and agents of Santa Maria del Monte'
    BarGraphPlot (male_actors, ('MariaDelMonte',), create_list, title, numbertitle)

# dispute distribution
religious = ('1', '2','5')
lay = ('3', '4')
def getTotalDispute (instituion, type):
    querySentence = '''
    SELECT COUNT (*) FROM alldocuments 
    WHERE doctype = '3'
    AND docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.monasteryname IN %s)
    AND docid  IN (
        SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        WHERE activity = '14'
    )
    '''
    queryDenounciation= '''
    SELECT COUNT (*) FROM alldocuments 
    WHERE doctype = '3'
    AND docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.monasteryname  IN %s)
    AND docid NOT  IN (
        SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        WHERE activity = '14'
    )
    '''
    querylay = '''
    SELECT COUNT (*) FROM alldocuments 
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE  monastery.monasteryname IN %s)
    AND docid  IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.type_instiution IN %s)
    AND doctype = '3'
    '''

    if type == 'd':
        dc.execute(queryDenounciation, [instituion])
        return dc.fetchall()[0][0]
    elif type == 's':
        dc.execute(querySentence, [instituion])
        return dc.fetchall()[0][0]
    if type == 'l':
        dc.execute(querylay, [instituion, lay])
        return dc.fetchall()[0][0]
    elif type == 'r':
        dc.execute(querylay, [instituion, religious])
        return dc.fetchall()[0][0]


def getSentenceNumber (institutions, classification):
    queryLeader = '''
        SELECT COUNT (*) FROM alldocuments 
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE  classification.classification IN %s AND monastery.monasteryname IN %s)
    AND docid  IN (
        SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        WHERE activity = '14'
    )
    AND doctype = '3'
    '''
    queryAgent = '''
        SELECT COUNT (*) FROM alldocuments 
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE  classification.classification IN %s AND monastery.monasteryname IN %s)
    AND docid IN (
        SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        WHERE activity = '14'
    )
    AND docid NOT IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE classification.classification IN %s AND monastery.monasteryname IN %s)
    AND doctype = '3'
    '''


    leadersentence = []
    agentsentence = []
   
    for mon in institutions:
        mon = (mon,)
        dc.execute(queryLeader, [classification[0][0], (mon,)])
        result = dc.fetchall()[0][0]
        if getTotalDispute(mon, 's') == 0:
            pass
        else:
            leadersentence.append(result/getTotalDispute(mon, 's'))
       
        dc.execute(queryAgent, [classification[1][0], (mon,), classification[0][0], (mon,)])
        result = dc.fetchall()[0][0]
        if getTotalDispute(mon, 's') == 0:
            pass
        else:
            agentsentence.append(result/getTotalDispute(mon, 's'))
        
       
    forframe = {'Leader sentence' : leadersentence, 'Agent sentence': agentsentence}
    frame = pd.DataFrame(forframe)
    return frame

def getdenounciationnum (institution, classification):
    queryLeaderD = '''
        SELECT COUNT (*) FROM alldocuments 
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE  classification.classification IN %s AND monastery.monasteryname IN %s)
    AND docid  NOT IN (
        SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        WHERE activity = '14'
    )
    AND doctype = '3'
    '''
    queryAgentd = '''
        SELECT COUNT (*) FROM alldocuments 
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE  classification.classification IN %s AND monastery.monasteryname IN %s)
    AND docid NOT IN (
        SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        WHERE activity = '14'
    )
    AND docid NOT IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE classification.classification IN %s AND monastery.monasteryname IN %s)
    AND doctype = '3'
    '''
    leaderden = []
    agentden = []
    for mon in institution:
        mon = (mon,)
        dc.execute(queryLeaderD, [classification[0][0], (mon,)])
        result = dc.fetchall()[0][0]

        if getTotalDispute(mon, 'd') == 0:
            pass
        else:
            leaderden.append(result/getTotalDispute(mon, 'd'))

        dc.execute(queryAgentd, [classification[1][0], (mon,), classification[0][0], (mon,)])
        result = dc.fetchall()[0][0]
        if getTotalDispute(mon, 'd') == 0:
            pass
        else:
            agentden.append(result/getTotalDispute(mon, 'd'))
    forframe = {'Leader denounciation' : leaderden, 'Agent denounciation': agentden}
    frame = pd.DataFrame(forframe)
    return frame

def laynumber (institutions, classification):
    queryLeader = '''
        SELECT COUNT (*) FROM alldocuments 
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE  classification.classification IN %s AND monastery.monasteryname IN %s)
    AND docid  IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.type_instiution IN %s)
    AND doctype = '3'
    '''
    queryAgent = '''
        SELECT COUNT (*) FROM alldocuments 
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE  classification.classification IN %s AND monastery.monasteryname IN %s)
    AND docid  IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.type_instiution IN %s)
    AND docid NOT IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE classification.classification IN %s AND monastery.monasteryname IN %s)
    AND doctype = '3'
    '''
   
    leaderreligious = []
    agentreligious = []
    leaderlay = []
    agentlay = []
    for mon in institutions:
        mon = (mon,)
        dc.execute(queryLeader, [classification[0][0], (mon,), religious])
        result = dc.fetchall()[0][0]
        if getTotalDispute(mon, 'r') == 0:
            pass
        else:
            leaderreligious.append(result/getTotalDispute(mon, 'r'))
       
        dc.execute(queryAgent, [classification[1][0], (mon,),religious, classification[0][0], (mon,)])
        result = dc.fetchall()[0][0]
        if getTotalDispute(mon, 'r') == 0:
            pass
        else:
            agentreligious.append(result/getTotalDispute(mon, 'r'))

        dc.execute(queryLeader, [classification[0][0], (mon,), lay])
        result = dc.fetchall()[0][0]
        if getTotalDispute(mon, 'l') == 0:
            pass
        else:
            leaderlay.append(result/getTotalDispute(mon, 'l'))
       
        dc.execute(queryAgent, [classification[1][0], (mon,),lay, classification[0][0], (mon,)])
        result = dc.fetchall()[0][0]
        if getTotalDispute(mon, 'l') == 0:
            pass
        else:
            agentlay.append(result/getTotalDispute(mon, 'l'))
    religiousframe = {'Leader religious' : leaderreligious, 'Agent religious': agentreligious}
    layframe = {'Leader lay' : leaderlay, 'Agent lay': agentlay}
    frame1 = pd.DataFrame(religiousframe)
    frame2 = pd.DataFrame(layframe)
    return frame1, frame2

def createDisputeDistributions(column, title, frames, labels, numbertitle):  
    ttest, pvalue = stats.ttest_ind(frames[column[0]],frames[column[1]], equal_var=False)
    fig, ax = plt.subplots(figsize = (7,7))
    standard_plot(ax,column[0], frames, labels[0])
    standard_plot(ax,column[1], frames, labels [1])
    ax.set_xlabel('Percentage',size=15)
    ax.set_ylabel('Frequency' ,size=15)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label=f'P-Value {pvalue.round(2)}'))
    ax.legend(handles=handles)
    plt.title(title)
    plt.savefig(f'{directoryToSaveGraphs}/{numbertitle}.jpeg')
    plt.close(fig)



def fiftythirdGraph(numbertitle):
    frame = getSentenceNumber(female, female_actors)
    columns = ['Leader sentence', 'Agent sentence']
    title = 'Distribution of sentences by abbesses and agents'
    lables = ['Abbesses', 'Agents']
    createDisputeDistributions(columns, title, frame, lables, numbertitle)


def fiftyfourthGraph(numbertitle):
    frame = getdenounciationnum(female, female_actors)
    columns = ['Leader denounciation', 'Agent denounciation']
    title = 'Distribution of denounciations by abbesses and agents'
    lables = ['Abbesses', 'Agents']
    createDisputeDistributions(columns, title, frame, lables,numbertitle)

def fiftyfifthGraph(numbertitle):
    frame = getSentenceNumber(male, male_actors)
    columns = ['Leader sentence', 'Agent sentence']
    title = 'Distribution of sentences by male leaders and agents'
    lables = ['Leader', 'Agents']
    createDisputeDistributions(columns, title, frame, lables, numbertitle)

def fiftysixthGraph(numbertitle):
    frame = getdenounciationnum(male, male_actors)
    columns = ['Leader denounciation', 'Agent denounciation']
    title = 'Distribution of denounciations by male leaders and agents'
    lables = ['Leader', 'Agents']
    createDisputeDistributions(columns, title, frame, lables, numbertitle)

def fiftyseventhGraph(numbertitle):
    religiousFrame, layFrame = laynumber(female, female_actors)
    columns = ['Leader lay', 'Agent lay']
    title = 'Distribution of lay disputes by abbesses and agents'
    lables = ['Abbesses', 'Agents']
    createDisputeDistributions(columns, title, layFrame, lables, numbertitle)

def fiftyeightGraph(numbertitle):
    religiousFrame, layFrame = laynumber(female, female_actors)
    columns = ['Leader religious', 'Agent religious']
    title = 'Distribution of religious disputes by abbesses and agents'
    lables = ['Abbesses', 'Agents']
    createDisputeDistributions(columns, title, religiousFrame, lables, numbertitle)

def fiftynineGraph(numbertitle):
    religiousFrame, layFrame = laynumber(male, male_actors)
    columns = ['Leader lay', 'Agent lay']
    title = 'Distribution of lay disputes by male leaders and agents'
    lables = ['Leaders', 'Agents']
    createDisputeDistributions(columns, title, layFrame, lables, numbertitle)

def sixtyGraph(numbertitle):
    religiousFrame, layFrame = laynumber(male, male_actors)
    columns = ['Leader religious', 'Agent religious']
    title = 'Distribution of religious disputes by male leaders and agents'
    lables = ['Leader', 'Agents']
    createDisputeDistributions(columns, title, religiousFrame, lables, numbertitle)

def sixtyoneGraph(numbertitle):
    classificationSet = [[female_leaders, 'Abbesses'], [femaleagent, 'Agents for nunneries']]
    title = 'Distance travelled by abbesses and agents for disputes'
    labels = ['Abbess', 'Agent']
    frames = [female, female]
    executePlots(frames, classificationSet, title, labels, doctype=['Sentence', 'Sentence'], numbertitle=numbertitle)

def sixtytwoGraph(numbertitle):
    classificationSet = [[male_leaders, 'Male Leaders'], [maleagent, 'Agents']]
    title = 'Distance travelled by male leaders and agents for disputes'
    labels = ['Leader', 'Agent']
    frames = [male, male]
    executePlots(frames, classificationSet, title, labels, doctype=['Sentence', 'Sentence'], numbertitle=numbertitle)

def doctype_data (doctype, monastery, fication, mask):
    query = '''
    SELECT ST_AsText(geocoordinates::geometry),coordinates.locations, count(*) FROM alldocuments
    JOIN coordinates ON coordinates.coordid = alldocuments.redaction
    JOIN doctype ON alldocuments.doctype = doctype.id
    WHERE doctype.translation = %s
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
    JOIN doctype ON alldocuments.doctype = doctype.id
    WHERE doctype.translation = %s
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
        dc.execute(query, [doctype, monastery, fication])
        frame_landloc = pd.DataFrame(dc.fetchall(), columns=('Geometry', 'location', 'Count'))
        geoframe_landloc = gpd.GeoDataFrame(frame_landloc, geometry=gpd.GeoSeries.from_wkt(frame_landloc ['Geometry'], crs='EPSG:4326'))
    else: 
        dc.execute(q_clergy, [doctype, monastery, fication, male_leaders, monastery])
        frame_landloc = pd.DataFrame(dc.fetchall(), columns=('Geometry', 'location', 'Count'))
        geoframe_landloc = gpd.GeoDataFrame(frame_landloc, geometry=gpd.GeoSeries.from_wkt(frame_landloc ['Geometry'], crs='EPSG:4326'))
    geoframe_landloc.sort_values('Count', ascending= False, inplace= True)
    return geoframe_landloc


def sixtythreeGraph(numbertitle):
    titles = ['By the abbess','By the agents', 'Redaction of Monastero Maggiore disputes'] 
    leaderFrame = doctype_data('Sentence',('Monastero Maggiore',),female_leaders, 'Abbess')
    agentFrame = doctype_data( 'Sentence',('Monastero Maggiore',),femaleagent , 'Agents')
    createMultiMap([leaderFrame, agentFrame], titles, directoryToSaveGraphs,numbertitle)

def sixtyfourGraph(numbertitle):
    classificationSet = [[female_leaders, 'Abbesses'], [femaleagent, 'Agents for nunneries']]
    title = 'Distance travelled for disputes of Monastero Maggiore'
    labels = ['Abbess', 'Agent']
    frames = [('Monastero Maggiore',),('Monastero Maggiore',)]
    executePlots(frames, classificationSet, title, labels, doctype=['Sentence', 'Sentence'], numbertitle=numbertitle)

def sixtyfiveGraph(numbertitle):
    classificationSet = [[femaleagent, 'Agents for nunneries'], [femaleagent, 'Agents for nunneries']]
    title = 'Travel of Monastero Maggiore agents for disputes and for all documents'
    labels = ['Normal agents', 'Agent for disputes']
    frames = [('Monastero Maggiore',),('Monastero Maggiore',)]
    executePlots(frames, classificationSet, title, labels, doctype=['Null', 'Sentence'], numbertitle=numbertitle)

def sixtysixthGraph(numbertitle):
    classificationSet = [[male_leaders, 'Male Leaders'], [maleagent, 'Agents']]
    title = 'Distance travelled for disputes of monastery Sant Ambrogio'
    labels = ['Abbot', 'Agent']
    frames = [('Ambrogio',),('Ambrogio',)]
    executePlots(frames, classificationSet, title, labels, doctype=['Sentence', 'Sentence'], numbertitle=numbertitle)

def sixtyeigthGraph(numbertitle):
    classificationSet = [[maleagent, 'Agents'], [maleagent, 'Agents']]
    title = 'Travel of Sant Ambrogio agents for disputes and for all documents'
    labels = ['Normal agents', 'Agent for disputes']
    frames = [('Ambrogio',),('Ambrogio',)]
    executePlots(frames, classificationSet, title, labels, doctype=['Null', 'Sentence'], numbertitle=numbertitle)
#leaderGraph
def doclist_per_leader (list, institutions, actors):
    number_query = '''
        SELECT COUNT(*), year as decade FROM alldocuments 
        WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
        JOIN actor ON actor.docid = alldocuments.docid 
        JOIN monastery ON monastery.monasteryid = actor.monastery
        JOIN classification ON classification.classid = actor.classification
        WHERE  classification.classification IN %s AND monastery.monasteryname IN %s)
        AND year BETWEEN %s::timestamp AND %s::timestamp
        GROUP BY decade
        ORDER BY decade
    '''
    query_without_abbot = '''
        SELECT COUNT(*), year as decade FROM alldocuments 
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
        AND year BETWEEN %s::timestamp AND %s::timestamp
        GROUP BY decade
        ORDER BY decade
    '''
    list_of_list = []
    for name, beg, end in list:
        beg = f'{beg}-01-01'
        end = f'{end}-01-01'
        main_count = []
        for classification, mask in actors:
            counts = []
            if classification == ('Clergymen',):
                dc.execute(query_without_abbot, [classification, institutions, male_leaders, institutions, beg, end])
                for count, decade in dc.fetchall():
                    counts.append([count, decade])
            elif classification == ('Nun',):
                dc.execute(query_without_abbot, [classification, institutions, female_leaders, institutions, beg, end])
                for count, decade in dc.fetchall():
                    counts.append([count, decade])
            else:
                dc.execute(number_query, [classification, institutions, beg, end])
                for count, decade in dc.fetchall():
                    counts.append([count, decade])
            main_count.append([counts, mask])
        list_of_list.append([name, main_count,])
    return list_of_list


def year_data (list):
    year = []
    number = []
    for num, ye in list:
        year.append(ye)
        number.append(num)
    return year, number

def create_plot (mask, list, title, directoryToSaveGraphs):
    fig, ax = plt.subplots(figsize= (7,7))
    year, number = year_data(list) 
    ax.plot(year, number, label=mask)
    plt.title(title)
    plt.legend()
    plt.savefig(f'{directoryToSaveGraphs}/{title}.jpeg')
    plt.close(fig)

def graph_per_leader (list, data, mon, numbertitle):   
        for name, lists in doclist_per_leader(list, mon, data):
            fig, ax = plt.subplots(figsize= (7,7))
            for counts, masks in lists:
                year, number = year_data(counts) 
                ax.plot (year, number, label= masks)
            ax.set_ylabel('Number of documents')
            ax.set_xlabel('Years')
            title = f'Document per year under {name}'
            plt.title(title)
            plt.legend()
            plt.savefig(f'{directoryToSaveGraphs}/{numbertitle}.jpeg')
            plt.close(fig)
            
def sixtynineGraph(numbertitle):
    maleActors =[[male_leaders, 'Male Leaders'], [male_clergy, 'Clergy'], [male_intermediaries, 'Intermediaries']]
    graph_per_leader ([['Guglielmo Cotta', '1235', '1267']], maleActors, ('Ambrogio',),numbertitle)

def seventyGraph(numbertitle):
    title = 'Documents led by abbess and nuns of Ambrogio of Rivolta'
    actors =[[female_leaders, 'Abbesses']  , [('Nun',), 'Agents for nunneries']]
    BarGraphPlot (female_actors, ('AmbrogioRivalta',), create_list, title, numbertitle)


def allGraphs():
    #Should be 11, women overtime
    firstGraph(11)
    #should be 12, doc with men adn women
    secondGraph(12)
    #should be 13 doc money overtime
    thirdGraph(13)
    #shuold be 14 lone men and son
    fourthGraph(14)
    #should be 15 ovrall distribution
    fifthGraph(15)
    #should be 16 overall top ten
    sixthGraph(16)
    # without outliers overall 17
    seventhGraph(17)
    #denari ten 18
    eigthGraph(18)
    # denari without out 19
    ninthGraph(19)
    # location side by side 20
    tenthGraph(20)
    # network side by side 21
    eleventhGraph(21)
    # broletto market 22
    twelfthGraph(22)
    # mm and canonica 23
    thirteenthGraph(23)
    # aurona margherita 24 
    fourteenthGraph(24)
    # four nunneries 25
    fifteenthGraph(25)
    # apollinare 26
    sixteenthGraph(26)
    # radegonda 27
    seventeenthGraph(27)
    #four male 28
    eighteenthGraph(28)
    #three dispute male 29
    nineteenthGraph(29)
    #chiaravalle mmonte 30
    twentiethGraph(30)
    #overall market 31
    twentyFirtstGraph(31)
    #humilati 32
    twentySecondGraph(32)
    #members all 33
    twentyThirdGraph(33)
    #members top 30, 34
    twentyfourthGraph(34)
    #members top 10, 34
    twentyfiveGraph(35)
    #memebership and wealth 36
    twentysixthGraph(36)
    #regression male leaders agents 37
    twentyseventhGraph(37)
    # regression female leaders 38
    twentyeightGraph(38)
    #regression mm 39
    twentyninthGraph(39)
    #regression ambrogio 40
    thirtiethGraph(40)
    #regression monte 41
    thirtyfirstGraph(41)
    #regression lentasion 42
    thirtysecondGraph(42)
    #regression apollinare 43
    thirtythirdGraph(43)
    #regression giorgio 44
    thirtyfourthGraph(44)
    #abbess agent overtime 45
    seventyfirstGraph(45)
    #leader agent overtime 46
    seventysecondGraph(46)
    # maria del monte overtime 47
    seventythirdGraph(47)
    #apollinare 48
    seventyfourthGraph(48)
    #distance travel nunneries 49
    thirtyfifthGraph(49)
    #distanfce male 50
    thirtysixthGraph(50)
    #abbesses male laders 51
    thirtyseventhGraph(51)
    #map MM 52
    thirtyeigthGraph(52)
    # mm distance travel 53
    thirtynineGraph(53)
    #map mm overtime 54
    fortyGraph(54)
    # map ambrogio 55
    fortyoneGraph(55)
    #ambrogio distance 56 
    fortytwoGraph(56)
    #giorgio distance 57
    fortythreeGraph(57)
    #mmonte distance 58
    fortyfourGraph(58)

    fortyfiveGraph(59)
    fortysixGraph(60)
    fortysevenGraph(61)
    fortyeightGraph(62)
    fortynineGraph(63)
    fiftyGraph(64)
    fiftyfirstGraph(65)
    fiftysecondGraph(66)
    fiftythirdGraph(67)
    fiftyfourthGraph(68)
    fiftyfifthGraph(69)
    fiftysixthGraph(70)
    fiftyseventhGraph(71)
    fiftyeightGraph(72)
    fiftynineGraph(73)
    sixtyGraph(74)
    sixtyoneGraph(75)
    sixtytwoGraph(76)
    sixtythreeGraph(77)
    sixtyfourGraph(78)
    sixtyfiveGraph(79)
    sixtysixthGraph(80)
    sixtyeigthGraph(81)
    sixtynineGraph(82)
    seventyGraph(83)


#allGraphs()