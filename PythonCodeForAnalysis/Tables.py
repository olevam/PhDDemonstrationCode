from Database_connecting import connecting as database
dc = database.connect('womeninmilanphd')
import pandas as pd
from matplotlib import pyplot as plt
import InfoMon as im
import numpy as np
import scipy.stats as stats

allinstitutions = im.get_monasteires('a')
maleMonasteries = ("Ambrogio", "CanonicaAmbrogio", "MilanArchdioces", "Chiaravalle", "Lorenzo", "Giorgio_Palazzo", "Simpliciano", "MariaDelMonte", "DecumaniMilanesi", "OspedaleBroletoMilano")
femaleMonasteries = ("Monastero Maggiore", "Apollinare", "Radegonda", "Lentasio", "MariaAurona", "Margherita", "Agnese", "SanFelice", "AmbrogioRivalta", "CappucineConcorezzo")
top20Frame = pd.read_excel('C:/Users/Olevam/OneDrive - University of Glasgow/PhD_WomenInMilan/Results/WealthData/Top20All.xlsx')
top20 = (maleMonasteries+femaleMonasteries)
top10Female = top20Frame[top20Frame['Monastery'].isin(femaleMonasteries)] 
top10Male = top20Frame[top20Frame  ['Monastery'].isin(maleMonasteries)]  
directory_excels = 'C:/Users/Olevam/OneDrive - University of Glasgow/PhD_WomenInMilan/Results/Lay/Excels'
genders = [['y', 'Female'], ['n', 'Male']]
male = im.get_monasteires('m')
female = im.get_monasteires('f')
male_leaders = ('Abbott', 'Preposto', 'Archpriest', 'Archbishop')
female_leaders = ('Abbess',)
male_clergy = ('Clergymen',)
female_clergy = ('Nun',)
male_intermediaries = ('Intermediary', 'Laybrother')
#male_actors = [[male_leaders, 'Male Leaders'], [male_clergy, 'Clergy'],[male_clergy, 'Clergy with leader'], [male_intermediaries, 'Intermediaries']]
#female_actors = [[female_leaders, 'Female Leaders'], [female_clergy, 'Nuns'],[female_clergy, 'Nuns with Abbess'], [male_intermediaries, 'Intermediaries of nunneries'], [male_clergy, 'Clergy']]


masks = [['Male Leaders', 'Female Leaders'], ['Clergy', 'Nuns'], ['Clergy with leader', 'Nuns with Abbess' ], ['Intermediaries','Intermediaries of nunneries'], ['Agent_male', 'Agent_female'] ]
maleagent = ('Intermediary', 'Laybrother', 'Clergymen')
femaleagent = ('Intermediary', 'Laybrother', 'Nun')
male_actors =   [[male_leaders, 'Male Leaders'], [maleagent, 'Agents']]
female_actors = [[female_leaders, 'Abbesses']  , [femaleagent, 'Agents for nunneries']]

class frameSet ():
    def __init__(self, frame1, frame2, label1, label2, type) -> None:
        self.frame1 = frame1
        self.frame2 = frame2
        self.label1 =label1
        self.label2 =label2
        self.type = type

topTenFrame = frameSet (top10Male, top10Female, 'Male institutions', 'Female institutions', 'Top ten')     
removeOutliersFrame = frameSet (top10Male.iloc[2:10], top10Female.iloc[2:10], 'Male institutions', 'Female institutions', 'Removing Outliers')          
locationFrame = frameSet(top10Male.iloc[1:10], top10Female.iloc[1:10], 'Male institutions', 'Female institutions', 'Removing Outliers')


def classificationTables (): 
    def get_frame_typeofwomem (gender):
        query = '''
            SELECT COUNT (*), classification.classification as type FROM actor
            JOIN monastery ON monastery.monasteryid = actor.monastery
            JOIN classification ON actor.classification = classification.classid
            WHERE female = %s
            AND type_instiution = '3'
            GROUP BY type
            ORDER BY COUNT DESC
            '''
        dc.execute(query, [gender])
        frame = pd.DataFrame(dc.fetchall(), columns= ['Count', 'classification'])
        total = frame['Count'].sum()
        frame['Percentage'] = frame['Count']/total
        print (frame)
        return frame.sort_values('Count', ascending=False)
    for type, name in genders:
        get_frame_typeofwomem(type).to_csv(f'{directory_excels}/clasification_type{name}.csv')
    
def get_number (classification, mask, inst):
        number_query = '''
            SELECT COUNT(*) FROM alldocuments 
            WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
            JOIN actor ON actor.docid = alldocuments.docid 
            JOIN monastery ON monastery.monasteryid = actor.monastery
            JOIN classification ON classification.classid = actor.classification
            WHERE  classification.classification IN %s AND monastery.monasteryname IN %s)
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
        '''
        query_with_abbot = '''
            SELECT COUNT(*) FROM alldocuments 
            WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
            JOIN actor ON actor.docid = alldocuments.docid 
            JOIN monastery ON monastery.monasteryid = actor.monastery
            JOIN classification ON classification.classid = actor.classification
            WHERE  classification.classification IN %s AND monastery.monasteryname IN %s )
            AND docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
            JOIN actor ON actor.docid = alldocuments.docid 
            JOIN classification ON classification.classid = actor.classification
            JOIN monastery ON monastery.monasteryid = actor.monastery
            WHERE classification.classification IN %s AND monastery.monasteryname IN %s)
        '''
        if classification == ('Clergymen',) and mask == 'Clergy':
            dc.execute(query_without_abbot, [classification, inst, male_leaders, inst])
            return dc.fetchall()[0][0]
        elif classification == ('Nun',) and mask == 'Nuns':
            dc.execute(query_without_abbot, [classification, inst, female_leaders, inst])
            return dc.fetchall()[0][0]
        elif classification == ('Clergymen',) and mask == 'Clergy with leader':
            dc.execute(query_with_abbot, [classification, inst, male_leaders, inst])
            return dc.fetchall()[0][0]
        elif classification == ('Nun',) and mask == 'Nuns with Abbess':
            dc.execute(query_with_abbot, [classification, inst, female_leaders, inst])
            return dc.fetchall()[0][0]
        elif mask == 'Agent_female':
            dc.execute(query_without_abbot, [classification, inst, female_leaders, inst])
            return dc.fetchall()[0][0]
        elif mask == 'Agent_male':
            dc.execute(query_without_abbot, [classification, inst, male_leaders, inst])
            return dc.fetchall()[0][0]
        else:
            dc.execute(number_query,[classification, inst])
            return dc.fetchall()[0][0]


def get_total (inst):
        number_query = '''
            SELECT COUNT(*) FROM alldocuments 
            WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
            JOIN actor ON actor.docid = alldocuments.docid 
            JOIN monastery ON monastery.monasteryid = actor.monastery
            JOIN classification ON classification.classid = actor.classification
            WHERE monastery.monasteryname IN %s)
        '''

        dc.execute(number_query,[ inst])
        return dc.fetchall()[0][0]
def numberOfReligiousLeadersAndOtherActors ():
    def class_table (actors, inst):
        frame = pd.DataFrame(index=['Number', 'Percentage'])
        total = get_total(inst)
        relative_total = 0
        for list, mask in actors:
            number = get_number(list, mask, inst)
            percentage = int((number/total)*100)
            frame[mask] = [number, percentage]
            relative_total += number
        frame['Relative total'] = [relative_total, (relative_total/total)*100]
        frame['Total'] = [total, (total/total)*100]
        return frame


    class_table(female_actors, female).to_excel(f'{directory_excels}/Number_femaleactor.xlsx')
    class_table(male_actors, male).to_excel(f'{directory_excels}/Number_maleactor.xlsx')

## for all the tables of wealth look at Wealth Rank 

def tableTTESTLeadershipAction ():

    maleActors = [[male_leaders, 'Male Leaders'], [male_clergy, 'Clergy'],[male_clergy, 'Clergy with leader'], [male_intermediaries, 'Intermediaries'], [maleagent, 'Agent_male']]
    femaleActors = [[female_leaders, 'Female Leaders'], [female_clergy, 'Nuns'],[female_clergy, 'Nuns with Abbess'], [male_intermediaries, 'Intermediaries of nunneries'], [femaleagent, 'Agent_female']]

    def getPercentList (inst, actors):
        frame = pd.DataFrame()
        for list, mask in actors:
            frame['Monastery'] = inst
            masklist = []
            for mon in inst:
                number = get_number(list, mask, (mon,))
                masklist.append(number)
            frame[mask] = masklist
        cols = frame.columns
        frame['TotalDocs'] = frame[cols[1]] + frame[cols[2]] + frame[cols[4]]
        percentageFrame = frame.iloc[:, 1:-1].apply(lambda x: x / frame['TotalDocs']).round(4)
        percentageFrame['Monastery'] = frame['Monastery']
        return percentageFrame


    def mean_std (sample):
        average = sample.mean()
        standard = np.std(sample)
        return average, standard

    males = getPercentList(maleMonasteries, maleActors)
    females = getPercentList(femaleMonasteries, femaleActors)

    def ttest_ind (column_1, column_2):
        return stats.ttest_ind(males[column_1], females[column_2], equal_var=False)

    def run_ttests ():
        results = []
        for mmask, fmask in masks:
            stat, p = ttest_ind(mmask, fmask)
            avg_men, std_men = mean_std(males[mmask])
            avg_wmen, std_wmen = mean_std(females[fmask])
            results.append([f'{mmask, fmask}', stat, p, avg_men, avg_wmen])

        ttest_frame = pd.DataFrame(results, columns= ['Column', 'T-statistic', 'P-value', 'Average male institutions', 'Average female institutions'])

        return ttest_frame
    run_ttests().to_excel(f'{directory_excels}/ttests_avgs.xlsx')
    getPercentList(femaleMonasteries, female_actors).to_excel(f'{directory_excels}/femalePercent.xlsx')
    getPercentList(maleMonasteries, male_actors).to_excel(f'{directory_excels}/malePercent.xlsx')

def regressionBusyiness():
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
        return res, res.rvalue **2
    
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

    def singleMonData (monastery, gender):
        agent, leader = get_numeber_years(gender, monastery)
        res, resS = regression(leader,agent)
        return res, resS

    def createFrame (lists):
        restulDitct = {}
        for inst, gender in lists:
            res, resS = singleMonData((inst,), gender) 
            restulDitct[inst] = [res.slope, res.rvalue, resS]
        frame = pd.DataFrame.from_dict(restulDitct, orient= 'index')
        frame.columns = ['Slope', 'R value', 'R Squared']
        return frame 
    sixInstitutions = [['Monastero Maggiore', 'f'],['Apollinare', 'f'],['Margherita', 'f'],['Ambrogio', 'm'],['Giorgio_Palazzo', 'm'],['MariaDelMonte', 'm']]
    #createFrame(sixInstitutions).to_excel(f'{directory_excels}/sixInstitutions.xlsx')
    sixClearInst =[['Monastero Maggiore', np.arange(110, 130, 1), 'f'], ['Apollinare', np.arange(122, 131, 1), 'f'], ['Lentasio', np.arange(120, 130, 1), 'f'], ['Ambrogio', np.arange(110, 130, 1), 'm'], ['Giorgio_Palazzo', np.arange(120, 130, 1), 'm'], ['MariaDelMonte', np.arange(119, 131, 1), 'm']]

    

    def regressionFrameSingleInst (years, first_list, second_list, monastery):
        res_leader, squaredLeader = regression(years, first_list)
        res_agent, squaredAgent = regression(years, second_list)
        leaderList = [res_leader.slope.round(2), res_leader.rvalue.round(2), squaredLeader.round(2), monastery]
        agentList = [res_agent.slope.round(2), res_agent.rvalue.round(2), squaredAgent.round(2), monastery]
        institutionFrame = pd.DataFrame([leaderList, agentList], columns =['Slope', 'R value', 'R Squared', 'Monastery'])
        return institutionFrame
    
    def multipleExcels (mon, year, gen):
        title = f'Document by male leaders and agents over time of {mon} '
        yearsList = year
        agent, abbess, yearsf = getPercentDecade(gen, (f'{mon}',), yearsList)
        return regressionFrameSingleInst(yearsf, abbess, agent, mon)
    frameList = []
    for mon, year, gen in sixClearInst:
        frameList.append(multipleExcels(mon, year, gen))
    finalFrame = pd.concat(frameList)
    finalFrame.to_excel(f'{directory_excels}/overtimeRegressionData.xlsx')
religious = ('1', '2','5')
lay = ('3', '4')
def getTotalDisputes (mons):
    query = '''
    SELECT COUNT (*) FROM alldocuments
    WHERE docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.monasteryname IN %s)
    AND doctype = 3
    '''
    dc.execute(query, [mons])
    print (dc.fetchall())
#getTotalDisputes(maleMonasteries)
#getTotalDisputes(femaleMonasteries)

def layecclesiasitcalframe (lisinstitutions,classification):
    religiousaddon = '''
    AND docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.monasteryname IN %s)
    '''
    layaddon = '''
    AND docid  IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE monastery.type_instiution IN ('3', '4'))
    '''
    queryLeader = '''
        SELECT COUNT (*) FROM alldocuments 
    WHERE doctype = 3
    AND docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
    JOIN actor ON actor.docid = alldocuments.docid 
    JOIN classification ON classification.classid = actor.classification
    JOIN monastery ON monastery.monasteryid = actor.monastery
    WHERE  classification.classification IN %s AND monastery.monasteryname IN %s)
    AND doctype = '3'
    '''
    queryAgent = '''
        SELECT COUNT (*) FROM alldocuments 
    WHERE doctype = 3
    AND docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments
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
    lay = ('3' ,'4')
    leaderreligious = 0
    agentreligious = 0
    leaderlay = 0
    agentlay = 0
    

    for institutions in lisinstitutions:
        allinstitutions.remove(institutions)
        otherInst = tuple(allinstitutions)
        institutions = (institutions,)
    

        dc.execute(queryLeader + religiousaddon, [classification[0][0], institutions, otherInst])
        result = dc.fetchall()[0][0]
        leaderreligious+=result
        dc.execute(queryAgent + religiousaddon, [classification[1][0], institutions, classification[0][0], institutions, otherInst])
        result = dc.fetchall()[0][0]
        agentreligious+=result

        dc.execute(queryLeader + layaddon, [classification[0][0], institutions])
        result = dc.fetchall()[0][0]
        leaderlay+=result
        dc.execute(queryAgent + layaddon, [classification[1][0], institutions, classification[0][0], institutions])
        result = dc.fetchall()[0][0] 
        agentlay+=result
    religiousSum = leaderreligious+agentreligious
    laySum = leaderlay+agentlay
    listValues = [leaderreligious,agentreligious,  leaderlay, agentlay]
    percentages = [leaderreligious/religiousSum, agentreligious/religiousSum, leaderlay/laySum, agentlay/laySum]
    frame = pd.DataFrame([listValues, percentages], columns=['Leader Religiuos', 'Agent Religious', 'Leader Lay', 'Agent Lay'])
    frame['Total Religious Disputes'] = [religiousSum, '1']
    frame['Total Lay Disputes'] = [laySum, '1']
    frame = frame[['Leader Religiuos', 'Agent Religious', 'Total Religious Disputes', 'Leader Lay', 'Agent Lay', 'Total Lay Disputes']]
    return frame



def MultiInstitutionFrame (list):
    instituinList = []
    bigFrameSentence =[]
    bigFrameLay = []
    for institutions, classifications in list:
        instituinList.append(institutions)
        bigFrameSentence.append(SentenceFrame(institutions, classifications))
        bigFrameLay.append(layecclesiasitcalframe(institutions, classifications))
    sentenceDenFrame = pd.concat(bigFrameSentence)
    sentenceDenFrame['Monastery'] = instituinList
    layEcclFrame = pd.concat(bigFrameLay)
    layEcclFrame['Monastery'] = instituinList
    return sentenceDenFrame, layEcclFrame



def finalEcclLayFrame ():
    arrays = [['Male Institutions', 'Male Institutions', 'Female Institutions', 'Female Institutions'], ['Number', 'Percentage', 'Number', 'Percentage']]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names = ['Institution', ''])
    columnArray = [['Religious Disputes', 'Religious Disputes', 'Religious Disputes', 'Lay Disputes', 'Lay Disputes', 'Lay Disputes'], ['Leader', 'Agent','Total', 'Leader', 'Agent', 'Total']]
    tupleColumn = list(zip(*columnArray))
    indexCol = pd.MultiIndex.from_tuples(tupleColumn, names = ['Type of Dispute', 'Actor'])
    malelayEcclesiastical = layecclesiasitcalframe(maleMonasteries, male_actors)
    femalelayEcclesiastical = layecclesiasitcalframe(femaleMonasteries, female_actors)
    allEcclLayframe = pd.concat([malelayEcclesiastical, femalelayEcclesiastical])
    allEcclLayframe.set_index(index, inplace=True)
    allEcclLayframe.columns= indexCol
    allEcclLayframe.to_excel(f'{directory_excels}/allEcclLayframe.xlsx')



def SentenceFrame (listinstitutions, classification):
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

    leadersentence = 0
    agentsentence = 0
    leaderden = 0
    agentden = 0
     
    

    for institutions in listinstitutions:
        institutions = (institutions,)
        dc.execute(queryLeader, [classification[0][0], institutions])
        result = dc.fetchall()[0][0]
        leadersentence+=result
        dc.execute(queryAgent, [classification[1][0], institutions, classification[0][0], institutions])
        result = dc.fetchall()[0][0]
        agentsentence+=result
        dc.execute(queryLeaderD, [classification[0][0], institutions])
        result = dc.fetchall()[0][0]
        leaderden+=result
        dc.execute(queryAgentd, [classification[1][0], institutions, classification[0][0], institutions])
        result = dc.fetchall()[0][0]
        agentden+=result

   
    #total = getTotalDispute(institutions)
    DenounciationSum = leaderden+agentden
    listDenounciations = [leaderden,agentden, DenounciationSum]
    percentageDenounciation = [leaderden/DenounciationSum, agentden/DenounciationSum, 1]
    SentenceSum = leadersentence+agentsentence
    listSentences = [leadersentence,  agentsentence, SentenceSum]
    percentagesSentences = [leadersentence/SentenceSum, agentsentence/SentenceSum, 1]
    frameDenounciations = pd.DataFrame([listDenounciations, percentageDenounciation], columns=['Leader Denounciation', 'Agent denounciation', 'Total'])
    frameSentences = pd.DataFrame([listSentences, percentagesSentences], columns= ['Leader Sentences', 'Agent Sentences', 'Total'])

    return frameDenounciations, frameSentences


def bringDenSentTogether ():
    maleden, malesen = SentenceFrame(maleMonasteries, male_actors) 
    femaleden, femalesen = SentenceFrame(femaleMonasteries, female_actors)
    arrays = [['Male Institutions', 'Male Institutions', 'Female Institutions', 'Female Institutions'], ['Number', 'Percentage', 'Number', 'Percentage']]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names = ['Institution', ''])
    denFrame = pd.concat([maleden, femaleden])
    denFrame.set_index(index, inplace=True)
    senFrame = pd.concat([malesen, femalesen])
    senFrame.set_index(index, inplace=True)
    return denFrame, senFrame


#denFrame, senFrame = bringDenSentTogether()
#denFrame.to_excel(f'{directory_excels}/denounciations.xlsx')
#senFrame.to_excel(f'{directory_excels}/sentences.xlsx')

##Table 14 I think I built this myself, using list leaders

def ExpenditureTable ():
    queryDoctype = ') AND doctype.translation = %s'

    def Doctypes():
        query = '''
            SELECT doctype.translation FROM doctype
        '''
        dc.execute(query)
        return dc.fetchall()

    # this function counts the sum of denari, given a condition
    def sumOfDenari (variables, begCondition = 'SELECT', endCondition = ')'):
        masterDenariQuery = '''
            SUM (price.price) FROM alldocuments
            JOIN doctype ON alldocuments.doctype = doctype.id
            JOIN price ON alldocuments.docid = price.docid 
            WHERE alldocuments.docid IN (SELECT DISTINCT alldocuments.docid FROM alldocuments 
            JOIN actor ON alldocuments.docid = actor.docid
            JOIN monastery ON monastery.monasteryid = actor.monastery
            WHERE monasteryname IN %s
        '''
        finalQuery = begCondition + masterDenariQuery + endCondition
        dc.execute(finalQuery, [variables[0],variables[1] ])
        return dc.fetchall()

    def singleMoneyDataFrames (listMon, name, beg = 'SELECT', end = ')'):
        list = []
        for monastery in listMon:
            result = sumOfDenari(monastery, beg, end)
            list.append([int(result[0][0]), monastery])
        frame = pd.DataFrame (list, columns=[name, monastery])
        return frame.sort_values(by=name, ascending=False)

    def doctypeDictionary (monasteryList, listDoctype, extraCondition = None):
        sumsPerDoctype = {}
        for doct in listDoctype:
            listPerDoc = []
            for monastery in monasteryList:
                if extraCondition == None:
                    result = sumOfDenari([monastery, doct[0]], endCondition= queryDoctype)
                else:
                    result = sumOfDenari([monastery, doct[0]], endCondition= extraCondition)
                if result[0][0] == None:
                    listPerDoc.append(0) 
                else:
                    listPerDoc.append(result[0][0])
            sumsPerDoctype[doct[0]]= listPerDoc
        return sumsPerDoctype


    def doctypeFrame (monList, names, specificDoctypes = Doctypes(),extraCon = None):
        dict = doctypeDictionary(monList, specificDoctypes, extraCon)
        frame = pd.DataFrame.from_dict(dict, orient='Index', columns=names)
        frame = frame.sort_values(by='Male', ascending=False)

        for col in frame.columns:
            frame[col] = frame[col].astype(float)
            frame[col + '_pct'] = (frame[col] / frame[col].sum() * 100).round(1)
        #return frame
        frame.to_excel(f'{directory_excels}/DoctypeDenariMaleFemale.xlsx')
    doctypeFrame([tuple(maleMonasteries), tuple(femaleMonasteries)], names = ['Male','Female'] )

managementDoc = ('11', '26', '1','7','15')
buyingDoc = ('6', '9', '17', '18')
disputeDoc = ('3','16','5')
doctypeList = [(managementDoc, 'Documents regarding management, (M in graphs)'),(buyingDoc,'Documents regarding expansion, (E in graphs)'), (disputeDoc, 'Documents regarding litigation, (L in graphs)')]
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

def allPercentageFrames ():
    arrays = [['All institutions','All institutions','Male Institutions', 'Male Institutions', 'Female Institutions', 'Female Institutions'], ['Count', 'Percentage', 'Count', 'Percentage', 'Count', 'Percentage']]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names = ['Type of Institution', ''])
    allManagement = getPercentageDoctype((maleMonasteries + femaleMonasteries))
    maleManagement = getPercentageDoctype(maleMonasteries)
    femaleManagement = getPercentageDoctype(femaleMonasteries)
    allframe = pd.concat([allManagement, maleManagement, femaleManagement], axis=1)
    allframe.columns= index
    return allframe

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

def chiSquare (mainData, compareData):
    excepted = mainData['Percentage']*compareData['Percentage'].sum()
    return stats.chisquare(compareData['Percentage'], excepted)


def chiSquareFrame (mainData, list):
    dictOfChi = {}
    for mon in list: 
        stat, pValue = chiSquare(mainData, singleMonData(mon))
        dictOfChi[mon] = [stat, round(pValue, 10)]
    frame = pd.DataFrame.from_dict(dictOfChi, 'Index', columns= [['xSquareStat', 'PValue']])
    return frame
def get_monasteries(mon):
    query = '''
        SELECT monasteryname, type_instiution.instiution FROM actor 
        JOIN monastery ON monastery.monasteryid = actor.monastery
        JOIN type_instiution ON type_instiution.type_id = monastery.type_instiution
            AND actor.docid IN (
                SELECT DISTINCT alldocuments.docid FROM alldocuments 
                JOIN actor ON alldocuments.docid = actor.docid
                JOIN monastery ON monastery.monasteryid = actor.monastery
                JOIN type_instiution ON type_instiution.type_id = monastery.type_instiution
                WHERE monasteryname = %s
            )
            AND monasteryname != %s
        ORDER BY type_instiution.type_id
        '''
    dc.execute(query, [mon, mon])
    return dc.fetchall()


def get_connected_monasteries(monasteries):
    typeDictionary = {}
    for monasteryname, typeInst in monasteries:
        if monasteryname == 'MilanArchdioces':
            continue
        elif typeDictionary.get(typeInst):
            typeDictionary[typeInst].append(monasteryname)
        else:
            typeDictionary[typeInst] = [monasteryname]
    return typeDictionary

def typeCount(mon):
    query = '''
        SELECT count(*) FROM alldocuments 
        WHERE alldocuments.docid IN (
            SELECT DISTINCT alldocuments.docid FROM alldocuments 
            JOIN actor ON alldocuments.docid = actor.docid
            JOIN monastery ON monastery.monasteryid = actor.monastery
            WHERE monasteryname = %s 
            )
        AND alldocuments.docid IN (
            SELECT DISTINCT alldocuments.docid FROM alldocuments 
            JOIN actor ON alldocuments.docid = actor.docid
            JOIN monastery ON monastery.monasteryid = actor.monastery
            JOIN type_instiution ON type_instiution.type_id = monastery.type_instiution
            WHERE type_instiution.instiution = %s AND monasteryname IN %s)
        '''
    types = []
    results = []
    listMons = get_connected_monasteries(get_monasteries(mon))
    for key, value in listMons.items():
        dc.execute(query, [mon, key, tuple(set(value))])
        results.append(dc.fetchall()[0][0])
        types.append(key)
    frame = pd.DataFrame([results], columns=types, index=[mon])
    return frame


def CreateDataframe (listmons):
    frameList = []
    for monastery in listmons:
        frameList.append(typeCount(monastery))
    frame = pd.concat(frameList,join='outer', axis=0)
    frame.fillna(0, inplace=True)
    frame =frame[['Simple_Lay', 'Religious_Male','Religious_Female']] 
    frame['TotalDocs'] = frame.sum(axis=1)
    percentageFrame = frame.iloc[:, :-1].apply(lambda x: x / frame['TotalDocs'] * 100).round(2)
    average_percentage = percentageFrame.mean()
    percentageFrame.fillna(0, inplace=True)  
    percentageFrame.sort_values(by='Religious_Female', ascending=False, inplace=True)
    percentageFrame.loc['Average'] = average_percentage.round(2)  
    return percentageFrame
## to get the average run this code for male and female groups 
## otherwise remove the line about the percentage and run the code as top20 for each institution