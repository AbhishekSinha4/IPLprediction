
# coding: utf-8

# In[1]:


from __future__ import print_function
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from numpy import array
import pandas as pd
import csv
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[2]:


from __future__ import print_function
spark = SparkSession.builder.appName('player-clustering').getOrCreate()

Bat_Clus = spark.read.options(header="true",
                               inferSchema="true",
                               nullValue="-",
                               mode="failfast").csv("Batting_Clusters.csv")

Bowl_Clus = spark.read.options(header="true",
                               inferSchema="true",
                               nullValue="-",
                               mode="failfast").csv("Bowling_Clusters.csv")

Over_Data = spark.read.options(header="true",
                               inferSchema="true",
                               nullValue="-",
                               mode="failfast").csv("Over_Data.csv")


# In[3]:


Over_Data.head()
Bowl_Clus = Bowl_Clus.toPandas()
Bat_Clus = Bat_Clus.toPandas()
Bowl_Clus.fillna(0)
Bat_Clus.fillna(0)
Bat_Clus.head()
Bowl_Clus.head()


# In[4]:


Over_df = Over_Data.rdd.repartition(5).toDF()
Over_df.fillna(0)
Over_df=Over_df.drop('Batsman')
Over_df=Over_df.drop('Bowler')
Over_df=Over_df.drop('NonStriker')
Over_R = Over_df.drop('TotalWickets')
Over_W = Over_df.drop('TotalRuns')
Over_W = Over_W.rdd
Over_R = Over_R.rdd
print(type(Over_W))


# In[5]:


def flatten(aList):
    t = []
    for i in aList:
        if not isinstance(i, list):
             t.append(i)
        else:
             t.extend(flatten(i))
    return t

def train_create(vec):
    res = vec[1]
    
    feat = []
    feat.append(vec[0])
    feat = feat + list(vec[2:])
    feat = flatten(feat)
    return LabeledPoint(res,feat)

wicketData = Over_W.map(train_create)
runsData = Over_R.map(train_create)

print(wicketData)


# In[6]:


def getBatsmanStat(name):
    player = Bat_Clus[Bat_Clus['Player Name']==name]
    if(player.empty):
        l = np.array(Bat_Clus.mean())
    else:
        player.drop('Player Name', axis=1, inplace=True)
        l = np.array(player.values[0])
    return l
        
def getBowlerStat(name):
    player = Bowl_Clus[Bowl_Clus['Player Name']==name]
    if(player.empty):
        l = np.array(Bowl_Clus.mean())
    else:
        player.drop('Player Name', axis=1, inplace=True)
        l = np.array(player.values[0])
    return l

#getBowlerStat('DJ Bravo')
getBatsmanStat('DJ Bravo')


# In[7]:


wicketModel = DecisionTree.trainRegressor(wicketData, impurity='variance',categoricalFeaturesInfo={}, maxDepth=30, maxBins=40)
print(wicketModel.toDebugString())


# In[ ]:


runsModel = DecisionTree.trainRegressor(runsData, impurity='variance',categoricalFeaturesInfo={}, maxDepth=30, maxBins=40)
print(runsModel.toDebugString())


# In[ ]:


import random
random.seed(20)

def matchSimulation(inn1Ba,inn1Bo,inn2Ba,inn2Bo):
    batting=[inn1Ba,inn2Ba]
    bowling=[inn2Bo,inn1Bo]
    match=[[],[]]
    runPerInn=[0,0]
    wicketCount=[0,0]
    strike={}
    offStrike={}
    bowler={}    
    for inn in range(2):
        strike['name']=batting[inn][0]
        offStrike['name']=batting[inn][1]
        
        for over in range(20):
            bowler['name']=bowling[inn][over%len(bowling[inn])]
            strike['data']= getBatsmanStat(strike['name'])
            offStrike['data']=getBatsmanStat(offStrike['name'])
            bowler['data']=getBowlerStat(bowler['name'])
            overData=[over]
            overData.extend(strike['data'])
            overData.extend(offStrike['data'])
            overData.extend(bowler['data'])
            
            runs=round(runsModel.predict(overData)*(random.uniform(1,1.2)))
            wickets=round(wicketModel.predict(overData)*(random.uniform(1,1.25)))
            match[inn].append([strike['name'],offStrike['name'],bowler['name'],runs,wickets])
            if(wickets):
                runs=round((runs/6)*(6-wickets))
                if((wicketCount[inn]+wickets)>=10):
                    wicketCount[inn]=10
                    runPerInn[inn]+=runs
                    match[inn].append([strike['name'],offStrike['name'],runs,'all out']) 
                    break;
                wicketCount[inn]+=wickets  
                strike['name']=batting[inn][wicketCount[inn]+1]
                
            runPerInn[inn]+=runs
            if(runPerInn[1]>runPerInn[0]):
                break
            strike,offStrike= offStrike,strike
    for i in match[0]:
        print(i)
    print('\n\n')
    for i in match[1]:
        print(i)
    print("\n1st Innings: ",runPerInn[0],"/",wicketCount[0])
    print("\n2nd Innings: ",runPerInn[1],"/",wicketCount[1])


# In[ ]:


t1bat=['LMP Simmons','PA Patel','AT Rayudu','RG Sharma','KH Pandya','KA Pollard','HH Pandya','KV Sharma','MG Johnson','JJ Bumrah','SL Malinga']
t2bowl=['JD Unadkat','Washington Sundar','JD Unadkat','Washington Sundar','JD Unadkat','DT Christian','SN Thakur','DT Christian','A Zampa','LH Ferguson','A Zampa','LH Ferguson','A Zampa','Washington Sundar','A Zampa','Washington Sundar','DT Christian','SN Thakur','JD Unadkat','DT Christian']

t2bat=['AM Rahane','RA Tripathi','SPD Smith','MS Dhoni','MK Tiwary','DT Christian','Washington Sundar','LH Ferguson','A Zampa','SN Thakur','JD Unadkat']
t1bowl=['KH Pandya','MG Johnson','JJ Bumrah','SL Malinga','KV Sharma','KH Pandya','MG Johnson','JJ Bumrah','SL Malinga','KV Sharma','KH Pandya','MG Johnson','JJ Bumrah','SL Malinga','KV Sharma','KH Pandya','MG Johnson','JJ Bumrah','SL Malinga','KV Sharma']

matchSimulation(t1bat, t1bowl, t2bat, t2bowl)


# In[ ]:


match1={
"team1_batting_order":["SR Watson","AT Rayudu","SK Raina","SW Billings","RA Jadeja","MS Dhoni","DJ Bravo","Harbhajan Singh","DL Chahar","SN Thakur","Imran Tahir"],
"team2_batting_order":["AS Yadav","SR Tendulkar","Ishan Kishan","RG Sharma","JP Duminy","KH Pandya","HH Pandya","BCJ Cutting","MJ McClenaghan","SL Malinga","JJ Bumrah"],
"team1_bowling_order":["DL Chahar","SN Thakur","DL Chahar","SR Watson","DL Chahar","SN Thakur","SR Watson","Harbhajan Singh","Imran Tahir","Harbhajan Singh","Imran Tahir","Harbhajan Singh","DJ Bravo","SR Watson","DJ Bravo","SN Thakur","DJ Bravo","SR Watson","SN Thakur","Imran Tahir"],
"team2_bowling_order":["MJ McClenaghan","JJ Bumrah","MJ McClenaghan","HH Pandya","KH Pandya","HH Pandya","KH Pandya","SL Malinga","BCJ Cutting","SL Malinga","MJ McClenaghan","KH Pandya","JJ Bumrah","KH Pandya","HH Pandya","SL Malinga","JJ Bumrah","MJ McClenaghan","JJ Bumrah","HH Pandya"]
}
#

#2.SRH vs CSK 27 may - 178/6 vs 181/2
match2={
"team1_batting_order":["SP Goswami","S Dhawan","KS Williamson","Shakib Al Hasan","YK Pathan","DJ Hooda","CR Brathwaite","Rashid Khan","B Kumar","S Kaul","Sandeep Sharma"],
"team2_batting_order":["SR Watson","F du Plessis","SK Raina","AT Rayudu","MS Dhoni","DJ Bravo","RA Jadeja","KV Sharma","DL Chahar","SN Thakur","L Ngidi"],
"team1_bowling_order":["B Kumar","Sandeep Sharma","B Kumar","Sandeep Sharma","B Kumar","Sandeep Sharma","S Kaul","Rashid Khan","S Kaul","Rashid Khan","Shakib Al Hasan","CR Brathwaite","Sandeep Sharma","CR Brathwaite","Rashid Khan","B Kumar","Rashid Khan","S Kaul","CR Brathwaite","S Kaul"],
"team2_bowling_order":["DL Chahar","L Ngidi","DL Chahar","L Ngidi","DL Chahar","SN Thakur","KV Sharma","DJ Bravo","RA Jadeja","DL Chahar","RA Jadeja","DJ Bravo","KV Sharma","SN Thakur","KV Sharma","DJ Bravo","L Ngidi","DJ Bravo","L Ngidi","SN Thakur"]
}

#3.SRH vs CSK 22 may - 139/7 vs 140/8
match3={
"team2_bowling_order":["DL Chahar","L Ngidi","DL Chahar","L Ngidi","SN Thakur","DL Chahar","DJ Bravo","RA Jadeja","DJ Bravo","RA Jadeja","DL Chahar","RA Jadeja","SN Thakur","RA Jadeja","DJ Bravo","L Ngidi","DJ Bravo","SN Thakur","L Ngidi","SN Thakur"],
"team1_bowling_order":["B Kumar","Sandeep Sharma","B Kumar","S Kaul","B Kumar","S Kaul","CR Brathwaite","Rashid Khan","CR Brathwaite","Rashid Khan","Shakib Al Hasan","Rashid Khan","Sandeep Sharma","Shakib Al Hasan","Sandeep Sharma","Rashid Khan","S Kaul","CR Brathwaite","S Kaul","B Kumar"],
"team1_batting_order":["S Dhawan","SP Goswami","KS Williamson","MK Pandey","Shakib Al Hasan","YK Pathan","CR Brathwaite","B Kumar","Rashid Khan","S Kaul","Sandeep Sharma"],
"team2_batting_order":["SR Watson","F du Plessis","SK Raina","AT Rayudu","MS Dhoni","DJ Bravo","RA Jadeja","DL Chahar","Harbhajan Singh","SN Thakur","L Ngidi"]
}



#4.SRH vs CSK 13 may - 179/4 vs 180/2
match4={
"team1_bowling_order":["Sandeep Sharma","B Kumar","Sandeep Sharma","B Kumar","Rashid Khan","Shakib Al Hasan","S Kaul","Shakib Al Hasan","Rashid Khan","Sandeep Sharma","S Kaul","Rashid Khan","B Kumar","Shakib Al Hasan","Sandeep Sharma","Shakib Al Hasan","Rashid Khan","S Kaul","B Kumar","S Kaul"],
"team2_bowling_order":["DL Chahar","SN Thakur","DJ Wiley","DL Chahar","SN Thakur","DL Chahar","Harbhajan Singh","DL Chahar","SR Watson","DJ Bravo","RA Jadeja","SR Watson","RA Jadeja","Harbhajan Singh","DJ Wiley","DJ Bravo","SN Thakur","DJ Bravo","SN Thakur","DJ Bravo"],
"team2_batting_order":["SR Watson","AT Rayudu","SK Raina","MS Dhoni","SW Billings","DJ Bravo","RA Jadeja","DJ Wiley","Harbhajan Singh","DL Chahar","SN Thakur"],
"team1_batting_order":["S Dhawan","AD Hales","KS Williamson","MK Pandey","DJ Hooda","Shakib Al Hasan","SP Goswami","Rashid Khan","B Kumar","S Kaul","Sandeep Sharma"]
}



#5.CSK vs SRH 22 Apr 182/3 vs 178/6
match5={
"team1_batting_order":["SR Watson","F du Plessis","SK Raina","AT Rayudu","MS Dhoni","SW Billings","DJ Bravo","RA Jadeja","DL Chahar","KV Sharma","SN Thakur"],
"team2_batting_order":["RK Bhui","KS Williamson","MK Pandey","DJ Hooda","Shakib Al Hasan","YK Pathan","WP Saha","Rashid Khan","B Kumar","B Stanlake","S Kaul"],
"team1_bowling_order":["DL Chahar","SN Thakur","DL Chahar","SN Thakur","DL Chahar","SR Watson","RA Jadeja","DL Chahar","RA Jadeja","SR Watson","KV Sharma","RA Jadeja","KV Sharma","RA Jadeja","KV Sharma","DJ Bravo","SN Thakur","DJ Bravo","SN Thakur","DJ Bravo"],
"team2_bowling_order":["B Kumar","B Stanlake","Shakib Al Hasan","B Kumar","B Stanlake","S Kaul","Shakib Al Hasan","Rashid Khan","S Kaul","B Kumar","DJ Hooda","Rashid Khan","Shakib Al Hasan","B Stanlake","Shakib Al Hasan","Rashid Khan","S Kaul","Rashid Khan","S Kaul","B Stanlake"]
}


# In[ ]:


matchSimulation(match1["team1_batting_order"],match1["team1_bowling_order"],match1["team2_batting_order"],match1["team2_bowling_order"])


# In[ ]:


matchSimulation(match2["team1_batting_order"],match2["team1_bowling_order"],match2["team2_batting_order"],match2["team2_bowling_order"])


# In[ ]:


matchSimulation(match3["team2_batting_order"],match3["team2_bowling_order"],match3["team1_batting_order"],match3["team1_bowling_order"])


# In[ ]:


matchSimulation(match4["team1_batting_order"],match4["team1_bowling_order"],match4["team2_batting_order"],match4["team2_bowling_order"])


# In[ ]:


matchSimulation(match5["team1_batting_order"],match5["team1_bowling_order"],match5["team2_batting_order"],match5["team2_bowling_order"])

