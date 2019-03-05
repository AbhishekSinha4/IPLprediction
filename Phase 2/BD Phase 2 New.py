
# coding: utf-8

# In[2]:

import findspark
findspark.init()


# In[6]:

from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import  KMeans
from pyspark.sql import SQLContext
import pandas as pd
import glob
import yaml
import csv
import numpy as np


# In[7]:

spark = SparkSession.builder.appName('player-clustering').getOrCreate()
df = spark.read.options(header="true",
                               inferSchema="true",
                               nullValue="-",
                               mode="failfast").csv("hdfs://localhost:9000/batting.csv")

df_d = spark.read.options(header="true",
                               inferSchema="true",
                               nullValue="-",
                               mode="failfast").csv("hdfs://localhost:9000/DebutantBattingData.csv")

df_d = df_d.select('Player','Inns','Runs','HS','Ave','BF','SR')
df.show()
df_d.show()


# In[8]:

df.rdd.getNumPartitions()


# In[9]:

dfBowl = spark.read.options(header="true",
                               inferSchema="true",
                               nullValue="-",
                               mode="failfast").csv("hdfs://localhost:9000/bowling.csv")

dfBowl_d = spark.read.options(header="true",
                               inferSchema="true",
                               nullValue="-",
                               mode="failfast").csv("hdfs://localhost:9000/DebutantBowlingData.csv")

dfBowl_d = dfBowl_d.select('Player','Inns','Runs','Wkts','Ave','Econ','SR')


# In[10]:

dfBowl.rdd.getNumPartitions()


# In[11]:

df5 = df.rdd.repartition(5).toDF()
df5.rdd.getNumPartitions()


# In[12]:

dfBowl5 = dfBowl.rdd.repartition(5).toDF()
dfBowl5.rdd.getNumPartitions()


# In[13]:

df5.printSchema()


# In[14]:

dfBowl5.printSchema()


# In[15]:

data = df5.select('Player','Year','Inns','Runs','HS','Ave','BF','SR')
#data.show()
#data2018=data[(data['Year']=='2018')]
#datan18=data[(data['Year']!='2018')]
data.printSchema()
data=data[(data['Year']!='2018')].toPandas()
data=data.drop(['Year'], axis=1)
df_d = df_d.toPandas()
data = data.append(df_d)
data['Runs']=data['Runs']/data['BF']
data = data.groupby('Player',as_index=False).agg({'Inns':sum,'Runs':sum,'HS':max,'Ave': np.mean,'BF':sum,'SR': np.mean})
print(data)
print(type(data))
print(data.info)
#data2018=data2018.toPandas()
#datan18=datan18.toPandas()
#print(data2018)
#count=0
#counts=[]
#datan18.groupby(['Player']).count()

#for i in range(len(datan18)):
#    continue
    
# data2018.iloc[i]['Player']


# In[16]:

dataBowl = dfBowl5.select('Player','Year','Inns','Runs','Wkts','Ave','Econ','SR') 
dataBowl = dataBowl[(dataBowl['Year']!='2018')] 
dataBowl = dataBowl[(dataBowl['Year']!='2018')].toPandas()
dataBowl = dataBowl.drop(['Year'], axis=1) 
dfBowl_d = dfBowl_d.toPandas()
dataBowl = dataBowl.append(dfBowl_d)
dataBowl['Runs'] = dataBowl['Runs']/dataBowl['Inns']
dataBowl['Wkts'] = dataBowl['Wkts']/dataBowl['Inns']
dataBowl = dataBowl.groupby('Player', as_index=False).agg({'Inns':sum,'Runs':sum,'Econ':np.mean,'Ave': np.mean,'SR': np.mean,'Wkts':sum})

print(dataBowl) 
print(type(dataBowl))

#dataBowl.show()


# In[17]:

from pyspark.sql.types import *

print(data.dtypes)
mySchema = StructType([StructField("Player", StringType(), True), StructField("Inns", LongType(), True), StructField("Runs", FloatType(), True), StructField("HS", LongType(), True), StructField("Ave", FloatType(), True), StructField("BF", LongType(), True), StructField("SR", FloatType(), True)])
data = spark.createDataFrame(data,schema=mySchema)
features=['Inns','Runs','HS','Ave','BF','SR']
for c in data.columns:
    if c in features:
        data= data.withColumn(c,data[c].cast('float'))
data.show()


# In[18]:

print(dataBowl.dtypes)
myBowlSchema = StructType([StructField("Player", StringType(), True), StructField("SR", FloatType(), True), StructField("Wkts", FloatType(), True), StructField("Econ", FloatType(), True), StructField("Inns", FloatType(), True), StructField("Runs", FloatType(), True), StructField("Ave", FloatType(), True)])
dataBowl = spark.createDataFrame(dataBowl,schema=myBowlSchema)

features_ball=['Inns','Runs','Wkts','Ave','Econ','SR']
for c in dataBowl.columns:
    if c in features_ball:
        dataBowl= dataBowl.withColumn(c,dataBowl[c].cast('float'))
dataBowl.show()


# In[19]:

features=['Runs','HS','Ave','BF','SR']
vec = VectorAssembler(inputCols=features, outputCol="features")
df_clus = vec.transform(data).select('Player','features')
df_clus.show()


# In[20]:

features_ball=['Runs','Wkts','Ave','Econ','SR']
vec_ball = VectorAssembler(inputCols=features_ball, outputCol="features_ball")
df_clus_ball = vec_ball.transform(dataBowl).select('Player','features_ball')
df_clus_ball.show()


# In[21]:

error = np.zeros(15)
for k in range(2,15):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(df_clus.sample(False,0.25, seed=1))
    error[k] = model.computeCost(df_clus) 


# In[22]:

errorBowl = np.zeros(15)
for k in range(2,15):
    kmeans_ball = KMeans().setK(k).setSeed(1).setFeaturesCol("features_ball")
    model_ball = kmeans_ball.fit(df_clus_ball.sample(False,0.25, seed=1))
    errorBowl[k] = model_ball.computeCost(df_clus_ball) 


# In[23]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()

ax.plot(range(2,15),error[2:15])
print(error)


# In[24]:

plt.style.use('seaborn-whitegrid')
fig_bowl = plt.figure()
ax_bowl = plt.axes()

ax_bowl.plot(range(2,15),errorBowl[2:15])
print(errorBowl)


# In[25]:

k = 4
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(df_clus)
centers = model.clusterCenters()

print("Centers: ")
for i in range(len(centers)):
    print(i+1,": ",centers[i])


# In[26]:

k = 5
kmeans_ball = KMeans().setK(k).setSeed(1).setFeaturesCol("features_ball")
model_ball = kmeans_ball.fit(df_clus_ball)
centers_ball = model_ball.clusterCenters()

print("Centers: ")
for i in range(len(centers_ball)):
    print(i+1,": ",centers_ball[i])


# In[27]:

bat_preds = model.transform(df_clus).select('Player','features','prediction')
bat_r = bat_preds.collect()
bat = pd.DataFrame(bat_r)
bat.columns = ['Player', 'Features', 'Cluster']
bat.head()


# In[28]:

ball_preds = model_ball.transform(df_clus_ball).select('Player','features_ball','prediction')
ball_r = ball_preds.collect()
ball = pd.DataFrame(ball_r)
ball.columns = ['Player', 'Features', 'Cluster']
ball.head()


# In[25]:

bat.groupby(['Cluster']).size()


# In[26]:

ball.groupby(['Cluster']).size()


# <h1>Phase 2

# In[27]:

ipl = spark.read.options(header="true",
                               inferSchema="true",
                               nullValue="-",
                               mode="failfast").csv("hdfs://localhost:9000/ball_data.csv")

ipl5 = ipl.rdd.repartition(5).toDF()
ipl5.rdd.getNumPartitions()
ipl5.printSchema()

#ipl5.head()


# In[28]:

ipl_data = ipl5.select('batsman','bowler','runs_by_batsman','wicket','same_player_out','extra')
ipl_data.show()
type(ipl_data)


# In[29]:

ipl_pd = ipl_data.toPandas()
ipl_pd.head()

ipl_pd = ipl_pd.drop(['same_player_out','extra'], axis=1)
ipl_pd.columns = ['batsman','bowler','runs','wicket']
ipl_pd.head()


# In[30]:

ipl_group = ipl_pd.groupby(['batsman','bowler'])
summary = ipl_group.agg({'runs':lambda x:(x==0).count()})
summary.head()


# In[31]:

ipl_group2 = ipl_pd.groupby(['batsman','bowler'])
x0 = ipl_group2['runs'].apply(lambda x: x[x == 0].count()).to_frame().reset_index()
x1 = ipl_group2['runs'].apply(lambda x: x[x == 1].count()).to_frame().reset_index()
x2 = ipl_group2['runs'].apply(lambda x: x[x == 2].count()).to_frame().reset_index()
x3 = ipl_group2['runs'].apply(lambda x: x[x == 3].count()).to_frame().reset_index()
x4 = ipl_group2['runs'].apply(lambda x: x[x == 4].count()).to_frame().reset_index()
x6 = ipl_group2['runs'].apply(lambda x: x[x == 6].count()).to_frame().reset_index()
xw = ipl_group2['wicket'].apply(lambda x: x[x == True].count()).to_frame().reset_index()
xb = ipl_group2['runs'].apply(lambda x: x.count()).to_frame().reset_index()


# In[32]:

from functools import reduce
dfs = [x0,x1,x2,x3,x4,x6,xw,xb]
new_df = reduce(lambda left,right: pd.merge(left,right, how='inner', on=['batsman','bowler']), dfs)
new_df.columns = ['batsman','bowler','0','1','2','3','4','6','wickets','balls']
print(new_df)


# In[44]:

def thresholdfilter(threshold):
    df_sub = new_df[new_df['balls']>threshold]
    df_sub['0'] = (df_sub['0']+1)/(df_sub['balls']+6)
    df_sub['1'] = (df_sub['1']+1)/(df_sub['balls']+6)
    df_sub['2'] = (df_sub['2']+1)/(df_sub['balls']+6)
    df_sub['3'] = (df_sub['3']+1)/(df_sub['balls']+6)
    df_sub['4'] = (df_sub['4']+1)/(df_sub['balls']+6)
    df_sub['6'] = (df_sub['6']+1)/(df_sub['balls']+6)
    df_sub['wickets'] = 1-((df_sub['wickets']+1)/(df_sub['balls']+1))
    l = [df_sub['0'], df_sub['1'], df_sub['2'], df_sub['3'], df_sub['4'], df_sub['6'], df_sub['wickets']]
    for i in range(len(l)-2):
        l[i+1] = l[i]+l[i+1]
        df_sub[df_sub.columns[i+3]]=l[i+1]
    return df_sub
      
filtered=thresholdfilter(9)
print(filtered)
df_fallprobs = [[np.mean(filtered['0']),np.mean(filtered['1']),np.mean(filtered['2']),np.mean(filtered['3']),np.mean(filtered['4']),np.mean(filtered['6'])]]
df_wicketprob = np.mean(filtered['wickets'])*1.05


# In[34]:

def pvpprob(batsman,bowler,random):
    df_sub = filtered[(filtered['batsman']==batsman) & (filtered['bowler']==bowler)]
    if len(df_sub)>0:
        #print(df_sub)
        l = [df_sub['0'], df_sub['1'], df_sub['2'], df_sub['3'], df_sub['4'], df_sub['6']]
        res = [0,1,2,3,4,6]
        index=0
        for i in range(len(l)):
            if (l[i].iloc[0]<random):
                index=i+1
        return [int(res[index]),df_sub['wickets'].iloc[0]]
    else:
        return clusterprob(batsman,bowler,random)


# In[35]:

pvpprob("A Ashish Reddy","AD Mathews",0.94215)


# In[36]:

def clusterprob(batsman,bowler,random):
    if((len(bat[bat['Player']==batsman])>0) and (len(ball[ball['Player']==bowler])>0)):
        
        batClus = bat[bat['Player']==batsman].iloc[0]['Cluster']
        ballClus = ball[ball['Player']==bowler].iloc[0]['Cluster']
        
        #each batsman of this cluster ka probability with that bowler
        batsmenInClus = bat[bat['Cluster']==batClus].loc[:]["Player"]
        bowlersInClus = ball[ball['Cluster']==ballClus].loc[:]["Player"]
        
        #print("Batsman Cluster Size",len(batsmenInClus))
        #print("Bowler Cluster Size",len(bowlersInClus))
        
        df=[]
        wicketProbs=[]
        for aBatsman in batsmenInClus:
            df_sub = filtered[(filtered['batsman']==aBatsman) & (filtered['bowler']==bowler)]
            if len(df_sub)>0:
                l = [df_sub['0'].iloc[0], df_sub['1'].iloc[0], df_sub['2'].iloc[0], df_sub['3'].iloc[0], df_sub['4'].iloc[0], df_sub['6'].iloc[0]]
                df.append(l)
                wicketProbs.append(df_sub['wickets'].iloc[0])
        for aBowler in bowlersInClus:
            df_sub = filtered[(filtered['batsman']==batsman) & (filtered['bowler']==aBowler)]
            if len(df_sub)>0:
                l = [df_sub['0'].iloc[0], df_sub['1'].iloc[0], df_sub['2'].iloc[0], df_sub['3'].iloc[0], df_sub['4'].iloc[0], df_sub['6'].iloc[0]]
                df.append(l)
                wicketProbs.append(df_sub['wickets'].iloc[0])
        if(df==[]):
            #print("hello",batsman,bowler)
            df=df_fallprobs 
            wicketProbs=df_wicketprob
        df=pd.DataFrame(df)
        runList=['0','1','2','3','4','6']
        df.columns=runList
        meanProb=df.mean()
        wicketProb=np.mean(wicketProbs)
        index=0
        for i in range(len(meanProb)):
            if (meanProb[i]<random):
                index=i+1
        return [int(runList[index%6]),wicketProb]
    else:
        df=df_fallprobs
        wicketProbs=df_wicketprob
        df=pd.DataFrame(df)
        runList=['0','1','2','3','4','6']
        df.columns=runList
        meanProb=df.mean()
        wicketProb=np.mean(wicketProbs)
        index=0
        for i in range(len(meanProb)):
            if (meanProb[i]<random):
                index=i+1
        return [int(runList[index]),wicketProb]
clusterprob("V Kohli","AD Mathews", 0.946)            
            
        
    


# In[37]:

import random
from copy import deepcopy
def matchSimulation(t1bat,t1bowl,t2bat,t2bowl):
    innings=2
    overs=20
    ballsPerOver=6
    runs=[0,0]
    wickets=[0,0]
    batting=[t1bat,t2bat]
    bowling=[t2bowl,t1bowl]
    bbbrec=[]
    p=0
    batsmenPlaying={'strike':{'name':batting[0][0], 'wicketProb': 1}, 'nonStrike':{'name':batting[0][1],'wicketProb': 1}}
    #print(batsmenPlaying)
    for i in range(innings):
        for j in range(overs):
            #random.seed(10)
            for k in range(ballsPerOver):  
                #print(j, " : ",p)
                p+=1
                randomvar = random.uniform(0, 1)
                #if j<6 or j>15:
                #    randomvar=random.uniform(0.2, 1)
                pvpRes=pvpprob(batsmenPlaying['strike']['name'],bowling[i][j%len(bowling[i])],randomvar)
                batsmenPlaying['strike']['wicketProb']*=pvpRes[1]
                if(batsmenPlaying['strike']['wicketProb']>0.3):
                    runs[i]+=pvpRes[0]
                    bbbrec.append(pvpRes[0])
                    if(pvpRes[0]%2==1):
                        temp=batsmenPlaying['nonStrike']
                        batsmenPlaying['nonStrike']=batsmenPlaying['strike']
                        batsmenPlaying['strike']=temp
                else:
                    wickets[i]+=1
                    bbbrec.append(deepcopy(batsmenPlaying['strike']['name']))
                    if(wickets[i]>=10):
                        break;
                    batsmenPlaying['strike']['name']=batting[i][wickets[i]+1]
                    batsmenPlaying['strike']['wicketProb']=1
                    
                if (runs[1]>runs[0]):
                    break
                
            else:
                temp=batsmenPlaying['nonStrike']
                batsmenPlaying['nonStrike']=batsmenPlaying['strike']
                batsmenPlaying['strike']=temp
                #print(bbbrec)
                continue
            break
    print(bbbrec)
    print("Number of balls: ",len(bbbrec))
    #winTeam= runs[0]<runs[1]?runs[1]:runs[0]
    print("Team 1: ",runs[0],"/",wickets[0],"\tTeam 2: ",runs[1],"/",wickets[1],sep="")
    if(runs[0]<runs[1]):
        winTeam = "Team 2"
    else:
        winTeam = "Team 1"
    return winTeam


# In[38]:

pvpprob("RG Sharma","Washington Sundar",random.random())


# In[39]:

t1bat=['LMP Simmons','PA Patel','AT Rayudu','RG Sharma','KH Pandya','KA Pollard','HH Pandya','KV Sharma','MG Johnson','JJ Bumrah','SL Malinga']
t2bowl=['JD Unadkat','Washington Sundar','JD Unadkat','Washington Sundar','JD Unadkat','DT Christian','SN Thakur','DT Christian','A Zampa','LH Ferguson','A Zampa','LH Ferguson','A Zampa','Washington Sundar','A Zampa','Washington Sundar','DT Christian','SN Thakur','JD Unadkat','DT Christian']

t2bat=['AM Rahane','RA Tripathi','SPD Smith','MS Dhoni','MK Tiwary','DT Christian','Washington Sundar','LH Ferguson','A Zampa','SN Thakur','JD Unadkat']
t1bowl=['KH Pandya','MG Johnson','JJ Bumrah','SL Malinga','KV Sharma','KH Pandya','MG Johnson','JJ Bumrah','SL Malinga','KV Sharma','KH Pandya','MG Johnson','JJ Bumrah','SL Malinga','KV Sharma','KH Pandya','MG Johnson','JJ Bumrah','SL Malinga','KV Sharma']

matchSimulation(t1bat,t1bowl,t2bat,t2bowl)


# In[56]:

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
"team1_batting_order":["S Dhawan","AH","KS Williamson","MK Pandey","DJ Hooda","Shakib Al Hasan","SP Goswami","Rashid Khan","B Kumar","S Kaul","Sandeep Sharma"]
}



#5.CSK vs SRH 22 Apr 182/3 vs 178/6
match5={
"team1_batting_order":["SR Watson","F du Plessis","SK Raina","AT Rayudu","MS Dhoni","SW Billings","DJ Bravo","RA Jadeja","DL Chahar","KV Sharma","SN Thakur"],
"team2_batting_order":["RB","KS Williamson","MK Pandey","DJ Hooda","Shakib Al Hasan","YK Pathan","WP Saha","Rashid Khan","B Kumar","B Stanlake","S Kaul"],
"team1_bowling_order":["DL Chahar","SN Thakur","DL Chahar","SN Thakur","DL Chahar","SR Watson","RA Jadeja","DL Chahar","RA Jadeja","SR Watson","KV Sharma","RA Jadeja","KV Sharma","RA Jadeja","KV Sharma","DJ Bravo","SN Thakur","DJ Bravo","SN Thakur","DJ Bravo"],
"team2_bowling_order":["B Kumar","B Stanlake","Shakib Al Hasan","B Kumar","B Stanlake","S Kaul","Shakib Al Hasan","Rashid Khan","S Kaul","B Kumar","DJ Hooda","Rashid Khan","Shakib Al Hasan","B Stanlake","Shakib Al Hasan","Rashid Khan","S Kaul","Rashid Khan","S Kaul","B Stanlake"]
}



# In[57]:

matchSimulation(match4["team1_batting_order"],match4["team1_bowling_order"],match4["team2_batting_order"],match4["team2_bowling_order"])


# In[58]:

matchSimulation(match1["team1_batting_order"],match1["team1_bowling_order"],match1["team2_batting_order"],match1["team2_bowling_order"])


# In[59]:

matchSimulation(match2["team1_batting_order"],match2["team1_bowling_order"],match2["team2_batting_order"],match2["team2_bowling_order"])


# In[63]:

matchSimulation(match3["team2_batting_order"],match3["team2_bowling_order"],match3["team1_batting_order"],match3["team1_bowling_order"])


# In[89]:

matchSimulation(match5["team1_batting_order"],match5["team1_bowling_order"],match5["team2_batting_order"],match5["team2_bowling_order"])


# In[40]:

m5_t1_bowl = ["R Vinay Kumar","PP Chawla","AD Russell","PP Chawla","AD Russell","TK Curran","AD Russell","SP Narine","Kuldeep Yadav","SP Narine","Kuldeep Yadav","SP Narine","PP Chawla","Kuldeep Yadav","TK Curran","SP Narine","PP Chawla","AD Russell","TK Curran","R Vinay Kumar"]

m5_t2_bat = ["SR Watson","AT Rayudu","SK Raina","MS Dhoni","SW Billings","RA Jadeja","DJ Bravo","DL Chahar","Harbhajan Singh","SN Thakur","Imran Tahir"]

m5_t2_bowl = ["DL Chahar","Harbhajan Singh","SR Watson","RA Jadeja","Imran Tahir","RA Jadeja","Harbhajan Singh","SN Thakur","SR Watson","SN Thakur","Imran Tahir","SR Watson","Imran Tahir","DJ Bravo","Imran Tahir","SN Thakur","DJ Bravo","SR Watson","DJ Bravo","SN Thakur"]

m5_t1_bat = ["CA Lynn","SP Narine","RV Uthappa","N Rana","KD Karthik","RK Singh","AD Russell","TK Curran","R Vinay Kumar","Kuldeep Yadav","PP Chawla"]


# In[45]:

matchSimulation(m5_t1_bat, m5_t1_bowl, m5_t2_bat, m5_t2_bowl)


# In[ ]:



