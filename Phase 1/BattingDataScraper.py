from bs4 import BeautifulSoup
import requests
page = requests.get("http://stats.espncricinfo.com/indian-premier-league-2016/engine/records/averages/batting.html?id=11001;type=tournament")
print(page.status_code)
soup= BeautifulSoup(page.content,'html.parser')
file= open("C://Users//sunri//Downloads//study material//sem 5//Big Data//Project//data.csv",'a')
soup=soup.table
headerList= soup.thead.tr.contents
headerList = [headerList[i].contents[0] for i in range(1,len(headerList),2)]
headerList.append("TeamName")
finalData=[]
finalData.append(headerList)

Rows= soup.tbody.find_all('tr')
statDataRows = [[Rows[i].contents[j].contents[0] for j in range(1,len(Rows[i].contents),2)] for i in range(0,len(Rows),2)]
for i in statDataRows:
    i[0]=i[0].contents[0]
    
teamDataRows = [Rows[i].contents[1].contents[0] for i in range(1,len(Rows),2)]
for i in range(len(teamDataRows)):
    teamDataRows[i]=teamDataRows[i].replace('(','')
    teamDataRows[i]=teamDataRows[i].replace(')','')
for i in range(len(statDataRows)):
    statDataRows[i].append(teamDataRows[i])

finalData.extend(statDataRows)
print(finalData)

import csv
with open("C://Users//sunri//Downloads//study material//sem 5//Big Data//Project//data.csv", "a", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(finalData)
f.close()
