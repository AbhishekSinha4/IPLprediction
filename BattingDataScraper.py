from bs4 import BeautifulSoup
import requests
import csv

urlList = [url.rstrip('\n') for url in open("BattingAveragesURLs.txt", "r")]    
#Reading all URLs into a list from an input text file (a url per line) and stripping them of '\n's 
headerNeeded=1
finalData=[]
for url in urlList:
    page = requests.get(url)
    #print(page.status_code)   (to check whether page was requested successfully or not (200 for success))
    soup= BeautifulSoup(page.content,'html.parser')
    soup=soup.table                                     #First table is the data table
    if(headerNeeded):                                   #Column headers are only needed once
        headerList= soup.thead.tr.contents              #thead tr contents has alternate '\n' and 'th' elements. '\n' must be skipped 
        headerList = [headerList[i].contents[0] for i in range(1,len(headerList),2)]     
        #Hence, skipping over the '\n' and taking the 1st content of the 'th' element i.e. the column name
        headerList.append("TeamName")                   #Adding column name for team name
        headerList.append("Year")                       #Adding a column for Year
        finalData.append(headerList)                    #Adding column headers list to final data
        headerNeeded=0
    Rows= soup.tbody.find_all('tr')                        
    statDataRows = [[Rows[i].contents[j].contents[0] for j in range(1,len(Rows[i].contents),2)] for i in range(0,len(Rows),2)]
    #Skipping over alternate tr's in tbody as only even rows contain the stats data
    #For each of these tr's td's are alternating with '\n's and '\n's need to be skipped. The content of the td's is taken .i.e the data
    for i in statDataRows:
        i[0]=i[0].contents[0]   #1st td in each row contained an a tag, the first content of which is the player name 
        
    teamDataRows = [Rows[i].contents[1].contents[0] for i in range(1,len(Rows),2)]
    #Skipping over alternate tr's in tbody as only odd rows contain the team name
    #For each of these tr's the team name 'td' is the 1st index content and its first content is the teamName 
    for i in range(len(teamDataRows)):  #removing the brackets from the teamNames
        teamDataRows[i]=teamDataRows[i].replace('(','')
        teamDataRows[i]=teamDataRows[i].replace(')','')
    for i in range(len(statDataRows)):                  #Adding Team Name and Year to each row for this URLs data
        statDataRows[i].append(teamDataRows[i])
        statDataRows[i].append(url.split('/')[3].split('-')[3])
    finalData.extend(statDataRows)                      #Appending all of this Pages' data to the Final Data List

with open("BattingAveragesData.csv", "w", newline='') as f:         #Writing the data scraped into a persistent Datafile in csv format
    writer = csv.writer(f)
    writer.writerows(finalData)
f.close()                                               #Closing the file 
