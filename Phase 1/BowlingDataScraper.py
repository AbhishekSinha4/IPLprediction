#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import pandas as pd

itr=0
player_values=[]
links=[]
with open("BowlingAveragesURLs.txt") as f:
    links=f.readlines()
    links = list(map(lambda x:x.rstrip("\n"),links))
    
for i in links:
    page = requests.get(i)
    print(page)
    soup = BeautifulSoup(page.text, 'html.parser')
    #print(soup.prettify())
    player_row = soup.find(class_="engineTable")
    #print(player_row)
    
    if itr==0:
        player_row_head = player_row.find("thead")
        player_row_hval = player_row_head.find_all("th")
        headers = []
        for i in player_row_hval:
            headers.append(i.contents[0])
        headers.append("Year")
    
    player_row_val = player_row.find("tbody")
    player_row_val = player_row.find_all("tr")
    for i in player_row_val[1::2]:
        td = i.find_all("td")
        name = td[0].find("a").contents[0]
        td2 = list(map(lambda x:x.contents[0],td[1:]))
        for i in range(len(td2)):
            try:
                td2[i]=float(td2[i])
            except ValueError:
                td2[i]=td2[i]
        td2.insert(0,name)
        td2.append(2018-itr)
        player_values.append(list(td2))
    itr+=1
        
bowlerdf = pd.DataFrame(player_values, columns=headers)

bowlerdf.to_csv("BowlingAveragesData.csv")