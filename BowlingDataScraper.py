#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import pandas as pd

i=0
player_values=[]

page = requests.get("http://stats.espncricinfo.com/indian-premier-league-2018/engine/records/averages/bowling.html?id=12210;type=tournament")
soup = BeautifulSoup(page.text, 'html.parser')

player_row = soup.find(class_="engineTable")
print(player_row)

if i==0:
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
    td2.append(2018)
    player_values.append(list(td2))
        
df2016 = pd.DataFrame(player_values, columns=headers)
