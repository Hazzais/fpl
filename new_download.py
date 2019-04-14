# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:46:36 2019

@author: harry
"""

import os
import datetime
import json
import requests
import pickle

season_id = 's201819'
out_dir = r'D:\Documents\PythonDoc\FantasyFootball\new_data'

session = requests.session()

url = 'https://users.premierleague.com/accounts/login/'

"""
Sites used:
    - https://www.oddsonfpl.com/how-to-get-data-from-the-fantasy-premier-league-api/
    - https://medium.com/@bram.vanherle1/fantasy-premier-league-api-
        authentication-guide-2f7aeb2382e4
    - https://github.com/vaastav/Fantasy-Premier-League

"""


#https://stackoverflow.com/questions/4803999/how-to-convert-a-file-into-a-dictionary
def load_credentials(file='credentials.txt'):
    payload = {}
    with open(file, 'r') as f:
        for line in f:
            (k, v) = line.split(',')
            payload[k] = v
    return payload

# Download data as a dictionary
def retrieve_data(link="https://fantasy.premierleague.com/drf/bootstrap-static"):
    response = requests.get(link)
    json_data = json.loads(response.text)
    return json_data

try:
    payload = load_credentials()
    session.post(url, data=payload)
    # Need authentication
    datafile_all = retrieve_data(link=
                                 "https://fantasy.premierleague.com/drf/bootstrap")
    datafile_ent = retrieve_data(link=
                                 "https://fantasy.premierleague.com/drf/entries")
    datafile_trn = retrieve_data(link=
                                 "https://fantasy.premierleague.com/drf/transfers")
    datafile_dyn = retrieve_data(link=
                                 "https://fantasy.premierleague.com/drf/bootstrap-dynamic")

except:
    print("Unable to load credentials. Will only get data in which this is not needed.")

# Main data
datafile_main = retrieve_data()

# Regions
datafile_regions = retrieve_data(link=
                             "https://fantasy.premierleague.com/drf/region")

# Fixtures
datafile_fixtures = retrieve_data(link=
                                  "https://fantasy.premierleague.com/drf/fixtures")



# More complicated - for each player - retrieve a dictionary of their data
datafile_players_deep = {}
for i, pl in enumerate(datafile_main['elements']):
    if i%10==0:
        print("Player number: " + str(i) + " of " + str(len(datafile_main['elements'])))

    player_id = pl['id']
    datafile_players_deep[player_id] = retrieve_data(link=
                                 "https://fantasy.premierleague.com/drf/element-summary/"
                                 + str(player_id))



# Define variables used to save raw data
current_event = datafile_main['current-event']
datetime_now = datetime.datetime.now()
datetime_now_int = round(datetime_now.timestamp())

save_dir = os.path.join(out_dir, season_id, 'GW' + str(current_event))

out_data = {'data_main': datafile_main,
            'data_regions': datafile_regions,
            'data_fixtures': datafile_fixtures,
            'data_players_deep': datafile_players_deep}

# Check folder for this gameweek exists. If it does not - create it
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Pickle each dataset
for fl, fldata in out_data.items():
    outfile = os.path.join(save_dir, fl + '_GW' + str(current_event).zfill(2) \
                           + '_' + str(datetime_now_int) + '.pkl')
    with open(outfile, 'wb') as out:
        pickle.dump(fldata, out)

#with open(outfile, 'rb') as test:
#    x = pickle.load(test)

