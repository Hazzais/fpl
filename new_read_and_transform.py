# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:58:20 2019

@author: harry
"""

import os
import pickle
import pandas as pd
import numpy as np
from copy import deepcopy

season_id = 's201819'
in_dir = r'D:\Documents\PythonDoc\FantasyFootball\new_data'
gameweek = 33

# Make automatic
datetime_id = '1554918903'


infile_main_ = os.path.join(in_dir, season_id, 'GW' + str(gameweek),\
                            'data_main_GW' + str(gameweek) + '_' + datetime_id +\
                            '.pkl')
with open(infile_main_, 'rb') as read:
    data_main = pickle.load(read)

infile_regions = os.path.join(in_dir, season_id, 'GW' + str(gameweek),\
                              'data_regions_GW' + str(gameweek) + '_' + datetime_id +\
                              '.pkl')
with open(infile_regions, 'rb') as read:
    data_regions = pickle.load(read)

infile_fixtures = os.path.join(in_dir, season_id, 'GW' + str(gameweek),\
                               'data_fixtures_GW' + str(gameweek) + '_' + datetime_id +\
                               '.pkl')
with open(infile_fixtures, 'rb') as read:
    data_fixtures = pickle.load(read)

# Replace values which are lists or NoneTypes with numpy nans
def replace_nonetype_in_dict(thedict):
    enddict = {k: (np.nan if v==None or isinstance(v, (list,)) else v) \
               for k, v in thedict.items()}
    return enddict

del infile_fixtures, infile_main_, infile_regions
#def extract_main(data):


current_gw = data_main['current-event']
positions = pd.DataFrame(data_main['element_types'])
#game-settings
final_gw = data_main['last-entry-event']
next_gw = data_main['next-event']

#stats
#stats_options


def get_players(data):
    col_order = ['id',
                 'code',
                 'element_type',
                 'first_name',
                 'second_name',
                 'team',
                 'team_code',
                 'chance_of_playing_next_round',
                 'chance_of_playing_this_round',
                 'total_points',
                 'now_cost',
                 'selected_by_percent',
                 'status',
                 'news',
                 'news_added',
                 'minutes',
                 'points_per_game',
                 'goals_scored',
                 'assists',
                 'bonus',
                 'goals_conceded',
                 'bps',
                 'cost_change_event',
                 'cost_change_start',
                 'event_points',
                 'form',
                 ]

    cols_all = col_order + [col for col in data[0].keys() \
                            if col not in col_order]

    players = pd.DataFrame(columns=cols_all)
    for pl in data:
        player_id = pl['id']
        player_row = pd.DataFrame(pl, index=[player_id])
        players = pd.concat([players, player_row], sort=False)

    players.rename(columns={'id': 'player_id',
                            'code': 'player_code',
                            'element_type': 'position_id',
                            'team': 'team_id'},
    inplace=True)

    return players


def get_events(data):
    cols_order = ['id',
                  'name',
                  'finished',
                  'data_checked',
                  'average_entry_score',
                  'highest_score',
                  'highest_scoring_entry',
                  'is_current',
                  'is_next',
                  'is_previous'
                  'deadline_time',
                  'deadline_time_epoch',
                  'deadline_time_formatted',
                  'deadline_time_game_offset',
                  ]

    events = pd.DataFrame(columns=cols_order)
    for pl in data:
        event_id = pl['id']
        event_row = pd.DataFrame(pl, index=[event_id])
        events = pd.concat([events, event_row], sort=False)

    events.rename(columns={'id': 'gameweek'}, inplace=True)
    return events



def get_next_events(data):
    cols_order = ['event',
                  'code',
                  'id',
                  'event_day',
                  'team_a',
                  'team_h',
                  'teams_h_score',
                  'team_a_score',
                  'deadline_time',
                  'deadline_time_formatted',
                  'finished',
                  'finished_provisional',
                  'kickoff_time',
                  'kickoff_time_formatted',
                  'minutes',
                  'provisional_start_time',
                  'started',
                  'stats'
                  ]



    next_events = pd.DataFrame(columns=cols_order)
    for pl in data:
        next_event_id = pl['id']
        pl_clean = replace_nonetype_in_dict(pl)
        next_event_row = pd.DataFrame(pl_clean, index=[next_event_id])
        next_events = pd.concat([next_events, next_event_row], sort=False)

    next_events.rename(columns={'event': 'gameweek',
                                'code': 'fixture_code',
                                'id': 'unknown_id'},
    inplace=True)
    return next_events


def get_teams(data):
    cols_order = ['id',
                     'short_name',
                     'name',
                     'strength',
                     'code',
                     'strength_overall_home',
                     'strength_overall_away',
                     'strength_attack_home',
                     'strength_attack_away',
                     'strength_defence_away',
                     'strength_defence_home',
                     'points',
                     'draw',
                     'loss',
                     'win',
                     'form',
                     'position',
                     'played',
                     'team_division',
                     'unavailable',
                     'link_url',
                     ]

    teams = pd.DataFrame(columns=cols_order)
    for pl in data:
        team_id = pl['id']

        teams.loc[team_id, 'id'] = team_id

        for k, v in pl.items():
            if k=='current_event_fixture' and v:
                v_curr = {'curr_game_' + x: y for (x, y) in pl[k][0].items()}
                for k2, v2 in v_curr.items():
                    teams.loc[team_id, k2] = v2

            elif k=='current_event_fixture':
                pass
            elif k=='next_event_fixture' and v:
                v_next = {'next_game_' + x: y for (x, y) in pl[k][0].items()}
                for k2, v2 in v_next.items():
                    teams.loc[team_id, k2] = v2
            elif k=='next_event_fixture':
                pass
            elif v==None:
                v = np.nan
                teams.loc[team_id, k] = v
            else:
                teams.loc[team_id, k] = v

    teams.rename(columns={'id': 'team_id',
                          'code': 'team_code'}, inplace=True)

    return teams



def get_fixtures(data):
    fixtures = pd.DataFrame()

    for vdict in data:

        vdict_nostats = deepcopy(vdict)
        del vdict_nostats['stats']

        game_id = vdict['code']
        df_row = pd.DataFrame(vdict_nostats, index=[game_id])

        fixtures = fixtures.append(df_row)

    fixtures.rename(columns={'code': 'fixture_code',
                             'id': 'curr_season_fixture_id',
                             'event': 'gameweek'},
        inplace=True)

    return fixtures


def get_fixture_stats(data):
    cols = ['code',
            'team_code',
            'stat',
            'element',
            'value']

    player_game_stats = pd.DataFrame(columns=cols)

    for vdict in data:

        # For each game
        stats_game = vdict['stats']
        game_id = vdict['code']
        home_code = vdict['team_h']
        away_code = vdict['team_a']

        for current_stat in stats_game:
            # For each statistic for the game

            stat_string = list(current_stat)[0]
            key_current_stat = current_stat[stat_string]

            home_stat_dict = key_current_stat['h']
            home_stat_df = pd.DataFrame(home_stat_dict)
            home_stat_df['stat'] = stat_string
            home_stat_df['code'] = game_id
            home_stat_df['team_code'] = home_code

            away_stat_dict = key_current_stat['a']
            away_stat_df = pd.DataFrame(away_stat_dict)
            away_stat_df['stat'] = stat_string
            away_stat_df['code'] = game_id
            away_stat_df['team_code'] = away_code

            player_game_stats = player_game_stats.append(home_stat_df, sort=True)
            player_game_stats = player_game_stats.append(away_stat_df, sort=True)


    player_game_stats_wide = player_game_stats.pivot_table(index=['code','element'],
                                                                 columns='stat',
                                                                values='value',
                                                                aggfunc='sum',
                                                                fill_value=0).reset_index()
    player_game_stats_wide.rename(columns={'code': 'fixture_code',
                                           'element': 'player_id'},
    inplace=True)
    return player_game_stats_wide


fixtures = get_fixtures(data_fixtures)
events = get_events(data_main['events'])
next_events = get_next_events(data_main['next_event_fixtures'])
teams = get_teams(data_main['teams'])
players = get_players(data_main['elements'])
player_game_stats = get_fixture_stats(data_fixtures)

# Add to teams (before adding to players)


# Add to fixtures


# Add to players
players2 = players.copy()
players2['element_type'] = players2['element_type'].astype(int)
players2 = players2.merge(positions.rename(columns=
                                           {'id':'element_type'})[
    ['element_type', 'singular_name_short']],
                          how='left',
                          on='element_type')

