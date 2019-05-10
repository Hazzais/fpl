# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:58:20 2019

@author: harry
"""

import os
import pickle
import pandas as pd
import numpy as np
import sqlite3
from copy import deepcopy

season_id = 's201819'
in_dir = r'D:\Documents\PythonDoc\FantasyFootball\new_data'

# TODO Make automatic
datetime_id = '1555250244' #'1554918903'
gameweek = 34 #33

# Read in raw saved data files
# Main current gameweek dataset
infile_main_ = os.path.join(in_dir, season_id, 'GW' + str(gameweek),\
                            'data_main_GW' + str(gameweek) + '_' + datetime_id +\
                            '.pkl')
with open(infile_main_, 'rb') as read:
    data_main = pickle.load(read)

# Player regions
infile_regions = os.path.join(in_dir, season_id, 'GW' + str(gameweek),\
                              'data_regions_GW' + str(gameweek) + '_' + datetime_id +\
                              '.pkl')
with open(infile_regions, 'rb') as read:
    data_regions = pickle.load(read)

# All fixtures
infile_fixtures = os.path.join(in_dir, season_id, 'GW' + str(gameweek),\
                               'data_fixtures_GW' + str(gameweek) + '_' + datetime_id +\
                               '.pkl')
with open(infile_fixtures, 'rb') as read:
    data_fixtures = pickle.load(read)

# Detailed player data
infile_players_deep = os.path.join(in_dir, season_id, 'GW' + str(gameweek),\
                               'data_players_deep_GW' + str(gameweek) + '_' + datetime_id +\
                               '.pkl')
with open(infile_players_deep, 'rb') as read:
    data_players_deep = pickle.load(read)

# Replace values which are lists or NoneTypes with numpy nans
def replace_nonetype_in_dict(thedict):
    enddict = {k: (np.nan if v==None or isinstance(v, (list,)) else v) \
               for k, v in thedict.items()}
    return enddict

del infile_fixtures, infile_main_, infile_regions, infile_players_deep

def get_players(data):
    # get current gameweek player data

    # order columns
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

    # Add unordered columns to end of ordered columns
    cols_all = col_order + [col for col in data[0].keys() \
                            if col not in col_order]

    # Convert dictionary with key as ID to DataFrame - better method than below
    # but causes a problem later on
#    players = pd.DataFrame(data)
#    players = players[cols_all]
#    players.rename(columns={'id': 'player_id',
#                            'code': 'player_code',
#                            'element_type': 'position_id',
#                            'team': 'team_id'},
#    inplace=True)

    # Convert dictionary with key as ID to DataFrame
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
    # Get all gameweek data

    cols_order = ['id',
                  'name',
                  'finished',
                  'data_checked',
                  'average_entry_score',
                  'highest_score',
                  'highest_scoring_entry',
                  'is_current',
                  'is_next',
                  'is_previous',
                  'deadline_time',
                  'deadline_time_epoch',
                  'deadline_time_formatted',
                  'deadline_time_game_offset',
                  ]

    # Convert dict to DataFrame - better method than below
    # but causes a problem later on
#    events = pd.DataFrame(data)
#    events = events[cols_order]

    # Convert dict to DataFrame
    events = pd.DataFrame(columns=cols_order)
    for pl in data:
        event_id = pl['id']
        event_row = pd.DataFrame(pl, index=[event_id])
        events = pd.concat([events, event_row], sort=False)

    events.rename(columns={'id': 'gameweek'}, inplace=True)
    return events



def get_next_events(data):
    # Get current gameweek data (including each fixture)
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

    # Convert to dict this way as some values have scores and others don't
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
    # Get teams

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

    # This complicated loop is needed to convert the dict to a DataFrame due to
    # the different types of values in it.
    teams = pd.DataFrame(columns=cols_order)
    for pl in data:

        # For each team (initial row)
        team_id = pl['id']

        # Add new row to initially empty DF
        teams.loc[team_id, 'id'] = team_id

        # For each item in the team's dictionary
        for k, v in pl.items():

            # Get data for coming fixture from list containing dict. If two
            # elements (double gameweek), take only the first (scope of this
            # project )
            if k=='current_event_fixture' and v:
                v_curr = {'curr_game_' + x: y for (x, y) in pl[k][0].items()}
                for k2, v2 in v_curr.items():
                    teams.loc[team_id, k2] = v2
            # If no current fixture (team has no game this gameweek), ignore
            elif k=='current_event_fixture':
                pass
            # Get data for next fixture from list containing dict. If two
            # elements (double gameweek), take only the first (scope of this
            # project )
            elif k=='next_event_fixture' and v:
                v_next = {'next_game_' + x: y for (x, y) in pl[k][0].items()}
                for k2, v2 in v_next.items():
                    teams.loc[team_id, k2] = v2
            # If no next fixture (team has no game this gameweek), ignore
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
    # Get all fixtures data
    fixtures = pd.DataFrame()

    # Make dict into DataFrame - stats can cause problems and is done separately
    for vdict in data:

        vdict_nostats = deepcopy(vdict)
        del vdict_nostats['stats']

        game_id = vdict['code']
        df_row = pd.DataFrame(vdict_nostats, index=[game_id])

        fixtures = fixtures.append(df_row)

    fixtures.rename(columns={'code': 'fixture_code',
                             'id': 'fixture_id',
                             'event': 'gameweek'},
        inplace=True)

    return fixtures


def get_fixture_stats(data):
    # Get all player-fixture stats (not really used)
    cols = ['code',
            'team_code',
            'stat',
            'element',
            'value']

    player_game_stats = pd.DataFrame(columns=cols)

    # For each gameweek
    for vdict in data:

        # For each game
        stats_game = vdict['stats']
        game_id = vdict['code']
        home_code = vdict['team_h']
        away_code = vdict['team_a']

        # For every stat in the gameweek dictionary
        for current_stat in stats_game:
            # For each statistic for the game

            # Name of stat
            stat_string = list(current_stat)[0]
            key_current_stat = current_stat[stat_string]

            # Home team player stats
            home_stat_dict = key_current_stat['h']
            home_stat_df = pd.DataFrame(home_stat_dict)
            home_stat_df['stat'] = stat_string
            home_stat_df['code'] = game_id
            home_stat_df['team_code'] = home_code

             # Away team player stats
            away_stat_dict = key_current_stat['a']
            away_stat_df = pd.DataFrame(away_stat_dict)
            away_stat_df['stat'] = stat_string
            away_stat_df['code'] = game_id
            away_stat_df['team_code'] = away_code

            # Append each player for each team (both home and away)
            player_game_stats = player_game_stats.append(home_stat_df, sort=True)
            player_game_stats = player_game_stats.append(away_stat_df, sort=True)

    # Reshape data from long-to-wide
    player_game_stats_wide = player_game_stats.pivot_table(index=['code','element'],
                                                                 columns='stat',
                                                                values='value',
                                                                aggfunc='sum',
                                                                fill_value=0).reset_index()
    player_game_stats_wide.rename(columns={'code': 'fixture_code',
                                           'element': 'player_id'},
    inplace=True)
    return player_game_stats_wide




def get_players_deep(data):
    # Get detailed player details

    cols = ['element',
            'round',
            'fixture',
            'selected',
            'value',
            'total_points',
            'minutes',
            'goals_scored',
            'bonus',
            'opponent_team',
            ]

    # Need to use same data to get two outputs, a past and future dataset
    player_history = pd.DataFrame(columns=cols)
    player_future = pd.DataFrame()
    for player_id, pdict in data.items():
        # pdict is a dictionary containing 'future' and 'history' values for
        # each player
        player_history = player_history.append(pd.DataFrame(pdict['history']), sort=False)
        temp_future = pd.DataFrame(pdict['fixtures'])
        temp_future['player_id'] = player_id
        player_future = player_future.append(temp_future)

    player_history.rename(columns={'element': 'player_id',
                                   'round': 'gameweek',
                                        'fixture': 'fixture_id',
                                        'id': 'playergw_id'
                                        }, inplace=True)
    player_future.rename(columns={'id': 'fixture_id',
                                        'code': 'fixture_code'
                                        }, inplace=True)

    return player_history, player_future



current_gw = data_main['current-event']
positions = pd.DataFrame(data_main['element_types'])
final_gw = data_main['last-entry-event']
next_gw = data_main['next-event']

fixtures = get_fixtures(data_fixtures)
events = get_events(data_main['events'])
next_events = get_next_events(data_main['next_event_fixtures'])
teams = get_teams(data_main['teams'])
player_summary = get_players(data_main['elements'])
player_game_stats = get_fixture_stats(data_fixtures)
player_history, player_future = get_players_deep(data_players_deep)


prev_matches_consider = 3


player_full_set = player_history.copy()


lag_cols = ['total_points',
            'minutes',
            'goals_scored',
            'bonus',
            'opponent_team',
            'assists',
            'attempted_passes',
            'big_chances_created',
            'big_chances_missed',
            'bps',
            'clean_sheets',
            'clearances_blocks_interceptions',
            'completed_passes',
            'creativity',
            'dribbles',
            'ea_index',
            'errors_leading_to_goal',
            'errors_leading_to_goal_attempt',
            'fouls',
            'goals_conceded',
            'ict_index',
            'playergw_id',
            'influence',
            'key_passes',
            'kickoff_time',
            'kickoff_time_formatted',
            'offside',
            'open_play_crosses',
            'own_goals',
            'penalties_conceded',
            'penalties_missed',
            'penalties_saved',
            'recoveries',
            'red_cards',
            'saves',
            'tackled',
            'tackles',
            'target_missed',
            'team_a_score',
            'team_h_score',
            'threat',
            'was_home',
            'winning_goals',
            'yellow_cards'
            ]

target_cols = ['total_points',
               'goals_scored',
               'goals_conceded',
               'minutes']

target_cols_rename = {col: 'target_' + str(col) for col in target_cols}

del_cols = [col for col in lag_cols if col not in target_cols] +\
['loaned_in', 'loaned_out', 'team_a', 'team_h']

lagged_cols = ['prev_' + str(col) for col in lag_cols]

player_full_set.sort_values(['player_id','fixture_id'], inplace=True)

# Add team
player_full_set['fixture_id'] = player_full_set['fixture_id'].astype(int)
player_full_set = player_full_set.merge(fixtures[['fixture_id','team_a','team_h']],\
                                        how='left', on='fixture_id')
temp_team_id = np.where(player_full_set['was_home'],\
               player_full_set['team_h'],
               player_full_set['team_a'])

player_full_set.insert(1, 'team_id', temp_team_id)




# Add extra row per player
final_player_row = player_full_set.groupby('player_id').tail(1)[['player_id',
                                          'team_id','gameweek']]
final_player_row['gameweek'] = final_player_row['gameweek']+1

fixt_cols = ['gameweek',
             'fixture_id',
             'team_h',
             'team_h_difficulty',
             'team_h_score',
             'team_a',
             'team_a_difficulty',
             'team_a_score',
             'kickoff_time',
             'event_day']



team_fixtures_results_home = fixtures[fixt_cols].rename(
        columns={'team_h':'team_id',
                 'team_a':'opponent_team',
                 'team_h_difficulty':'team_difficulty',
                 'team_a_difficulty':'opponent_difficulty',
                 'team_h_score':'team_scored',
                 'team_a_score':'team_conceded'})
team_fixtures_results_home['is_home'] = True
team_fixtures_results_away = fixtures[fixt_cols].rename(
        columns={'team_a':'team_id',
                 'team_h':'opponent_team',
                 'team_a_difficulty':'team_difficulty',
                 'team_h_difficulty':'opponent_difficulty',
                 'team_a_score':'team_scored',
                 'team_h_score':'team_conceded'})
team_fixtures_results_away['is_home'] = False

team_fixtures_results = pd.concat([team_fixtures_results_home, team_fixtures_results_away], sort=False)
team_fixtures_results.sort_values(['team_id','gameweek','kickoff_time'], inplace=True)
team_fixtures_results_single = team_fixtures_results.groupby(['team_id','gameweek']).head(1).drop(columns=['team_scored','team_conceded'])
team_fixtures_results[['team_scored','team_conceded']] = team_fixtures_results[['team_scored','team_conceded']].astype(float)
team_fixtures_results['team_win'] = team_fixtures_results['team_scored']>team_fixtures_results['team_conceded']
team_fixtures_results['team_draw'] = team_fixtures_results['team_scored']==team_fixtures_results['team_conceded']
team_fixtures_results['team_loss'] = team_fixtures_results['team_scored']<team_fixtures_results['team_conceded']
team_fixtures_results['points'] = np.where(~team_fixtures_results['team_scored'].isna(),
                     3*team_fixtures_results['team_win'] + team_fixtures_results['team_draw'],
                     np.nan)

unique_scorers = player_full_set.loc[player_full_set.goals_scored>=1, ['team_id','player_id', 'gameweek']]
n_scorers = unique_scorers.groupby(['team_id','gameweek']).size().reset_index().rename(columns={0:'unique_scorers'})

unique_players = player_full_set.loc[player_full_set.minutes>0, ['team_id','player_id', 'gameweek', 'total_points']]
unique_players['total_points'] = unique_players['total_points'].astype(int)
total_scores = unique_players.groupby(['team_id','gameweek'])['total_points'].agg(
        ['mean','sum']).reset_index().rename(columns={'mean':'team_mean_points', 'sum':'team_total_points'})


team_fixtures_results = team_fixtures_results.merge(total_scores, how='left', on=['team_id','gameweek'])
team_fixtures_results = team_fixtures_results.merge(n_scorers, how='left', on=['team_id','gameweek'])
team_fixtures_results.loc[~team_fixtures_results['team_scored'].isna(),
                          'unique_scorers'] =\
                          team_fixtures_results.loc[~team_fixtures_results['team_scored'].isna(), 'unique_scorers'].fillna(0)



team_fixtures_results[['roll_team_scored','roll_team_conceded','roll_team_points', 'roll_unique_scorers','roll_mean_points','roll_total_points']] = \
team_fixtures_results.groupby('team_id')['team_scored','team_conceded','points','unique_scorers','team_mean_points','team_total_points'].apply(
        lambda x: x.rolling(center=False, window=prev_matches_consider).mean())



roll_cols = [col for col in team_fixtures_results.columns if col.startswith('roll_')]

add_team_cols = ['points',
                 'team_mean_points',
                 'team_total_points',
                 'unique_scorers',
                 ]

team_stats_add = team_fixtures_results[['team_id','gameweek'] + add_team_cols + roll_cols].copy()
team_stats_add[add_team_cols + roll_cols] = team_stats_add.groupby(['team_id'])[add_team_cols + roll_cols].shift(1)
team_stats_add.rename(columns={'points':'team_prev_result_points',
                                                 'team_mean_points':'team_prev_mean_points',
                                                 'team_total_points':'team_prev_total_points',
                                                 'unique_scorers':'team_prev_unique_scorers'}, inplace=True)




final_player_row.gameweek = final_player_row.gameweek.astype(int)
final_player_row = final_player_row.merge(
        team_fixtures_results_single[['team_id','gameweek','fixture_id']],
                                           how='left', on=['team_id','gameweek'])
add_latest = player_summary[['player_id',
                             'now_cost',
                             'selected_by_percent',
                             'chance_of_playing_this_round',
                             'chance_of_playing_next_round',
                             'status',
                             'news',
                             'transfers_in',
                             'transfers_out']].copy()

total_players = data_main['total-players']
tmp = add_latest['selected_by_percent'].astype(float)/100
add_latest.loc[:,'selected'] = np.round(total_players*tmp).astype(int)
add_latest.rename(columns={'now_cost':'value'}, inplace=True)
add_latest['transfers_balance'] = add_latest['transfers_in'] - add_latest['transfers_out']
add_latest.drop(columns=['selected_by_percent'], inplace=True)

final_player_row = final_player_row.merge(add_latest,
                                           how='left',
                                           on='player_id')

player_full_set = pd.concat([player_full_set,final_player_row], sort=False)

player_full_set.sort_values(['player_id','fixture_id'], inplace=True)

player_full_set[lagged_cols] = player_full_set.groupby('player_id')[lag_cols].shift(1)
player_full_set.drop(columns=del_cols, inplace=True)
player_full_set.rename(columns=target_cols_rename, inplace=True)

player_full_set = player_full_set.merge(team_fixtures_results[['fixture_id',
             'team_id',
             'team_difficulty',
             'opponent_team',
             'opponent_difficulty',
             'kickoff_time',
             'event_day',
             'is_home']], how='left',
                                          on=['team_id', 'fixture_id'])


player_full_set['prev_team_score'] = np.where(player_full_set.prev_was_home,\
               player_full_set.prev_team_h_score, player_full_set.prev_team_a_score)
player_full_set['prev_opponent_score'] = np.where(player_full_set.prev_was_home==False,\
               player_full_set.prev_team_h_score, player_full_set.prev_team_a_score)
player_full_set['prev_win'] = player_full_set.prev_team_score>player_full_set.prev_opponent_score
player_full_set['prev_draw'] = player_full_set.prev_team_score==player_full_set.prev_opponent_score
player_full_set['prev_loss'] = player_full_set.prev_team_score<player_full_set.prev_opponent_score

player_full_set.drop(columns=['prev_was_home',
                              'prev_team_a_score',
                              'prev_team_h_score'], inplace=True)





cols_player_details = ['player_id',
                       'position_id',
                       'first_name',
                       'second_name',
                        ]

player_full_set = player_full_set.merge(player_summary[cols_player_details],
                                          how='left',
                                          on='player_id')

player_full_set['position_id'] = player_full_set['position_id'].astype(int)
player_full_set = player_full_set.merge(
        positions[['id','singular_name_short']].rename(columns={'singular_name_short':'position'}),
        how='left',
        left_on='position_id',
        right_on='id').drop(columns=['id','position_id'])

cols_teams = ['team_id',
              'short_name',
              'name',
              'strength',
              ] + [col for col in teams if col.startswith('strength_')]

player_full_set = player_full_set.merge(
        teams[cols_teams].rename(columns={'short_name':'team_short',
             'name':'team_name',
             'strength':'team_strength'}),
        how='left',
        on='team_id')

player_full_set['team_strength_ha_overall'] = np.where(player_full_set.is_home,
                player_full_set['strength_overall_home'],
                player_full_set['strength_overall_away'])
player_full_set['team_strength_ha_attack'] = np.where(player_full_set.is_home,
                player_full_set['strength_attack_home'],
                player_full_set['strength_attack_away'])
player_full_set['team_strength_ha_defence'] = np.where(player_full_set.is_home,
                player_full_set['strength_defence_home'],
                player_full_set['strength_defence_away'])
player_full_set.drop(
        columns=[col for col in player_full_set if col.startswith('strength_')],
        inplace=True)

player_full_set = player_full_set.merge(
        teams[cols_teams].rename(columns={'short_name':'opponent_team_short',
             'name':'opponent_team_name',
             'strength':'opponent_team_strength',
             'team_id':'opponent_team'}),
        how='left',
        on='opponent_team')
player_full_set['opponent_strength_ha_overall'] = np.where(player_full_set.is_home,
                player_full_set['strength_overall_home'],
                player_full_set['strength_overall_away'])
player_full_set['opponent_strength_ha_attack'] = np.where(player_full_set.is_home,
                player_full_set['strength_attack_home'],
                player_full_set['strength_attack_away'])
player_full_set['opponent_strength_ha_defence'] = np.where(player_full_set.is_home,
                player_full_set['strength_defence_home'],
                player_full_set['strength_defence_away'])
player_full_set.drop(
        columns=[col for col in player_full_set if col.startswith('strength_')],
        inplace=True)

player_full_set['kickoff_datetime'] = pd.to_datetime(player_full_set['kickoff_time'],
          errors='coerce')
player_full_set['prev_kickoff_datetime'] = pd.to_datetime(player_full_set['prev_kickoff_time'],
          errors='coerce')

def hour_to_bin(h):
    if h<12: r = 'morning'
    elif h<15: r = 'midday'
    elif h<19: r = 'afternoon'
    else: r = 'evening'
    return r

player_full_set['kickoff_hour'] = player_full_set['kickoff_datetime'].dt.hour
player_full_set['kickoff_hour_bin'] = player_full_set['kickoff_hour'].apply(hour_to_bin)
player_full_set['kickoff_weekday'] = player_full_set['kickoff_datetime'].dt.weekday
player_full_set['prev_kickoff_hour'] = player_full_set['prev_kickoff_datetime'].dt.hour
player_full_set['prev_kickoff_hour_bin'] = player_full_set['prev_kickoff_hour'].apply(hour_to_bin)
player_full_set['prev_kickoff_weekday'] = player_full_set['prev_kickoff_datetime'].dt.weekday


# Cyclic features (day, hours)
h_const = 2*np.pi/24
player_full_set['kickoff_hour_cos'] = np.cos(h_const*player_full_set['kickoff_hour'].astype(float))
player_full_set['kickoff_hour_sin'] = np.sin(h_const*player_full_set['kickoff_hour'].astype(float))
player_full_set['prev_kickoff_hour_cos'] = np.cos(h_const*player_full_set['prev_kickoff_hour'].astype(float))
player_full_set['prev_kickoff_hour_sin'] = np.sin(h_const*player_full_set['prev_kickoff_hour'].astype(float))

w_const = 2*np.pi/7
player_full_set['kickoff_weekday_cos'] = np.cos(h_const*player_full_set['kickoff_weekday'].astype(float))
player_full_set['kickoff_weekday_sin'] = np.sin(h_const*player_full_set['kickoff_weekday'].astype(float))
player_full_set['prev_kickoff_weekday_cos'] = np.cos(h_const*player_full_set['prev_kickoff_weekday'].astype(float))
player_full_set['prev_kickoff_weekday_sin'] = np.sin(h_const*player_full_set['prev_kickoff_weekday'].astype(float))



team_stats_add.gameweek = team_stats_add.gameweek.astype(object)
player_full_set = player_full_set.merge(team_stats_add.groupby(['team_id','gameweek']).head(1), how='left', on=['team_id','gameweek'])

player_full_set['value_change'] = player_full_set.groupby('player_id')['value'].diff(1)
player_full_set[['custom_form','roll_minutes','roll_goals_scored']] = \
player_full_set.groupby('player_id')['prev_bps', 'prev_minutes', 'prev_goals_scored'].apply(
        lambda x: x.rolling(center=False, window=prev_matches_consider).mean())



all_cols = list(player_full_set.columns)


imp_col_order = [
 'player_id',
 'first_name',
 'second_name',
 'position',
 'team_id',
 'team_short',
 'team_name',
 'team_difficulty',
 'gameweek',
 'kickoff_time',
 'kickoff_hour',
 'kickoff_hour_cos',
 'kickoff_hour_sin',
 'kickoff_hour_bin',
 'kickoff_weekday',
 'kickoff_weekday_cos',
 'kickoff_weekday_sin',
 'event_day',
 'fixture_id',
 'is_home',
 'opponent_team',
 'opponent_team_short',
 'opponent_team_name',
 'opponent_team_strength',
 'opponent_difficulty',
 'opponent_strength_ha_overall',
 'opponent_strength_ha_attack',
 'opponent_strength_ha_defence',
 'target_total_points',
 'target_minutes',
 'target_goals_scored',
 'target_goals_conceded',
 'selected',
 'value',
 'value_change',
 'custom_form',
 'transfers_balance',
 'transfers_in',
 'transfers_out',
 'team_strength',
 'team_strength_ha_overall',
 'team_strength_ha_attack',
 'team_strength_ha_defence',
]

new_col_order = imp_col_order + [col for col in all_cols if col not in imp_col_order]



cols_to_numeric = [ 'target_total_points',
                             'target_minutes',
                             'target_goals_scored',
                             'selected',
                             'value',
                             'value_change',
                             'custom_form',
                             'transfers_balance',
                             'transfers_in',
                             'transfers_out',
                             'chance_of_playing_this_round',
                             'chance_of_playing_next_round',
                             'prev_total_points',
                             'prev_minutes',
                             'prev_goals_scored',
                             'prev_bonus',
                             'prev_creativity',
                             'prev_ict_index',
                             'prev_influence',
                             'prev_threat',
                             'team_prev_result_points',
                             'team_prev_mean_points',
                             'team_prev_total_points',
                             'team_prev_unique_scorers',
                             'roll_team_scored',
                             'roll_team_conceded',
                             'roll_team_points',
                             'roll_unique_scorers',
                             'roll_mean_points',
                             'roll_total_points',
                             'roll_minutes',
                             'roll_goals_scored',
                             'kickoff_hour_cos',
                             'kickoff_hour_sin',
                             'kickoff_weekday_cos',
                             'kickoff_weekday_sin',
                             'prev_kickoff_hour_cos',
                             'prev_kickoff_hour_sin',
                             'prev_kickoff_weekday_cos',
                             'prev_kickoff_weekday_sin',
                             ]

cols_to_categorical = ['player_id',
                       'position',
                       'team_id',
                       'team_short',
                       'team_name',
                       'gameweek',
                       'kickoff_hour',
                       'kickoff_hour_bin',
                       'kickoff_weekday',
                       'event_day',
                       'fixture_id',
                       'opponent_team',
                       'opponent_team_short',
                       'opponent_team_name',
                       'status',
                       'prev_opponent_team',
                       'prev_playergw_id',
                       'prev_kickoff_hour',
                       'prev_kickoff_hour_bin',
                       'prev_kickoff_weekday',
                      ]

player_full_set[cols_to_numeric] = player_full_set[cols_to_numeric].astype(float)
player_full_set[cols_to_categorical] = player_full_set[cols_to_categorical].astype('category')

use_dtypes = player_full_set.dtypes

df_final = player_full_set[new_col_order].copy()
df_final = df_final.astype(use_dtypes)

x = df_final[df_final['player_id'].isin([302])].copy()

corr = df_final.corr()

corr_target = corr.loc[:, 'target_total_points']

# =============================================================================
# TRAIN TEST SPLIT SCRIPT
# =============================================================================

train_fraction = 0.7

player_ids = df_final['player_id'].unique()
random_numbers = np.random.random(len(player_ids))

player_train_test_num = pd.DataFrame([player_ids, random_numbers], columns=['player_id','randomnumber'])
player_train_test_num['split'] = np.where(player_train_test_num['randomnumber']<train_fraction, 'TRAIN', 'TEST')


df_final2 = df_final.merge(player_train_test_num, how='left', on='player_id')


# Quantiles for selected OR standardised (by gameweek) selected

### GET QUANTILES WITH TRAIN SET ONLY - APPLY TO TEST SET!

def get_selected_quantiles(data):
    quants = data.groupby('gameweek')['selected'].quantile(\
                                    [0.1*round(i/10, 2) for i in range(100)]).reset_index().rename(columns={'level_1':'selected_quantile',
                                    'selected':'lower_bound'})
    quants['upper_bound'] = quants.groupby('gameweek')['lower_bound'].shift(-1).fillna(99999999)
    return quants

data=df_final2.loc[df_final2.split=='TRAIN'].copy()
quants = get_selected_quantiles(data)

conn = sqlite3.connect(':memory:')
df_final2.to_sql('left_df', conn, index=False)
quants.to_sql('right_df', conn, index=False)

query = """
    select distinct a.*, b.selected_quantile
    from left_df as a
    left join right_df as b
    on a.gameweek=b.gameweek and a.selected>=b.lower_bound and a.selected<b.upper_bound

"""

df_test = pd.read_sql_query(query, conn)

def get_std_params(data):
    vals = data.groupby('gameweek')['selected'].agg(['mean','std']).reset_index()
    vals.rename(columns={'mean':'selected_mean',
                         'std':'selected_std'}, inplace=True)
    return vals

data = df_test.loc[df_final2.split=='TRAIN'].copy()
gw_selected_stats = get_std_params(data)

df_test = df_test.merge(gw_selected_stats, how='left', on='gameweek')
df_test['selected_norm'] = (df_test['selected']-df_test['selected_mean'])/df_test['selected_std']
df_test.drop(columns=['selected_mean','selected_std'],inplace=True)


df_small = df_test[['player_id','second_name','gameweek','selected','selected_quantile']].copy()


df_test['selected_norm'] = df_test.groupby('gameweek')['selected'].transform(lambda x: (x-x.mean())/x.std())

corr = df_test.corr()

corr_target = corr.loc[:, 'target_total_points']

x2 = df_test[df_test['player_id']=='302'].copy()

# =============================================================================
# TODO
# =============================================================================
# Clean up
# Some EDA
# Categorical response
# Choose features
# Build models






