# -*- coding: utf-8 -*-
"""
Created on Sun May 12 08:42:50 2019

@author: harry
"""

import os
import pickle
import pandas as pd
import numpy as np
from copy import deepcopy

def raw_data_load(in_dir,
                  in_prefix,
                  gameweek,
                  season_id,
                  datetime_id='latest'):

    # If requested, use the latest created file for the gameweek. Otherwise use
    # the supplied details to construct the correct file
    if datetime_id=='latest':
        full_in_dir = os.path.join(in_dir,
                                   season_id,
                                   'GW' + str(gameweek).zfill(2))
        files = [os.path.join(full_in_dir, file) for file in\
                 os.listdir(full_in_dir) if file.find(
                         in_prefix + str(gameweek).zfill(2))!=-1]
        use_file = max(files, key=os.path.getctime)
    else:
        if in_prefix==None:
            raise ValueError('in_prefix must be specified')
        elif season_id==None:
            raise ValueError('season_id must be specified')
        elif datetime_id==None:
            raise ValueError('datetime_id must be specified')
        else:
            use_file = os.path.join(in_dir, season_id, 'GW' + str(gameweek),\
                                    in_prefix + str(gameweek) + '_' + \
                                    datetime_id + '.pkl')

    with open(use_file, 'rb') as read:
        data_raw = pickle.load(read)
    return data_raw

# Replace values which are lists or NoneTypes with numpy nans
def replace_nonetype_in_dict(thedict):
    enddict = {k: (np.nan if v==None or isinstance(v, (list,)) else v) \
               for k, v in thedict.items()}
    return enddict

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

# TODO. Originally had as a script. Go back and split up into smaller functions
def clean_data(player_history,
               teams,
               fixtures,
               player_summary,
               positions,
               total_players,
               prev_matches_consider=3,
               gameweek_start=1,
               gameweek_end='latest'):


    def add_fixture_team(player_full_set, fixtures):
        # Add team. For merging some casting of keys to integers is needed due to
        # the way values are being stored. Merge based on the fixture and whether
        # the player played home or away that fixture.
        player_full_set['fixture_id'] = player_full_set['fixture_id'].astype(int)
        player_full_set = player_full_set.merge(fixtures[['fixture_id','team_a','team_h']],\
                                                how='left', on='fixture_id')
        temp_team_id = np.where(player_full_set['was_home'],\
                       player_full_set['team_h'],
                       player_full_set['team_a'])
        # Add team based upon the fixture and home/away
        player_full_set.insert(1, 'team_id', temp_team_id)
        return player_full_set

    def team_detailed_data(fixtures, player_full_set, prev_matches_consider,
                           gameweek_start_true, gameweek_end):
        # Make a new dataframe containing one row per team per game showing stats
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

        # Subset to gameweeks to use
        use_fixtures = fixtures.loc[
            (fixtures['gameweek'] >= gameweek_start_true)
            & (fixtures['gameweek'] <= gameweek_end + 1)]

        # Need to concatenate home and away data to get both teams
        team_fixtures_results_home = use_fixtures[fixt_cols].rename(
                columns={'team_h':'team_id',
                         'team_a':'opponent_team',
                         'team_h_difficulty':'team_difficulty',
                         'team_a_difficulty':'opponent_difficulty',
                         'team_h_score':'team_scored',
                         'team_a_score':'team_conceded'})
        team_fixtures_results_home['is_home'] = True
        team_fixtures_results_away = use_fixtures[fixt_cols].rename(
                columns={'team_a':'team_id',
                         'team_h':'opponent_team',
                         'team_a_difficulty':'team_difficulty',
                         'team_h_difficulty':'opponent_difficulty',
                         'team_a_score':'team_scored',
                         'team_h_score':'team_conceded'})
        team_fixtures_results_away['is_home'] = False

        team_fixtures_results = pd.concat([team_fixtures_results_home, team_fixtures_results_away], sort=False)
        team_fixtures_results.sort_values(['team_id','gameweek','kickoff_time'], inplace=True)

        # Get first row for team and gameweek only. This is done due to double
        # gameweeks which otherwise mess things up. For now, we just want to
        # predict the next game (rather than next gameweek)
        team_fixtures_results_single = team_fixtures_results.groupby(
                ['team_id','gameweek']).head(1).drop(
                        columns=['team_scored','team_conceded'])

        # Add additional stats including goals, results, points, and number of
        # players and scorers.
        team_fixtures_results[['team_scored','team_conceded']] = team_fixtures_results[['team_scored','team_conceded']].astype(float)
        team_fixtures_results['team_win'] = team_fixtures_results['team_scored']>team_fixtures_results['team_conceded']
        team_fixtures_results['team_draw'] = team_fixtures_results['team_scored']==team_fixtures_results['team_conceded']
        team_fixtures_results['team_loss'] = team_fixtures_results['team_scored']<team_fixtures_results['team_conceded']
        team_fixtures_results['points'] = np.where(~team_fixtures_results['team_scored'].isna(),
                             3*team_fixtures_results['team_win'] + team_fixtures_results['team_draw'],
                             np.nan)
        # Determine number of players playing and scoring
        unique_scorers = player_full_set.loc[player_full_set.goals_scored>=1, ['team_id','player_id', 'gameweek']]
        n_scorers = unique_scorers.groupby(['team_id','gameweek']).size().reset_index().rename(columns={0:'unique_scorers'})
        unique_players = player_full_set.loc[player_full_set.minutes>0, ['team_id','player_id', 'gameweek', 'total_points']]
        unique_players['total_points'] = unique_players['total_points'].astype(int)

        # Get number and mean points per team per game
        total_scores = unique_players.groupby(['team_id','gameweek'])['total_points'].agg(
                ['mean','sum']).reset_index().rename(columns={'mean':'team_mean_points', 'sum':'team_total_points'})

        # Add the above to the results dataframe
        team_fixtures_results = team_fixtures_results.merge(total_scores, how='left', on=['team_id','gameweek'])
        team_fixtures_results = team_fixtures_results.merge(n_scorers, how='left', on=['team_id','gameweek'])
        team_fixtures_results.loc[~team_fixtures_results['team_scored'].isna(),
                                  'unique_scorers'] =\
                                  team_fixtures_results.loc[~team_fixtures_results['team_scored'].isna(), 'unique_scorers'].fillna(0)


        # Determine the average stats value across the last several games for each
        # team
        roll_cols = ['roll_team_scored',
                       'roll_team_conceded',
                       'roll_team_points',
                       'roll_unique_scorers',
                       'roll_mean_points',
                       'roll_total_points']
        team_fixtures_results[roll_cols] = team_fixtures_results.\
        groupby('team_id')['team_scored','team_conceded','points','unique_scorers','team_mean_points','team_total_points'].apply(
                lambda x: x.rolling(center=False, window=prev_matches_consider).mean())


        # Full team stats per gameweek including previous averages
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
        return team_fixtures_results_single, team_fixtures_results, team_stats_add

    def add_predict_player_row(total_players, player_full_set):
        # Add extra row per player. This will be added to the bottom of each to
        # represent the next game (which we are predicting for)
        final_player_row = player_full_set.groupby('player_id').tail(1)[['player_id',
                                                  'team_id','gameweek']]
        final_player_row['gameweek'] = final_player_row['gameweek']+1
        # Add to the final player row the team's next fixture details (not in
        # original data for this row as we have created it).
        final_player_row.gameweek = final_player_row.gameweek.astype(int)
        final_player_row = final_player_row.merge(
                team_fixtures_results_single[['team_id','gameweek','fixture_id']],
                                                   how='left', on=['team_id','gameweek'])
        # Add to the new rows the estimated percentage ownership, absolute
        # ownership, and transfer stats
        add_latest = player_summary[['player_id',
                                     'now_cost',
                                     'selected_by_percent',
                                     'chance_of_playing_this_round',
                                     'chance_of_playing_next_round',
                                     'status',
                                     'news',
                                     'transfers_in',
                                     'transfers_out']].copy()

        # Calculate number selecting from total players and mean
        tmp = add_latest['selected_by_percent'].astype(float)/100
        add_latest.loc[:,'selected'] = np.round(total_players*tmp).astype(int)
        add_latest.rename(columns={'now_cost':'value'}, inplace=True)
        add_latest['transfers_balance'] = add_latest['transfers_in'] - add_latest['transfers_out']
        add_latest.drop(columns=['selected_by_percent'], inplace=True)
        final_player_row = final_player_row.merge(add_latest,
                                                   how='left',
                                                   on='player_id')
        # Add the new player rows and sort so these are at the end for each player
        player_full_set = pd.concat([player_full_set,final_player_row], sort=False)
        player_full_set.sort_values(['player_id', 'gameweek', 'fixture_id'], inplace=True)

        return player_full_set


    def add_lagged_columns(player_full_set):
    # Columns in which we need to lag the values (i.e. bring to player's next
        # row). I.e. these features for a game should be those from the previous
        # game
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

        # Columns we may potentially predict. Add prefix to mark them.
        target_cols = ['total_points',
                       'goals_scored',
                       'goals_conceded',
                       'minutes']
        target_cols_rename = {col: 'target_' + str(col) for col in target_cols}

        # Get rid of columns. All those columns to delete include the original
        # names we are lagging
        del_cols = [col for col in lag_cols if col not in target_cols] +\
        ['loaned_in', 'loaned_out', 'team_a', 'team_h']

        # Add prefix to mark lagged columns as values from the previous gameweek
        lagged_cols = ['prev_' + str(col) for col in lag_cols]

        # Perform the lag, drop the original columns, and rename the targets
        player_full_set[lagged_cols] = player_full_set.groupby('player_id')[lag_cols].shift(1)
        player_full_set.drop(columns=del_cols, inplace=True)
        player_full_set.rename(columns=target_cols_rename, inplace=True)
        return player_full_set


    def add_team_details(player_full_set, team_fixtures_results):
        # Add team details to the player dataset
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
        return player_full_set


    def add_player_reference_data(player_full_set, player_summary, positions):
        # Add player summary columns, including reference info like names as well
        # as position. This is constant data throughout the season
        cols_player_details = ['player_id',
                               'position_id',
                               'first_name',
                               'second_name',
                                ]
        player_full_set = player_full_set.merge(player_summary[cols_player_details],
                                                  how='left',
                                                  on='player_id')

        # Add position
        player_full_set['position_id'] = player_full_set['position_id'].astype(int)
        player_full_set = player_full_set.merge(
                positions[['id','singular_name_short']].rename(columns={'singular_name_short':'position'}),
                how='left',
                left_on='position_id',
                right_on='id').drop(columns=['id','position_id'])
        return player_full_set


    def add_team_reference_data(player_full_set, teams):
        # Add team reference data and strength
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

        # Split up team strength between home and away for each player's home and
        # away fixtures
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

        # Do the above but for the opponent
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
        return player_full_set



    def add_time_features(player_full_set):
        # Make features from kickoff times - first convert text to datetime
        player_full_set['kickoff_datetime'] = pd.to_datetime(player_full_set['kickoff_time'],
                  errors='coerce')
        player_full_set['prev_kickoff_datetime'] = pd.to_datetime(player_full_set['prev_kickoff_time'],
                  errors='coerce')

        # Function to bin hours
        def hour_to_bin(h):
            if h<12: r = 'morning'
            elif h<15: r = 'midday'
            elif h<19: r = 'afternoon'
            else: r = 'evening'
            return r

        # Determine the hour, bin, and weekday of this and the previous game
        player_full_set['kickoff_hour'] = player_full_set['kickoff_datetime'].dt.hour
        player_full_set['kickoff_hour_bin'] = player_full_set['kickoff_hour'].apply(hour_to_bin)
        player_full_set['kickoff_weekday'] = player_full_set['kickoff_datetime'].dt.weekday
        player_full_set['prev_kickoff_hour'] = player_full_set['prev_kickoff_datetime'].dt.hour
        player_full_set['prev_kickoff_hour_bin'] = player_full_set['prev_kickoff_hour'].apply(hour_to_bin)
        player_full_set['prev_kickoff_weekday'] = player_full_set['prev_kickoff_datetime'].dt.weekday


        # Convert features into cyclic ones (hours)
        h_const = 2*np.pi/24
        player_full_set['kickoff_hour_cos'] = np.cos(h_const*player_full_set['kickoff_hour'].astype(float))
        player_full_set['kickoff_hour_sin'] = np.sin(h_const*player_full_set['kickoff_hour'].astype(float))
        player_full_set['prev_kickoff_hour_cos'] = np.cos(h_const*player_full_set['prev_kickoff_hour'].astype(float))
        player_full_set['prev_kickoff_hour_sin'] = np.sin(h_const*player_full_set['prev_kickoff_hour'].astype(float))

        # Convert features into cyclic ones (weekdays)
        w_const = 2*np.pi/7
        player_full_set['kickoff_weekday_cos'] = np.cos(w_const*player_full_set['kickoff_weekday'].astype(float))
        player_full_set['kickoff_weekday_sin'] = np.sin(w_const*player_full_set['kickoff_weekday'].astype(float))
        player_full_set['prev_kickoff_weekday_cos'] = np.cos(w_const*player_full_set['prev_kickoff_weekday'].astype(float))
        player_full_set['prev_kickoff_weekday_sin'] = np.sin(w_const*player_full_set['prev_kickoff_weekday'].astype(float))
        return player_full_set


    def add_rolling_stats(player_full_set, team_stats_add, prev_matches_consider):
        # Add previous and rolling team stats
        team_stats_add.gameweek = team_stats_add.gameweek.astype(object)
        player_full_set = player_full_set.merge(team_stats_add.groupby(['team_id','gameweek']).head(1),
                                                how='left',
                                                on=['team_id','gameweek'])

        # Add change in value
        player_full_set['value_change'] = player_full_set.groupby('player_id')['value'].diff(1)

        # New variables created by average of previous ones. Create a 'form'
        # variable from previous bps. Also see how much they previously played and
        # scored.
        player_full_set[['custom_form','roll_minutes','roll_goals_scored']] = \
        player_full_set.groupby('player_id')['prev_bps', 'prev_minutes', 'prev_goals_scored'].apply(
                lambda x: x.rolling(center=False, window=prev_matches_consider).mean())
        return player_full_set

    gameweek_start_true = np.max(
            (np.min((gameweek_start, gameweek_start-prev_matches_consider)),
            1))

    if gameweek_end=='latest':
        gameweek_end = player_history['gameweek'].max()
    elif not (isinstance(gameweek_end, int) and \
                    gameweek_end <= 38 and \
                    gameweek_end >= gameweek_start):
        raise ValueError("gameweek_end must be either 'latest' or a integer"
                         "equal to or greater than gameweek_start.")

    player_full_set = player_history.copy()
    player_full_set = player_full_set.loc[
            (player_full_set['gameweek'] >= gameweek_start_true)
            & (player_full_set['gameweek'] <= gameweek_end + 1)]
    player_full_set = add_fixture_team(player_full_set, fixtures)
    team_fixtures_results_single, team_fixtures_results, team_stats_add = \
    team_detailed_data(fixtures, player_full_set, prev_matches_consider,
                           gameweek_start_true, gameweek_end)
    player_full_set = add_predict_player_row(total_players, player_full_set)
    player_full_set = add_lagged_columns(player_full_set)
    player_full_set = add_team_details(player_full_set, team_fixtures_results)
    player_full_set = add_player_reference_data(player_full_set, player_summary, positions)
    player_full_set = add_team_reference_data(player_full_set, teams)
    player_full_set = add_time_features(player_full_set)
    player_full_set = add_rolling_stats(player_full_set, team_stats_add, prev_matches_consider)


    # Not totally necessary, but I like the columns ordered so it's easier to
    # inspect the data
    all_cols = list(player_full_set.columns)

    # Important columns will go first in this order
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

    # Add important columns and then those left over
    new_col_order = imp_col_order + [col for col in all_cols if col not in imp_col_order]


    # Those columns which should be treated as numeric
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
    player_full_set[cols_to_numeric] = player_full_set[cols_to_numeric].astype(float)

    # Those columns which should be treated as categorical
    cols_to_categorical = ['player_id',
                           'position',
                           'team_id',
                           'team_short',
                           'team_name',
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
    player_full_set[cols_to_categorical] = player_full_set[cols_to_categorical].astype('category')

    # Final output is our player dataset with the columns ordered. Also take
    # into account the requested start gameweek here (as rolling values may
    # require earlier gameweeks when created in the code above)
    player_full_set = player_full_set.loc[
            (player_full_set['gameweek'] >= gameweek_start)
            & (player_full_set['gameweek'] <= gameweek_end + 1),
                                          new_col_order]

    return player_full_set
