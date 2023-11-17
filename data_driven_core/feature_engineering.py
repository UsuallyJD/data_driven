'''
Functions for creating new features
- points to teammate
- calculate lead driver
'''

import pandas as pd

def get_quali_times(f1_data):
    '''
    reformatting function to get qualifying times in ms from qualifying session
    ie official time from q1, q2, q3 format

    Parameters:
        f1_data: (pd.DataFrame) dataset

    Returns:
        f1_data: (pd.DataFrame) Dataset with added quali_time feature
    '''

    def format_time(time):
        time_str = str(time)
        if time_str is not None:
            time_components = time_str.replace('.', ':').split(':')
            if len(time_components) >= 3:
                minutes, seconds, milliseconds = map(int, time_components[:3])
                return (minutes * 60 + seconds) * 1000 + milliseconds
        return None
    
    def get_quali_time(row):
        q1_ms = format_time(row['q1'])
        q2_ms = format_time(row['q2'])
        q3_ms = format_time(row['q3'])
    
        if q3_ms is not None:
            return q3_ms
        elif q2_ms is not None:
            return q2_ms
        elif q1_ms is not None:
            return q1_ms
        else:
            return None

    f1_data['quali_time'] = f1_data.apply(get_quali_time, axis=1)

    return f1_data

def add_driver_age(f1_data):
    '''
    calculate age for each driver

    Parameters:
        f1_data dataset- DataFrame

    Returns:
        f1_data dataset- DataFrame with added age column, removed dob column
    '''

    f1_data.dropna(subset='dob', inplace=True)
    f1_data['age'] = ((f1_data['date'] - f1_data['dob']).dt.days/365.25).astype(int)
    f1_data.drop(columns='dob', axis=1, inplace=True)

    return f1_data


def add_lead_driver(f1_data):
    '''
    calculate which driver has 'first driver' status (assumed to be higher qualifier)

    Parameters:
        f1_data dataset- DataFrame

    ReturnsL
        f1_data dataset- DataFrame with added age column, removed dob column
    '''

    team_lead = []

    # make this next chunk its own function for use in the points difference function
    for race in f1_data['raceId'].value_counts().index:
        df_race = f1_data.loc[f1_data['raceId'] == race]
        for team in df_race['constructorRef'].value_counts().index:
            df_team = df_race.loc[df_race['constructorRef'] == team]
            df_team.reset_index(inplace=True)

            if df_team.shape[0] == 1:
                team_lead.append([race, df_team['driverRef'][0], True])

            if df_team.shape[0] == 2:
                if df_team['grid'][0] < df_team['grid'][1]:
                    team_lead.append([race, df_team['driverRef'][0], True])
                    team_lead.append([race, df_team['driverRef'][1], False])

                if df_team['grid'][1] < df_team['grid'][0]:
                    team_lead.append([race, df_team['driverRef'][1], True])
                    team_lead.append([race, df_team['driverRef'][0], False])

    #Add feature to DataFrame
    team_lead = pd.DataFrame(team_lead, columns=['raceId', 'driverRef', 'lead_driver'])
    f1_data = f1_data.merge(team_lead, how='left', on=['raceId', 'driverRef'])

    return f1_data


def add_points_difference_to_teammate(f1_data):
    '''
    calculate points difference to teammate

    Parameters:
        f1_data dataset- DataFrame

    ReturnsL
        f1_data dataset- DataFrame with added age column, removed dob column
    '''

    points_diff = []

    for race in f1_data['raceId'].value_counts().index:
        df_race = f1_data.loc[f1_data['raceId'] == race]
        for team in df_race['constructorRef'].value_counts().index:
            df_team = df_race.loc[df_race['constructorRef'] == team]
            df_team.reset_index(inplace=True)

            if df_team.shape[0] == 1:
                if points_diff[-1][2] == df_team['driverRef'][0]:
                    points_diff.append([race, df_team['driverRef'][0],
                                        points_diff[-1][3] + df_team['points'][0]])

                elif points_diff[-2][2] == df_team['driverRef'][0]:
                    points_diff.append([race, df_team['driverRef'][0],
                                        points_diff[-2][3] + df_team['points'][0]])

                else:
                    points_diff.append([race, df_team['driverRef'][0],
                                        df_team['points'][0]])

            if df_team.shape[0] == 2:
                points_diff.append([race, df_team['driverRef'][0],
                                    df_team['points'][0] - df_team['points'][1]])
                points_diff.append([race, df_team['driverRef'][1],
                                    df_team['points'][1] - df_team['points'][0]])

    #Add feature to DataFrame
    points_diff = pd.DataFrame(points_diff, columns=['raceId', 'driverRef', 'points_to_teammate'])
    f1_data = f1_data.merge(points_diff, how='left', on=['raceId', 'driverRef'])

    #Drop points column from primary data
    f1_data.drop(columns='points', axis=1, inplace=True)

    return f1_data
