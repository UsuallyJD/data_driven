'''
Helper functions for data cleaning
- read raw data
- merge raw data
- drop missing observations
'''

import pandas as pd

def read_raw_data():
    '''
    import function that reads in data from local csv files

    Parameters:
        None
    
    Returns:
        (list) of dataframes
    '''

    races = pd.read_csv('Datasets/races.csv',
                        parse_dates = ['date'],
                        usecols = ['date', 'round', 'raceId', 'circuitId'])

    circuits = pd.read_csv('Datasets/circuits.csv',
                           usecols = ['circuitId', 'circuitRef', 'country', 'alt'])

    qualifying = pd.read_csv('Datasets/qualifying.csv',
                             usecols = ['raceId', 'driverId', 'constructorId', 'position', 'q1', 'q2', 'q3'])
    #Feature name will become redundant, change 'position' to 'grid', ie grid_position
    qualifying.rename(columns={'position':'grid'}, inplace=True)

    times = pd.read_csv('Datasets/lap_times.csv')
    # calculate average lap time rounded to tenths of a second (ie hundreds of milliseconds)
    times = times.pivot(index = ['raceId', 'driverId'], columns='lap', values='milliseconds')
    times.reset_index(inplace=True)
    times['avg_lap'] = round((times.mean(axis=1).astype(int)), -2)

    results = pd.read_csv('Datasets/results.csv',
                          usecols = ['raceId', 'driverId', 'constructorId', 'grid', 'position'])

    drivers = pd.read_csv('Datasets/drivers.csv',
                          parse_dates = ['dob'],
                          usecols = ['driverId', 'driverRef', 'dob', 'nationality'])

    driver_table = pd.read_csv('Datasets/driver_standings.csv',
                               usecols = ['raceId', 'driverId', 'points', 'position'])
    driver_table.rename(columns={'position': 'table_position'}, inplace=True)
    #Adjust driver table so points are from before the race
    driver_table['raceId'] = driver_table['raceId'] + 1

    stops = pd.read_csv('Datasets/pit_stops.csv')
    #Extract strategy data
    stops = stops.loc[lambda stops: stops['stop'] <= 3]
    stops = stops.pivot(index = ['raceId', 'driverId'], columns='stop', values=['lap'])
    stops.reset_index(inplace=True)
    #Fix column names
    stops.columns = stops.columns.droplevel()
    stops.columns = ['raceId', 'driverId', 'one_stop', 'two_stop', 'three_stop']
    stops.dropna(subset='one_stop', inplace=True)
    #Remove lap counter from strategy board, map to binary (OHE for strategies)
    stops['stops'] = stops['one_stop'].notnull().astype(int) + \
                     stops['two_stop'].notnull().astype(int) + \
                     stops['three_stop'].notnull().astype(int)

    teams = pd.read_csv('Datasets/constructors.csv',
                        usecols = ['constructorId', 'constructorRef'])
    # adjust for changing team names
    teams['constructorRef'].replace('toro_rosso', 'alpha_tauri', inplace=True)
    teams['constructorRef'].replace('alfa', 'sauber', inplace=True)
    teams['constructorRef'].replace(['lotus_f1', 'renault'], 'alpine', inplace=True)
    teams['constructorRef'].replace(['force_india', 'racing_point'], 'aston_martin', inplace=True)
    teams['constructorRef'].replace(['virgin', 'marussia'], 'manor', inplace=True)
    teams['constructorRef'].replace('lotus_racing', 'caterham', inplace=True)

    return [races, circuits, qualifying, times, results, drivers, driver_table, stops, teams]


def merge_raw_data(files):
    '''
    Merges all the relevant data from separate dataframes into one

    Parameters:
        files: (list) of dataframes
    
    Returns:
        proto_data: (pd.DataFrame) of merged data
    '''

    # merge dataframes on race, circuit, driver, and constructor IDs
    proto_data = files[0] #races

    proto_data = proto_data.merge(files[1], # circuits
                             how='left',
                             on='circuitId')

    proto_data = proto_data.merge(files[2], # qualifying
                                  how='left',
                                  on='raceId')

    proto_data = proto_data.merge(files[3][['raceId', 'driverId', 'avg_lap']], # times
                                  how='left',
                                  on=['raceId', 'driverId'])

    proto_data = proto_data.merge(files[4].drop(columns='grid'), # results
                                  how = 'left',
                                  on = ['raceId', 'driverId', 'constructorId'])

    proto_data = proto_data.merge(files[5], # drivers
                                  how='left',
                                  on='driverId')

    proto_data = proto_data.merge(files[6][['raceId', 'driverId', 'points', 'table_position']], # driver_table
                                  how='left',
                                  on=['raceId', 'driverId'])

    proto_data = proto_data.merge(files[7][['raceId', 'driverId', 'stops']], # stops
                                  how='left',
                                  on=['raceId', 'driverId'])

    proto_data = proto_data.merge(files[8], # teams
                            how='left',
                            on='constructorId')
    
    # fix points column so season beginning points are zero
    proto_data.loc[proto_data['round'] == 1,['points']] = \
        proto_data.loc[proto_data['round'] == 1, ['points']].values[:] = 0

    # replace empty points and table_position with zero
    proto_data['points'].fillna(0, inplace=True)
    proto_data['table_position'].fillna(0, inplace=True)

    return proto_data


def drop_missing_observations(f1_data):
    '''
    filtering function to remove observations with missing data

    parameters:
        f1_data: (pd.DataFrame) of raw data

    returns:
        f1_data: (pd.DataFrame) of filtered data
    '''

    # replace new line command (from read-in)
    f1_data.replace(r'\\N', 0, inplace=True, regex=True)

    # remove observations bad observations:
    f1_data.dropna(subset=['driverRef'], inplace = True)    # no driver name
    f1_data.dropna(subset=['points_to_teammate'], inplace = True) #no points to teammate
    f1_data.dropna(subset=['quali_time'], inplace = True) # no qualifying time

    f1_data.dropna(subset=['stops'], inplace = True) # no stop data
    f1_data = f1_data.loc[f1_data['stops'].astype('int32') > 0] # remove obs with invalid data

    f1_data.dropna(subset=['position'], inplace = True) # no finishing position (ie DNFs excluded)
    f1_data = f1_data.loc[f1_data['position'].astype('int32') <= 20] # remove obs from drivers who finish improbably low
    f1_data = f1_data.loc[f1_data['position'].astype('int32') > 0] # remove obs with invalid data


    return f1_data
