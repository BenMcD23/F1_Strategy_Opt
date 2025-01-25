import fastf1 as ff1
from fastf1 import plotting
from fastf1.core import DataNotLoadedError
import logging
logging.getLogger('fastf1').setLevel(logging.WARNING)

import pandas as pd
from datetime import datetime
import time
from models import init_db, Circuit, Season, RacingWeekend, Driver, Session, SessionResult, Lap, TyreDeg

import numpy as np

db_engine, db_session = init_db()

# ff1.Cache.set_disabled()
ff1.Cache.enable_cache(r'D:\FastF1_Data\cache')




def correct_fuel_effect(df, max_fuel_kg=110, fuel_effect_per_kg=0.03):
    # Adjust lap times in a race for the effect of fuel weight
    max_laps = df['lap_num'].max()
    df.loc[:, 'fuel_weight'] = max_fuel_kg - (df['lap_num'] - 1) * (max_fuel_kg / max_laps)
    df.loc[:, 'fuel_correction'] = df['fuel_weight'] * fuel_effect_per_kg
    df.loc[:, 'fuel_corrected_lap_time'] = df['lap_time'] - df['fuel_correction']
    return df

def assign_stint_numbers(df):
    df = df.copy()


    # Assign stint numbers to laps based on pit stops for each driver
    df.loc[:, 'stint'] = np.nan

    for driver in df['driver_id'].unique():
        driver_data = df[df['driver_id'] == driver]
        stint_number = 1
        for i in driver_data.index:
            if driver_data.loc[i, 'pit'] and i != driver_data.index[0]:
                stint_number += 1
            df.loc[i, 'stint'] = stint_number
    df.loc[:, 'stint'] = df['stint'].astype(int)
    return df

def analyze_tyre_degradation(df, race_id, db_session: Session):
    # Correct for fuel effect
    df = correct_fuel_effect(df)

    # Normalize lap times based on the fastest lap and filter out outliers
    fastest_lap_time = df['fuel_corrected_lap_time'].min()
    df = df[df['fuel_corrected_lap_time'] <= 1.03 * fastest_lap_time]  # Filter out lap times more than 3% slower

    # Calculate the difference from the fastest lap for each lap
    df.loc[:, 'time_diff'] = df['fuel_corrected_lap_time'] - fastest_lap_time
    # Assign stint numbers
    df = assign_stint_numbers(df)

    # Store polynomial coefficients for each driver and tyre type
    driver_tyre_poly_coeffs = {}

    # Loop over each driver
    for driver in df['driver_id'].unique():
        driver_data = df[df['driver_id'] == driver]
        driver_tyre_poly_coeffs[driver] = {}

        # Loop over each tyre type
        for tyre in driver_data['tyre'].unique():
            tyre_data = driver_data[driver_data['tyre'] == tyre]
            poly_coeffs_list = []

            # Loop through each stint for the given tyre type
            for stint in tyre_data['stint'].unique():
                stint_data = tyre_data[tyre_data['stint'] == stint]
                x = stint_data['lap_num']
                y = stint_data['time_diff']

                # Polynomial regression (2nd degree)
                if len(x) > 2:
                    poly_coeffs = np.polyfit(x, y, 2)
                    poly_coeffs_list.append(poly_coeffs)

            # Average the polynomial coefficients for the tyre type across stints for this driver
            if poly_coeffs_list:
                avg_poly_coeffs = np.mean(poly_coeffs_list, axis=0)
                driver_tyre_poly_coeffs[driver][tyre] = avg_poly_coeffs

    # Save the polynomial coefficients to the TyreDeg table
    for driver, tyre_coeffs in driver_tyre_poly_coeffs.items():

        # driverEntry = db_session.query(Driver).filter_by(driver_id=int(driver)).first()

        for tyre, coeffs in tyre_coeffs.items():
            tyre_deg_entry = TyreDeg(
                race_id=race_id,
                driver_id=int(driver),
                tyre_type=int(tyre),
                a=coeffs[0],  # x^2 coefficient
                b=coeffs[1],  # x coefficient
                c=coeffs[2]   # constant term
            )
            db_session.add(tyre_deg_entry)

    # Commit changes to the database
    db_session.commit()






def convert_lap_time(LapTime):
    total_seconds = LapTime.total_seconds()
    
    # total time as a float in SS.MM format
    return round(total_seconds, 3)  # Round to 3 decimal places for milliseconds

def get_weather_for_lap(lap_row, weather_data):
    lap_start = lap_row['LapStartTime']
    lap_end = lap_start + lap_row['LapTime']
    relevant_weather = weather_data[(weather_data['Time'] >= lap_start) & (weather_data['Time'] <= lap_end)]
    if not relevant_weather.empty:
        return relevant_weather['Rainfall'].max(), relevant_weather['TrackTemp'].mean()
    return 0, None  # Default: No rainfall, unknown track temp

tyre_mapping = {
    'SOFT': 1,
    'MEDIUM': 2,
    'HARD': 3,
    'INTERMEDIATE': 4,
    'WET': 5,
}

start_time = time.time()

years = range(2019, 2020)

available_years = []


# go over every year/season
for year in years:

    season = db_session.query(Season).filter_by(year=year).first()
    if not season:
        season = Season(year=year)
        db_session.add(season)
        db_session.commit()

    # get the event schedule for the season
    schedule = ff1.get_event_schedule(year)

    # get rid of pre-season testing
    schedule = schedule[schedule['RoundNumber'] != 0]

    count = 0
    for _, event in schedule.iterrows():
        count += 1

        if count == 2:
            break
        CurrentRoundNumber = event['RoundNumber']


        circuit = db_session.query(Circuit).filter_by(circuit_name=event['Location']).first()
        # add if doesnt exist
        if not circuit:
            circuit = Circuit(circuit_name=event['Location'])
            db_session.add(circuit)
            db_session.commit()
        

        racing_weekend = db_session.query(RacingWeekend).filter_by(year=year, round=CurrentRoundNumber).first()
        # add if doesnt exist
        if not racing_weekend:
            racing_weekend = RacingWeekend(year=year, round=CurrentRoundNumber, circuit=circuit)
            db_session.add(racing_weekend)
            db_session.commit()

        # no sprint races/quali
        for session_name in ['Practice 1', 'Practice 2', 'Practice 3', 'Qualifying', 'Race']:

            try:
                # load the session data
                SessionData = ff1.get_session(year, CurrentRoundNumber, session_name)
                SessionData.load()

                print(f"Year:{year}  Round:{CurrentRoundNumber}  Session:{session_name}")

                session = db_session.query(Session).filter_by(weekend_id=racing_weekend.racing_weekend_id, session_type=session_name).first()
                if not session:
                    session = Session(racing_weekend=racing_weekend, session_type=session_name)
                    db_session.add(session)
                    db_session.commit()

                # done like this as drivers can have the same number depending on season

                CurrentSessionDrivers = {}
                # add all the drivers that took part in session
                for DriverNum in SessionData.drivers:

                    DriverDetails = SessionData.get_driver(DriverNum)

                    driver = db_session.query(Driver).filter_by(driver_name=DriverDetails.FullName).first()
                    if not driver:
                        driver = Driver(driver_num=DriverDetails.DriverNumber, driver_name=DriverDetails.FullName, driver_short=DriverDetails.Abbreviation)
                        db_session.add(driver)
                        db_session.commit()

                    CurrentSessionDrivers.update({int(DriverNum): driver.driver_id})

                SessionData.load()
                results = SessionData.results
                for result in results.itertuples():

                    if not pd.isna(result.Position):

                        driver = db_session.query(Driver).filter_by(driver_id=CurrentSessionDrivers[int(result.DriverNumber)]).first()
                        if not driver:
                            print("bad")

                        session_result = db_session.query(SessionResult).filter_by(session=session, driver=driver).first()
                        if not session_result:
                            session_result = SessionResult(session=session, driver=driver, position=int(result.Position))
                            db_session.add(session_result)
                            db_session.commit()
                
                laps = SessionData.laps

                weather = laps.get_weather_data()

                # Convert laps to DataFrame to handle weather data mapping
                laps_df = laps.copy()
                laps_df['Rainfall'], laps_df['TrackTemp'] = zip(*laps_df.apply(get_weather_for_lap, axis=1, weather_data=weather))

                # Iterate over laps with rainfall data
                for lap in laps_df.itertuples():
                    # Make sure laptime is not NaN
                    if not pd.isna(lap.LapTime):
                        # Check if driver exists
                        driver = db_session.query(Driver).filter_by(driver_id=CurrentSessionDrivers[int(lap.DriverNumber)]).first()
                        if not driver:
                            print("Driver not found in session drivers map.")

                        # Create lap record
                        lap_record = Lap(
                            session=session,
                            driver=driver,
                            lap_num=int(lap.LapNumber),
                            lap_time=convert_lap_time(lap.LapTime),
                            position=lap.Position,
                            tyre=tyre_mapping.get(lap.Compound, -1),
                            tyre_laps=int(lap.TyreLife),
                            pit=not pd.isna(lap.PitOutTime),
                            rainfall=lap.Rainfall,
                            track_temp=lap.TrackTemp
                        )
                        db_session.add(lap_record)
                db_session.commit()

                # TYRE DEG
                if session_name == "Race":

                    query = db_session.query(
                        Driver.driver_id,
                        Lap.lap_num,
                        Lap.pit,
                        Lap.lap_time,
                        Lap.tyre,
                        Lap.tyre_laps,
                    ).join(RacingWeekend.sessions) \
                    .join(Session.laps) \
                    .join(Lap.driver) \
                    .filter(
                        Session.session_id == session.session_id  # Ensure session_type is 'Race'
                    ).all()

                    # Convert result to a list of dictionaries
                    data = [
                        {
                            'driver_id': row.driver_id,
                            'lap_num': row.lap_num,
                            'pit': row.pit,
                            'tyre': row.tyre,
                            'tyre_laps': row.tyre_laps,
                            'lap_time': row.lap_time
                        }
                        for row in query
                    ]

                    # Create a DataFrame
                    df = pd.DataFrame(data)

                    # Show the DataFrame

                    analyze_tyre_degradation(df, race_id=session.session_id, db_session=db_session)

            except ValueError:
                print("Sprint Weekend, no P2/3")

            except DataNotLoadedError:
                print("Not loaded for some reason")


end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time/60)

db_session.commit()
db_session.close()


