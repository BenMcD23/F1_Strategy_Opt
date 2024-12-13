import fastf1 as ff1
from fastf1 import plotting
from fastf1.core import DataNotLoadedError
import logging
logging.getLogger('fastf1').setLevel(logging.WARNING)

import pandas as pd
from datetime import datetime
import time
from models import init_db, Circuit, Season, RacingWeekend, Driver, Session, SessionResult, Lap

db_engine, db_session = init_db()

# ff1.Cache.set_disabled()
ff1.Cache.enable_cache(r'D:\FastF1_Data\cache')

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

years = range(2019, 2025)

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

    # count = 0
    for _, event in schedule.iterrows():
        # count += 1

        # if count == 3:
        #     break
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

                session = db_session.query(Session).filter_by(weekend_id=racing_weekend.id, session_type=session_name).first()
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

                    CurrentSessionDrivers.update({int(DriverNum): driver.id})

                SessionData.load()
                results = SessionData.results
                for result in results.itertuples():

                    if not pd.isna(result.Position):

                        driver = db_session.query(Driver).filter_by(id=CurrentSessionDrivers[int(result.DriverNumber)]).first()
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
                        driver = db_session.query(Driver).filter_by(id=CurrentSessionDrivers[int(lap.DriverNumber)]).first()
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

            except ValueError:
                print("Sprint Weekend, no P2/3")

            except DataNotLoadedError:
                print("Not loaded for some reason")


end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time/60)

db_session.commit()
db_session.close()


