import fastf1 as ff1
import logging
logging.getLogger('fastf1').setLevel(logging.WARNING)

import pandas as pd
from datetime import datetime
import time
from models import init_db, Circuit, Season, RacingWeekend, Driver, Session, SessionResult, Lap



ff1.Cache.enable_cache(r'D:\FastF1_Data\cache')

def convert_lap_time(LapTime):
    total_seconds = LapTime.total_seconds()
    
    # total time as a float in SS.MM format
    return round(total_seconds, 3)  # Round to 3 decimal places for milliseconds

years = range(2024, 2025)

available_years = []

# go over every year/season
for year in years:


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



        # no sprint races/quali
        for session_name in ['Practice 1', 'Practice 2', 'Practice 3', 'Qualifying', 'Race']:
            try:
                # load the session data
                SessionData = ff1.get_session(year, CurrentRoundNumber, session_name)
                SessionData.load()

                print(f"Year:{year}  Round:{CurrentRoundNumber}  Session:{session_name}")

            except ValueError:
                print("Sprint Weekend, no P2/3")


          


