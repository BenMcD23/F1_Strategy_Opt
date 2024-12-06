import os
import pickle
import pandas as pd

from models import init_db, Circuit, Season, RacingWeekend, Driver, Session, SessionResult, Lap

db_engine, db_session = init_db()

def convert_lap_time(LapTime):
    total_seconds = LapTime.total_seconds()

    # total time as a float in SS.MM format
    return round(total_seconds, 3)  # Round to 3 decimal places for milliseconds

# Define the path to the cache directory
currentDir = os.getcwd()

# Define the path to the cache directory relative to the current directory
cacheDir = os.path.join(currentDir, 'cache')
yes = False
# Loop through each season (e.g., 2018, 2019, ...)
for seasonYear in os.listdir(cacheDir):
    seasonPath = os.path.join(cacheDir, seasonYear)

    seasonDB = db_session.query(Season).filter_by(year=seasonYear).first()
    if not seasonDB:
        seasonDB = Season(year=seasonYear)
        db_session.add(seasonDB)
        db_session.commit()

    print(f"Processing Season: {seasonYear}")

    # if yes is False:
    #     yes = True
    #     continue
    # Loop through each race weekend in the season
    for roundNum, raceWeekend in enumerate(os.listdir(seasonPath), start=1):
        raceWeekendPath = os.path.join(seasonPath, raceWeekend)

        # Ensure we are processing only directories (race weekends)
        if os.path.isdir(raceWeekendPath):
            for session in os.listdir(raceWeekendPath):
                print(session)
                sessionPath = os.path.join(raceWeekendPath, session)

                # now at this points get the file session_info.pkl and print it
                sessionInfoPath = os.path.join(sessionPath, 'session_info.ff1pkl')
                # Load and print session info
                with open(sessionInfoPath, 'rb') as file:
                    sessionInfo = pickle.load(file)

                circuitName = sessionInfo['data']['Meeting']['Circuit']['ShortName']
                circuitDB = db_session.query(Circuit).filter_by(circuit_name=circuitName).first()
                # add if doesnt exist
                if not circuitDB:
                    circuitDB = Circuit(circuit_name=circuitName)
                    db_session.add(circuitDB)
                    db_session.commit()

                # add race weekend
                racingWeekendDB = db_session.query(RacingWeekend).filter_by(year=seasonYear, round=roundNum).first()
                if not racingWeekendDB:
                    racingWeekendDB = RacingWeekend(year=seasonYear, round=roundNum, circuit=circuitDB)
                    db_session.add(racingWeekendDB)
                    db_session.commit()

                # add session
                sessionName = sessionInfo['data']['Name']
                session = db_session.query(Session).filter_by(weekend_id=racingWeekendDB.id, session_type=sessionName).first()
                if not session:
                    session = Session(racing_weekend=racingWeekendDB, session_type=sessionName)
                    db_session.add(session)
                    db_session.commit()

                driversPath = os.path.join(sessionPath, 'driver_info.ff1pkl')
                # Load and print session info
                with open(driversPath, 'rb') as file:
                    driverData = pickle.load(file)
                driverData = driverData.get('data', {})

                currentSessionDrivers = {}

                for driverID, driverInfo in driverData.items():
                    # has got team if needed

                    driverDB = db_session.query(Driver).filter_by(driver_name=driverInfo["FullName"]).first()
                    if not driverDB:
                        driverDB = Driver(driver_num=driverInfo["RacingNumber"], driver_name=driverInfo["FullName"], driver_short=driverInfo["Tla"])
                        db_session.add(driverDB)
                        db_session.commit()

                    currentSessionDrivers.update({int(driverInfo["RacingNumber"]): driverDB.id})


                timingPath = os.path.join(sessionPath, 'timing_app_data.ff1pkl')
                # Load and print session info
                with open(timingPath, 'rb') as file:
                    timingData = pickle.load(file)


                df1 = pd.DataFrame(timingData['data'])

                timingPath = os.path.join(sessionPath, '_extended_timing_data.ff1pkl')
                # Load and print session info
                with open(timingPath, 'rb') as file:
                    timingData = pickle.load(file)

                df2 = pd.DataFrame(timingData['data'][1])
                print(df1)
                print(df2)
                df3 = pd.DataFrame(timingData['data'][0])
                print(df3)
                # merged_df = pd.merge(df1, df2[['Time', 'Position']], on='Time', how='left')

                print(f"Number of rows in df1: {df1.shape[0]}")
                print(f"Number of rows in df2: {df2.shape[0]}")


                # This will add the 'Position' from df2 to df1 based on matching 'Time' values
                # # Process the laps (adjust this based on your timing data structure)
                # for lap in df1.itertuples():


                #     # make sure laptime is not NaN
                #     if not pd.isna(lap.LapTime):
                #         # check if driver exits
                #         driver = db_session.query(Driver).filter_by(id=currentSessionDrivers[int(lap.Driver)]).first()
                #         if not driver:
                #             print("bad")

                #         # Create lap record
                #         lap_record = Lap(
                #             session=session,
                #             driver=driver,
                #             lap_num=int(lap.LapNumber),
                #             lap_time=convert_lap_time(lap.LapTime),
                #             position=lap.Position,
                #             pit=not pd.isna(lap.PitOutTime)
                #         )
                #         db_session.add(lap_record)
                #         db_session.commit()


            break
        break
    break

                    # driver = db_session.query(Driver).filter_by(driver_name=DriverDetails.FullName).first()
                    # if not driver:
                    #     driver = Driver(driver_num=DriverDetails.DriverNumber, driver_name=DriverDetails.FullName, driver_short=DriverDetails.Abbreviation)
                    #     db_session.add(driver)
                    #     db_session.commit()

                    # CurrentSessionDrivers.update({int(DriverNum): driver.id})











