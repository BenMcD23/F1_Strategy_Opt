import fastf1 as ff1
from fastf1 import plotting

import logging
import pandas as pd
from datetime import datetime
import time
from models import init_db, Circuit, Season, RacingWeekend, Driver, Session, SessionResult, Lap, Team, DriverTeamSession, PitStop
import numpy as np
import os
import math

os.remove('f1_data.db')

# Initialize logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger('fastf1').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Database initialization
db_engine, db_session = init_db()

# FastF1 cache
ff1.Cache.enable_cache(r'C:\Users\mcdon\OneDrive - University of Leeds\Desktop\individual-project-BenMcD23\cache')

def add_stint_laps_column(df):
	if 'DriverNumber' not in df.columns or 'Stint' not in df.columns or 'LapNumber' not in df.columns:
		raise ValueError("The DataFrame must contain 'DriverNumber', 'Stint', and 'LapNumber' columns.")

	# Group by 'DriverNumber' and 'Stint', then calculate the lap number within each group
	df['stint_laps'] = df.groupby(['DriverNumber', 'Stint']).cumcount() + 1

	return df


def convert_lap_time(lap_time):
	return round(lap_time.total_seconds(), 4)


def get_weather_for_lap(lap_row, weather_data):
	lap_start = lap_row['LapStartTime']
	lap_end = lap_start + lap_row['LapTime']
	relevant_weather = weather_data[(weather_data['Time'] >= lap_start) & (weather_data['Time'] <= lap_end)]
	if not relevant_weather.empty:
		return relevant_weather['Rainfall'].max(), relevant_weather['TrackTemp'].mean()
	return 0, None


# Main Processing Loop
start_time = time.time()
years = range(2022, 2025)

tyre_mapping = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3, 'INTERMEDIATE': 4, 'WET': 5}

for year in years:

	season = db_session.query(Season).filter_by(year=year).first()
	if not season:
		season = Season(year=year)
		db_session.add(season)
		db_session.flush()

	schedule = ff1.get_event_schedule(year)
	schedule = schedule[schedule['RoundNumber'] != 0]

	# schedule = schedule[schedule['RoundNumber'] == 1]

	for _, event in schedule.iterrows():

		currentRoundNum = event['RoundNumber']

		circuit = db_session.query(Circuit).filter_by(circuit_name=event['Location']).first()
		if not circuit:
			circuit = Circuit(circuit_name=event['Location'])
			db_session.add(circuit)
			db_session.flush()

		racing_weekend = db_session.query(RacingWeekend).filter_by(year=year, round=currentRoundNum).first()
		if not racing_weekend:
			racing_weekend = RacingWeekend(year=year, round=currentRoundNum, circuit=circuit)
			db_session.add(racing_weekend)
			db_session.flush()

		for session_name in ['Practice 1', 'Practice 2', 'Practice 3', 'Qualifying', 'Race']:
			try:
				sessionData = ff1.get_session(year, currentRoundNum, session_name)
				sessionData.load()

				print(f"Loaded {session_name} for {year} Round {currentRoundNum}")

				session = db_session.query(Session).filter_by(weekend_id=racing_weekend.racing_weekend_id, session_type=session_name).first()
				if not session:
					# need to add wet later on
					session = Session(racing_weekend=racing_weekend, session_type=session_name, wet=False)
					db_session.add(session)
					db_session.flush()


				CurrentSessionDrivers = {}
				# add all the drivers that took part in session
				for DriverNum in sessionData.drivers:

					DriverDetails = sessionData.get_driver(DriverNum)

					driver = db_session.query(Driver).filter_by(driver_name=DriverDetails.FullName).first()
					if not driver:
						driver = Driver(driver_num=DriverDetails.DriverNumber, driver_name=DriverDetails.FullName, driver_short=DriverDetails.Abbreviation)
						db_session.add(driver)
						db_session.flush()

					CurrentSessionDrivers.update({int(DriverNum): driver.driver_id})

				results = sessionData.results

				# add team and link drivers and sessions and teams
				team_drivers = results.groupby('TeamName')['FullName'].apply(list)

				for team_name, drivers in team_drivers.items():
					team_name_mapping = {
						"Alfa Romeo": "Kick Sauber",
						"AlphaTauri": "RB",
						"Red Bull": "Red Bull Racing"
					}
					team_color_mapping = {
						"Kick Sauber": "#00e700",
						"RB": "#364aa9",
						"Red Bull Racing": "#0600ef",
					}

					# Update the team name using the dictionary
					team_name = team_name_mapping.get(team_name, team_name)

					# Check if team exists
					team = db_session.query(Team).filter_by(team_name=team_name).first()

					if not team:
						try:
							# Get official team color from FastF1
							team_color = plotting.get_team_color(
								identifier=team_name,
								session=sessionData,
								exact_match=True  # Requires exact team name match
							)
						except:
							# Fallback if team name isn't recognized
							team_color = team_color_mapping.get(team_name, '#000000')  # Default black if not found

						# Create new team with dynamic color
						team = Team(
							team_name=team_name,
							TeamColor=team_color
						)

						db_session.add(team)

					for driver in drivers:
						driver_database = db_session.query(Driver).filter_by(driver_name=driver).first()

						driver_team_session = db_session.query(DriverTeamSession).filter_by(team_id=team.team_id, session_id=session.session_id, driver_id=driver_database.driver_id).first()

						if not driver_team_session:
							driver_team_session = DriverTeamSession(
								team_id=team.team_id,
								session_id=session.session_id,
								driver_id=driver_database.driver_id
							)
							db_session.add(driver_team_session)


				db_session.flush()


				for result in results.itertuples():

					if not pd.isna(result.Position):

						driver = db_session.query(Driver).filter_by(driver_id=CurrentSessionDrivers[int(result.DriverNumber)]).first()
						if not driver:
							print("bad")

						session_result = db_session.query(SessionResult).filter_by(session=session, driver=driver).first()
						if not session_result:

							session_result = SessionResult(session=session,
										driver=driver,
										position=int(result.Position),
										result_classified_pos=result.ClassifiedPosition,
										grid_pos=int(result.GridPosition) if not pd.isna(result.GridPosition) else None,
										end_status=result.Status
										)
							db_session.add(session_result)
							db_session.flush()
				sessionData.load()
				laps = sessionData.laps

				weather = laps.get_weather_data()

				# Convert laps to DataFrame to handle weather data mapping
				laps_df = laps.copy()
				laps_df['Rainfall'], laps_df['TrackTemp'] = zip(*laps_df.apply(get_weather_for_lap, axis=1, weather_data=weather))

				laps_df['LapTimeSeconds'] = laps_df['LapTime'].apply(convert_lap_time)

				laps_df['Sector1TimeSecs'] = laps_df['Sector1Time'].apply(convert_lap_time)
				laps_df['Sector2TimeSecs'] = laps_df['Sector2Time'].apply(convert_lap_time)
				laps_df['Sector3TimeSecs'] = laps_df['Sector3Time'].apply(convert_lap_time)

				laps_df['pit'] = ~pd.isna(laps_df['PitOutTime'])
				laps_df['tyre'] = laps_df['Compound'].map(lambda x: tyre_mapping.get(x, -1))


				# calculating pit time
				laps_df['AlignedPitInTime'] = laps_df.groupby('DriverNumber')['PitInTime'].shift(1)

				laps_df['PitTime'] = (
					laps_df['PitOutTime'] - laps_df['AlignedPitInTime']
				).dt.total_seconds()

				laps_df.drop(columns=['AlignedPitInTime'], inplace=True)

				if session_name == "Qualifying":
					laps_df = laps_df[laps_df["Deleted"] != True]

				if session_name == "Race":
					laps_df = add_stint_laps_column(laps_df)

				if laps_df['Rainfall'].any():  # Check if any lap had rainfall
					session = db_session.query(Session).filter_by(
						weekend_id=racing_weekend.racing_weekend_id,
						session_type=session_name
					).first()

					if session:
						session.wet = True
						db_session.flush()

				if session_name == "Race":

					for lap in laps_df.itertuples():

						# Check if driver exists
						driver = db_session.query(Driver).filter_by(driver_id=CurrentSessionDrivers[int(lap.DriverNumber)]).first()
						if not driver:
							print("Driver not found in session drivers map.")

						# Create lap record
						lap_record = Lap(
							session=session,
							driver=driver,
							lap_num=int(lap.LapNumber),
							stint_num=int(lap.Stint),
							stint_lap=int(lap.stint_laps) if not pd.isna(lap.stint_laps) else None,
							lap_time=float(lap.LapTimeSeconds) if not pd.isna(lap.LapTimeSeconds) else None,
							s1_time=float(lap.Sector1TimeSecs) if not pd.isna(lap.Sector1TimeSecs) else None,
							s2_time=float(lap.Sector2TimeSecs) if not pd.isna(lap.Sector2TimeSecs) else None,
							s3_time=float(lap.Sector3TimeSecs) if not pd.isna(lap.Sector3TimeSecs) else None,
							position=lap.Position,
							tyre_type=lap.tyre,
							tyre_laps=lap.TyreLife,
							pit=True if not pd.isna(lap.PitOutTime) else False,
							track_status=lap.TrackStatus,
							rainfall=lap.Rainfall
						)
						db_session.add(lap_record)

						db_session.flush()

						# if we got a pit on this lap
						if not pd.isna(lap.PitTime):
							pit_record = PitStop(
								lap_id=lap_record.lap_id,
								pit_time=float(lap.PitTime)
							)
							db_session.add(pit_record)

					db_session.flush()

				else:
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
								lap_time=float(lap.LapTimeSeconds),
								s1_time=float(lap.Sector1TimeSecs),
								s2_time=float(lap.Sector2TimeSecs),
								s3_time=float(lap.Sector3TimeSecs),
								position=lap.Position,
								tyre_type=lap.tyre,
								tyre_laps=lap.TyreLife,
								pit=lap.pit,
								rainfall=lap.Rainfall
							)
							db_session.add(lap_record)

					db_session.flush()



			except Exception as e:

				print(f"Error processing {session_name} for {year} Round {currentRoundNum}: {e}")
		db_session.commit()


db_session.commit()


end_time = time.time()
print(f"Script completed in {end_time - start_time:.2f} seconds.")
