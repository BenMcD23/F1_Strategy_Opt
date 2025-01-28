import fastf1 as ff1
from fastf1 import plotting

import logging
import pandas as pd
from datetime import datetime
import time
from models import init_db, Circuit, Season, RacingWeekend, Driver, Session, SessionResult, Lap, TyreRaceData, Team, DriverTeamSession, TeamCircuitStats
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import os
# os.remove('f1_data.db')

# Initialize logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger('fastf1').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Database initialization
db_engine, db_session = init_db()

# FastF1 cache
ff1.Cache.enable_cache(r'C:\Users\mcdon\OneDrive - University of Leeds\Desktop\individual-project-BenMcD23\cache')


# Function Definitions
def correct_fuel_effect(df, max_fuel_kg=110, fuel_effect_per_kg=0.03):
	# Find the maximum number of laps completed by any driver in the race
	max_laps_race = df['LapNumber'].max()

	# Group by driver to process each driver's laps individually
	def _correct_fuel_for_driver(driver_df):
		# Make sure we modify the original DataFrame using .loc to avoid SettingWithCopyWarning
		driver_df.loc[:, 'fuel_weight'] = max_fuel_kg - (driver_df['LapNumber'] - 1) * (max_fuel_kg / max_laps_race)
		driver_df.loc[:, 'fuel_correction'] = driver_df['fuel_weight'] * fuel_effect_per_kg
		driver_df.loc[:, 'fuel_corrected_lap_time'] = driver_df['LapTimeSeconds'] - driver_df['fuel_correction']
		return driver_df

	# Apply the correction to each driver's laps using groupby and avoid deprecated behavior with group_keys=False
	df = df.groupby('DriverNumber', group_keys=False)[df.columns].apply(_correct_fuel_for_driver).reset_index(drop=True)
	return df


def assign_stint_numbers(df):
	# Assign stint numbers to laps based on pit stops for each driver
	df['stint'] = np.nan
	for driver in df['DriverNumber'].unique():
		driver_data = df[df['DriverNumber'] == driver]
		stint_number = 1
		for i in driver_data.index:
			if driver_data.loc[i, 'pit'] and i != driver_data.index[0]:
				stint_number += 1
			df.loc[i, 'stint'] = stint_number
	df['stint'] = df['stint'].astype(int)
	return df

def remove_laps_outside_percent(df, percentage=5):
	# Group by driver and apply the filtering logic to each driver's laps
	def _filter_driver_laps(driver_df):
		# Calculate the threshold based on the fastest lap time for the driver
		fastest_lap_time = driver_df['fuel_corrected_lap_time'].min()
		threshold = fastest_lap_time * (1 + percentage / 100)

		# Remove laps not within the specified percentage of the fastest lap time
		driver_df = driver_df[driver_df['fuel_corrected_lap_time'] <= threshold]

		return driver_df

	# Apply the filtering logic to each driver's laps
	df = df.groupby('DriverNumber')[df.columns].apply(_filter_driver_laps).reset_index(drop=True)

	return df

def normalise_lap_times(df):
	# Group by driver and calculate the fastest lap time for each driver
	df['fastest_lap_time'] = df.groupby('DriverNumber')['fuel_corrected_lap_time'].transform('min')

	# Normalise lap times by subtracting the fastest lap time
	df['normalised_lap_time'] = df['fuel_corrected_lap_time'] - df['fastest_lap_time']

	# Drop the fastest_lap_time as it's no longer needed
	df = df.drop(columns=['fastest_lap_time'])

	return df

def add_tyre_deg(df, CurrentSessionDrivers, session_id, db_session):
	# Normalize lap times
	df = correct_fuel_effect(df)
	df = remove_laps_outside_percent(df)
	df = normalise_lap_times(df)

	pd.set_option('display.max_rows', None)  # Show all rows
	pd.set_option('display.max_columns', None)  # Show all columns
	pd.set_option('display.width', 1000)  # Adjust the width to avoid line breaks
	pd.set_option('display.colheader_justify', 'left')  # Align headers to the left

	# Get the list of all drivers
	drivers = df['DriverNumber'].unique()

	# Dictionary to store the results
	degradation_curves = {}

	# Define minimum laps required for each tyre type
	min_laps_by_tyre = {
		1: 8,    # Tyre Type 1 (e.g., Soft)
		2: 14,    # Tyre Type 2 (e.g., Medium)
		3: 24     # Tyre Type 3 (e.g., Hard)
	}

	# Process each driver
	for driver in drivers:
		# Filter data for the current driver
		df_driver = df[df['DriverNumber'] == driver]

		# Group by tyre type
		tyre_types = df_driver['tyre'].unique()

		# Dictionary to store polynomial coefficients for each tyre type
		tyre_coefficients = {tyre: [] for tyre in tyre_types}

		# Fit polynomial curves for each tyre type
		for tyre in tyre_types:
			tyre_data = df_driver[df_driver['tyre'] == tyre]

			# Group by stint to handle multiple stints with the same tyre type
			for stint, stint_data in tyre_data.groupby('stint'):
				# Check if the stint meets the minimum lap requirement for the tyre type
				min_laps = min_laps_by_tyre.get(tyre, 0)  # Default to 0 if tyre type not in dictionary
				if len(stint_data) < min_laps:
					# print(f"Skipping stint for {driver} on {tyre} tyres: Only {len(stint_data)} laps (minimum {min_laps} required).")
					continue

				x = stint_data['TyreLife'].values.reshape(-1, 1)  # Tyre laps as x-axis
				y = stint_data['normalised_lap_time'].values       # Normalized lap times as y-axis

				# Fit a polynomial of degree 2
				poly = PolynomialFeatures(degree=2)
				x_poly = poly.fit_transform(x)
				model = LinearRegression()
				model.fit(x_poly, y)

				# Store the coefficients (a, b, c) for the polynomial ax^2 + bx + c
				coefficients = [model.coef_[2], model.coef_[1], model.intercept_]  # [a, b, c]
				tyre_coefficients[tyre].append(coefficients)

		# Average coefficients for each tyre type
		averaged_coefficients = {}
		for tyre, coefficients_list in tyre_coefficients.items():
			if coefficients_list:  # Check if there are coefficients for this tyre type
				averaged_coefficients[tyre] = np.mean(coefficients_list, axis=0)

		# Store the averaged coefficients in the result dictionary
		degradation_curves[driver] = averaged_coefficients
	# print(degradation_curves)

	# After calculating degradation_curves:
	for driver_num, tyre_data in degradation_curves.items():
		# print(driver_num)
		# print(tyre_data)
		driver = db_session.query(Driver).filter_by(driver_id=CurrentSessionDrivers[int(driver_num)]).first()
		if not driver:
			print(f"Driver {driver_num} not found. Skipping tyre data insertion.")
			continue
		
		for tyre_type, coefficients in tyre_data.items():
			if len(coefficients) != 3:
				print(f"Invalid coefficients for {driver_num} tyre {tyre_type}. Skipping.")
				continue

			# Check for existing entry to avoid duplicates
			existing_entry = db_session.query(TyreRaceData).filter_by(
				race_id=session_id,
				driver_id=driver.driver_id,
				tyre_type=tyre_type
			).first()

			if not existing_entry:
				new_entry = TyreRaceData(
					race_id=session_id,
					driver_id=driver.driver_id,
					tyre_type=int(tyre_type),
					a=float(coefficients[0]),
					b=float(coefficients[1]),
					c=float(coefficients[2])
				)
				db_session.add(new_entry)
	
	db_session.commit()  # Commit all new entries

	return df


def add_circuit_stats(df, fastest_laps_quali, circuit_id, db_session):
	# Calculate Pitstop Time
	df['PitstopTime'] = (df['PitOutTime'] - df['PitInTime']).dt.total_seconds()
	
	# Shift PitOutTime column by 1 lap to align with the previous lap
	df['NextPitOutTime'] = df['PitOutTime'].shift(-1)
	
	# Filter rows where PitInTime is not null and the next row has PitOutTime
	valid_pits = df[df['PitInTime'].notna() & df['NextPitOutTime'].notna()].copy()
	valid_pits['PitstopTime'] = (valid_pits['NextPitOutTime'] - valid_pits['PitInTime']).dt.total_seconds()
	
	# Calculate average pitstop time for each team
	average_pitstop_by_team = valid_pits.groupby('Team')['PitstopTime'].mean().to_dict()
	
	# Calculate percentage difference from qualifying to race times
	quali_race_diff = {}
	team_differences = {}

	# Map DriverNumber to Teams
	driver_to_team = df[['DriverNumber', 'Team']].drop_duplicates().set_index('DriverNumber')['Team'].to_dict()
	

	# find fastest laps in race
	fastest_laps_by_driver = (
		df.groupby(['Team', 'DriverNumber'])['LapTimeSeconds']
		.min()
		.reset_index()
	)

	# Group by Team and calculate the average fastest lap time across the two drivers
	fastest_laps_race = (
		fastest_laps_by_driver.groupby('Team')['LapTimeSeconds']
		.mean()
		.to_dict()
	)
	
	percent_diff = {}

	# Iterate through the teams in the qualifying dictionary
	for team, quali_time in fastest_laps_quali.items():
		# Ensure the team is also in the race dictionary
		if team in fastest_laps_race:
			race_time = fastest_laps_race[team]
			# Calculate the percentage difference
			percent_diff[team] = ((race_time - quali_time) / quali_time) * 100

	for team, avg_pitstop_time in average_pitstop_by_team.items():
		team_database = db_session.query(Team).filter_by(team_name=team).first()
		if team in percent_diff:
			quali_to_race_diff = percent_diff[team]

			# Check if the stats for this circuit and team already exist
			existing_stats = db_session.query(TeamCircuitStats).filter_by(
				circuit_id=circuit_id, team_id=team_database.team_id
			).first()

			if existing_stats:
				# If the stats exist, average the new and old values
				new_pit_time = (existing_stats.pit_time + avg_pitstop_time) / 2
				new_quali_to_race_diff = (existing_stats.quali_to_race_percent_diff + quali_to_race_diff) / 2

				# Update the existing entry
				existing_stats.pit_time = new_pit_time
				existing_stats.quali_to_race_percent_diff = new_quali_to_race_diff
			else:
				# If the stats don't exist, create a new entry
				new_stats = TeamCircuitStats(
					circuit_id=circuit_id,
					team_id=team_database.team_id,
					pit_time=avg_pitstop_time,
					quali_to_race_percent_diff=quali_to_race_diff
				)
				db_session.add(new_stats)

	# Commit the changes to the database
	db_session.commit()


def convert_lap_time(lap_time):
	return round(lap_time.total_seconds(), 3)


def get_weather_for_lap(lap_row, weather_data):
	lap_start = lap_row['LapStartTime']
	lap_end = lap_start + lap_row['LapTime']
	relevant_weather = weather_data[(weather_data['Time'] >= lap_start) & (weather_data['Time'] <= lap_end)]
	if not relevant_weather.empty:
		return relevant_weather['Rainfall'].max(), relevant_weather['TrackTemp'].mean()
	return 0, None


# Main Processing Loop
start_time = time.time()
years = range(2019, 2025)

tyre_mapping = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3, 'INTERMEDIATE': 4, 'WET': 5}

for year in years:
	season = db_session.query(Season).filter_by(year=year).first()
	if not season:
		season = Season(year=year)
		db_session.add(season)
		db_session.commit()

	schedule = ff1.get_event_schedule(year)
	schedule = schedule[schedule['RoundNumber'] != 0]

	# schedule = schedule[schedule['RoundNumber'] == 1]

	for _, event in schedule.iterrows():

		currentRoundNum = event['RoundNumber']

		circuit = db_session.query(Circuit).filter_by(circuit_name=event['Location']).first()
		if not circuit:
			circuit = Circuit(circuit_name=event['Location'])
			db_session.add(circuit)
			db_session.commit()

		racing_weekend = db_session.query(RacingWeekend).filter_by(year=year, round=currentRoundNum).first()
		if not racing_weekend:
			racing_weekend = RacingWeekend(year=year, round=currentRoundNum, circuit=circuit)
			db_session.add(racing_weekend)
			db_session.commit()

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
					db_session.commit()


				CurrentSessionDrivers = {}
				# add all the drivers that took part in session
				for DriverNum in sessionData.drivers:

					DriverDetails = sessionData.get_driver(DriverNum)

					driver = db_session.query(Driver).filter_by(driver_name=DriverDetails.FullName).first()
					if not driver:
						driver = Driver(driver_num=DriverDetails.DriverNumber, driver_name=DriverDetails.FullName, driver_short=DriverDetails.Abbreviation)
						db_session.add(driver)
						db_session.commit()

					CurrentSessionDrivers.update({int(DriverNum): driver.driver_id})

				results = sessionData.results

				# add team and link drivers and sessions and teams
				team_drivers = results.groupby('TeamName')['FullName'].apply(list)

				for team_name, drivers in team_drivers.items():
					# print(drivers)
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
						except ValueError:
							# Fallback if team name isn't recognized
							team_color = '#000000'  # Default black
							
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
							

				db_session.commit()


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
				sessionData.load()
				laps = sessionData.laps

				weather = laps.get_weather_data()

				# Convert laps to DataFrame to handle weather data mapping
				laps_df = laps.copy()
				laps_df['Rainfall'], laps_df['TrackTemp'] = zip(*laps_df.apply(get_weather_for_lap, axis=1, weather_data=weather))
				
				laps_df['LapTimeSeconds'] = laps_df['LapTime'].apply(convert_lap_time)
				laps_df['pit'] = ~pd.isna(laps_df['PitOutTime'])
				laps_df['tyre'] = laps_df['Compound'].map(lambda x: tyre_mapping.get(x, -1))
				laps_df = assign_stint_numbers(laps_df)


				if session_name == "Qualifying":
					# Group by both Team and DriverNumber to calculate the fastest lap for each driver
					fastest_laps_by_driver = (
						laps_df.groupby(['Team', 'DriverNumber'])['LapTimeSeconds']
						.min()
						.reset_index()
					)
				
					# Group by Team and calculate the average fastest lap time across the two drivers
					fastest_laps_quali = (
						fastest_laps_by_driver.groupby('Team')['LapTimeSeconds']
						.mean()
						.to_dict()
					)

				if session_name == "Race":
					add_tyre_deg(laps_df, CurrentSessionDrivers, session.session_id, db_session)
					add_circuit_stats(laps_df, fastest_laps_quali, circuit.circuit_id, db_session)

				if laps_df['Rainfall'].any():  # Check if any lap had rainfall
					session = db_session.query(Session).filter_by(
						weekend_id=racing_weekend.racing_weekend_id,
						session_type=session_name
					).first()

					if session:
						session.wet = True
						db_session.commit()
						
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
							stint_num=lap.stint,
							lap_time=float(lap.LapTimeSeconds),
							position=lap.Position,
							tyre=lap.tyre,
							tyre_laps=lap.TyreLife,
							pit=lap.pit,
							rainfall=lap.Rainfall
						)
						db_session.add(lap_record)
				db_session.commit()


			except Exception as e:

				logger.error(f"Error processing {session_name} for {year} Round {currentRoundNum}: {e}")

end_time = time.time()
print(f"Script completed in {end_time - start_time:.2f} seconds.")
