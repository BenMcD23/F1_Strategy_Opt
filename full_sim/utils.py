from scipy.optimize import minimize
import numpy as np
import pandas as pd
from DB.models import init_db, Circuit, Season, RacingWeekend, Driver, Session, SessionResult, Lap, Team, DriverTeamSession, PitStop
from sqlalchemy import func

# --------------------------------------
# Mainly for calculating tyre deg

def correct_fuel_effect(df, max_lap=None, max_fuel_kg=110, fuel_effect_per_kg=0.03):
	# Find the maximum number of laps completed by any driver in the race
	if max_lap is None:
		max_lap = df["lap_num"].max()

	# Group by driver to process each driver"s laps individually
	def _correct_fuel_for_driver(driver_df):
		# Make sure we modify the original DataFrame using .loc to avoid SettingWithCopyWarning
		driver_df.loc[:, "fuel_weight"] = max_fuel_kg - (driver_df["lap_num"] - 1) * (max_fuel_kg / max_lap)
		driver_df.loc[:, "fuel_correction"] = (driver_df["fuel_weight"] * fuel_effect_per_kg) / 3
		driver_df.loc[:, "fuel_corrected_sector_time"] = driver_df["sector_time"] - driver_df["fuel_correction"]
		return driver_df

	# Apply the correction to each driver"s laps using groupby and avoid deprecated behavior with group_keys=False
	df = df.groupby("driver_number", group_keys=False)[df.columns].apply(_correct_fuel_for_driver).reset_index(drop=True)
	return df


def assign_stint_numbers(df):
	# Assign stint numbers to laps based on pit stops for each driver
	df["stint"] = np.nan
	for driver in df["driver_number"].unique():
		driver_data = df[df["driver_number"] == driver]
		stint_number = 1
		for i in driver_data.index:
			if driver_data.loc[i, "pit"] and i != driver_data.index[0]:
				stint_number += 1
			df.loc[i, "stint"] = stint_number
	df["stint"] = df["stint"].astype(int)
	return df


def remove_laps_outside_percent(df, percentage=5):
	# Group by driver and apply the filtering logic to each driver"s laps
	def _filter_driver_sector_laps(driver_sector_df):
		# Calculate the threshold based on the fastest lap time for the driver and sector
		fastest_lap_time = driver_sector_df["fuel_corrected_sector_time"].min()
		threshold = fastest_lap_time * (1 + percentage / 100)
		
		# Remove laps not within the specified percentage of the fastest lap time
		filtered_df = driver_sector_df[driver_sector_df["fuel_corrected_sector_time"] <= threshold]
		return filtered_df

	# Group by driver and sector, then apply the filtering logic
	df = (
		df.groupby(["driver_number", "sector"], group_keys=False)
		.apply(_filter_driver_sector_laps)
		.reset_index(drop=True)
	)

	return df


def normalise_lap_times_by_sector(df):
	# Normalise lap times by subtracting the fastest sector time
	df['normalised_sector_time'] = (
		df['fuel_corrected_sector_time'] - df['base_sector_time']
	)

	
	return df
# --------------------------------------



def get_tyre_deg_per_driver(df):
	# Pre process
	df = assign_stint_numbers(df)
	df = correct_fuel_effect(df)
	df = remove_laps_outside_percent(df)
	df = normalise_lap_times_by_sector(df)
	
	# Define minimum laps required for each tyre type
	min_laps_by_tyre = {
		1: 0,  # Soft
		2: 0,  # Med
		3: 0   # Hard
	}

	# Dictionary to store tyre coefficients for each driver
	driver_tyre_coefficients = {}

	# Process each driver
	for driver in df["driver_number"].unique():
		# Filter data for the current driver
		df_driver = df[df["driver_number"] == driver]

		# Dictionary to store coefficients for this driver
		driver_coeffs = {}

		# Group by tyre type and sector
		for tyre in df_driver["tyre_type"].unique():
			tyre_data = df_driver[df_driver["tyre_type"] == tyre]

			# Dictionary to store coefficients for each sector
			tyre_sector_coeffs = {}

			for sector in df_driver["sector"].unique():
				sector_data = tyre_data[tyre_data["sector"] == sector]

				# List to store coefficients for this tyre type and sector
				tyre_sector_coefficients = []

				# Group by stint to handle multiple stints with the same tyre type and sector
				for stint, stint_data in sector_data.groupby("stint"):
					# Check if the stint meets the minimum lap requirement for the tyre type
					min_laps = min_laps_by_tyre.get(tyre, 0)  # Default to 0 if tyre type not in dictionary
					if len(stint_data) < min_laps:
						continue

					# Fit a constrained polynomial model
					x = stint_data["tyre_laps"].values
					y = stint_data["normalised_sector_time"].values

					# Define the objective function for least squares
					def objective(coeffs):
						a, b, c = coeffs
						return np.sum((a * x**2 + b * x + c - y)**2)

					# Define constraints
					constraints = (
						{"type": "ineq", "fun": lambda coeffs: coeffs[0]},  # Ensure a >= 0
					)

					# Initial guess for coefficients
					initial_guess = [0.001, 0.1, y[0]]  # Small positive values for [a, b, c]

					# Perform constrained optimization
					result = minimize(objective, initial_guess, constraints=constraints)
					a, b, c = result.x

					# Store the coefficients
					coefficients = [a, b, c]
					tyre_sector_coefficients.append(coefficients)

				# Average coefficients for this tyre type and sector (if there are any)
				if tyre_sector_coefficients:
					tyre_sector_coeffs[sector] = np.mean(tyre_sector_coefficients, axis=0).tolist()

			# Store the coefficients for this tyre type
			if tyre_sector_coeffs:
				driver_coeffs[tyre] = tyre_sector_coeffs

		# Store the driver's coefficients
		driver_tyre_coefficients[driver] = driver_coeffs

	# Step 2: Fallback to global averages if a driver is missing tyre types or sectors
	# Calculate global averages for each tyre type and sector
	global_tyre_coefficients = {}
	for tyre in min_laps_by_tyre.keys():
		tyre_global_coeffs = {}
		for sector in df["sector"].unique():
			all_coeffs = [
				driver_coeffs[tyre][sector]
				for driver_coeffs in driver_tyre_coefficients.values()
				if tyre in driver_coeffs and sector in driver_coeffs[tyre]
			]
			if all_coeffs:
				tyre_global_coeffs[sector] = np.mean(all_coeffs, axis=0).tolist()
		if tyre_global_coeffs:
			global_tyre_coefficients[tyre] = tyre_global_coeffs

	# Fill in missing tyre types and sectors for each driver
	for driver, driver_coeffs in driver_tyre_coefficients.items():
		for tyre in min_laps_by_tyre.keys():
			if tyre not in driver_coeffs:
				driver_coeffs[tyre] = {}
			for sector in df["sector"].unique():
				if sector not in driver_coeffs[tyre] and tyre in global_tyre_coefficients and sector in global_tyre_coefficients[tyre]:
					driver_coeffs[tyre][sector] = global_tyre_coefficients[tyre][sector]

	# {driverNum: {tyreType: {sector1Deg: [], sector2Deg: [], sector3Deg: []}}}
	return driver_tyre_coefficients


def get_base_sector_times(year, circuit, db_session):
	# Query to find the qualifying session
	quali_session = (
		db_session.query(Session)
		.join(RacingWeekend, Session.weekend_id == RacingWeekend.racing_weekend_id)
		.join(Circuit, RacingWeekend.circuit_id == Circuit.circuit_id)
		.filter(
			RacingWeekend.year == year,
			Circuit.circuit_name == circuit,
			Session.session_type == "Qualifying"
		)
		.first()
	)

	if not quali_session:
		raise ValueError("No qualifying session found for the given year and circuit.")

	# Query to find the minimum sector times for each driver and sector
	min_sector_times = (
		db_session.query(
			Lap.driver_id,
			Driver.driver_num,
			func.min(Lap.s1_time).label("min_s1"),
			func.min(Lap.s2_time).label("min_s2"),
			func.min(Lap.s3_time).label("min_s3")
		)
		.join(Driver, Lap.driver_id == Driver.driver_id)
		.filter(
			Lap.session_id == quali_session.session_id,
			Lap.s1_time.isnot(None),  # Ensure sector times are not null
			Lap.s2_time.isnot(None),
			Lap.s3_time.isnot(None)
		)
		.group_by(Lap.driver_id, Driver.driver_num)
		.all()
	)

	# Convert results into a dictionary for easier access
	base_sector_times = {
		row.driver_num: {
			1: row.min_s1,
			2: row.min_s2,
			3: row.min_s3,
		}
		for row in min_sector_times
	}

	return base_sector_times




def extract_driver_strategies(df):
	# Initialize the dictionary to store strategies
	driver_strategies = {}
	
	# Group the data by driver
	for driver in df["driver_number"].unique():
		# Filter data for the current driver
		driver_data = df[df["driver_number"] == driver]
		
		# Get the starting tyre (tyre used on lap 1)
		starting_tyre = driver_data[driver_data["lap_num"] == 1]["tyre_type"].values[0]
				
		# Store pits as a dictionary: {lap_number: tyre}
		pits_dict = (
			driver_data[driver_data["pit"]]
			.set_index("lap_num")["tyre_type"]
			.astype(int)
			.to_dict()
		)
		
		pits_dict[1] = int(starting_tyre)
		# Store the strategy in the dictionary
		driver_strategies[driver] = pits_dict
	
	return driver_strategies


def get_race_session(year, circuit, db_session):
	race_session = (db_session.query(Session)
				.join(RacingWeekend, Session.weekend_id == RacingWeekend.racing_weekend_id)
				.join(Circuit, RacingWeekend.circuit_id == Circuit.circuit_id)
				.filter(
					RacingWeekend.year == year,
					Circuit.circuit_name == circuit,
					Session.session_type == "Race"
				)
				.first())
	
	return race_session


# calcualtes on average how much slower safety car laps are
def get_safety_car_penalty(year, circuit_name, db_session):
	# Step 1: Get the circuit ID for the given circuit name
	circuit = db_session.query(Circuit).filter_by(circuit_name=circuit_name).first()
	if not circuit:
		raise ValueError(f"Circuit '{circuit_name}' not found in the database.")
	circuit_id = circuit.circuit_id

	# Step 2: Query all sessions for the given circuit and years
	sessions = (
		db_session.query(Session)
		.join(RacingWeekend)
		.filter(
			RacingWeekend.circuit_id == circuit_id,
			RacingWeekend.year < year
		)
		.all()
	)

	if not sessions:
		raise ValueError(f"No sessions found for circuit '{circuit_name}' before year {year}.")

	# Step 3: Collect lap times for safety car and normal laps
	safety_car_lap_times = []
	normal_lap_times = []

	for session in sessions:
		laps = db_session.query(Lap).filter_by(session_id=session.session_id).all()
		for lap in laps:
			if lap.lap_time is None:
				continue  # Skip laps with missing lap times
			if lap.track_status == 4:  # Safety car lap
				safety_car_lap_times.append(lap.lap_time)
			else:  # Normal lap
				normal_lap_times.append(lap.lap_time)

	# Step 4: Calculate average lap times
	avg_safety_car_lap_time = sum(safety_car_lap_times) / len(safety_car_lap_times) if safety_car_lap_times else None
	avg_normal_lap_time = sum(normal_lap_times) / len(normal_lap_times) if normal_lap_times else None

	if avg_safety_car_lap_time is None or avg_normal_lap_time is None:
		raise ValueError("Insufficient data to calculate safety car penalty.")

	# Step 5: Calculate the penalty
	penalty_percentage = ((avg_safety_car_lap_time - avg_normal_lap_time) / avg_normal_lap_time) + 1

	return penalty_percentage


def get_driver_pit_time(year, circuit_name, db_session):
	# Step 1: Get the circuit ID for the given circuit name
	circuit = db_session.query(Circuit).filter_by(circuit_name=circuit_name).first()
	if not circuit:
		raise ValueError(f"Circuit '{circuit_name}' not found in the database.")
	circuit_id = circuit.circuit_id

	# Step 2: Query all sessions for the given circuit and years before the specified year
	past_sessions = (
		db_session.query(Session)
		.join(RacingWeekend)
		.filter(
			RacingWeekend.circuit_id == circuit_id,
			RacingWeekend.year < year
		)
		.all()
	)

	if not past_sessions:
		raise ValueError(f"No sessions found for circuit '{circuit_name}' before year {year}.")

	# Step 3: Collect pit stop times for each team
	team_pit_times = {}  # Dictionary to store lists of pit stop times for each team

	for session in past_sessions:
		# Get all DriverTeamSession entries for this session
		driver_team_sessions = db_session.query(DriverTeamSession).filter_by(session_id=session.session_id).all()

		for dts in driver_team_sessions:
			team_id = dts.team_id
			laps = db_session.query(Lap).filter_by(session_id=session.session_id, driver_id=dts.driver_id).all()

			for lap in laps:

				for pit_stop in lap.pit_stop:
					pit_time = pit_stop.pit_time

					# Skip laps with missing pit times
					if pit_time is None:
						continue

					# Add pit time to the team's list
					if team_id not in team_pit_times:
						team_pit_times[team_id] = []
					team_pit_times[team_id].append(pit_time)

	# Step 4: Calculate average pit stop times for each team
	team_avg_pit_times = {}
	
	for team_id, pit_times in team_pit_times.items():
		avg_pit_time = round(sum(pit_times) / len(pit_times), 3)
		team_avg_pit_times[team_id] = avg_pit_time

	# Step 5: Get the current year's session for the circuit
	current_session = (
		db_session.query(Session)
		.join(RacingWeekend)
		.filter(
			RacingWeekend.circuit_id == circuit_id,
			RacingWeekend.year == year
		)
		.first()
	)

	if not current_session:
		raise ValueError(f"No session found for circuit '{circuit_name}' in year {year}.")

	# Step 6: Map teams to drivers for the current session
	driver_pit_times = {}
	current_driver_team_sessions = db_session.query(DriverTeamSession).filter_by(session_id=current_session.session_id).all()

	for dts in current_driver_team_sessions:
		team_id = dts.team_id
		driver = db_session.query(Driver).filter_by(driver_id=dts.driver_id).first()

		if driver and team_id in team_avg_pit_times:
			driver_pit_times[driver.driver_num] = team_avg_pit_times[team_id]

	return driver_pit_times
