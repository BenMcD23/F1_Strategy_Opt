from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import sys
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
from DB.models import init_db, Circuit, Season, RacingWeekend, Driver, Session, SessionResult, Lap, Team, DriverTeamSession, PitStop
# from utils import correct_fuel_effect, extract_driver_strategies, get_base_sector_times, get_tyre_deg_per_driver, get_race_session, get_safety_car_penalty, get_driver_pit_time

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from itertools import product

from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey, Boolean, Text
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

pd.set_option('future.no_silent_downcasting', True)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

import numpy as np


class SessionNotFoundError(Exception):
	"""Custom exception for when a session is not found"""
	pass



class DatabaseOperations:
	_Session = None

	# classmethod so the session isnt recreated every time we create a new instance of the class
	@classmethod
	def init_db(cls):
		if cls._Session is None:
			Base = declarative_base()
			cls._engine = create_engine('sqlite:////home/ben/Individual_Project/DB/f1_data_V4.db')
			cls._Session = sessionmaker(bind=cls._engine)
			Base.metadata.create_all(cls._engine)


	def __init__(self, year, circuit):
		# Init db if it isnt already
		if DatabaseOperations._Session is None:
			DatabaseOperations.init_db()
		self.year = year
		self.circuit = circuit
		self.db_session = DatabaseOperations._Session()
		self.race_session_db = self._get_race_session_db()
		self.race_session_results_db = self._get_session_results_db()
		self.quali_session_db = self._get_quali_session_db()
	
	def _get_race_session_db(self):
		race_session = (self.db_session.query(Session)
			.join(RacingWeekend, Session.weekend_id == RacingWeekend.racing_weekend_id)
			.join(Circuit, RacingWeekend.circuit_id == Circuit.circuit_id)
			.filter(
				RacingWeekend.year == self.year,
				Circuit.circuit_name == self.circuit,
				Session.session_type == "Race"
			)
			.first())
		
		if race_session is None:
			raise SessionNotFoundError(f"No race session found for year {self.year} at circuit {self.circuit}")
		
		self.race_session_db = race_session
		return race_session
	
	def _get_quali_session_db(self):
		quali_session = (
			self.db_session.query(Session)
			.join(RacingWeekend, Session.weekend_id == RacingWeekend.racing_weekend_id)
			.join(Circuit, RacingWeekend.circuit_id == Circuit.circuit_id)
			.filter(
				RacingWeekend.year == self.year,
				Circuit.circuit_name == self.circuit,
				Session.session_type == "Qualifying"
			)
			.first()
		)

		if not quali_session:
			raise SessionNotFoundError(f"No qualifying session found for year {self.year} at circuit {self.circuit}")
		
		return quali_session


	def _get_session_results_db(self):
		
		session_results = (
			self.db_session.query(SessionResult.grid_pos, Driver.driver_num, SessionResult.end_status)
			.join(Session, Session.session_id == SessionResult.session_id)
			.join(Driver, Driver.driver_id == SessionResult.driver_id)
			.filter(SessionResult.session_id == self.race_session_db.session_id)
			.all()
		)

		return session_results




class RaceDataframe:
	def __init__(self, db_operations):
		self.db_operations = db_operations    # this is an object of the DatabaseOperations class
		self.db_session = db_operations.db_session

		self.race_session_db = self.db_operations.race_session_db
		self.quali_session_db = self.db_operations.quali_session_db
		
		self.base_sector_times = self._get_base_sector_times()
		self.race_df = self._get_race_df()

	# not 100% happy about where this is
	def _get_base_sector_times(self):
		# Query to find the minimum sector times for each driver and sector
		min_sector_times = (
			self.db_session.query(
				Lap.driver_id,
				Driver.driver_num,
				func.min(Lap.s1_time).label("min_s1"),
				func.min(Lap.s2_time).label("min_s2"),
				func.min(Lap.s3_time).label("min_s3")
			)
			.join(Driver, Lap.driver_id == Driver.driver_id)
			.filter(
				Lap.session_id == self.quali_session_db.session_id,
				Lap.s1_time.isnot(None),  # Ensure sector times are not null
				Lap.s2_time.isnot(None),
				Lap.s3_time.isnot(None)
			)
			.group_by(Lap.driver_id, Driver.driver_num)
			.all()
		)

		# Convert results into a dict
		base_sector_times = {
			row.driver_num: {
				1: row.min_s1,
				2: row.min_s2,
				3: row.min_s3,
			}
			for row in min_sector_times
		}

		return base_sector_times

	def _get_raw_race_df(self):
		"""Reutrns the raw race race_df with no added info from the database

		Returns:
			Pandas DF: 
		"""
		laps = self.race_session_db.laps
		# Convert to DataFrame
		laps_data = []
		for lap in laps:
			for sector in range(1, 4):
				# get the time for this sector
				sector_time = getattr(lap, f"s{sector}_time")

				# add a row for each sector
				laps_data.append({
					"lap_num": lap.lap_num,
					"lap_time": lap.lap_time,
					"sector": sector,
					"stint_num": lap.stint_num,
					"stint_lap": lap.stint_lap,
					"position": lap.position,
					"driver_name": lap.driver.driver_name,
					"driver_number": lap.driver.driver_num,
					"sector_time": sector_time,
					"tyre_type": lap.tyre_type,
					"tyre_laps": lap.tyre_laps,
					"pit": lap.pit,
					"pit_time": lap.pit_stop[0].pit_time if lap.pit_stop else None,
					"track_status": lap.track_status,
					"base_sector_time": self.base_sector_times[lap.driver.driver_num][sector]
				})

		
		# Create a DataFrame from the list of dicts
		race_df = pd.DataFrame(laps_data)
		

		race_df = race_df.sort_values(["lap_num", "sector", "position"]).reset_index(drop=True)

		return race_df
	
	def _add_race_data(self, race_df):
		"""Adds extra info the race dataframe such as gaps and info for overtakes. Added to train the overtake model

		Returns:
			Pandas DF: _description_
		"""
		# Calculate cumulative race time for each driver
		race_df["cumulative_time"] = race_df.groupby("driver_name")["sector_time"].cumsum()
	
		# Calculate rolling pace (average lap time over the last 5 laps)
		race_df["pace"] = (
			race_df.groupby(["driver_name", "sector"])["sector_time"]
			.rolling(window=5, min_periods=1)
			.mean()
			.reset_index(level=[0, 1], drop=True)
		)

		# Get car ahead"s cumulative time (car immediately ahead in position for each lap)
		race_df["front_cumulative_time"] = race_df.groupby(["lap_num", "sector"])["cumulative_time"].shift(1)
		# This gap is calculated only for drivers who are not in the lead position (position > 1)
		race_df["gap"] = race_df["cumulative_time"] - race_df["front_cumulative_time"]
		race_df["gap"] = race_df["gap"].fillna(0)  # Leader has no car ahead, so gap is 0

		# Calculate tyre difference (compared to car immediately ahead in THIS Sector)
		race_df["front_tyre"] = race_df.groupby(["lap_num", "sector"])["tyre_type"].shift(1)
		race_df["tyre_diff"] = race_df["front_tyre"] - race_df["tyre_type"]
		race_df["tyre_diff"] = race_df["tyre_diff"].fillna(0)  # Leader has no car ahead

		# Calculate tyre age difference (compared to car immediately ahead in THIS Sector)
		race_df["front_laps"] = race_df.groupby(["lap_num", "sector"])["stint_lap"].shift(1)
		race_df["stint_laps_diff"] = race_df["front_laps"] - race_df["stint_lap"]
		race_df["stint_laps_diff"] = race_df["stint_laps_diff"].fillna(0)  # Leader has no car ahead

		# Calculate DRS availability (within 1s of car ahead IN THIS Sector)
		race_df["drs_available"] = (
			(race_df["gap"] <= 1) &
			(race_df["position"] > 1) &
			(race_df["lap_num"] > 1)
		)


		# race_df["next_position"] = race_df.groupby("driver_name")["position"].shift(1) 
		# race_df["overtaken"] = ((race_df["next_position"] < race_df["position"]) | 
		# 				  (race_df["next_position"].isna()))


		# Shift the "position" and "pit" columns
		race_df["next_position"] = race_df.groupby("driver_name")["position"].shift(1)
		race_df["next_pit"] = race_df.groupby("driver_name")["pit"].shift(-1)

		# Handle NaN values in 'next_pit' by filling them with False
		race_df["next_pit"] = race_df["next_pit"].fillna(False)

		# Define the "overtaken" column
		race_df["overtaken"] = (
			((race_df["next_position"] < race_df["position"]) | (race_df["next_position"].isna()))  # Original condition
			& (~race_df["next_pit"])  # Ensure the driver in the next position is not pitting
		)

		# Cleanup and final sorting
		race_df = race_df.drop(columns=["front_cumulative_time", "front_tyre", "next_position"])
		# race_df = race_df.sort_values(["lap_num", "sector", "position"]).reset_index(drop=True)

		# Set as a new order so its a bit easier to read
		# new_order = [
		# 	"lap_num", "sector", "base_sector_time", "stint_num", "stint_lap", "position", "driver_name",
		# 	"driver_number", "sector_time", "gap", "cumulative_time", "tyre_type", "tyre_laps", 
		# 	"pit", "pit_time", "drs_available", "overtaken", "tyre_diff", "front_laps", "stint_laps_diff", "track_status", "pace"
		# ]

		# race_df = race_df[new_order]

		return race_df
	
	def _get_race_df(self):
		raw_df = self._get_raw_race_df()
		
		race_df = self._add_race_data(raw_df)


		return race_df
	

class RaceDataSetup:
	def __init__(self, db_operations_obj, race_df_obj):
		self.db_operations_obj = db_operations_obj      # this is an object of the DatabaseOperations class
		self.race_dataframe_obj = race_df_obj      # this is an object of the RaceDataframe class

		self.race_df = self.race_dataframe_obj.race_df
		self.driver_tyre_coefficients = self._get_driver_tyre_coefficients()
		self.driver_strategies = self.extract_driver_strategies()
		self.max_laps = self.race_df["lap_num"].max()
		self.drivers = self.race_df["driver_number"].unique()
		self.driver_names = {
			driver: self.race_df[self.race_df["driver_number"] == driver]["driver_name"].iloc[0]
			for driver in self.drivers
			}
		self.starting_positions = self.get_starting_positions()
		self.base_sector_times = self.race_dataframe_obj.base_sector_times
		self.fuel_corrections = self.calc_fuel_corrections()
		self.safety_car_laps = self.get_safety_car_laps()
		self.retirements_by_lap = self.get_retirements_per_lap()
		self.safety_car_penalty_percentage = self.get_safety_car_penaltly()
		self.driver_pit_times = self.get_pit_stop_time()

	# ----------------------- Mainly for calculating tyre deg -----------------------
	#									   START
	@staticmethod
	def _correct_fuel_effect(race_df, max_lap=None, max_fuel_kg=110, fuel_effect_per_kg=0.03):
		# Find the maximum number of laps completed by any driver in the race
		if max_lap is None:
			max_lap = race_df["lap_num"].max()

		# Group by driver to process each driver"s laps individually
		def _correct_fuel_for_driver(driver_df):
			# Make sure we modify the original DataFrame using .loc to avoid SettingWithCopyWarning
			driver_df.loc[:, "fuel_weight"] = max_fuel_kg - (driver_df["lap_num"] - 1) * (max_fuel_kg / max_lap)
			driver_df.loc[:, "fuel_correction"] = (driver_df["fuel_weight"] * fuel_effect_per_kg) / 3
			driver_df.loc[:, "fuel_corrected_sector_time"] = driver_df["sector_time"] - driver_df["fuel_correction"]
			return driver_df

		# Apply the correction to each driver"s laps using groupby and avoid deprecated behavior with group_keys=False
		race_df = race_df.groupby("driver_number", group_keys=False)[race_df.columns].apply(_correct_fuel_for_driver).reset_index(drop=True)
		return race_df

	@staticmethod
	def _assign_stint_numbers(race_df):
		# Assign stint numbers to laps based on pit stops for each driver
		race_df["stint"] = np.nan
		for driver in race_df["driver_number"].unique():
			driver_data = race_df[race_df["driver_number"] == driver]
			stint_number = 1
			for i in driver_data.index:
				if driver_data.loc[i, "pit"] and i != driver_data.index[0]:
					stint_number += 1
				race_df.loc[i, "stint"] = stint_number
		race_df["stint"] = race_df["stint"].astype(int)
		return race_df

	@staticmethod
	def _remove_laps_outside_percent(race_df, percentage=5):
		def _filter_driver_sector_laps(driver_sector_df):
			# Calculate the threshold based on the fastest lap time for the driver and sector
			fastest_lap_time = driver_sector_df["fuel_corrected_sector_time"].min()
			threshold = fastest_lap_time * (1 + percentage / 100)
			
			# Remove laps not within the specified percentage of the fastest lap time
			filtered_df = driver_sector_df[driver_sector_df["fuel_corrected_sector_time"] <= threshold]
			return filtered_df

		# Group by driver and sector, then apply the filtering logic
		race_df = (
			race_df.groupby(["driver_number", "sector"], group_keys=False)
			.apply(_filter_driver_sector_laps)
			.reset_index(drop=True)
		)


		return race_df

	@staticmethod
	def _normalise_lap_times_by_sector(race_df):
		# Normalise lap times by subtracting the fastest sector time
		race_df['normalised_sector_time'] = (
			race_df['fuel_corrected_sector_time'] - race_df['base_sector_time']
		)

		return race_df
	#									   END
	# ----------------------- Mainly for calculating tyre deg -----------------------

	def _get_driver_tyre_coefficients(self):
		# Pre process
		race_df = self.race_df
		race_df = self._assign_stint_numbers(race_df)
		race_df = self._correct_fuel_effect(race_df)
		race_df = self._remove_laps_outside_percent(race_df)
		race_df = self._normalise_lap_times_by_sector(race_df)
		
		# Define minimum laps required for each tyre type
		min_laps_by_tyre = {
			1: 0,  # Soft
			2: 0,  # Med
			3: 0   # Hard
		}

		# Dictionary to store tyre coefficients for each driver
		driver_tyre_coefficients = {}

		# Process each driver
		for driver in race_df["driver_number"].unique():
			# Filter data for the current driver
			df_driver = race_df[race_df["driver_number"] == driver]

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
			for sector in race_df["sector"].unique():
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
				for sector in race_df["sector"].unique():
					if sector not in driver_coeffs[tyre] and tyre in global_tyre_coefficients and sector in global_tyre_coefficients[tyre]:
						driver_coeffs[tyre][sector] = global_tyre_coefficients[tyre][sector]

		# {driverNum: {tyreType: {sector1Deg: [], sector2Deg: [], sector3Deg: []}}}
		return driver_tyre_coefficients


	def extract_driver_strategies(self):
		# Initialize the dictionary to store strategies
		driver_strategies = {}
		
		# Group the data by driver
		for driver in self.race_df["driver_number"].unique():
			# Filter data for the current driver
			driver_data = self.race_df[self.race_df["driver_number"] == driver]
			
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

	def get_starting_positions(self):
		return {driver_num: grid_pos for grid_pos, driver_num, _ in self.db_operations_obj.race_session_results_db}

	def calc_fuel_corrections(self, max_fuel_kg=110, fuel_effect_per_kg=0.03):

		fuel_corrections = {
			lap: (max_fuel_kg - (lap - 1) * (max_fuel_kg / self.max_laps)) * fuel_effect_per_kg
			for lap in range(1, self.max_laps + 1)
		}

		return fuel_corrections
	
	def get_safety_car_laps(self):
		return self.race_df[(self.race_df["track_status"] != 1) & (self.race_df["position"]==1)]["lap_num"].unique().tolist()
	
	def get_safety_car_penaltly(self):
		safety_car_lap_time_mean = self.race_df[self.race_df["lap_num"].isin(self.safety_car_laps)]["lap_time"].mean()
		normal_lap_time_mean = self.race_df[~self.race_df["lap_num"].isin(self.safety_car_laps)]["lap_time"].mean()

		penalty_percentage = ((safety_car_lap_time_mean - normal_lap_time_mean) / normal_lap_time_mean) + 1

		return penalty_percentage

	def get_retirements_per_lap(self):
			# Initialize an empty dictionary to store retirements by lap
		retirements_per_lap = {}
	
		# Iterate through session results to determine retirements
		for driver_id, driver_num, end_status in self.db_operations_obj.race_session_results_db:
			# Check if the driver retired (end_status is not "Finished" or "+1 Lap")
			if end_status and not (end_status.startswith("Finished") or end_status.startswith("+")):
				# Find the maximum lap number for the driver (last recorded lap)
				lap_retired = self.race_df[self.race_df["driver_number"] == driver_num]["lap_num"].max()
				
				# Add the driver to the list of retirees for the corresponding lap
				if lap_retired not in retirements_per_lap:
					retirements_per_lap[lap_retired] = []
				retirements_per_lap[lap_retired].append(driver_num)

		return retirements_per_lap
	
	def get_pit_stop_time(self):
		# Filter rows where pit_time is not None (i.e., rows with valid pit stops)
		pit_stops = self.race_df[self.race_df["pit_time"].notna()]

		# Initialize an empty dictionary to store the results
		average_pit_stop_times = {}

		# Group by driver_number and calculate the mean pit_time for each group
		for driver_number, group in pit_stops.groupby("driver_number"):
			# Calculate the average pit stop time for the driver
			average_pit_time = group["pit_time"].mean()
			
			# Add the driver's average pit stop time to the dictionary
			average_pit_stop_times[driver_number] = average_pit_time

		return average_pit_stop_times
	

class OvertakingModel:
	def __init__(self, race_df):
		self.race_df = race_df
		self.feature_names = [
			"gap",
			"sector",
			"tyre_diff",
			"stint_laps_diff",
			"drs_available",
			"cumulative_time",
			"sector_time",
			"pace",
			"pit"
		]
		self.imputer = SimpleImputer(strategy='mean')  # Imputer for missing values
		self.model = self._train_overtaking_model()

	def _train_overtaking_model(self):
		# Prepare feature matrix (X) and target vector (y)
		X = self.race_df[self.feature_names].values  # Convert to NumPy array immediately
		y = self.race_df["overtaken"].values

		# Handle missing values using the imputer
		X = self.imputer.fit_transform(X)

		# Resample the data using SMOTE
		smote = SMOTE(random_state=42)
		X_resampled, y_resampled = smote.fit_resample(X, y)

		# Train the GradientBoostingClassifier
		gbc = GradientBoostingClassifier(
			n_estimators=200,
			learning_rate=0.05,
			max_depth=3,
			subsample=0.8,
			random_state=42
		)

		# Calibrate for better probabilities
		model = CalibratedClassifierCV(gbc, method="sigmoid", cv=3)
		model.fit(X_resampled, y_resampled)

		return model

	def extract_features(self, driver_data):
		"""
		Extracts features from the active_drivers list of dictionaries and returns a NumPy array.
		
		Args:
			active_drivers (list): List of dictionaries containing driver data.
		
		Returns:
			np.ndarray: A 2D NumPy array where each row corresponds to a driver and columns correspond to features.
		"""
		# Extract features for each driver
		features = []
		for driver in driver_data:
			driver_features = [driver[feature] for feature in self.feature_names]
			features.append(driver_features)
		
		# Convert to a NumPy array
		return np.array(features, dtype=float)

	def predict_overtake(self, data):
		"""
		Predict overtakes for a NumPy array of feature data.
		
		Args:
			data (np.ndarray): A 2D NumPy array where each row represents a sample and columns correspond to features.
		
		Returns:
			np.ndarray: Predictions for each sample.
		"""
		# Ensure all features are numeric
		data = np.array([
			[int(x) if isinstance(x, bool) else x for x in row]  # Convert booleans to integers
			for row in data
		], dtype=float)  # Ensure the array is of type float

		# Handle missing values using the same imputer used during training
		data_filled = self.imputer.transform(data)

		# Make predictions using the trained model
		predictions = self.model.predict(data_filled)
		return predictions

	def handle_overtake_prediction(self, driver_data):
		driver_features = self.extract_features(driver_data)
		predicted_overtakes = self.predict_overtake(driver_features)
	
		return predicted_overtakes
	
	def get_model_accuracy(self):
		# Split the data into features and target
		X_test = self.race_df[self.feature_names].values
		y_test = self.race_df["overtaken"].values

		# Predict overtakes
		predicted_overtakes = self.predict_overtake(X_test)

		# Evaluate performance
		accuracy = accuracy_score(y_test, predicted_overtakes)

		# Generate classification report
		report = classification_report(
			y_test,
			predicted_overtakes,
			target_names=["No Overtake", "Overtaken"]
		)
		return accuracy, report
	


from collections import deque
import pandas as pd


class RaceSimulation:
	def __init__(self, race_data, overtake_model, given_driver=None, simulated_strategy=None):
		self.race_data = race_data      # a RaceDataSetup object
		self.overtake_model = overtake_model    # a OvertakingModel object


		# Update strategies if a specific driver and strategy are provided
		if given_driver and simulated_strategy:
			self.race_data.driver_strategies[given_driver] = simulated_strategy

		# Initialize drivers' data
		self.drivers_data = self._initialize_drivers_data()

		# Rolling pace tracking
		self.driver_pace_per_sec = {
			driver: {sector: deque(maxlen=5) for sector in range(1, 4)}  # Sectors 1, 2, 3
			for driver in self.race_data.drivers
		}

		# Track overtakes
		self.num_overtakes = 0

	def _initialize_drivers_data(self):
		"""
		Initialize the data structure for each driver.
		"""
		drivers_data = []
		for driver in self.race_data.drivers:
			drivers_data.append({
				"driver_number": driver,
				"driver_name": self.race_data.driver_names[driver],
				"pit_schedule": {key: value for key, value in self.race_data.driver_strategies[driver].items() if key != 1},
				"tyre_type": self.race_data.driver_strategies[driver][1],
				"lap_num": 0,
				"sector": 0,
				"sector_time": 0.0,
				"stint_lap": 0,
				"cumulative_time": 0.0,
				"gap": 0.0,
				"pit": False,
				"pace": 0,
				"position": self.race_data.starting_positions[driver],
				"starting_pos": self.race_data.starting_positions[driver],
				"base_sector_times": self.race_data.base_sector_times[driver],
				"tyre_diff": 0,  # Initialize tyre difference as 0
				"stint_laps_diff": 0,  # Initialize stint laps difference as 0
				"drs_available": False,  # Initialize DRS availability as False
				"retired": False,
			})
		return drivers_data

	def simulate(self):
		"""
		Simulate the race and return the final drivers' data.
		"""
		for lap in range(1, self.race_data.max_laps + 1):
			self._process_lap(lap)

		print(f"Number of overtakes: {self.num_overtakes}")
		return self.drivers_data

	def _process_lap(self, lap):
		"""
		Process a lap, including itterating the lapnumber and handling retirements and safety cars
		"""
		# Increment lap and stint lap counters
		for d in self.drivers_data:
			d["lap_num"] += 1
			d["stint_lap"] += 1

		# Check for safety car and retirements
		safety_car = lap in self.race_data.safety_car_laps

		if lap in self.race_data.retirements_by_lap:
			self._handle_retirements(lap)

		# Process each sector
		for sector in range(1, 4):
			self._process_sector(sector, lap, safety_car)

	def _handle_retirements(self, lap):
		"""
		Handle driver retirements at the given lap.
		"""
		retiring_drivers = self.race_data.retirements_by_lap[lap]

		# Move all drivers behind the retiring drivers up by 1 position
		for driver in retiring_drivers:
			retiring_position = next(
				d["position"] for d in self.drivers_data if d["driver_number"] == driver
			)

			for d in self.drivers_data:
				if d["position"] > retiring_position:
					d["position"] -= 1

		# Mark retiring drivers as retired
		for d in self.drivers_data:
			if d["driver_number"] in retiring_drivers:
				d["retired"] = True
				d["position"] = 999

	def _process_sector(self, sector, lap, safety_car):
		"""
		Process a single sector for all drivers.
		"""
		for d in self.drivers_data:
			if d["retired"]:
				continue

			d["sector"] = sector

			# Calculate sector time based on tyre degradation, fuel correction, and safety car penalty
			a, b, c = self.race_data.driver_tyre_coefficients[d["driver_number"]][d["tyre_type"]][sector]
			sector_time = (
				d["base_sector_times"][sector]  # Base sector time for specific driver
				+ (a * d["stint_lap"]**2 + b * d["stint_lap"] + c)  # Tyre degradation
				+ self.race_data.fuel_corrections[lap]  # Fuel effect
			)
			if safety_car:
				sector_time *= self.race_data.safety_car_penalty_percentage

			# Update sector time and cumulative time
			d["sector_time"] = sector_time
			d["cumulative_time"] += sector_time

			# Add to rolling pace tracker
			self.driver_pace_per_sec[d["driver_number"]][sector].append(sector_time)

			# Handle pit stops at the start of a lap (sector 1)
			if sector == 1 and lap in d["pit_schedule"]:
				d["pit"] = True
				d["cumulative_time"] += self.race_data.driver_pit_times[d["driver_number"]]
				d["stint_lap"] = 1
				d["tyre_type"] = d["pit_schedule"][lap]
			else:
				d["pit"] = False

		# Re-sort drivers by cumulative time and update positions
		active_drivers = [d for d in self.drivers_data if not d["retired"]]
		active_drivers.sort(key=lambda x: x["cumulative_time"])
		for i, d in enumerate(active_drivers):
			d["position"] = i + 1

		# Skip overtakes during safety car
		if safety_car:
			return

		# Handle overtakes
		for d in self.drivers_data:
			if d["retired"]:
				continue

			ahead_pos = d["position"] - 1
			if ahead_pos > 0:
				ahead_driver = next(a_d for a_d in active_drivers if a_d["position"] == ahead_pos)
				current_driver_time = d["cumulative_time"]
				ahead_driver_time = ahead_driver["cumulative_time"]

				# Fix cumulative times if out of order
				if ahead_driver_time > current_driver_time:
					new_ahead_time = current_driver_time - 1
					ahead_driver["cumulative_time"] = new_ahead_time
					ahead_driver_time = new_ahead_time

				gap = current_driver_time - ahead_driver_time
				d["gap"] = gap

				# Calculate rolling pace
				sector_times = self.driver_pace_per_sec[d["driver_number"]][sector]
				d["pace"] = sum(sector_times) / len(sector_times) if sector_times else 0.0

				# Update other features
				d["tyre_diff"] = ahead_driver["tyre_type"] - d["tyre_type"]
				d["stint_laps_diff"] = ahead_driver["stint_lap"] - d["stint_lap"]
				d["drs_available"] = True
			else:
				d["gap"] = 0

		# Predict overtakes
		active_drivers = [d for d in self.drivers_data if not d["retired"]]
		
		predicted_overtakes = self.overtake_model.handle_overtake_prediction(active_drivers)

		# can use active_drivers and dicts are mutable
		for i, driver in enumerate(active_drivers):
			driver["predicted_overtake"] = predicted_overtakes[i]

		for driver in active_drivers:
			if driver["retired"]:
				continue

			ahead_pos = driver["position"] - 1
			if driver["gap"] < 1 and ahead_pos > 0 and driver["predicted_overtake"]:
				self.num_overtakes += 1
				ahead_driver = next(d for d in active_drivers if d["position"] == ahead_pos)
				# Swap positions and cumulative times
				driver["position"], ahead_driver["position"] = ahead_driver["position"], driver["position"]
				driver["cumulative_time"], ahead_driver["cumulative_time"] = (
					ahead_driver["cumulative_time"] - 1, driver["cumulative_time"]
				)

	
	def get_results_as_dataframe(self):
		"""
		Return the final results as a Pandas DataFrame.
		"""
		sim_df = pd.DataFrame(self.drivers_data)
		sim_df = sim_df.sort_values(by="position", ascending=True).reset_index(drop=True)
		return sim_df
	
	

# db1 = DatabaseOperations(2024, "Sakhir")
# race1 = RaceDataframe(db1)
# race_data1 = RaceDataSetup(db1, race1)
# overtake1 = OvertakingModel(race1.race_df)

# race_sim = RaceSimulation(race_data1, overtake1)
# race_sim.simulate()

# race_sim.get_results_as_dataframe()