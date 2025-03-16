import numpy as np
from scipy.optimize import minimize
import pandas as pd

class RaceDataSetup:
	def __init__(self, db_operations_obj, race_df_obj):
		self.__db_operations_obj = db_operations_obj      # this is an object of the DatabaseOperations class
		self.__race_dataframe_obj = race_df_obj      # this is an object of the RaceDataframe class

		self.race_df = self.__race_dataframe_obj.race_df
		self.driver_tyre_coefficients = self.get_driver_tyre_coefficients()
		self.driver_strategies = self.extract_driver_strategies()
		self.max_laps = self.race_df["lap_num"].max()
		self.drivers = self.race_df["driver_number"].unique()
		self.driver_names = {
			driver: self.race_df[self.race_df["driver_number"] == driver]["driver_name"].iloc[0]
			for driver in self.drivers
			}
		self.starting_positions = self.get_starting_positions()
		self.base_sector_times = self.__race_dataframe_obj.base_sector_times
		self.fuel_corrections = self.calc_fuel_corrections()
		self.safety_car_laps = self.get_safety_car_laps()
		self.retirements_by_lap = self.get_retirements_per_lap()
		self.safety_car_penalty_percentage = self.get_safety_car_penaltly()
		self.driver_pit_times = self.get_pit_stop_time()

	# ----------------------- Mainly for calculating tyre deg -----------------------
	#									   START
	@staticmethod
	def correct_fuel_effect(race_df, max_lap=None, max_fuel_kg=110, fuel_effect_per_kg=0.03):
		""" Assigns new columns for fuel effects at the sector level.

		Args:
			race_df (pd.DataFrame): dataframe of the race
			max_lap (int, optional): the max number of laps in the race. Defaults to None.
			max_fuel_kg (int, optional): how much fuel the cars are starting with in KG. Defaults to 110.
			fuel_effect_per_kg (float, optional): the time lost per lap, for each kg of fuel. Defaults to 0.03.

		Returns:
			pd.DataFrame: df with new fuel corrected columns
		"""

		if max_lap is None:
			max_lap = race_df["lap_num"].max()

		fuel_reduction_per_lap = max_fuel_kg / max_lap
		fuel_reduction_per_sector = fuel_reduction_per_lap / 3
		fuel_time_per_sector = fuel_effect_per_kg / 3

		def _correct_fuel_for_driver(driver_df):
			# Calculate fuel weight for each sector
			driver_df.loc[:, "fuel_weight_sector"] = max_fuel_kg - (
				(driver_df["lap_num"] - 1) * fuel_reduction_per_lap +
				(driver_df["sector"] - 1) * fuel_reduction_per_sector
			)

			# Calculate fuel correction for each sector
			driver_df.loc[:, "fuel_correction_sector"] = (
				driver_df["fuel_weight_sector"] * fuel_time_per_sector
			)

			# Apply fuel correction to sector times
			driver_df.loc[:, "fuel_corrected_sector_time"] = (
				driver_df["sector_time"] - driver_df["fuel_correction_sector"]
			)

			return driver_df

		race_df = race_df.groupby("driver_number", group_keys=False).apply(_correct_fuel_for_driver).reset_index(drop=True)

		return race_df

	@staticmethod
	def assign_stint_numbers(race_df):
		"""Assigns stint numbers to laps based on pit stops for each driver

		Args:
			race_df (pd.DataFrame): dataframe of the race

		Returns:
			pd.DataFrame: df with new stint column
		"""

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
	def remove_laps_outside_percent(race_df, percentage=5):
		""" Removes laps where fuel corrected sector times is above the given percentage

		Args:
			race_df (pd.DataFrame): dataframe of the race
			percentage (float, optional): the threshold percentage above the fastest laptime

		Returns:
			pd.DataFrame: df with outliers removed
		"""
			
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
	def normalise_lap_times_by_sector(race_df):
		""" Normalises by taking the drivers fastest time in that sector in quali from the sector time
		This means we can get deg curves that arent dependant on the laptime, they are how much time deg adds on

		Args:
			race_df (pd.DataFrame): race df

		Returns:
			pd.DataFrame: df with the normalised time
		"""

		# Normalise lap times by subtracting the fastest sector time
		race_df['normalised_sector_time'] = (
			race_df['fuel_corrected_sector_time'] - race_df['base_sector_time']
		)

		return race_df
	#									   END
	# ----------------------- Mainly for calculating tyre deg -----------------------

	def get_driver_tyre_coefficients(self):
		""" Calculates tyre degradation coefficients for each driver, tyre type, and sector.

		The race df is taken and pre processed, then tyre deg is done on the normalised sector times
		If a driver hasnt used a tyre type, an average of every other driver is added

		Returns:
			dict: {driverNum: {tyreType: {sector1Deg: [], sector2Deg: [], sector3Deg: []}}}
		"""
		# Pre process
		race_df = self.race_df
		race_df = self.assign_stint_numbers(race_df)
		race_df = self.correct_fuel_effect(race_df)
		race_df = self.remove_laps_outside_percent(race_df)
		race_df = self.normalise_lap_times_by_sector(race_df)

		# Define minimum laps required for each tyre type
		min_laps_by_tyre = {
			1: 2,  # Soft
			2: 4,  # Med
			3: 6   # Hard
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
				if tyre in global_tyre_coefficients:
					if tyre not in driver_coeffs:
						driver_coeffs[tyre] = {}
					for sector in race_df["sector"].unique():
						if sector not in driver_coeffs[tyre] and tyre in global_tyre_coefficients and sector in global_tyre_coefficients[tyre]:
							driver_coeffs[tyre][sector] = global_tyre_coefficients[tyre][sector]

		# {driverNum: {tyreType: {sector1Deg: [], sector2Deg: [], sector3Deg: []}}}
		return driver_tyre_coefficients
	
	def get_unique_tyre_types(self):
		""" Extract all unique tyre types for which we have degradation data across all drivers
		
		Parameters:
			driver_tyre_coefficients (dict): output of get_driver_tyre_coefficients.
		
		Returns:
			set: A set of unique tyre types (e.g., {1, 2, 3} for Hard, Medium, Soft).
		"""
		# Initialize a set to store unique tyre types
		unique_tyre_types = set()
		# Iterate over each driver's coefficients
		for driver_coeffs in self.driver_tyre_coefficients.values():
			# Add all tyre types for this driver to the set
			unique_tyre_types.update(driver_coeffs.keys())
		
		return unique_tyre_types

	def extract_driver_strategies(self):
		""" Gets strategies of all drivers in the race

		Returns:
			dict: {driver_number: {lap_number: tyre_type}}
		"""
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

	def extract_driver_strategy(self, given_driver):
		""" Gets the strategy for a single given driver

		Args:
			given_driver (int): the number of the given driver

		Raises:
			ValueError: if we cant find the driver or if the user didnt give a valid number

		Returns:
			dict: {lap_number: tyre_type}
		"""
		if given_driver is None:
			raise ValueError("No driver specified. Please provide a valid driver number.")
	
		# Filter data for the given driver
		driver_data = self.race_df[self.race_df["driver_number"] == given_driver]
		
		# Ensure the driver exists in the dataset
		if driver_data.empty:
			raise ValueError(f"Driver {given_driver} not found in the dataset.")

		driver_data = self.race_df[self.race_df["driver_number"] == given_driver]
		starting_tyre = driver_data[driver_data["lap_num"] == 1]["tyre_type"].values[0]
		pits_dict = (
			driver_data[driver_data["pit"]]
			.set_index("lap_num")["tyre_type"]
			.astype(int)
			.to_dict()
		)
		pits_dict[1] = int(starting_tyre)
		sorted_strategy = dict(sorted(pits_dict.items()))
		return sorted_strategy
	
	def get_starting_positions(self):
		""" Gets the positions of each drivers starting pos

		Returns:
			dict: {driver_num: grid_pos}
		"""
		return {driver_num: grid_pos for grid_pos, driver_num, _ in self.__db_operations_obj.race_session_results_db}

	def calc_fuel_corrections(self, max_fuel_kg=110, fuel_effect_per_kg=0.03):
		""" Pre calcs the fuel correction values for each lap in race

		Args:
			max_fuel_kg (int, optional): Fuel level in kg. Defaults to 110.
			fuel_effect_per_kg (float, optional): time lost per lap per each kg of fuel. Defaults to 0.03.

		Returns:
			dict: fuel correction values for each lap
		"""
		fuel_reduction_per_lap = max_fuel_kg / self.max_laps
		fuel_reduction_per_sector = fuel_reduction_per_lap / 3

		# Initialize the dictionary to store fuel corrections for each sector
		fuel_corrections = {}

		# Loop through each lap and sector to calculate fuel corrections
		for lap in range(1, self.max_laps + 1):
			for sector in range(1, 4):  # Sectors are numbered 1, 2, 3
				# Calculate the fuel weight at the start of this sector
				fuel_weight_at_sector_start = (
					max_fuel_kg - 
					((lap - 1) * fuel_reduction_per_lap) - 
					((sector - 1) * fuel_reduction_per_sector)
				)

				# Calculate the fuel correction for this sector
				fuel_correction = fuel_weight_at_sector_start * fuel_effect_per_kg

				# Store the fuel correction in the dictionary
				fuel_corrections[(lap, sector)] = fuel_correction

		return fuel_corrections
	
	def get_safety_car_laps(self):
		""" Gets all the laps where the safety car is deployed

		Based on whos in position 1 only

		Returns:
			array: list of lap numbers where its deployed
		"""
		return self.race_df[(self.race_df["track_status"] != 1) & (self.race_df["position"]==1)]["lap_num"].unique().tolist()
	
	def get_safety_car_penaltly(self):
		""" Calcs the average penalty of how much slower laps are under the safety car

		Returns:
			float: the penalty (decimal) due to the safety car laps - like 1.__ so can multiply straight away
		"""
		safety_car_lap_time_mean = self.race_df[self.race_df["lap_num"].isin(self.safety_car_laps)]["lap_time"].mean()
		normal_lap_time_mean = self.race_df[~self.race_df["lap_num"].isin(self.safety_car_laps)]["lap_time"].mean()

		penalty = ((safety_car_lap_time_mean - normal_lap_time_mean) / normal_lap_time_mean) + 1

		return penalty

	def get_retirements_per_lap(self):
		""" Gets the drivers who retired and the laps they retired on

		Returns:
			dict: a dict with lap number being the key and then the value being 
				an array of the drivers who retired {lap_num: [drivers]}
		"""
		retirements_per_lap = {}
	
		# Iterate through session results to determine retirements
		for driver_id, driver_num, end_status in self.__db_operations_obj.race_session_results_db:
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
		""" Calcualtes the average pit stops time for each driver
		And the average as well, for a fallback

		Returns:
			dict: a dict with the average pit stop times for each driver - {driver_num: pit_time}
		"""

		# get rows where pit stops are
		pit_stops = self.race_df[self.race_df["pit_time"].notna()]

		average_pit_stop_times = {}

		# group by driver_number and calculate the mean pit_time for each group
		for driver_number, group in pit_stops.groupby("driver_number"):
			# calc the average pit stop time for the driver
			average_pit_time = group["pit_time"].mean()
			
			# add average pit stop time to the dictionary
			average_pit_stop_times[driver_number] = average_pit_time
		
		# calc average
		overall_average_pit_time = pit_stops["pit_time"].mean()
		average_pit_stop_times[0] = overall_average_pit_time


		return average_pit_stop_times
	