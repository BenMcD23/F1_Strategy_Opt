import numpy as np
from scipy.optimize import minimize
import pandas as pd

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
				if tyre in global_tyre_coefficients:
					if tyre not in driver_coeffs:
						driver_coeffs[tyre] = {}
					for sector in race_df["sector"].unique():
						if sector not in driver_coeffs[tyre] and tyre in global_tyre_coefficients and sector in global_tyre_coefficients[tyre]:
							driver_coeffs[tyre][sector] = global_tyre_coefficients[tyre][sector]

		# {driverNum: {tyreType: {sector1Deg: [], sector2Deg: [], sector3Deg: []}}}
		return driver_tyre_coefficients
	
	def get_unique_tyre_types(self):
		"""
		Extract all unique tyre types for which we have degradation data across all drivers.
		
		Parameters:
			driver_tyre_coefficients (dict): The output of _get_driver_tyre_coefficients.
				Structure: {driver_number: {tyre_type: {sector: [a, b, c]}}}
		
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
		
		overall_average_pit_time = pit_stops["pit_time"].mean()
		average_pit_stop_times[0] = overall_average_pit_time


		return average_pit_stop_times
	

	def get_driver_finishing_position(self, given_driver):
		"""
		Retrieve the finishing position for a given driver from the session results.

		Parameters:
			driver_number (int): The driver number for which to retrieve the finishing position.

		Returns:
			int: The finishing position of the driver.

		Raises:
			ValueError: If the driver is not found in the session results.
			SessionNotFoundError: If no race results are found for the session.
		"""
		# Step 1: Get the session results
		session_results = self.db_operations_obj.race_session_results_db

		# Step 2: Filter the results for the given driver
		for result in session_results:
			grid_pos, driver_num, end_status = result
			if driver_num == given_driver:
				if end_status and (end_status.startswith("Finished") or end_status.startswith("+")):
					return grid_pos  # Return the finishing position
				else:
					raise ValueError(f"Driver {given_driver} did not finish the race. End status: {end_status}")

		# Step 3: If the driver is not found, raise an error
		raise ValueError(f"Driver {given_driver} not found in the session results.")