from scipy.optimize import minimize
import numpy as np
import pandas as pd

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
	# Group by driver, sector, then calculate the fastest lap time for each group
	df['fastest_sector_time'] = (
		df.groupby(['driver_number', 'sector'])['fuel_corrected_sector_time']
		  .transform('min')
	)
	
	# Normalise lap times by subtracting the fastest sector time
	df['normalised_sector_time'] = (
		df['fuel_corrected_sector_time'] - df['fastest_sector_time']
	)
	
	# Drop the fastest_sector_time column as it's no longer needed
	df = df.drop(columns=['fastest_sector_time'])
	
	return df
# --------------------------------------



def get_tyre_deg_per_driver(df):
	# Normalize lap times
	df = assign_stint_numbers(df)
	df = correct_fuel_effect(df)
	df = remove_laps_outside_percent(df)
	df = normalise_lap_times_by_sector(df)
	
	# Define minimum laps required for each tyre type
	min_laps_by_tyre = {
		1: 2,  # Soft
		2: 4,  # Med
		3: 6   # Hard
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

	return driver_tyre_coefficients


def calculate_base_sector_times(race_df):
    # Unique drivers and sectors
    drivers = race_df["driver_number"].unique()
    sectors = race_df["sector"].unique()
    
    # Step 1: Calculate the fastest sector time for each driver and sector
    base_sector_times = {
        driver: {
            sector: race_df[
                (race_df["driver_number"] == driver) &
                (race_df["sector"] == sector)
            ]["fuel_corrected_sector_time"].min()
            for sector in sectors
        }
        for driver in drivers
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