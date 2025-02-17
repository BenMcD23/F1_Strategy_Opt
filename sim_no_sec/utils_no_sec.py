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
		driver_df.loc[:, "fuel_corrected_lap_time"] = driver_df["lap_time"] - driver_df["fuel_correction"]
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
	def _filter_driver_laps(driver_df):
		# Calculate the threshold based on the fastest lap time for the driver and sector
		fastest_lap_time = driver_df["fuel_corrected_lap_time"].min()
		threshold = fastest_lap_time * (1 + percentage / 100)
		
		# Remove laps not within the specified percentage of the fastest lap time
		filtered_df = driver_df[driver_df["fuel_corrected_lap_time"] <= threshold]
		return filtered_df

	# Group by driver and sector, then apply the filtering logic
	df = (
		df.groupby(["driver_number"], group_keys=False)
		.apply(_filter_driver_laps)
		.reset_index(drop=True)
	)

	return df


def normalise_lap_times(df):
	# Group by driver, sector, then calculate the fastest lap time for each group
	df['fastest_lap_time'] = (
		df.groupby(['driver_number'])['fuel_corrected_lap_time']
		  .transform('min')
	)
	
	# Normalise lap times by subtracting the fastest sector time
	df['normalised_lap_time'] = (
		df['fuel_corrected_lap_time'] - df['fastest_lap_time']
	)
	
	# Drop the fastest_lap_time column as it's no longer needed
	df = df.drop(columns=['fastest_lap_time'])
	
	return df
# --------------------------------------


def get_tyre_deg_per_driver(df):
    # Normalize lap times
    df = assign_stint_numbers(df)
    df = correct_fuel_effect(df)
    df = remove_laps_outside_percent(df)
    df = normalise_lap_times(df)
    
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
        driver_df = df[df["driver_number"] == driver]
        
        # Dictionary to store coefficients for this driver
        driver_coeffs = {}
        
        # Group by tyre type
        for tyre in driver_df["tyre_type"].unique():
            tyre_coeffs = []
            
            # Group by stint to handle multiple stints with the same tyre type
            for stint, stint_data in driver_df[driver_df["tyre_type"] == tyre].groupby("stint"):
                # Check if the stint meets the minimum lap requirement for the tyre type
                min_laps = min_laps_by_tyre.get(tyre, 0)  # Default to 0 if tyre type not in dictionary
                if len(stint_data) < min_laps:
                    continue
                
                # Fit a constrained polynomial model
                x = stint_data["tyre_laps"].values
                y = stint_data["normalised_lap_time"].values
                
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
                tyre_coeffs.append([a, b, c])
            
            # Average coefficients for this tyre type (if there are any)
            if tyre_coeffs:
                driver_coeffs[tyre] = np.mean(tyre_coeffs, axis=0).tolist()
        
        # Store the driver's coefficients
        driver_tyre_coefficients[driver] = driver_coeffs

 	# Step 2: Calculate global averages for each tyre type
    global_tyre_coefficients = {}
    for tyre in min_laps_by_tyre.keys():
        all_coeffs = [
            driver_coeffs[tyre]
            for driver_coeffs in driver_tyre_coefficients.values()
            if tyre in driver_coeffs
        ]
        if all_coeffs:
            global_tyre_coefficients[tyre] = np.mean(all_coeffs, axis=0).tolist()
    
    # Step 3: Add global averages to drivers missing specific tyre types
    for driver, driver_coeffs in driver_tyre_coefficients.items():
        for tyre in min_laps_by_tyre.keys():
            if tyre not in driver_coeffs and tyre in global_tyre_coefficients:
                driver_coeffs[tyre] = global_tyre_coefficients[tyre]
                
    return driver_tyre_coefficients


def calculate_base_lap_times(race_df):
    # Unique drivers
    drivers = race_df["driver_number"].unique()
    
    # Step 1: Calculate the fastest lap time for each driver
    base_lap_times = {
        driver: race_df[
            (race_df["driver_number"] == driver)
        ]["fuel_corrected_lap_time"].min()
        for driver in drivers
    }
    
    return base_lap_times





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