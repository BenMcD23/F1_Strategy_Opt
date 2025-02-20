import pandas as pd

class RaceSimEvaluation:
	def __init__(self, race_sim_obj, race_df_obj, database_obj):
		self.race_sim_obj = race_sim_obj
		self.race_df_obj = race_df_obj
		self.database_obj = database_obj
		
		self.sim_df = race_sim_obj.get_results_as_dataframe()
		self.actual_race_df = race_df_obj.race_df

	def compare_total_cumaltive_times(self):
		comparison_results = []
			
		# Get unique drivers from the simulated DataFrame
		drivers = self.sim_df["driver_number"].unique()
		
		# Calculate cumulative times for each driver in both simulated and actual data
		for driver in drivers:
			# Simulated cumulative time for the driver
			sim_data = self.sim_df[self.sim_df["driver_number"] == driver]
			sim_cumulative_time = sim_data["cumulative_time"].max()
			driver_name = sim_data["driver_name"].iloc[0]
			
			# Actual cumulative time for the driver
			actual_data = self.actual_race_df[self.actual_race_df["driver_number"] == driver]
			actual_cumulative_time = actual_data["cumulative_time"].max()
			driver_name = actual_data["driver_name"].iloc[0]

			# Calculate the absolute error for the driver
			absolute_error = abs(sim_cumulative_time - actual_cumulative_time)
			
			# Store the results for the driver
			comparison_results.append({
				"driver_number": driver,
				"driver_name": driver_name,
				"simulated_cumulative_time": sim_cumulative_time,
				"actual_cumulative_time": actual_cumulative_time,
				"absolute_error": absolute_error
			})
		
		# Convert the results to a DataFrame
		comparison_df = pd.DataFrame(comparison_results)
		
		# Calculate the total Mean Absolute Error (MAE)
		total_mae = comparison_df["absolute_error"].mean()
		
		return comparison_df, total_mae
	
	# def get_mean_cumulative_time_error(self):


	def get_position_accuracy_final_class(self):
		# Extract simulated final positions
		sim_results = (
			self.sim_df[self.sim_df["retired"] == False]  # Exclude retired drivers
			.groupby("driver_number")["position"]
			.last()
			.to_dict()
		)
		
		session_results = self.database_obj.race_session_results_db
		# Extract actual final positions
		actual_results = {driver_num: position for position, driver_num, end_status in session_results}

		# Ensure both results have the same drivers
		common_drivers = set(sim_results.keys()).intersection(actual_results.keys())
		if not common_drivers:
			raise ValueError("No common drivers found between actual and simulated results")
		
		# Filter results to only include common drivers
		actual_positions = [actual_results[driver] for driver in common_drivers]
		sim_positions = [sim_results[driver] for driver in common_drivers]
		
		# Calculate accuracy metrics
		position_accuracy = sum(1 for a, s in zip(actual_positions, sim_positions) if a == s) / len(common_drivers)
		top_3_accuracy = sum(1 for a, s in zip(actual_positions, sim_positions) if (a <= 3 and s <= 3)) / 3
		mean_error = sum(abs(a - s) for a, s in zip(actual_positions, sim_positions)) / len(common_drivers)
		total_error = sum(abs(a - s) for a, s in zip(actual_positions, sim_positions))
		
		# Return accuracy metrics
		return {
			"position_accuracy": position_accuracy,
			"top_3_accuracy": top_3_accuracy,
			"mean_error": mean_error,
			"total_error": total_error,
		}


	def get_position_accuracy_end_of_race(self):
		# Extract simulated final positions
		sim_results = (
			self.sim_df[self.sim_df["retired"] == False]  # Exclude retired drivers
			.groupby("driver_number")["position"]
			.last()
			.to_dict()
		)
		
		# Extract actual final positions
		actual_results = (
			self.actual_race_df[self.actual_race_df["retired"] == False]  # Exclude retired drivers
			.groupby("driver_number")["position"]
			.last()
			.to_dict()
		)
		
		# Ensure both results have the same drivers
		common_drivers = set(sim_results.keys()).intersection(actual_results.keys())
		if not common_drivers:
			raise ValueError("No common drivers found between actual and simulated results")
		
		# Filter results to only include common drivers
		actual_positions = [actual_results[driver] for driver in common_drivers]
		sim_positions = [sim_results[driver] for driver in common_drivers]
		
		# Calculate accuracy metrics
		position_accuracy = sum(1 for a, s in zip(actual_positions, sim_positions) if a == s) / len(common_drivers)
		top_3_accuracy = sum(1 for a, s in zip(actual_positions, sim_positions) if (a <= 3 and s <= 3)) / 3
		mean_error = sum(abs(a - s) for a, s in zip(actual_positions, sim_positions)) / len(common_drivers)
		total_error = sum(abs(a - s) for a, s in zip(actual_positions, sim_positions))
		
		# Return accuracy metrics
		return {
			"position_accuracy": position_accuracy,
			"top_3_accuracy": top_3_accuracy,
			"mean_error": mean_error,
			"total_error": total_error,
		}


	def compare_total_to_front(self):
		def _get_retired_drivers():
			# Initialize an empty list to store retired drivers
			retired_drivers = []

			# Iterate through session results to determine retirements
			for _, driver_num, end_status in self.database_obj.race_session_results_db:
				# Check if the driver retired (end_status is not "Finished" or "+1 Lap")
				if end_status and not (end_status.startswith("Finished") or end_status.startswith("+")):
					# Add the driver to the list of retirees
					retired_drivers.append(driver_num)
			
			return retired_drivers

		# Get the list of retired drivers
		retired_drivers = _get_retired_drivers()

		# Simulated data
		sim_leader_time = (
			self.sim_df[self.sim_df["position"] == 1]  # Leader (first position)
			.groupby("driver_number")["cumulative_time"]
			.max()
			.min()  # In case of ties, take the minimum time
		)
		sim_gaps_to_front = (
			self.sim_df[~self.sim_df["driver_number"].isin(retired_drivers)]  # Exclude retired drivers
			.groupby("driver_number")["cumulative_time"]
			.max()
			.apply(lambda x: x - sim_leader_time)
		)
		total_sim_gap = sim_gaps_to_front.sum()

		# Actual data
		actual_leader_time = (
			self.actual_race_df[self.actual_race_df["position"] == 1]  # Leader (first position)
			.groupby("driver_number")["cumulative_time"]
			.max()
			.min()  # In case of ties, take the minimum time
		)
		actual_gaps_to_front = (
			self.actual_race_df[~self.actual_race_df["driver_number"].isin(retired_drivers)]  # Exclude retired drivers
			.groupby("driver_number")["cumulative_time"]
			.max()
			.apply(lambda x: x - actual_leader_time)
		)
		total_actual_gap = actual_gaps_to_front.sum()

		# Return the total gaps
		return {
			"total_simulated_gap_to_front": total_sim_gap,
			"total_actual_gap_to_front": total_actual_gap,
		}
	


class EvaluateMany:
	def __init__(self):
		pass