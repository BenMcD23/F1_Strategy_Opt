import pandas as pd

class RaceSimEvaluation:
	def __init__(self, race_sim_obj, race_df_obj, database_obj):
		self.race_sim_obj = race_sim_obj
		self.race_df_obj = race_df_obj
		self.database_obj = database_obj
		
		self.sim_df = race_sim_obj.get_results_as_dataframe()
		self.actual_race_df = race_df_obj.race_df

	def compare_total_cumulative_times(self):
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
        """
        Initialize the EvaluateMany class.
        """
        self.race_evaluations = []  # List to store individual RaceSimEvaluation objects
        self.results = []           # List to store evaluation results for each race

    def add_race(self, race_sim_obj, race_df_obj, database_obj):
        """
        Add a race to the evaluation pipeline.

        Args:
            race_sim_obj: The RaceSimulation object for the race.
            race_df_obj: The RaceDataframe object for the race.
            database_obj: The DatabaseOperations object for the race.
        """
        # Create a RaceSimEvaluation object for the race
        race_evaluation = RaceSimEvaluation(race_sim_obj, race_df_obj, database_obj)
        self.race_evaluations.append(race_evaluation)

    def evaluate_all_races(self):
        """
        Evaluate all added races and store their results.
        """
        for i, race_evaluation in enumerate(self.race_evaluations):
            print(f"Evaluating race {i + 1}...")
            # Perform evaluations for each metric
            cumulative_times = race_evaluation.compare_total_cumulative_times()
            position_accuracy = race_evaluation.get_position_accuracy_final_class()
            total_to_front = race_evaluation.compare_total_to_front()

            # Store the results
            self.results.append({
                "cumulative_times": cumulative_times,
                "position_accuracy": position_accuracy,
                "total_to_front": total_to_front,
            })

    def get_aggregated_results(self):
        """
        Aggregate results across all races and calculate averages.

        Returns:
            dict: Aggregated results with averages for each metric.
        """
        if not self.results:
            raise ValueError("No races have been evaluated yet.")

        # Initialize aggregated metrics
        total_mae = 0
        total_position_accuracy = 0
        total_top_3_accuracy = 0
        total_mean_error = 0
        total_total_error = 0
        total_simulated_gap_to_front = 0
        total_actual_gap_to_front = 0
        num_races = len(self.results)
		
        for result in self.results:
            # Cumulative times
            total_mae += result["cumulative_times"][1]  # Extract MAE from the tuple

            # Position accuracy
            total_position_accuracy += result["position_accuracy"]["position_accuracy"]
            total_top_3_accuracy += result["position_accuracy"]["top_3_accuracy"]
            total_mean_error += result["position_accuracy"]["mean_error"]
            total_total_error += result["position_accuracy"]["total_error"]

            # Total gaps to front
            total_simulated_gap_to_front += result["total_to_front"]["total_simulated_gap_to_front"]
            total_actual_gap_to_front += result["total_to_front"]["total_actual_gap_to_front"]

        # Calculate averages
        avg_mae = total_mae / num_races
        avg_position_accuracy = total_position_accuracy / num_races
        avg_top_3_accuracy = total_top_3_accuracy / num_races
        avg_mean_error = total_mean_error / num_races
        avg_total_error = total_total_error / num_races
        avg_simulated_gap_to_front = total_simulated_gap_to_front / num_races
        avg_actual_gap_to_front = total_actual_gap_to_front / num_races

        # Return aggregated results
        return {
            "average_mae": avg_mae,
            "average_position_accuracy": avg_position_accuracy,
            "average_top_3_accuracy": avg_top_3_accuracy,
            "average_mean_error": avg_mean_error,
            "average_total_error": avg_total_error,
            "average_simulated_gap_to_front": avg_simulated_gap_to_front,
            "average_actual_gap_to_front": avg_actual_gap_to_front,
        }