import pandas as pd

class RaceSimEvaluation:
	def __init__(self, race_sim_obj, race_df_obj, database_obj):
		"""Initializes the RaceSimEvaluation object.

		Args:
			race_sim_obj: Object of RaceSimulator class
			race_df_obj: Object of RaceDataframe class - this is the actual race dataframe
			database_obj: Object of class DatabaseOperations class
		"""
		self.__database_obj = database_obj
		
		self.__sim_df = race_sim_obj.get_results_as_dataframe()
		self.__actual_race_df = race_df_obj.race_df

	def get_comparison_df(self):
		# Extract relevant columns from the simulated DataFrame
		sim_positions = self.__sim_df[["driver_name", "position"]].rename(
			columns={"position": "simulated_position"}
		)

		# Filter the actual DataFrame to get the last lap for each driver
		actual_final_laps = (
			self.__actual_race_df
			.sort_values(by=["driver_name", "lap_num"])  # Sort by driver and lap number
			.groupby("driver_name")  # Group by driver
			.tail(1)  # Get the last lap for each driver
		)
		actual_positions = actual_final_laps[["driver_name", "position"]].rename(
			columns={"position": "actual_position"}
		)

		# Merge the two DataFrames on "driver_name"
		driver_positions_df = pd.merge(
			sim_positions,
			actual_positions,
			on="driver_name",
			how="inner"
		)

		# Mark retirements as R
		driver_positions_df.loc[driver_positions_df['simulated_position'] > 21, 'simulated_position'] = "R"

		driver_positions_df.loc[driver_positions_df['simulated_position'] == "R", 'actual_position'] = "R"

		# Add the maximum lap completed by each driver in the actual race
		max_laps_completed = (
			self.__actual_race_df
			.groupby("driver_name")["lap_num"]  # Group by driver and get lap numbers
			.max()  # Get the maximum lap number for each driver
		)
		driver_positions_df = driver_positions_df.merge(
			max_laps_completed.rename("laps_completed"),
			left_on="driver_name",
			right_index=True,
			how="left"
		)

		# Add the maximum cumulative time from the simulated data
		max_cumulative_time_sim = (
			self.__sim_df
			.groupby("driver_name")["cumulative_time"]  # Group by driver and get cumulative times
			.max()  # Get the maximum cumulative time for each driver
		)
		driver_positions_df = driver_positions_df.merge(
			max_cumulative_time_sim.rename("cumulative_time_sim"),
			left_on="driver_name",
			right_index=True,
			how="left"
		)

		# Add the maximum cumulative time from the actual data
		max_cumulative_time_actual = (
			self.__actual_race_df
			.groupby("driver_name")["cumulative_time"]  # Group by driver and get cumulative times
			.max()  # Get the maximum cumulative time for each driver
		)
		driver_positions_df = driver_positions_df.merge(
			max_cumulative_time_actual.rename("cumulative_time_actual"),
			left_on="driver_name",
			right_index=True,
			how="left"
		)

		driver_positions_df = self.calculate_gaps_to_leader(driver_positions_df)

		driver_positions_df = self.calculate_errors(driver_positions_df)


		return driver_positions_df

	def calculate_gaps_to_leader(self, driver_positions_df):
		active_drivers = driver_positions_df[driver_positions_df['simulated_position'] != "R"]["driver_name"]

		sim_leader_time = self.__sim_df[self.__sim_df["position"] == 1]["cumulative_time"].values[0]

		sim_gaps_to_leader = {
			driver: self.__sim_df[self.__sim_df["driver_name"] == driver]["cumulative_time"].max() - sim_leader_time
			for driver in active_drivers
		}

		actual_leader_times = (
			self.__actual_race_df[
				(self.__actual_race_df["position"] == 1) &  # Leader (first position)
				(self.__actual_race_df["sector"] == 3)      # Sector 3
			]
			.set_index("lap_num")["cumulative_time"]  # Set lap_num as index and select cumulative_time
		)

		actual_gaps_to_leader = {}
		for driver in active_drivers:
			actual_driver_data = self.__actual_race_df[self.__actual_race_df["driver_name"] == driver]
			if not actual_driver_data.empty:
				max_lap_actual = actual_driver_data["lap_num"].max()
				actual_driver_time = actual_driver_data["cumulative_time"].max()
				actual_leader_time = actual_leader_times.get(max_lap_actual, None)
				actual_gaps_to_leader[driver] = actual_driver_time - actual_leader_time

		# Add gaps to the leader to the main DataFrame
		driver_positions_df["gap_to_leader_sim"] = driver_positions_df["driver_name"].map(sim_gaps_to_leader)
		driver_positions_df["gap_to_leader_actual"] = driver_positions_df["driver_name"].map(actual_gaps_to_leader)

		# Handle NaN values for retired drivers
		driver_positions_df["gap_to_leader_sim"] = driver_positions_df["gap_to_leader_sim"].fillna("R")
		driver_positions_df["gap_to_leader_actual"] = driver_positions_df["gap_to_leader_actual"].fillna("R")

		return driver_positions_df

	def calculate_errors(self, driver_positions_df):
		"""
		Calculates the error between simulated and actual data for cumulative times and gaps to the leader.
		Handles cases where values are "R" or NaN.

		Args:
			driver_positions_df (pd.DataFrame): The main DataFrame containing driver data.

		Returns:
			pd.DataFrame: The updated DataFrame with error columns.
		"""
		# Calculate cumulative time error
		driver_positions_df["cumulative_time_error"] = (
			driver_positions_df.apply(
				lambda row: row["cumulative_time_sim"] - row["cumulative_time_actual"]
				if isinstance(row["cumulative_time_sim"], (int, float)) and isinstance(row["cumulative_time_actual"], (int, float))
				else "R",
				axis=1
			)
		)

		# Calculate gap error
		driver_positions_df["gap_error"] = (
			driver_positions_df.apply(
				lambda row: row["gap_to_leader_sim"] - row["gap_to_leader_actual"]
				if isinstance(row["gap_to_leader_sim"], (int, float)) and isinstance(row["gap_to_leader_actual"], (int, float))
				else "R",
				axis=1
			)
		)

		return driver_positions_df


	


class EvaluateMany:
	def __init__(self):
		""" Initialize the EvaluateMany class.
		"""
		self.race_evaluations = []  # List to store individual RaceSimEvaluation objects
		self.results = []           # List to store evaluation results for each race

	def add_race(self, race_sim_obj, race_df_obj, database_obj):
		""" Add a race to the evaluation pipeline.

		Args:
			race_sim_obj: The RaceSimulator object for the race.
			race_df_obj: The RaceDataframe object for the race.
			database_obj: The DatabaseOperations object for the race.
		"""
		# Create a RaceSimEvaluation object for the race
		race_evaluation = RaceSimEvaluation(race_sim_obj, race_df_obj, database_obj)
		self.race_evaluations.append(race_evaluation)

	def evaluate_all_races(self):
		""" Evaluate all added races and store their results.
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
		""" Aggregate results across all races and calculate averages.

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