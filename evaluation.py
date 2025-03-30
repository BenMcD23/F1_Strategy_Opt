import pandas as pd
from scipy.stats import wilcoxon, spearmanr
import matplotlib.pyplot as plt
import numpy as np

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

		self.comparison_df = self.get_comparison_df()

	def get_comparison_df(self):
		""" Using the simulated comparison_df and the actual race comparison_df, it combines certain elements

		Returns:
			DataFrame: comparison_df of the combined results, comparison
		"""
		

		def _calculate_gaps_to_leader(comparison_df):
			active_drivers = comparison_df[comparison_df['position_sim'] != "R"]["driver_name"]
			
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
			comparison_df["gap_to_leader_sim"] = comparison_df["driver_name"].map(sim_gaps_to_leader)
			comparison_df["gap_to_leader_actual"] = comparison_df["driver_name"].map(actual_gaps_to_leader)

			# Handle NaN values for retired drivers
			comparison_df["gap_to_leader_sim"] = comparison_df["gap_to_leader_sim"].fillna("R")
			comparison_df["gap_to_leader_actual"] = comparison_df["gap_to_leader_actual"].fillna("R")

			return comparison_df

		def _calculate_errors(comparison_df):
			valid_rows = comparison_df[
				(comparison_df["cumulative_time_sim"] != "R") &
				(comparison_df["cumulative_time_actual"] != "R") &
				(comparison_df["gap_to_leader_sim"] != "R") &
				(comparison_df["gap_to_leader_actual"] != "R") &
				(comparison_df["position_sim"] != "R") &
				(comparison_df["position_actual"] != "R")
			].copy()
			
			# Calculate cumulative time error
			valid_rows["cumulative_time_error"] = valid_rows["cumulative_time_sim"] - valid_rows["cumulative_time_actual"]

			# Calculate gap error
			valid_rows["gap_error"] = valid_rows["gap_to_leader_sim"] - valid_rows["gap_to_leader_actual"]

			# Calculate position error
			valid_rows["position_error"] = valid_rows["position_sim"] - valid_rows["position_actual"]

			valid_rows["overtake_error"] = valid_rows["overtakes_sim"] - valid_rows["overtakes_actual"]

			# Merge the calculated errors back into the original DataFrame
			comparison_df = comparison_df.merge(
				valid_rows[["driver_name", "cumulative_time_error", "gap_error", "position_error", "overtake_error"]],
				on="driver_name",
				how="left"
			)

			comparison_df["overtake_error"]  = comparison_df["overtakes_sim"] - comparison_df["overtakes_actual"]

			# Fill NaN values in error columns with "R" (for rows that were filtered out)
			comparison_df[["cumulative_time_error", "gap_error", "position_error"]] = comparison_df[["cumulative_time_error", "gap_error", "position_error"]].fillna("R")

			return comparison_df


		# Filling any missing times with the average for that driver
		avg_sector_times = (
			self.__actual_race_df.groupby(["driver_name", "sector"])["sector_time"]
			.mean()
			.reset_index()
			.rename(columns={"sector_time": "avg_sector_time"})
		)

		# Merge the average sector times back into the original DataFrame
		self.__actual_race_df = self.__actual_race_df.merge(avg_sector_times, on=["driver_name", "sector"], how="left")

		# Fill missing sector times in the original column with the average sector time
		self.__actual_race_df["sector_time"] = self.__actual_race_df["sector_time"].fillna(self.__actual_race_df["avg_sector_time"])
		self.__actual_race_df = self.__actual_race_df.drop(columns=["avg_sector_time"])
		self.__actual_race_df ["cumulative_time"] = self.__actual_race_df .groupby("driver_name")["sector_time"].cumsum()

		# columns wanted from sim
		sim_positions = self.__sim_df[["driver_name", "position", "overtakes"]].rename(
			columns={"position": "position_sim", "overtakes": "overtakes_sim"}
		)

		# The number of laps each driver completed
		actual_final_laps = (
			self.__actual_race_df
			.sort_values(by=["driver_name", "lap_num"])  # Sort by driver and lap number
			.groupby("driver_name")  # Group by driver
			.tail(1)  # Get the last lap for each driver
		)

		# columns wanted from actual comparison_df
		actual_positions = actual_final_laps[["driver_name", "position"]].rename(
			columns={"position": "position_actual"}
		)

		# Merge the two DataFrames
		comparison_df = pd.merge(
			sim_positions,
			actual_positions,
			on="driver_name",
			how="inner"
		)

		# Change to obj as have floats and str's (R for retired)
		comparison_df["position_sim"] = comparison_df["position_sim"].astype(object)
		comparison_df["position_actual"] = comparison_df["position_actual"].astype(object)

		# If the finishing position is above 20 then they retired
		comparison_df.loc[comparison_df['position_sim'] > 20, 'position_sim'] = "R"

		comparison_df.loc[comparison_df['position_sim'] == "R", 'position_actual'] = "R"

		# add how many laps each driver completed
		max_laps_completed = (
			self.__actual_race_df
			.groupby("driver_name")["lap_num"]
			.max()
		)

		# merge into one comparison_df
		comparison_df = comparison_df.merge(
			max_laps_completed.rename("laps_completed"),
			left_on="driver_name",
			right_index=True,
			how="left"
		)

		# Add cumulative time from the simulated data
		max_cumulative_time_sim = (
			self.__sim_df
			.groupby("driver_name")["cumulative_time"]
			.max()
		)

		# merge into comparsion again
		comparison_df = comparison_df.merge(
			max_cumulative_time_sim.rename("cumulative_time_sim"),
			left_on="driver_name",
			right_index=True,
			how="left"
		)

		# Add cumulative time from the actual data
		max_cumulative_time_actual = (
			self.__actual_race_df
			.groupby("driver_name")["cumulative_time"]
			.max()
		)

		# merge again into main comparison_df
		comparison_df = comparison_df.merge(
			max_cumulative_time_actual.rename("cumulative_time_actual"),
			left_on="driver_name",
			right_index=True,
			how="left"
		)

		# Count actual overtakes for each driver
		actual_overtakes_count = (
			self.__actual_race_df[
				(self.__actual_race_df["overtake"] == True) & 
				~((self.__actual_race_df["lap_num"] == 1) & (self.__actual_race_df["sector"] == 1))
			]
			.drop_duplicates(subset=["driver_name", "lap_num"])
			.groupby("driver_name")
			.size()
			.rename("overtakes_actual")
		)

		comparison_df = comparison_df.merge(
			actual_overtakes_count,
			left_on="driver_name",
			right_index=True,
			how="left"
		)

		comparison_df["overtakes_actual"] = comparison_df["overtakes_actual"].fillna(0).astype(int)
		
		comparison_df = _calculate_gaps_to_leader(comparison_df)
		comparison_df = _calculate_errors(comparison_df)

		# to sort the retired drivers by lap number
		comparison_df['is_retired'] = comparison_df['position_sim'] == "R"
		comparison_df = comparison_df.sort_values(
			by=['is_retired', 'position_sim', 'laps_completed'],
			ascending=[True, True, False]
		)
		comparison_df = comparison_df.drop(columns=['is_retired'])



		return comparison_df[["driver_name", "laps_completed", "position_sim", "position_actual", "position_error", "overtakes_sim", "overtakes_actual", "overtake_error",
							"cumulative_time_sim", "cumulative_time_actual", "cumulative_time_error", "gap_to_leader_sim", "gap_to_leader_actual", "gap_error"]]


	def calculate_general_errors(self):
		# filter out retirement drivers
		valid_rows = self.comparison_df[self.comparison_df["position_sim"] != "R"].copy()

		# use the error column and make absolute and calc mean
		position_errors = valid_rows["position_error"].abs()
		position_mae = position_errors.mean()
		# add up all the errors
		total_absolute_position_error = position_errors.sum()

		# use the error column and make absolute and calc mean
		overtake_errors = valid_rows["overtake_error"].abs()
		overtake_mae = overtake_errors.mean()
		# add up all the errors
		total_absolute_overtake_error = overtake_errors.sum()

		# use the error column and make absolute and calc mean
		cumulative_time_errors = valid_rows["cumulative_time_error"].abs()
		cumulative_time_mae = cumulative_time_errors.mean()

		# use the error column and make absolute and calc mean
		gap_errors = valid_rows["gap_error"].abs()
		gap_mae = gap_errors.mean()

		return {
			"total_absolute_position_error": total_absolute_position_error,
			"position_mae": position_mae,
			"total_absolute_overtake_error": total_absolute_overtake_error,
			"overtake_mae": overtake_mae,
			"cumulative_time_mae": cumulative_time_mae,
			"gap_mae": gap_mae
		}

	
	def calculate_spearman(self):
		# filter out retirement drivers
		valid_rows = self.comparison_df[self.comparison_df["position_sim"] != "R"].copy()

		# Calculate Spearman correlation for cumulative times
		spearman_cumulative = spearmanr(
			valid_rows["cumulative_time_actual"],
			valid_rows["cumulative_time_sim"]
		)

		# Calculate Spearman correlation for gaps to the leader
		spearman_gaps = spearmanr(
			valid_rows["gap_to_leader_actual"],
			valid_rows["gap_to_leader_sim"]
		)

		return {
			"cumulative_times": {
				"correlation": spearman_cumulative.correlation,
				"p_value": spearman_cumulative.pvalue
			},
			"gaps_to_leader": {
				"correlation": spearman_gaps.correlation,
				"p_value": spearman_gaps.pvalue
			}
		}

	def calculate_wilcoxon(self):
		# filter out retirement drivers
		valid_rows = self.comparison_df[self.comparison_df["position_sim"] != "R"].copy()

		# make the columns numeric, error otherwise
		valid_rows["cumulative_time_error"] = pd.to_numeric(valid_rows["cumulative_time_error"], errors="coerce")
		valid_rows["gap_error"] = pd.to_numeric(valid_rows["gap_error"], errors="coerce")

		valid_rows = valid_rows.dropna(subset=["cumulative_time_error", "gap_error"])

		# Calculate Wilcoxon test for cumulative times
		wilcoxon_cumulative = wilcoxon(valid_rows["cumulative_time_error"])
		n_cumulative = valid_rows["cumulative_time_error"].astype(bool).sum()  # Count non-zero values
		expected_value_cumulative = n_cumulative * (n_cumulative + 1) / 4

		# Calculate Wilcoxon test for gaps to the leader
		wilcoxon_gaps = wilcoxon(valid_rows["gap_error"])
		n_gaps = valid_rows["gap_error"].astype(bool).sum()  # Count non-zero values
		expected_value_gaps = n_gaps * (n_gaps + 1) / 4

		return {
			"cumulative_times": {
				"statistic": wilcoxon_cumulative.statistic,
				"expected_value": expected_value_cumulative,
				"p_value": wilcoxon_cumulative.pvalue
			},
			"gaps_to_leader": {
				"statistic": wilcoxon_gaps.statistic,
				"expected_value": expected_value_gaps,
				"p_value": wilcoxon_gaps.pvalue
			}
		}


	def plot_evaluation_results(self):
		# filter out retirement drivers
		valid_rows = self.comparison_df[self.comparison_df["position_sim"] != "R"].copy()

		# Convert columns to numeric
		valid_rows["cumulative_time_sim"] = pd.to_numeric(valid_rows["cumulative_time_sim"], errors="coerce")
		valid_rows["cumulative_time_actual"] = pd.to_numeric(valid_rows["cumulative_time_actual"], errors="coerce")
		valid_rows["gap_to_leader_sim"] = pd.to_numeric(valid_rows["gap_to_leader_sim"], errors="coerce")
		valid_rows["gap_to_leader_actual"] = pd.to_numeric(valid_rows["gap_to_leader_actual"], errors="coerce")
		
		valid_rows = valid_rows.dropna(subset=["cumulative_time_sim", "cumulative_time_actual",
											"gap_to_leader_sim", "gap_to_leader_actual"])

		# Create subplots
		# fig, axes = plt.subplots(1, 3, figsize=(20, 5))
		fig, axes = plt.subplots(1, 2, figsize=(20, 5))

		axes[0].scatter(valid_rows['position_actual'], valid_rows['position_sim'], alpha=0.7)
		axes[0].set_xlabel('Actual Finishing Position')
		axes[0].set_ylabel('Simulated Finishing Position')
		axes[0].set_title('Actual vs. Simualted Finishing Position')

		# Add a perfect fit line
		axes[0].plot([valid_rows['position_actual'].min(), valid_rows['position_actual'].max()],
					[valid_rows['position_actual'].min(), valid_rows['position_actual'].max()],
					color='red', linestyle='--', label='Perfect Fit')

		# Ensure ticks are sequential (1, 2, 3, ...)
		x_ticks = range(int(valid_rows['position_actual'].min()), int(valid_rows['position_actual'].max()) + 1)
		y_ticks = range(int(valid_rows['position_sim'].min()), int(valid_rows['position_sim'].max()) + 1)

		axes[0].set_xticks(x_ticks)
		axes[0].set_yticks(y_ticks)

		# Add legend and grid
		axes[0].legend()
		axes[0].grid(True)


		mean_values = (valid_rows['gap_to_leader_actual'] + valid_rows['gap_to_leader_sim']) / 2
		differences = valid_rows['gap_to_leader_sim'] - valid_rows['gap_to_leader_actual']

		# Calculate mean difference and std deviations
		mean_difference = np.mean(differences)
		std_difference = np.std(differences)
		upper_limit = mean_difference + 1.96 * std_difference
		lower_limit = mean_difference - 1.96 * std_difference

		# Scatter plot
		axes[1].scatter(mean_values, differences, alpha=0.7)
		axes[1].axhline(y=0, color='r', linestyle='--')  # Add a reference line at zero
		axes[1].axhline(y=upper_limit, color='g', linestyle='--')  # Upper limit
		axes[1].axhline(y=lower_limit, color='g', linestyle='--')  # Lower limit

		y_min, y_max = axes[1].get_ylim()
		y_range = y_max - y_min

		offset = 0.02 * y_range

		axes[1].text(
			max(mean_values), mean_difference + offset,
			f'Mean: {mean_difference:.2f}', color='red', va='center', ha='right'
		)
		axes[1].text(
			max(mean_values), upper_limit + offset,
			f'+1.96 SD: {upper_limit:.2f}', color='green', va='center', ha='right'
		)
		axes[1].text(
			max(mean_values), lower_limit + offset,
			f'-1.96 SD: {lower_limit:.2f}', color='green', va='center', ha='right'
		)

		axes[1].set_xlabel('Mean of Actual and Simulated Gaps')
		axes[1].set_ylabel('Difference (Simulated - Actual)')
		axes[1].set_title('Bland-Altman Plot for Gaps to First Place')

		plt.tight_layout()
		plt.show()

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