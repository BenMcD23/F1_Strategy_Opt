from DB.models import init_db, Circuit, Season, RacingWeekend, Driver, Session, SessionResult, Lap, Team, DriverTeamSession, PitStop
from sqlalchemy import create_engine, func
import pandas as pd

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
	
	def _add_race_data(self, df):
		"""Adds extra info the race dataframe such as gaps and info for overtakes. Added to train the overtake model

		Returns:
			Pandas DF: _description_
		"""
		df["cumulative_time"] = df.groupby("driver_name")["sector_time"].cumsum()
	
		# Calculate rolling pace (average lap time over the last 5 laps)
		df["pace"] = (
			df.groupby(["driver_name", "sector"])["sector_time"]
			.rolling(window=5, min_periods=1)
			.mean()
			.reset_index(level=[0, 1], drop=True)
		)

		# Get car ahead"s cumulative time (car immediately ahead in position for each lap)
		df["front_cumulative_time"] = df.groupby(["lap_num", "sector"])["cumulative_time"].shift(1)
		# This gap is calculated only for drivers who are not in the lead position (position > 1)
		df["gap"] = df["cumulative_time"] - df["front_cumulative_time"]
		df["gap"] = df["gap"].fillna(0)  # Leader has no car ahead, so gap is 0

		# Calculate tyre difference (compared to car immediately ahead in THIS Sector)
		df["front_tyre"] = df.groupby(["lap_num", "sector"])["tyre_type"].shift(1)
		df["tyre_diff"] = df["front_tyre"] - df["tyre_type"]
		df["tyre_diff"] = df["tyre_diff"].fillna(0)  # Leader has no car ahead

		# Calculate tyre age difference (compared to car immediately ahead in THIS Sector)
		df["front_laps"] = df.groupby(["lap_num", "sector"])["stint_lap"].shift(1)
		df["stint_laps_diff"] = df["front_laps"] - df["stint_lap"]
		df["stint_laps_diff"] = df["stint_laps_diff"].fillna(0)  # Leader has no car ahead

		# Calculate DRS availability (within 1s of car ahead IN THIS Sector)
		df["drs_available"] = (
			(df["gap"] <= 1) &
			(df["position"] > 1) &
			(df["lap_num"] > 1)
		)


		# df["next_position"] = df.groupby("driver_name")["position"].shift(1) 
		# df["overtaken"] = ((df["next_position"] < df["position"]) | 
		# 				  (df["next_position"].isna()))


		# Shift the "position" and "pit" columns
		df["next_position"] = df.groupby("driver_name")["position"].shift(1)
		df["next_pit"] = df.groupby("driver_name")["pit"].shift(-1)

		# Handle NaN values in 'next_pit' by filling them with False
		df["next_pit"] = df["next_pit"].fillna(False)

		# Define the "overtaken" column
		df["overtaken"] = (
			((df["next_position"] < df["position"]) | (df["next_position"].isna()))  # Original condition
			& (~df["next_pit"])  # Ensure the driver in the next position is not pitting
		)

		# # Create target variable for overtaking model)
		# # Step 1: Detect position improvement on the next sector
		# df.loc[:, "next_position"] = df.groupby("driver_name")["position"].shift(-1)
		# df.loc[:, "position_improved"] = (
		# 	(df["next_position"] < df["position"]) |  # Position improved
		# 	(df["next_position"].isna())  # Handle NaN values
		# )

		# # Step 2: Identify the sector with the minimum gap for each driver and lap
		# df.loc[:, "min_gap_sector"] = df[df["position_improved"]].groupby(["driver_name", "lap_num"])["gap"].transform("idxmin")

		# # Step 3: Propagate the overtaken flag to the sector with the minimum gap
		# df.loc[:, "overtaken"] = False  # Initialize the overtake flag
		# for idx in df[df["position_improved"]].index:
		# 	driver = df.loc[idx, "driver_name"]
		# 	lap = df.loc[idx, "lap_num"]
			
		# 	# Find the sector with the minimum gap for this driver and lap
		# 	min_gap_idx = df.loc[
		# 		(df["driver_name"] == driver) & 
		# 		(df["lap_num"] == lap) & 
		# 		(df["position_improved"]), "min_gap_sector"
		# 	].iloc[0]
			
		# 	# Set the overtaken flag only for the sector with the minimum gap
		# 	if idx == min_gap_idx:
		# 		df.loc[idx, "overtaken"] = True

		# # Cleanup intermediate columns
		# df = df.drop(columns=["position_improved", "min_gap_sector"], errors="ignore")
		

		# Cleanup and final sorting
		df = df.drop(columns=["front_cumulative_time", "front_tyre", "next_position"])
		# df = df.sort_values(["lap_num", "sector", "position"]).reset_index(drop=True)

		return df
	
	def _get_race_df(self):
		raw_df = self._get_raw_race_df()
		
		race_df = self._add_race_data(raw_df)


		return race_df