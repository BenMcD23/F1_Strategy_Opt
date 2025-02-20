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