from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
import sys
import os
import pandas as pd


# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))
from DB.models import init_db, Circuit, Season, RacingWeekend, Driver, Session, SessionResult, Lap, Team, DriverTeamSession, PitStop
# from utils import correct_fuel_effect, extract_driver_strategies, get_base_sector_times, get_tyre_deg_per_driver, get_race_session, get_safety_car_penalty, get_driver_pit_time


pd.set_option('future.no_silent_downcasting', True)


from collections import deque
import pandas as pd


class RaceSimulator:
	def __init__(self, race_data, given_driver=None, simulated_strategy=None):
		self.__race_data = race_data      # a RaceDataSetup object


		# Update strategies if a specific driver and strategy are provided
		if given_driver and simulated_strategy:
			self.__race_data.driver_strategies[given_driver] = simulated_strategy

		# Initialize drivers' data
		self.sim_data = self.__initialize_drivers_data()

		# Rolling pace tracking
		self.__driver_pace_per_sec = {
			driver: {sector: deque(maxlen=5) for sector in range(1, 4)}  # Sectors 1, 2, 3
			for driver in self.__race_data.drivers
		}

		# Track overtakes
		self.num_overtakes = 0
		self.has_safety_car = bool(self.__race_data.safety_car_laps)

	def __initialize_drivers_data(self):
		"""
		Initialize the data structure for each driver.
		"""
		sim_data = []
		for driver in self.__race_data.drivers:
			sim_data.append({
				"driver_number": driver,
				"driver_name": self.__race_data.driver_names[driver],
				# "pit_schedule": {key: value for key, value in self.__race_data.driver_strategies[driver].items() if key != 1},
				"pit_schedule": self.__race_data.driver_strategies[driver],
				"tyre_type": self.__race_data.driver_strategies[driver][1],
				"lap_num": 0,
				"sector": 0,
				"sector_time": 0.0,
				"stint_lap": 0,
				"cumulative_time": 0.0,
				"gap": 0.0,
				"pit": False,
				"pace": 0,
				"position": self.__race_data.starting_positions[driver],
				"starting_pos": self.__race_data.starting_positions[driver],
				"base_sector_times": self.__race_data.base_sector_times[driver],
				"tyre_diff": 0,
				"stint_laps_diff": 0,
				"drs_available": False,
				"retired": False,
			})
		return sim_data

	def simulate(self):
		"""
		Simulate the race and return the final drivers' data.
		"""
		for lap in range(1, self.__race_data.max_laps + 1):
			self.__process_lap(lap)

		return self.sim_data

	def __process_lap(self, lap):
		"""
		Process a lap, including itterating the lapnumber and handling retirements and safety cars
		"""
		# Increment lap and stint lap counters
		for d in self.sim_data:
			d["lap_num"] += 1
			d["stint_lap"] += 1

		# Check for safety car and retirements
		safety_car = lap in self.__race_data.safety_car_laps

		if lap in self.__race_data.retirements_by_lap:
			self.__handle_retirements(lap)

		# Process each sector
		for sector in range(1, 4):
			self.__process_sector(sector, lap, safety_car)

	def __handle_retirements(self, lap):
		"""
		Handle driver retirements at the given lap.
		"""
		retiring_drivers = self.__race_data.retirements_by_lap[lap]

		# Move all drivers behind the retiring drivers up by 1 position
		for driver in retiring_drivers:
			retiring_position = next(
				d["position"] for d in self.sim_data if d["driver_number"] == driver
			)

			for d in self.sim_data:
				if d["position"] > retiring_position:
					d["position"] -= 1

		# Mark retiring drivers as retired
		for d in self.sim_data:
			if d["driver_number"] in retiring_drivers:
				d["retired"] = True
				d["position"] = 999

	def __process_sector(self, sector, lap, safety_car):
		"""
		Process a single sector for all drivers.
		"""

		for d in self.sim_data:
			if d["retired"]:
				continue

			d["sector"] = sector

			# Calculate sector time based on tyre degradation, fuel correction, and safety car penalty
			a, b, c = self.__race_data.driver_tyre_coefficients[d["driver_number"]][d["tyre_type"]][sector]
			sector_time = (
				d["base_sector_times"][sector]  # Base sector time for specific driver
				+ (a * d["stint_lap"]**2 + b * d["stint_lap"] + c)  # Tyre degradation
				+ self.__race_data.fuel_corrections[(lap, sector)]  # Fuel effect
			)
			if safety_car:
				sector_time *= self.__race_data.safety_car_penalty_percentage

			# Update sector time and cumulative time
			d["sector_time"] = sector_time
			d["cumulative_time"] += sector_time

			# Add to rolling pace tracker
			self.__driver_pace_per_sec[d["driver_number"]][sector].append(sector_time)

			# Handle pit stops at the start of a lap (sector 1)
			if sector == 1 and lap in d["pit_schedule"]:
				if lap == 1:
					continue
				d["pit"] = True
				if d["driver_number"] in self.__race_data.driver_pit_times:
					# Add the driver's specific pit time if available
					d["cumulative_time"] += self.__race_data.driver_pit_times[d["driver_number"]]
				else:
					# Use the overall average pit time (key 0) if the driver doesn't have a specific pit time
					d["cumulative_time"] += self.__race_data.driver_pit_times[0]

				d["stint_lap"] = 1
				d["tyre_type"] = d["pit_schedule"][lap]   # change tyre 
			else:
				d["pit"] = False

		# Re-sort drivers by cumulative time and update positions
		active_drivers = [d for d in self.sim_data if not d["retired"]]
		active_drivers.sort(key=lambda x: x["cumulative_time"])
		for i, d in enumerate(active_drivers):
			d["position"] = i + 1

		# Skip overtakes during safety car
		if safety_car:
			return



	
	def get_results_as_dataframe(self):
		"""
		Return the final results as a Pandas DataFrame.
		"""
		sim_df = pd.DataFrame(self.sim_data)
		sim_df = sim_df.sort_values(by="position", ascending=True).reset_index(drop=True)
		return sim_df
	
	