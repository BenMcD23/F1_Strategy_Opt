from collections import deque
import pandas as pd


class RaceSimulation:
    def __init__(self, race_data, overtake_model, given_driver, simulated_strategy):
        self.race_data = race_data      # a RaceDataSetup object
        self.overtake_model = overtake_model    # a OvertakingModel object


        # Update strategies if a specific driver and strategy are provided
        if given_driver and simulated_strategy:
            self.race_data.driver_strategies[given_driver] = simulated_strategy

        # Initialize drivers' data
        self.drivers_data = self._initialize_drivers_data()

        # Rolling pace tracking
        self.driver_pace_per_sec = {
            driver: {sector: deque(maxlen=5) for sector in range(1, 4)}  # Sectors 1, 2, 3
            for driver in self.drivers
        }

        # Track overtakes
        self.num_overtakes = 0

    def _initialize_drivers_data(self):
        """
        Initialize the data structure for each driver.
        """
        drivers_data = []
        for driver in self.drivers:
            drivers_data.append({
                "driver_number": driver,
                "driver_name": self.driver_names[driver],
                "pit_schedule": {key: value for key, value in self.driver_strategies[driver].items() if key != 1},
                "tyre_type": self.driver_strategies[driver][1],
                "lap_num": 0,
                "sector": 0,
                "sector_time": 0.0,
                "stint_lap": 0,
                "cumulative_time": 0.0,
                "gap": 0.0,
                "pit": False,
                "pace": 0,
                "position": self.initial_positions[driver],
                "starting_pos": self.initial_positions[driver],
                "base_sector_times": self.base_sector_times[driver],
                "tyre_diff": 0,  # Initialize tyre difference as 0
                "stint_laps_diff": 0,  # Initialize stint laps difference as 0
                "drs_available": False,  # Initialize DRS availability as False
                "retired": False,
            })
        return drivers_data

    def simulate(self):
        """
        Simulate the race and return the final drivers' data.
        """
        for lap in range(1, self.race_data.max_laps + 1):
            self._process_lap(lap)

        print(f"Number of overtakes: {self.num_overtakes}")
        return self.drivers_data

    def _process_lap(self, lap):
        """
        Process a lap, including itterating the lapnumber and handling retirements and safety cars
        """
        # Increment lap and stint lap counters
        for d in self.drivers_data:
            d["lap_num"] += 1
            d["stint_lap"] += 1

        # Check for safety car and retirements
        safety_car = lap in self.race_data.safety_car_laps
        if lap in self.race_data.retirement_laps:
            self._handle_retirements(lap)

        # Process each sector
        for sector in range(1, 4):
            self._process_sector(sector, lap, safety_car)

    def _handle_retirements(self, lap):
        """
        Handle driver retirements at the given lap.
        """
        retiring_drivers = self.race_data.retirement_laps[lap]

        # Move all drivers behind the retiring drivers up by 1 position
        for driver in retiring_drivers:
            retiring_position = next(
                d["position"] for d in self.drivers_data if d["driver_number"] == driver
            )

            for d in self.drivers_data:
                if d["position"] > retiring_position:
                    d["position"] -= 1

        # Mark retiring drivers as retired
        for d in self.drivers_data:
            if d["driver_number"] in retiring_drivers:
                d["retired"] = True
                d["position"] = 999

    def _process_sector(self, sector, lap, safety_car):
        """
        Process a single sector for all drivers.
        """
        for d in self.drivers_data:
            if d["retired"]:
                continue

            d["sector"] = sector

            # Calculate sector time based on tyre degradation, fuel correction, and safety car penalty
            a, b, c = self.race_data.driver_tyre_coefficients[d["driver_number"]][d["tyre_type"]][sector]
            sector_time = (
                d["base_sector_times"][sector]  # Base sector time for specific driver
                + (a * d["stint_lap"]**2 + b * d["stint_lap"] + c)  # Tyre degradation
                + self.fuel_corrections[lap]  # Fuel effect
            )
            if safety_car:
                sector_time *= self.race_data.safety_car_penalty_percentage

            # Update sector time and cumulative time
            d["sector_time"] = sector_time
            d["cumulative_time"] += sector_time

            # Add to rolling pace tracker
            self.driver_pace_per_sec[d["driver_number"]][sector].append(sector_time)

            # Handle pit stops at the start of a lap (sector 1)
            if sector == 1 and lap in d["pit_schedule"]:
                d["pit"] = True
                d["cumulative_time"] += self.race_data.driver_pit_times[d["driver_number"]]
                d["stint_lap"] = 1
                d["tyre_type"] = d["pit_schedule"][lap]
            else:
                d["pit"] = False

        # Re-sort drivers by cumulative time and update positions
        active_drivers = [d for d in self.drivers_data if not d["retired"]]
        active_drivers.sort(key=lambda x: x["cumulative_time"])
        for i, d in enumerate(active_drivers):
            d["position"] = i + 1

        # Skip overtakes during safety car
        if safety_car:
            return

        # Handle overtakes
        for d in self.drivers_data:
            if d["retired"]:
                continue

            ahead_pos = d["position"] - 1
            if ahead_pos > 0:
                ahead_driver = next(a_d for a_d in active_drivers if a_d["position"] == ahead_pos)
                current_driver_time = d["cumulative_time"]
                ahead_driver_time = ahead_driver["cumulative_time"]

                # Fix cumulative times if out of order
                if ahead_driver_time > current_driver_time:
                    new_ahead_time = current_driver_time - 1
                    ahead_driver["cumulative_time"] = new_ahead_time
                    ahead_driver_time = new_ahead_time

                gap = current_driver_time - ahead_driver_time
                d["gap"] = gap

                # Calculate rolling pace
                sector_times = self.driver_pace_per_sec[d["driver_number"]][sector]
                d["pace"] = sum(sector_times) / len(sector_times) if sector_times else 0.0

                # Update other features
                d["tyre_diff"] = ahead_driver["tyre_type"] - d["tyre_type"]
                d["stint_laps_diff"] = ahead_driver["stint_lap"] - d["stint_lap"]
                d["drs_available"] = True
            else:
                d["gap"] = 0

        # Predict overtakes
        active_drivers = [d for d in self.drivers_data if not d["retired"]]
        
        predicted_overtakes = self.overtake_model.handle_overtake_prediction(active_drivers)

        # can use active_drivers and dicts are mutable
        for i, driver in enumerate(active_drivers):
            driver["predicted_overtake"] = predicted_overtakes[i]

        for driver in active_drivers:
            if driver["retired"]:
                continue

            ahead_pos = driver["position"] - 1
            if driver["gap"] < 1 and ahead_pos > 0 and driver["predicted_overtake"]:
                self.num_overtakes += 1
                ahead_driver = next(d for d in active_drivers if d["position"] == ahead_pos)
                # Swap positions and cumulative times
                driver["position"], ahead_driver["position"] = ahead_driver["position"], driver["position"]
                driver["cumulative_time"], ahead_driver["cumulative_time"] = (
                    ahead_driver["cumulative_time"] - 1, driver["cumulative_time"]
                )

    
    def get_results_as_dataframe(self):
        """
        Return the final results as a Pandas DataFrame.
        """
        sim_df = pd.DataFrame(self.drivers_data)
        sim_df = sim_df.sort_values(by="position", ascending=True).reset_index(drop=True)
        return sim_df
    
    