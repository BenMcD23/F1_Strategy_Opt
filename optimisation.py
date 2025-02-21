from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd

from race_sim import RaceSimulation

class Optimisation:
	def __init__(self, race_data, overtake_model):
		"""
		Initialize the Optimization class with race data and an overtaking model.
		"""
		self.race_data = race_data
		self.overtake_model = overtake_model

	def bayesian_strategy_optimization(self, given_driver, max_iterations=50):
		"""
		Perform Bayesian Optimization to find the most optimal strategy for a given driver.
		Enforces F1 rules: at least two different tyre compounds must be used.
		Allows up to 3 pit stops but does not require all 3.
		Explores around the driver's initial strategy.
		"""
		# Step 1: Get unique tyre types and convert to a sorted list
		unique_tyre_types = sorted(self.race_data.get_unique_tyre_types())  # Convert set to sorted list
		
		# Step 2: Extract the initial strategy for the given driver
		initial_strategy = self.race_data.driver_strategies[given_driver]
		
		# Step 3: Validate and adjust the initial strategy if necessary
		initial_tyre_types_used = set(initial_strategy.values())
		if len(initial_tyre_types_used) < 2:
			# Modify the initial strategy to satisfy the F1 rule
			print("Initial strategy violates F1 rules. Adjusting...")
			# Add a second tyre type if only one is used
			additional_tyre = [tyre for tyre in unique_tyre_types if tyre not in initial_tyre_types_used][0]
			initial_strategy[list(initial_strategy.keys())[1]] = additional_tyre
		
		# Step 4: Define parameter bounds for Bayesian Optimization
		pbounds = {
			"pit1_lap": (2, self.race_data.max_laps - 1),  # First pit stop lap
			"pit2_lap": (2, self.race_data.max_laps - 1),  # Second pit stop lap
			"pit3_lap": (2, self.race_data.max_laps - 1),  # Third pit stop lap
		}
		
		# Step 5: Map initial strategy to optimization parameters
		def map_strategy_to_params(strategy):
			"""Map a strategy dictionary to optimization parameters."""
			pit_laps = sorted([lap for lap in strategy.keys() if lap > 1])  # Exclude lap 1 (starting tyre)
			tyres = [strategy[lap] for lap in pit_laps]
			
			# Ensure there are up to 3 pit stops
			pit_laps += [self.race_data.max_laps] * (3 - len(pit_laps))  # Pad with max laps if fewer than 3 stops
			tyres += [tyres[-1]] * (3 - len(tyres))  # Repeat the last tyre if fewer than 3 stops
			
			# Map tyre types to indices
			tyre_indices = [unique_tyre_types.index(tyre) for tyre in tyres]
			
			return {
				"pit1_lap": pit_laps[0],
				"pit1_tyre_idx": tyre_indices[0],
				"pit2_lap": pit_laps[1],
				"pit2_tyre_idx": tyre_indices[1],
				"pit3_lap": pit_laps[2],
				"pit3_tyre_idx": tyre_indices[2],
			}
		
		initial_params = map_strategy_to_params(initial_strategy)
		
		# Step 6: Define the objective function for Bayesian Optimization
		def objective_function(pit1_lap, pit1_tyre_idx, pit2_lap, pit2_tyre_idx, pit3_lap, pit3_tyre_idx):
			# Map tyre indices to actual tyre types
			starting_tyre = unique_tyre_types[int(pit1_tyre_idx) % len(unique_tyre_types)]
			pit1_tyre = unique_tyre_types[int(pit1_tyre_idx) % len(unique_tyre_types)]
			pit2_tyre = unique_tyre_types[int(pit2_tyre_idx) % len(unique_tyre_types)]
			pit3_tyre = unique_tyre_types[int(pit3_tyre_idx) % len(unique_tyre_types)]
			
			# Ensure pit laps are unique and within valid range
			pit_laps = sorted(set([int(pit1_lap), int(pit2_lap), int(pit3_lap)]))
			tyres = [pit1_tyre, pit2_tyre, pit3_tyre]
			
			# Filter out invalid pit laps (e.g., out of bounds or overlapping)
			valid_pit_laps = []
			valid_tyres = []
			for lap, tyre in zip(pit_laps, tyres):
				if 2 <= lap < self.race_data.max_laps:  # Only include valid pit laps
					valid_pit_laps.append(lap)
					valid_tyres.append(tyre)
			
			# Construct the strategy dictionary
			strategy = {1: starting_tyre}  # Starting tyre
			for lap, tyre in zip(valid_pit_laps, valid_tyres):
				strategy[lap] = tyre
			
			# Enforce F1 rule: At least two distinct tyre types must be used
			tyre_types_used = set(strategy.values())
			if len(tyre_types_used) < 2:
				return -20.0  # Penalize heavily for violating the rule
			
			# Evaluate the strategy using the race simulation
			try:
				sim = RaceSimulation(self.race_data, self.overtake_model, given_driver=given_driver, simulated_strategy=strategy)
				sim_data = sim.simulate()
				sim_df = pd.DataFrame(sim_data)
				final_position = sim_df[sim_df["driver_number"] == given_driver]["position"].iloc[-1]
				if final_position == 1:  # Stop if the driver achieves position 1
			   		raise StopIteration(f"Optimization stopped early: Achieved position {final_position}")
			
				return -final_position  # Negative because BayesianOptimization maximizes by default
			

				
			except Exception as e:
				print(f"Error during simulation: {e}")
				return -20.0  # Return worst position in case of errors
		
		# Step 7: Initialize the Bayesian Optimizer
		optimizer = BayesianOptimization(
			f=objective_function,
			pbounds={
				**pbounds,
				"pit1_tyre_idx": (0, len(unique_tyre_types) - 1),  # Index for tyre type selection
				"pit2_tyre_idx": (0, len(unique_tyre_types) - 1),  # Index for tyre type selection
				"pit3_tyre_idx": (0, len(unique_tyre_types) - 1),  # Index for tyre type selection
			},
			verbose=2,
			random_state=42
		)
		
		# Step 8: Add the initial strategy as an initial point
		optimizer.probe(
			params=initial_params,
			lazy=True  # Evaluate the initial point lazily
		)
		
		# Step 9: Perform the optimization
		optimizer.maximize(init_points=5, n_iter=max_iterations)
		
		# Step 10: Extract the best strategy from the optimizer
		best_params = optimizer.max["params"]
		best_starting_tyre = unique_tyre_types[int(best_params["pit1_tyre_idx"]) % len(unique_tyre_types)]
		pit_laps = sorted(set([int(best_params["pit1_lap"]), int(best_params["pit2_lap"]), int(best_params["pit3_lap"])]))
		tyres = [
			unique_tyre_types[int(best_params["pit1_tyre_idx"]) % len(unique_tyre_types)],
			unique_tyre_types[int(best_params["pit2_tyre_idx"]) % len(unique_tyre_types)],
			unique_tyre_types[int(best_params["pit3_tyre_idx"]) % len(unique_tyre_types)]
		]
		
		# Construct the best strategy dictionary
		best_strategy = {1: best_starting_tyre}
		for lap, tyre in zip(pit_laps, tyres):
			if 2 <= lap < self.race_data.max_laps:  # Only include valid pit laps
				best_strategy[lap] = tyre
		
		# Validate the best strategy (should already satisfy the rule due to enforcement)
		tyre_types_used = set(best_strategy.values())
		if len(tyre_types_used) < 2:
			raise ValueError("Best strategy violates F1 rules: fewer than 2 tyre types used.")
		
		# Evaluate the best strategy to get the finishing position
		sim = RaceSimulation(self.race_data, self.overtake_model, given_driver=given_driver, simulated_strategy=best_strategy)
		sim_data = sim.simulate()
		sim_df = pd.DataFrame(sim_data)
		best_position = sim_df[sim_df["driver_number"] == given_driver]["position"].iloc[-1]
		
		return best_strategy, best_position