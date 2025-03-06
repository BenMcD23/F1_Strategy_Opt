from bayes_opt import BayesianOptimization

from race_sim import RaceSimulation

from deap import base, creator, tools, algorithms
import random

from scipy.optimize import dual_annealing
from scipy.optimize import minimize

import math
class Optimisation:
	def __init__(self, race_data, overtake_model, given_driver):
		"""
		Initialize the optimisation class with race data and an overtaking model.
		"""
		self.race_data = race_data
		self.overtake_model = overtake_model

		self.initial_strategy = self.race_data.driver_strategies[given_driver]
		self.given_driver = given_driver
		self.unique_tyre_types = sorted(self.race_data.get_unique_tyre_types())

	def _map_to_nearest_tyre(self, value):
		# Find the closest tyre type by minimizing the absolute difference
		closest_tyre = min(self.unique_tyre_types, key=lambda tyre: abs(tyre - value))
		
		return closest_tyre

	def bayesian_strategy_optimisation(self, max_iterations=50):
		"""
		Perform Bayesian optimisation to find the most optimal strategy for a given driver.
		Explores the starting tyre and enforces F1 rules: at least two different tyre compounds must be used.
		Allows up to 3 pit stops but does not require all 3.
		"""

		
		# parameter bounds for Bayesian optimisation
		pbounds = {
			"start_tyre": (1, 3),  # Index for starting tyre selection
			"num_pit_stops": (1, 3),  # Number of pit stops (1 to 3)
			"pit1_lap": (2, self.race_data.max_laps - 1),  # First pit stop lap
			"pit2_lap": (2, self.race_data.max_laps - 1),  # Second pit stop lap
			"pit3_lap": (2, self.race_data.max_laps - 1),  # Third pit stop lap
			"pit1_tyre": (1, 3),
			"pit2_tyre": (1, 3),
			"pit3_tyre": (1, 3),
		}

		# Define the objective function for Bayesian optimisation
		def objective_function(start_tyre, num_pit_stops, pit1_lap, pit2_lap, pit3_lap, pit1_tyre, pit2_tyre, pit3_tyre):
			# Map continuous values to discrete tyre types
			start_tyre = self._map_to_nearest_tyre(start_tyre)
			pit1_tyre = self._map_to_nearest_tyre(pit1_tyre)
			pit2_tyre = self._map_to_nearest_tyre(pit2_tyre)
			pit3_tyre = self._map_to_nearest_tyre(pit3_tyre)
			
			# Determine the number of pit stops
			num_pit_stops = int(num_pit_stops)
			
			# Collect pit stop laps and tyre types
			pit_laps = sorted([int(pit1_lap), int(pit2_lap), int(pit3_lap)])[:num_pit_stops]
			pit_tyres = [pit1_tyre, pit2_tyre, pit3_tyre][:num_pit_stops]
			
			# Construct the strategy dictionary
			strategy = {1: start_tyre}  # Starting tyre
			for lap, tyre in zip(pit_laps, pit_tyres):
				if 2 <= lap < self.race_data.max_laps:  # Can't pit after race is finished
					strategy[lap] = tyre
			
			# Enforce F1 rule - At least two distinct tyre types must be used
			tyre_types_used = set(strategy.values())
			if len(tyre_types_used) < 2:
				return -20.0  # Penalize strategies that violate rules
			
			# Evaluate the strategy using the race simulation
			try:
				sim = RaceSimulation(self.race_data, self.overtake_model, given_driver=self.given_driver, simulated_strategy=strategy)
				sim_data = sim.simulate()
				final_position = next(d["position"] for d in sim_data if d["driver_number"] == self.given_driver)
				return -final_position  # Negative because Bayesianoptimisation maximizes by default
			except Exception as e:
				print(f"Error during simulation: {e}")
				return -20.0  # Penalize errors
			
		# Initialize the Bayesian optimiser
		optimiser = BayesianOptimization(
			f=objective_function,
			pbounds=pbounds,
			verbose=2,
			random_state=42
		)
		

		# Sort the keys of the initial strategy by lap number
		sorted_laps = sorted(self.initial_strategy.keys())

		# Construct the initial parameters dictionary
		initial_params = {
			"start_tyre": self.initial_strategy[1],  # Starting tyre (lap 1)
			"num_pit_stops": len(self.initial_strategy) - 1,  # Number of pit stops
			"pit1_lap": sorted_laps[1] if len(sorted_laps) > 1 else 0,  # First pit stop lap
			"pit2_lap": sorted_laps[2] if len(sorted_laps) > 2 else 0,  # Second pit stop lap
			"pit3_lap": sorted_laps[3] if len(sorted_laps) > 3 else 0,  # Third pit stop lap
			"pit1_tyre": self.initial_strategy[sorted_laps[1]] if len(sorted_laps) > 1 else 0,  # First pit stop tyre
			"pit2_tyre": self.initial_strategy[sorted_laps[2]] if len(sorted_laps) > 2 else 0,  # Second pit stop tyre
			"pit3_tyre": self.initial_strategy[sorted_laps[3]] if len(sorted_laps) > 3 else 0,  # Third pit stop tyre
		}

		optimiser.probe(params=initial_params, lazy=True)
		
		optimiser.maximize(init_points=5, n_iter=max_iterations)
		

		# Get the top 10 strategies from the runs
		top_10_runs = sorted(optimiser.res, key=lambda x: x["target"], reverse=True)[:10]
		top_10_strategies = []
		for run in top_10_runs:
			params = run["params"]
			start_tyre = self._map_to_nearest_tyre(params["start_tyre"])
			num_pit_stops = int(params["num_pit_stops"])
			pit_laps = sorted([int(params["pit1_lap"]), int(params["pit2_lap"]), int(params["pit3_lap"])])[:num_pit_stops]
			pit_tyres = [
				self._map_to_nearest_tyre(params["pit1_tyre"]),
				self._map_to_nearest_tyre(params["pit2_tyre"]),
				self._map_to_nearest_tyre(params["pit3_tyre"])
			][:num_pit_stops]
			strategy = {1: start_tyre}
			for lap, tyre in zip(pit_laps, pit_tyres):
				if 2 <= lap < self.race_data.max_laps:
					strategy[lap] = tyre
			top_10_strategies.append({
				"strategy": strategy,
				"position": -run["target"]  # Convert back to positive position
			})

		return top_10_strategies
	


	def genetic_algorithm_optimisation(self, population_size=50, generations=50):
		"""
		Perform Genetic Algorithm optimisation to find the top 10 strategies for a given driver.
		"""
		# Unique tyre types
		unique_tyre_types = sorted(self.race_data.get_unique_tyre_types())
		
		# Define the fitness and individual classes
		creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize final position
		creator.create("Individual", list, fitness=creator.FitnessMin)
		
		toolbox = base.Toolbox()
		
		# Define strategy representation
		def create_strategy():
			"""
			Create a random strategy as an individual.
			[starting_tyre, pit1_lap, pit1_tyre, pit2_lap, pit2_tyre, pit3_lap, pit3_tyre]
			"""
			return [
				random.choice(unique_tyre_types),  # Starting tyre
				random.randint(2, self.race_data.max_laps - 1),  # Pit1 lap
				random.choice(unique_tyre_types),  # Pit1 tyre
				random.randint(2, self.race_data.max_laps - 1),  # Pit2 lap
				random.choice(unique_tyre_types),  # Pit2 tyre
				random.randint(2, self.race_data.max_laps - 1),  # Pit3 lap
				random.choice(unique_tyre_types),  # Pit3 tyre
			]
		
		toolbox.register("individual", tools.initIterate, creator.Individual, create_strategy)
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		
		# Fitness function
		def evaluate(individual):
			"""
			Evaluate the fitness of an individual strategy.
			"""
			starting_tyre, pit1_lap, pit1_tyre, pit2_lap, pit2_tyre, pit3_lap, pit3_tyre = individual
			
			# Construct the strategy dictionary
			strategy = {
				1: starting_tyre,
				pit1_lap: pit1_tyre,
				pit2_lap: pit2_tyre,
				pit3_lap: pit3_tyre,
			}
			
			# Extract the unique tyre types used in the strategy
			used_tyres = {tyre for lap, tyre in strategy.items() if lap > 0 and tyre != 0}
			
			# Penalize strategies that use fewer than two unique tyre types
			if len(used_tyres) < 2:
				return (20,)  # Worst position as penalty

			# Simulate the race with the strategy
			sim = RaceSimulation(self.race_data, self.overtake_model, given_driver=self.given_driver, simulated_strategy=strategy)
			sim_data = sim.simulate()
			final_position = next(d["position"] for d in sim_data if d["driver_number"] == self.given_driver)
			
			return (final_position,)  # Return as a tuple for DEAP
		
		toolbox.register("evaluate", evaluate)
		toolbox.register("mate", tools.cxTwoPoint)  # Crossover
		
		# Custom mutation function
		def mutate_strategy(individual):
			"""
			Custom mutation function to ensure valid tyre types.
			"""
			for i in range(len(individual)):
				if i % 2 == 0:  # Mutate tyre types (even indices)
					individual[i] = random.choice(unique_tyre_types)
				else:  # Mutate pit stop laps (odd indices)
					individual[i] = random.randint(2, self.race_data.max_laps - 1)
			return individual,
		
		toolbox.register("mutate", mutate_strategy)
		toolbox.register("select", tools.selTournament, tournsize=3)
		
		# Parse the initial strategy using initial_params
		sorted_laps = sorted([lap for lap in self.initial_strategy.keys() if lap > 1])  # Sort pit stop laps
		
		initial_params = {
			"start_tyre": self.initial_strategy[1],  # Starting tyre (lap 1)
			"num_pit_stops": len(sorted_laps),  # Number of pit stops
			"pit1_lap": sorted_laps[0] if len(sorted_laps) > 0 else 0,  # First pit stop lap
			"pit2_lap": sorted_laps[1] if len(sorted_laps) > 1 else 0,  # Second pit stop lap
			"pit3_lap": sorted_laps[2] if len(sorted_laps) > 2 else 0,  # Third pit stop lap
			"pit1_tyre": self.initial_strategy[sorted_laps[0]] if len(sorted_laps) > 0 else 0,  # First pit stop tyre
			"pit2_tyre": self.initial_strategy[sorted_laps[1]] if len(sorted_laps) > 1 else 0,  # Second pit stop tyre
			"pit3_tyre": self.initial_strategy[sorted_laps[2]] if len(sorted_laps) > 2 else 0,  # Third pit stop tyre
		}
		
		def create_initial_strategy():
			"""
			Create an individual based on the initial_params with slight variations.
			"""
			# Extract the initial strategy parameters
			start_tyre = initial_params["start_tyre"]
			pit1_lap = initial_params["pit1_lap"]
			pit1_tyre = initial_params["pit1_tyre"]
			pit2_lap = initial_params["pit2_lap"]
			pit2_tyre = initial_params["pit2_tyre"]
			pit3_lap = initial_params["pit3_lap"]
			pit3_tyre = initial_params["pit3_tyre"]
			
			# Add slight variations to the initial strategy
			return [
				random.choice(unique_tyre_types) if random.random() < 0.2 else start_tyre,  # Starting tyre
				random.randint(max(2, pit1_lap - 5), min(pit1_lap + 5, self.race_data.max_laps - 1)) if pit1_lap > 0 else 0,  # Pit1 lap
				random.choice(unique_tyre_types) if random.random() < 0.2 else pit1_tyre if pit1_tyre != 0 else random.choice(unique_tyre_types),  # Pit1 tyre
				random.randint(max(2, pit2_lap - 5), min(pit2_lap + 5, self.race_data.max_laps - 1)) if pit2_lap > 0 else 0,  # Pit2 lap
				random.choice(unique_tyre_types) if random.random() < 0.2 else pit2_tyre if pit2_tyre != 0 else random.choice(unique_tyre_types),  # Pit2 tyre
				random.randint(max(2, pit3_lap - 5), min(pit3_lap + 5, self.race_data.max_laps - 1)) if pit3_lap > 0 else 0,  # Pit3 lap
				random.choice(unique_tyre_types) if random.random() < 0.2 else pit3_tyre if pit3_tyre != 0 else random.choice(unique_tyre_types),  # Pit3 tyre
			]
		
		# Initialize part of the population with the initial strategy
		initial_population_size = int(population_size * 0.2)  # 20% of the population
		initial_population = [creator.Individual(create_initial_strategy()) for _ in range(initial_population_size)]
		
		# Fill the rest of the population with random strategies
		random_population = toolbox.population(n=population_size - initial_population_size)
		
		# Combine the initial and random populations
		population = initial_population + random_population
		
		# Use Hall of Fame to store the top 10 strategies
		hof = tools.HallOfFame(10)
		
		# Run the genetic algorithm
		print("Running Genetic Algorithm...")
		stats = tools.Statistics(lambda ind: ind.fitness.values)
		stats.register("avg", lambda x: sum(val[0] for val in x) / len(x))
		stats.register("min", lambda x: min(val[0] for val in x))
		stats.register("max", lambda x: max(val[0] for val in x))
		
		population, logbook = algorithms.eaSimple(
			population,
			toolbox,
			cxpb=0.7,  # Crossover probability
			mutpb=0.2,  # Mutation probability
			ngen=generations,
			stats=stats,
			halloffame=hof,  # Pass the Hall of Fame here
			verbose=True
		)
		
		# Extract the top 10 strategies from the Hall of Fame
		top_strategies = []
		for individual in hof:
			starting_tyre, pit1_lap, pit1_tyre, pit2_lap, pit2_tyre, pit3_lap, pit3_tyre = individual
			
			strategy = {
				1: starting_tyre,
				pit1_lap: pit1_tyre,
				pit2_lap: pit2_tyre,
				pit3_lap: pit3_tyre,
			}
			
			# Simulate the race with the strategy to get the final position - HOF doesn't save result
			sim = RaceSimulation(self.race_data, self.overtake_model, given_driver=self.given_driver, simulated_strategy=strategy)
			sim_data = sim.simulate()
			final_position = next(d["position"] for d in sim_data if d["driver_number"] == self.given_driver)
			
			# Store the strategy and its final position
			top_strategies.append({
				"strategy": strategy,
				"final_position": final_position
			})
		
		return top_strategies
	

	def simulated_annealing_optimisation(self, max_iterations=50):
		# Define bounds for the parameters
		unique_tyre_types = sorted(self.race_data.get_unique_tyre_types())
		num_tyres = len(unique_tyre_types)

		bounds = {
			"start_tyre": (1, num_tyres),  # Index for starting tyre selection
			"num_pit_stops": (1, 3),       # Number of pit stops (1 to 3)
			"pit1_lap": (2, self.race_data.max_laps - 1),  # First pit stop lap
			"pit2_lap": (2, self.race_data.max_laps - 1),  # Second pit stop lap
			"pit3_lap": (2, self.race_data.max_laps - 1),  # Third pit stop lap
			"pit1_tyre": (1, num_tyres),   # Index for first pit stop tyre
			"pit2_tyre": (1, num_tyres),   # Index for second pit stop tyre
			"pit3_tyre": (1, num_tyres),   # Index for third pit stop tyre
		}

		# List to log all evaluated strategies and their performance
		evaluated_strategies = []

		# Objective function
		def objective_function(params):
			# Extract parameters
			start_tyre = self._map_to_nearest_tyre(params["start_tyre"])
			num_pit_stops = int(params["num_pit_stops"])
			pit1_lap = int(params["pit1_lap"])
			pit2_lap = int(params["pit2_lap"])
			pit3_lap = int(params["pit3_lap"])
			pit1_tyre = self._map_to_nearest_tyre(params["pit1_tyre"])
			pit2_tyre = self._map_to_nearest_tyre(params["pit2_tyre"])
			pit3_tyre = self._map_to_nearest_tyre(params["pit3_tyre"])

			# Collect pit stop laps and tyre types
			pit_laps = sorted([pit1_lap, pit2_lap, pit3_lap])[:num_pit_stops]
			pit_tyres = [pit1_tyre, pit2_tyre, pit3_tyre][:num_pit_stops]

			# Construct the strategy dictionary
			strategy = {1: start_tyre}  # Starting tyre
			for lap, tyre in zip(pit_laps, pit_tyres):
				if 2 <= lap < self.race_data.max_laps:  # Only include valid pit laps
					strategy[lap] = tyre

			# Enforce F1 rule: At least two distinct tyre types must be used
			tyre_types_used = set(strategy.values())
			if len(tyre_types_used) < 2:
				final_position = float('inf')  # Penalize strategies that violate F1 rules
			else:
				# Evaluate the strategy using the race simulation
				try:
					sim = RaceSimulation(self.race_data, self.overtake_model, given_driver=self.given_driver, simulated_strategy=strategy)
					sim_data = sim.simulate()
					final_position = next(d["position"] for d in sim_data if d["driver_number"] == self.given_driver)
				except Exception as e:
					print(f"Error during simulation: {e}")
					final_position = float('inf')  # Penalize errors

			# Log the strategy and its performance
			evaluated_strategies.append({
				"strategy": strategy,
				"position": final_position
			})

			return final_position  # Minimize the final position

		# Convert bounds to a list for dual_annealing
		bounds_list = list(bounds.values())

		# Convert the initial strategy into parameter values
		start_tyre = unique_tyre_types.index(self.initial_strategy[1]) + 1  # Starting tyre index
		sorted_laps = sorted([lap for lap in self.initial_strategy.keys() if lap > 1])
		num_pit_stops = len(sorted_laps)
		pit1_lap = sorted_laps[0] if num_pit_stops >= 1 else 2
		pit2_lap = sorted_laps[1] if num_pit_stops >= 2 else 2
		pit3_lap = sorted_laps[2] if num_pit_stops >= 3 else 2
		pit1_tyre = unique_tyre_types.index(self.initial_strategy[sorted_laps[0]]) + 1 if num_pit_stops >= 1 else 1
		pit2_tyre = unique_tyre_types.index(self.initial_strategy[sorted_laps[1]]) + 1 if num_pit_stops >= 2 else 1
		pit3_tyre = unique_tyre_types.index(self.initial_strategy[sorted_laps[2]]) + 1 if num_pit_stops >= 3 else 1

		# Create the initial guess
		x0 = [
			start_tyre,
			num_pit_stops,
			pit1_lap,
			pit1_tyre,
			pit2_lap,
			pit2_tyre,
			pit3_lap,
			pit3_tyre
		]
	
		# Run simulated annealing
		result = dual_annealing(
			lambda x: objective_function(dict(zip(bounds.keys(), x))),
			bounds_list,
			x0=x0,  # Use the initial strategy as the starting point
			maxiter=max_iterations
		)

		unique_strategies = []
		seen_strategies = set()

		for strategy_data in evaluated_strategies:
			strategy_tuple = tuple(sorted(strategy_data["strategy"].items()))  # Convert strategy dict to a hashable tuple
			if strategy_tuple not in seen_strategies:
				seen_strategies.add(strategy_tuple)
				unique_strategies.append(strategy_data)

		# Sort and extract the top 10 unique strategies
		unique_strategies.sort(key=lambda x: x["position"])
		top_10_strategies = unique_strategies[:10]

		return top_10_strategies

	


	def powell(self, max_iterations=50, initial_strategy=None):
		# Use the provided initial strategy or fall back to the default one
		if initial_strategy is None:
			initial_strategy = self.initial_strategy

		bounds = [
			(1, 3),  # starting tyre selection
			(1, 3),  # Number of pit stops (1 to 3)
			(2, self.race_data.max_laps - 1),  # 1st pit stop lap
			(2, self.race_data.max_laps - 1),  # 2md pit stop lap
			(2, self.race_data.max_laps - 1),  # 3rd pit stop lap
			(1, 3),  # 1st pit stop tyre
			(1, 3),  # 2nd pit stop tyre
			(1, 3),  # 3rd pit stop tyre
		]

		# List to store all evaluated strategies and their final positions
		evaluated_strategies = []

		# Define the objective function
		def objective_function(params):
			# Map continuous values to discrete tyre types
			start_tyre = self._map_to_nearest_tyre(params[0])
			num_pit_stops = int(params[1])
			pit_laps = sorted([int(params[2]), int(params[3]), int(params[4])])[:num_pit_stops]
			pit_tyres = [self._map_to_nearest_tyre(params[5]), self._map_to_nearest_tyre(params[6]), self._map_to_nearest_tyre(params[7])][:num_pit_stops]

			# Construct the strategy dictionary
			strategy = {1: start_tyre}  # Starting tyre
			for lap, tyre in zip(pit_laps, pit_tyres):
				if lap > 1 and tyre != 0:  # Ensure valid pit stops
					strategy[lap] = tyre

			# Simulate the race with the strategy
			sim = RaceSimulation(self.race_data, self.overtake_model, given_driver=self.given_driver, simulated_strategy=strategy)
			sim_data = sim.simulate()
			final_position = next(d["position"] for d in sim_data if d["driver_number"] == self.given_driver)

			# add the strategy and the position
			evaluated_strategies.append({
				"strategy": strategy,
				"final_position": final_position
			})

			return final_position  # Minimize finishing position

		# Initial guess based on the provided or default initial strategy
		sorted_laps = sorted([lap for lap in initial_strategy.keys() if lap > 1])  # Sort pit stop laps
		initial_guess = [
			initial_strategy[1],  # Starting tyre
			len(sorted_laps),     # Number of pit stops
			sorted_laps[0] if len(sorted_laps) > 0 else 2,  # 1st pit stop lap
			sorted_laps[1] if len(sorted_laps) > 1 else 2,  # 2nd pit stop lap
			sorted_laps[2] if len(sorted_laps) > 2 else 2,  # 3rd pit stop lap
			initial_strategy[sorted_laps[0]] if len(sorted_laps) > 0 else 1,  # 1st pit stop tyre
			initial_strategy[sorted_laps[1]] if len(sorted_laps) > 1 else 1,  # 2nd pit stop tyre
			initial_strategy[sorted_laps[2]] if len(sorted_laps) > 2 else 1,  # 3rd pit stop tyre
		]

		result = minimize(
			fun=objective_function,
			x0=initial_guess,
			bounds=bounds,
			method="Powell",  # Similar to Hooke and Jeeves
			options={"maxiter": max_iterations}
		)

		# Sort the evaluated strategies by final position (lower is better)
		top_strategies = sorted(evaluated_strategies, key=lambda x: x["final_position"])[:10]

		return {
			"top_strategies": top_strategies
		}
		


	def monte_carlo_tree_search(self, max_iterations=100, exploration_weight=1.414):
		"""
		Perform Monte Carlo Tree Search (MCTS) to find optimal strategies.
		Enforces F1 rules: at least two distinct tyre compounds must be used.
		Returns the top 10 strategies based on simulated performance.
		"""
		class Node:
			def __init__(self, strategy, parent=None):
				self.strategy = strategy.copy()
				self.parent = parent
				self.children = []
				self.visits = 0
				self.total_score = 0  # Higher score is better

			def ucb(self, exploration_weight):
				if self.visits == 0:
					return float('inf') if self.parent else float('inf')
				exploitation = self.total_score / self.visits
				if self.parent:
					exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
				else:
					exploration = exploration_weight * math.sqrt(math.log(self.visits + 1) / (self.visits + 1))
				return exploitation + exploration

		def _is_valid_strategy(strategy):
			"""
			Check if the strategy uses at least two distinct tyre compounds.
			"""
			tyre_types_used = set(strategy.values())
			return len(tyre_types_used) >= 2

		def _generate_child_strategies(current_strategy):
			"""
			Generate valid child strategies that enforce F1 rules.
			"""
			child_strategies = []
			current_pits = sorted([lap for lap in current_strategy.keys() if lap != 1])
			num_pits = len(current_pits)
			
			# Add a new pit stop if possible
			if num_pits < 3:
				last_pit = current_pits[-1] if num_pits > 0 else 1
				min_lap = last_pit + 1
				max_lap = self.race_data.max_laps - 1
				if min_lap <= max_lap:
					new_lap = random.randint(min_lap, max_lap)
					available_tyres = [tyre for tyre in self.unique_tyre_types if tyre != current_strategy[last_pit]]
					if not available_tyres:
						available_tyres = self.unique_tyre_types
					new_tyre = random.choice(available_tyres)
					new_strat = current_strategy.copy()
					new_strat[new_lap] = new_tyre
					if _is_valid_strategy(new_strat):
						child_strategies.append(new_strat)
			
			# Modify existing pit stops
			for lap in list(current_strategy.keys()):
				if lap == 1:
					continue  # Skip starting tyre
				# Change lap
				new_lap = lap + random.randint(-5, 5)
				new_lap = max(2, min(new_lap, self.race_data.max_laps - 1))
				if new_lap not in current_strategy or new_lap == lap:
					new_strat = current_strategy.copy()
					del new_strat[lap]
					new_strat[new_lap] = current_strategy[lap]
					if _is_valid_strategy(new_strat):
						child_strategies.append(new_strat)
				# Change tyre
				current_tyre = current_strategy[lap]
				available_tyres = [tyre for tyre in self.unique_tyre_types if tyre != current_tyre]
				if available_tyres:
					new_tyre = random.choice(available_tyres)
					new_strat = current_strategy.copy()
					new_strat[lap] = new_tyre
					if _is_valid_strategy(new_strat):
						child_strategies.append(new_strat)
			
			# Remove a pit stop
			if current_pits:
				lap_to_remove = random.choice(current_pits)
				new_strat = current_strategy.copy()
				del new_strat[lap_to_remove]
				if _is_valid_strategy(new_strat):
					child_strategies.append(new_strat)
			
			# Deduplicate
			unique = []
			seen = set()
			for strat in child_strategies:
				key = frozenset(strat.items())
				if key not in seen:
					seen.add(key)
					unique.append(strat)
			return unique

		root = Node(self.initial_strategy)

		for _ in range(max_iterations):
			node = root
			path = [node]

			# Selection
			while node.children:
				node = max(node.children, key=lambda n: n.ucb(exploration_weight))
				path.append(node)

			# Expansion
			if node.visits > 0 or node == root:
				children_strategies = _generate_child_strategies(node.strategy)
				for strat in children_strategies:
					child = Node(strat, parent=node)
					node.children.append(child)
				if node.children:
					node = random.choice(node.children)
					path.append(node)

			# Simulation
			try:
				sim = RaceSimulation(self.race_data, self.overtake_model, self.given_driver, node.strategy)
				sim_data = sim.simulate()
				position = next(d["position"] for d in sim_data if d["driver_number"] == self.given_driver)
				score = -position  # Higher score is better
			except Exception as e:
				score = -20  # Penalize invalid

			# Backpropagation
			for n in path:
				n.visits += 1
				n.total_score += score

		# Collect all strategies
		all_strategies = []
		stack = [root]
		while stack:
			current = stack.pop()
			if current.visits > 0:
				all_strategies.append((current.strategy, current.total_score / current.visits))
			stack.extend(current.children)

		# Deduplicate and sort
		seen = set()
		unique_strategies = []
		for strategy, score in sorted(all_strategies, key=lambda x: x[1], reverse=True):
			key = frozenset(strategy.items())
			if key not in seen:
				seen.add(key)
				unique_strategies.append(strategy)
			if len(unique_strategies) >= 10:
				break

		# Evaluate final positions
		top_10 = []
		for strat in unique_strategies[:10]:
			try:
				sim = RaceSimulation(self.race_data, self.overtake_model, self.given_driver, strat)
				sim_data = sim.simulate()
				pos = next(d["position"] for d in sim_data if d["driver_number"] == self.given_driver)
			except:
				pos = 20
			top_10.append({"strategy": strat, "position": pos})

		# Sort by position
		top_10.sort(key=lambda x: x["position"])
		return top_10[:10]