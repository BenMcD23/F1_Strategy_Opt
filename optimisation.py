from bayes_opt import BayesianOptimization

from race_sim import RaceSimulator

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

		self.__initial_strategy = self.race_data.driver_strategies[given_driver]
		self.__given_driver = given_driver
		self.__unique_tyre_types = sorted(self.race_data.get_unique_tyre_types())

	def __map_to_nearest_tyre(self, value):
		# Find the closest tyre type by minimizing the absolute difference
		closest_tyre = min(self.__unique_tyre_types, key=lambda tyre: abs(tyre - value))
		
		return closest_tyre

	def bayesian_optimisation(self, max_iterations=50):
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
			start_tyre = self.__map_to_nearest_tyre(start_tyre)
			pit1_tyre = self.__map_to_nearest_tyre(pit1_tyre)
			pit2_tyre = self.__map_to_nearest_tyre(pit2_tyre)
			pit3_tyre = self.__map_to_nearest_tyre(pit3_tyre)
			
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
				sim = RaceSimulator(self.race_data, self.overtake_model, given_driver=self.__given_driver, simulated_strategy=strategy)
				sim_data = sim.simulate()
				final_position = next(d["position"] for d in sim_data if d["driver_number"] == self.__given_driver)
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
		
		optimiser.maximize(init_points=5, n_iter=max_iterations)
		
		# Get the top 10 strategies from the runs
		top_10_runs = sorted(optimiser.res, key=lambda x: x["target"], reverse=True)[:10]
		top_10_strategies = []
		for run in top_10_runs:
			params = run["params"]
			start_tyre = self.__map_to_nearest_tyre(params["start_tyre"])
			num_pit_stops = int(params["num_pit_stops"])
			pit_laps = sorted([int(params["pit1_lap"]), int(params["pit2_lap"]), int(params["pit3_lap"])])[:num_pit_stops]
			pit_tyres = [
				self.__map_to_nearest_tyre(params["pit1_tyre"]),
				self.__map_to_nearest_tyre(params["pit2_tyre"]),
				self.__map_to_nearest_tyre(params["pit3_tyre"])
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
		__unique_tyre_types = sorted(self.race_data.get_unique_tyre_types())
		
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
				random.choice(__unique_tyre_types),  # Starting tyre
				random.randint(2, self.race_data.max_laps - 1),  # Pit1 lap
				random.choice(__unique_tyre_types),  # Pit1 tyre
				random.randint(2, self.race_data.max_laps - 1),  # Pit2 lap
				random.choice(__unique_tyre_types),  # Pit2 tyre
				random.randint(2, self.race_data.max_laps - 1),  # Pit3 lap
				random.choice(__unique_tyre_types),  # Pit3 tyre
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
			sim = RaceSimulator(self.race_data, self.overtake_model, given_driver=self.__given_driver, simulated_strategy=strategy)
			sim_data = sim.simulate()
			final_position = next(d["position"] for d in sim_data if d["driver_number"] == self.__given_driver)
			
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
					individual[i] = random.choice(__unique_tyre_types)
				else:  # Mutate pit stop laps (odd indices)
					individual[i] = random.randint(2, self.race_data.max_laps - 1)
			return individual,
		
		toolbox.register("mutate", mutate_strategy)
		toolbox.register("select", tools.selTournament, tournsize=3)
		

		# Fill the rest of the population with random strategies
		random_population = toolbox.population(n=population_size)

		population = random_population
		
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
			sim = RaceSimulator(self.race_data, self.overtake_model, given_driver=self.__given_driver, simulated_strategy=strategy)
			sim_data = sim.simulate()
			final_position = next(d["position"] for d in sim_data if d["driver_number"] == self.__given_driver)
			
			# Store the strategy and its final position
			top_strategies.append({
				"strategy": strategy,
				"final_position": final_position
			})
		
		return top_strategies
	


	def powell(self, max_iterations=50, __initial_strategy=None):
		# Use the provided initial strategy or fall back to the default one
		if __initial_strategy is None:
			__initial_strategy = self.__initial_strategy

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
			start_tyre = self.__map_to_nearest_tyre(params[0])
			num_pit_stops = int(params[1])
			pit_laps = sorted([int(params[2]), int(params[3]), int(params[4])])[:num_pit_stops]
			pit_tyres = [self.__map_to_nearest_tyre(params[5]), self.__map_to_nearest_tyre(params[6]), self.__map_to_nearest_tyre(params[7])][:num_pit_stops]

			# Construct the strategy dictionary
			strategy = {1: start_tyre}  # Starting tyre
			for lap, tyre in zip(pit_laps, pit_tyres):
				if lap > 1 and tyre != 0:  # Ensure valid pit stops
					strategy[lap] = tyre

			# Simulate the race with the strategy
			sim = RaceSimulator(self.race_data, self.overtake_model, given_driver=self.__given_driver, simulated_strategy=strategy)
			sim_data = sim.simulate()
			final_position = next(d["position"] for d in sim_data if d["driver_number"] == self.__given_driver)

			# add the strategy and the position
			evaluated_strategies.append({
				"strategy": strategy,
				"final_position": final_position
			})

			return final_position  # Minimize finishing position

		# Initial guess based on the provided or default initial strategy
		sorted_laps = sorted([lap for lap in __initial_strategy.keys() if lap > 1])  # Sort pit stop laps
		initial_guess = [
			__initial_strategy[1],  # Starting tyre
			len(sorted_laps),     # Number of pit stops
			sorted_laps[0] if len(sorted_laps) > 0 else 2,  # 1st pit stop lap
			sorted_laps[1] if len(sorted_laps) > 1 else 2,  # 2nd pit stop lap
			sorted_laps[2] if len(sorted_laps) > 2 else 2,  # 3rd pit stop lap
			__initial_strategy[sorted_laps[0]] if len(sorted_laps) > 0 else 1,  # 1st pit stop tyre
			__initial_strategy[sorted_laps[1]] if len(sorted_laps) > 1 else 1,  # 2nd pit stop tyre
			__initial_strategy[sorted_laps[2]] if len(sorted_laps) > 2 else 1,  # 3rd pit stop tyre
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

		return top_strategies