from race_sim import RaceSimulator

from bayes_opt import BayesianOptimization

from deap import base, creator, tools, algorithms
import random

class Optimisation:
	def __init__(self, race_data, overtake_model, given_driver):
		"""
		Initialize the optimisation class with race data and an overtaking model.
		"""
		self.__race_data = race_data
		self.__overtake_model = overtake_model

		self.__given_driver = given_driver
		self.__unique_tyre_types = sorted(self.__race_data.get_unique_tyre_types())

	def get_actual_strategy(self):

		actual_strat = self.__race_data.extract_driver_strategy(self.__given_driver)
		actual_finishing_pos = self.__race_data.get_driver_finishing_position(self.__given_driver)

		sim = RaceSimulator(self.__race_data, self.__overtake_model, given_driver=self.__given_driver, simulated_strategy=actual_strat)
		sim_data = sim.simulate()
		sim_final_position = next(d["position"] for d in sim_data if d["driver_number"] == self.__given_driver)

		return {
			"actual_strat": actual_strat,
			"actual_finishing_pos": actual_finishing_pos,
			"simualted_finishing_pos": sim_final_position
		}

	def __map_to_nearest_tyre(self, value):
		# Find the closest tyre type by minimising the absolute difference
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
			"pit1_lap": (2, self.__race_data.max_laps - 1),  # First pit stop lap
			"pit2_lap": (2, self.__race_data.max_laps - 1),  # Second pit stop lap
			"pit3_lap": (2, self.__race_data.max_laps - 1),  # Third pit stop lap
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
				if 2 <= lap < self.__race_data.max_laps:  # Can't pit after race is finished
					strategy[lap] = tyre

			# Enforce F1 rule - At least two distinct tyre types must be used
			tyre_types_used = set(strategy.values())
			if len(tyre_types_used) < 2:
				return -20.0  # Penalize strategies that violate rules

			# Evaluate the strategy using the race simulation
			try:
				sim = RaceSimulator(self.__race_data, self.__overtake_model, given_driver=self.__given_driver, simulated_strategy=strategy)
				sim_data = sim.simulate()
				final_position = next(d["position"] for d in sim_data if d["driver_number"] == self.__given_driver)
				return -final_position  # Negative because Bayesianoptimisation maximizes by default
			except Exception as e:
				print(f"Error during simulation: {e}")
				return -20.0  # Penalize errors

		# Initialize the Bayesian optimiser
		optimiser = BayesianOptimization(     # bayes_opt library
			f=objective_function,
			pbounds=pbounds,
			verbose=2,
			random_state=42
		)

		optimiser.maximize(init_points=5, n_iter=max_iterations)

		# compares to see if a strategy is unique
		def is_unique_strategy(new_strategy, existing_strategies):
			for existing_strategy in existing_strategies:
				if new_strategy == existing_strategy:
					return False
			return True

		# Get the top 10 unique strategies from the runs
		top_10_strategies = []
		for run in sorted(optimiser.res, key=lambda x: x["target"], reverse=True):
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
				if 2 <= lap < self.__race_data.max_laps:
					strategy[lap] = tyre

			# Check if the strategy is unique
			if is_unique_strategy(strategy, [s["strategy"] for s in top_10_strategies]):
				top_10_strategies.append({
					"strategy": strategy,
					"position": -run["target"]  # Convert back to positive position
				})

			# Stop once we have 10 unique strategies
			if len(top_10_strategies) == 10:
				break

		return top_10_strategies



	def genetic_algorithm_optimisation(self, population_size=50, generations=50):
		"""
		Perform Genetic Algorithm optimisation to find the top 10 strategies for a given driver.
		"""
		# Unique tyre types
		__unique_tyre_types = sorted(self.__race_data.get_unique_tyre_types())

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
				random.randint(2, self.__race_data.max_laps - 1),  # Pit1 lap
				random.choice(__unique_tyre_types),  # Pit1 tyre
				random.randint(2, self.__race_data.max_laps - 1),  # Pit2 lap
				random.choice(__unique_tyre_types),  # Pit2 tyre
				random.randint(2, self.__race_data.max_laps - 1),  # Pit3 lap
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
			sim = RaceSimulator(self.__race_data, self.__overtake_model, given_driver=self.__given_driver, simulated_strategy=strategy)
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
					individual[i] = random.randint(2, self.__race_data.max_laps - 1)
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

		population, logbook = algorithms.eaSimple(     # deap library
			population,
			toolbox,
			cxpb=0.7,  # Crossover probability
			mutpb=0.2,  # Mutation probability
			ngen=generations,
			stats=stats,
			halloffame=hof,  # Pass the Hall of Fame here
			verbose=True
		)

		# compares to see if a strategy is unique
		def is_unique_strategy(new_strategy, existing_strategies):
			for existing_strategy in existing_strategies:
				if new_strategy == existing_strategy:
					return False
			return True

		# Extract the top 10 unique strategies from the Hall of Fame
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
			sim = RaceSimulator(self.__race_data, self.__overtake_model, given_driver=self.__given_driver, simulated_strategy=strategy)
			sim_data = sim.simulate()
			final_position = next(d["position"] for d in sim_data if d["driver_number"] == self.__given_driver)

			# Check if the strategy is unique
			if is_unique_strategy(strategy, [s["strategy"] for s in top_strategies]):
				top_strategies.append({
					"strategy": strategy,
					"final_position": final_position
				})

			# Stop once we have 10 unique strategies
			if len(top_strategies) == 10:
				break

		return top_strategies
