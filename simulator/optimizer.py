import random
import numpy as np
import time
from datetime import datetime
import json
import copy


class RailSystemOptimizer:
    """
    Optimizer for rail transit systems.
    Uses various algorithms to optimize train schedules, headways, and operations.
    """

    # Optimization objective types
    OBJECTIVE_PASSENGER_WAIT = "passenger_wait_time"
    OBJECTIVE_ENERGY = "energy_consumption"
    OBJECTIVE_CAPACITY = "capacity_utilization"
    OBJECTIVE_BALANCED = "balanced"

    def __init__(self, simulator, config=None):
        """
        Initialize the optimizer.

        Args:
            simulator: RailSystemSimulator object to use for evaluating solutions
            config (dict, optional): Configuration parameters for the optimization
        """
        self.simulator = simulator

        # Default configuration
        self.config = {
            'algorithm': 'genetic',  # genetic, simulated_annealing, hill_climbing
            'objective': self.OBJECTIVE_BALANCED,  # What to optimize for
            'population_size': 20,  # For genetic algorithm
            'generations': 30,  # For genetic algorithm
            'mutation_rate': 0.2,  # For genetic algorithm
            'crossover_rate': 0.7,  # For genetic algorithm
            'initial_temperature': 100.0,  # For simulated annealing
            'cooling_rate': 0.95,  # For simulated annealing
            'min_temperature': 0.1,  # For simulated annealing
            'max_iterations': 100,  # For hill climbing and simulated annealing
            'simulation_duration': 120,  # Duration for each simulation in minutes
            'weights': {  # Weights for multi-objective optimization
                'passenger_wait_time': 0.4,
                'energy_consumption': 0.3,
                'capacity_utilization': 0.3
            },
            'constraints': {
                'min_headway': 3,  # Minimum headway in minutes
                'max_headway': 15,  # Maximum headway in minutes
                'min_trains': 1,  # Minimum number of trains
                'max_trains': None,  # Maximum number of trains (None for unlimited)
                'max_energy_consumption': None  # Maximum allowed energy consumption
            }
        }

        # Update configuration with provided values
        if config:
            self.config.update(config)
            # Also update nested dictionaries
            if 'weights' in config:
                self.config['weights'].update(config['weights'])
            if 'constraints' in config:
                self.config['constraints'].update(config['constraints'])

        # Optimization state
        self.best_solution = None
        self.best_score = float('-inf')
        self.current_generation = 0
        self.history = []
        self.optimization_time = 0

    def optimize(self, callback=None):
        """
        Run the optimization process using the selected algorithm.

        Args:
            callback (function, optional): Function to call after each iteration

        Returns:
            dict: Optimization results
        """
        start_time = time.time()

        # Select algorithm
        if self.config['algorithm'] == 'genetic':
            result = self._genetic_algorithm(callback)
        elif self.config['algorithm'] == 'simulated_annealing':
            result = self._simulated_annealing(callback)
        elif self.config['algorithm'] == 'hill_climbing':
            result = self._hill_climbing(callback)
        else:
            raise ValueError(f"Unknown algorithm: {self.config['algorithm']}")

        self.optimization_time = time.time() - start_time
        return result

    def _genetic_algorithm(self, callback=None):
        """
        Implement genetic algorithm for optimization.

        Args:
            callback (function, optional): Function to call after each generation

        Returns:
            dict: Optimization results
        """
        # Initialize population
        population = self._initialize_population(self.config['population_size'])

        # Evaluate initial population
        fitness_scores = [self._evaluate_solution(solution) for solution in population]

        # Find initial best solution
        best_idx = np.argmax(fitness_scores)
        self.best_solution = copy.deepcopy(population[best_idx])
        self.best_score = fitness_scores[best_idx]

        # Main loop for genetic algorithm
        for generation in range(self.config['generations']):
            self.current_generation = generation

            # Create new generation
            new_population = []

            # Elitism: Keep the best solution
            elite_idx = np.argmax(fitness_scores)
            new_population.append(population[elite_idx])

            # Create rest of the new population
            while len(new_population) < self.config['population_size']:
                # Selection
                parent1 = self._selection(population, fitness_scores)
                parent2 = self._selection(population, fitness_scores)

                # Crossover
                if random.random() < self.config['crossover_rate']:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                if random.random() < self.config['mutation_rate']:
                    child1 = self._mutate(child1)
                if random.random() < self.config['mutation_rate']:
                    child2 = self._mutate(child2)

                # Add children to new population
                new_population.append(child1)
                if len(new_population) < self.config['population_size']:
                    new_population.append(child2)

            # Update population
            population = new_population

            # Evaluate new population
            fitness_scores = [self._evaluate_solution(solution) for solution in population]

            # Update best solution
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > self.best_score:
                self.best_solution = copy.deepcopy(population[current_best_idx])
                self.best_score = fitness_scores[current_best_idx]

            # Record history
            self.history.append({
                'generation': generation,
                'best_score': self.best_score,
                'avg_score': np.mean(fitness_scores),
                'best_solution': copy.deepcopy(self.best_solution)
            })

            # Call callback if provided
            if callback:
                callback_result = callback(generation, self.best_solution, self.best_score, population, fitness_scores)
                # Allow early stopping
                if callback_result is False:
                    break

        # Return best solution
        return self._format_result()

    def _simulated_annealing(self, callback=None):
        """
        Implement simulated annealing algorithm for optimization.

        Args:
            callback (function, optional): Function to call after each iteration

        Returns:
            dict: Optimization results
        """
        # Initialize with a random solution
        current_solution = self._generate_random_solution()
        current_score = self._evaluate_solution(current_solution)

        # Set initial best solution
        self.best_solution = copy.deepcopy(current_solution)
        self.best_score = current_score

        # Initialize temperature
        temperature = self.config['initial_temperature']

        # Main loop for simulated annealing
        iteration = 0
        while temperature > self.config['min_temperature'] and iteration < self.config['max_iterations']:
            # Generate a neighbor
            neighbor = self._get_neighbor(current_solution)
            neighbor_score = self._evaluate_solution(neighbor)

            # Decide whether to accept the neighbor
            if neighbor_score > current_score:
                # Always accept better solutions
                current_solution = neighbor
                current_score = neighbor_score
            else:
                # Sometimes accept worse solutions based on temperature
                acceptance_probability = np.exp((neighbor_score - current_score) / temperature)
                if random.random() < acceptance_probability:
                    current_solution = neighbor
                    current_score = neighbor_score

            # Update best solution if needed
            if current_score > self.best_score:
                self.best_solution = copy.deepcopy(current_solution)
                self.best_score = current_score

            # Record history
            self.history.append({
                'iteration': iteration,
                'temperature': temperature,
                'current_score': current_score,
                'best_score': self.best_score,
                'acceptance_probability': acceptance_probability if neighbor_score <= current_score else 1.0
            })

            # Call callback if provided
            if callback:
                callback_result = callback(iteration, self.best_solution, self.best_score, temperature)
                # Allow early stopping
                if callback_result is False:
                    break

            # Cool down
            temperature *= self.config['cooling_rate']
            iteration += 1

        # Return best solution
        return self._format_result()

    def _hill_climbing(self, callback=None):
        """
        Implement hill climbing algorithm for optimization.

        Args:
            callback (function, optional): Function to call after each iteration

        Returns:
            dict: Optimization results
        """
        # Initialize with a random solution
        current_solution = self._generate_random_solution()
        current_score = self._evaluate_solution(current_solution)

        # Set initial best solution
        self.best_solution = copy.deepcopy(current_solution)
        self.best_score = current_score

        # Main loop for hill climbing
        for iteration in range(self.config['max_iterations']):
            # Generate a neighbor
            neighbor = self._get_neighbor(current_solution)
            neighbor_score = self._evaluate_solution(neighbor)

            # Move to neighbor if it's better
            if neighbor_score > current_score:
                current_solution = neighbor
                current_score = neighbor_score

                # Update best solution
                self.best_solution = copy.deepcopy(current_solution)
                self.best_score = current_score

            # Record history
            self.history.append({
                'iteration': iteration,
                'current_score': current_score,
                'best_score': self.best_score
            })

            # Call callback if provided
            if callback:
                callback_result = callback(iteration, self.best_solution, self.best_score)
                # Allow early stopping
                if callback_result is False:
                    break

        # Return best solution
        return self._format_result()

    def _initialize_population(self, size):
        """
        Initialize a population of random solutions.

        Args:
            size (int): Population size

        Returns:
            list: List of solutions
        """
        return [self._generate_random_solution() for _ in range(size)]

    def _generate_random_solution(self):
        """
        Generate a random solution (schedule).

        Returns:
            dict: A random solution
        """
        constraints = self.config['constraints']

        # Get current system parameters for reference
        original_trains = len(self.simulator.trains)

        # Determine number of trains
        min_trains = constraints['min_trains']
        max_trains = constraints['max_trains'] if constraints['max_trains'] is not None else original_trains * 2
        num_trains = random.randint(min_trains, max_trains)

        # Generate headways between trains
        min_headway = constraints['min_headway']
        max_headway = constraints['max_headway']

        # Generate random initial positions and headways
        stations = len(self.simulator.line.stations)

        train_params = []
        for i in range(num_trains):
            train_params.append({
                'initial_station': random.randint(0, stations - 1),
                'headway': random.uniform(min_headway, max_headway),
                'dwell_time_factor': random.uniform(0.8, 1.2),  # Modifier for standard dwell time
                'speed_factor': random.uniform(0.9, 1.1)  # Modifier for standard speed
            })

        # Build solution
        solution = {
            'num_trains': num_trains,
            'train_params': train_params,
            'global_headway': random.uniform(min_headway, max_headway) if num_trains > 1 else 0,
            'global_speed_factor': random.uniform(0.9, 1.1)
        }

        return solution

    def _evaluate_solution(self, solution):
        """
        Evaluate a solution by running a simulation.

        Args:
            solution (dict): Solution to evaluate

        Returns:
            float: Fitness score for the solution
        """
        # Create a copy of the simulator for evaluation
        simulator_copy = copy.deepcopy(self.simulator)

        # Apply solution to simulator
        self._apply_solution_to_simulator(simulator_copy, solution)

        # Run simulation
        sim_results = simulator_copy.run_simulation(
            duration=self.config['simulation_duration']
        )

        # Extract relevant metrics
        stats = sim_results['statistics']

        # Calculate fitness score based on objective
        if self.config['objective'] == self.OBJECTIVE_PASSENGER_WAIT:
            # Lower wait time is better, so we negate it
            score = -stats['average_waiting_time']

        elif self.config['objective'] == self.OBJECTIVE_ENERGY:
            # Lower energy consumption is better, so we negate it
            # Normalize by distance
            if stats['total_distance_traveled'] > 0:
                energy_per_km = stats['total_energy_consumed'] / stats['total_distance_traveled']
                score = -energy_per_km
            else:
                score = float('-inf')

        elif self.config['objective'] == self.OBJECTIVE_CAPACITY:
            # Higher utilization is better (but not over 100%)
            score = stats['train_utilization'] if stats['train_utilization'] <= 1.0 else 2 - stats['train_utilization']

        elif self.config['objective'] == self.OBJECTIVE_BALANCED:
            # Balanced approach using weighted sum
            weights = self.config['weights']

            # Normalize metrics to roughly 0-1 range
            wait_score = min(1.0, 30 / max(1, stats['average_waiting_time']))

            energy_score = 0
            if stats['total_distance_traveled'] > 0:
                # Lower energy per km is better (assuming typical range 5-20 kWh/km)
                energy_per_km = stats['total_energy_consumed'] / stats['total_distance_traveled']
                energy_score = max(0, 1 - (energy_per_km - 5) / 15)

            capacity_score = stats['train_utilization'] if stats['train_utilization'] <= 0.9 else 1.8 - stats[
                'train_utilization']

            # Combine scores with weights
            score = (weights['passenger_wait_time'] * wait_score +
                     weights['energy_consumption'] * energy_score +
                     weights['capacity_utilization'] * capacity_score)

        else:
            raise ValueError(f"Unknown objective: {self.config['objective']}")

        # Apply penalty for constraint violations
        constraints = self.config['constraints']

        # Check max energy constraint
        if constraints['max_energy_consumption'] is not None:
            if stats['total_energy_consumed'] > constraints['max_energy_consumption']:
                # Apply penalty based on how much the constraint is violated
                penalty = (stats['total_energy_consumed'] - constraints['max_energy_consumption']) / constraints[
                    'max_energy_consumption']
                score -= penalty * 2  # Adjust penalty weight as needed

        return score

    def _apply_solution_to_simulator(self, simulator, solution):
        """
        Apply a solution to a simulator configuration.

        Args:
            simulator: RailSystemSimulator instance
            solution (dict): Solution to apply
        """
        # Reset simulator to clean state
        simulator.reset_simulation()

        # Adjust number of trains if needed
        current_trains = len(simulator.trains)
        target_trains = solution['num_trains']

        # Remove trains if too many
        if current_trains > target_trains:
            simulator.trains = simulator.trains[:target_trains]

        # Add trains if needed
        elif current_trains < target_trains:
            # Get train template from first train
            template_train = copy.deepcopy(simulator.trains[0])

            # Add new trains
            for i in range(current_trains, target_trains):
                new_train = copy.deepcopy(template_train)
                new_train.id = i + 1  # Ensure unique ID
                simulator.trains.append(new_train)

        # Apply train-specific parameters
        for i, train in enumerate(simulator.trains):
            if i < len(solution['train_params']):
                params = solution['train_params'][i]

                # Set initial position
                train.arrive_at_station(params['initial_station'], 1)

                # Set speed factor (affects travel time calculation)
                train.max_speed *= solution['global_speed_factor'] * params.get('speed_factor', 1.0)

                # Other parameters could be set here

        # Apply global parameters to simulator config
        if 'global_headway' in solution:
            simulator.config['min_headway'] = solution['global_headway']

        # Re-initialize train positions to avoid conflicts
        simulator._initialize_train_positions()

    def _selection(self, population, fitness_scores):
        """
        Select a solution from the population for genetic algorithm.
        Uses tournament selection.

        Args:
            population (list): List of solutions
            fitness_scores (list): Corresponding fitness scores

        Returns:
            dict: Selected solution
        """
        # Tournament selection
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]

        return population[winner_idx]

    def _crossover(self, parent1, parent2):
        """
        Perform crossover between two parent solutions.

        Args:
            parent1 (dict): First parent solution
            parent2 (dict): Second parent solution

        Returns:
            tuple: Two child solutions
        """
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        # Crossover global parameters
        if random.random() < 0.5:
            child1['global_headway'], child2['global_headway'] = child2['global_headway'], child1['global_headway']

        if random.random() < 0.5:
            child1['global_speed_factor'], child2['global_speed_factor'] = child2['global_speed_factor'], child1[
                'global_speed_factor']

        # Crossover number of trains
        if random.random() < 0.5:
            child1['num_trains'], child2['num_trains'] = child2['num_trains'], child1['num_trains']

            # Adjust train parameters array length
            if len(child1['train_params']) > child1['num_trains']:
                child1['train_params'] = child1['train_params'][:child1['num_trains']]
            elif len(child1['train_params']) < child1['num_trains']:
                # Copy from parent2 or generate new
                while len(child1['train_params']) < child1['num_trains']:
                    if len(parent2['train_params']) > len(child1['train_params']):
                        child1['train_params'].append(
                            copy.deepcopy(parent2['train_params'][len(child1['train_params'])]))
                    else:
                        # Generate a new random train parameter set
                        child1['train_params'].append({
                            'initial_station': random.randint(0, len(self.simulator.line.stations) - 1),
                            'headway': random.uniform(self.config['constraints']['min_headway'],
                                                      self.config['constraints']['max_headway']),
                            'dwell_time_factor': random.uniform(0.8, 1.2),
                            'speed_factor': random.uniform(0.9, 1.1)
                        })

            # Do the same for child2
            if len(child2['train_params']) > child2['num_trains']:
                child2['train_params'] = child2['train_params'][:child2['num_trains']]
            elif len(child2['train_params']) < child2['num_trains']:
                while len(child2['train_params']) < child2['num_trains']:
                    if len(parent1['train_params']) > len(child2['train_params']):
                        child2['train_params'].append(
                            copy.deepcopy(parent1['train_params'][len(child2['train_params'])]))
                    else:
                        child2['train_params'].append({
                            'initial_station': random.randint(0, len(self.simulator.line.stations) - 1),
                            'headway': random.uniform(self.config['constraints']['min_headway'],
                                                      self.config['constraints']['max_headway']),
                            'dwell_time_factor': random.uniform(0.8, 1.2),
                            'speed_factor': random.uniform(0.9, 1.1)
                        })

        # Crossover individual train parameters
        min_params = min(len(child1['train_params']), len(child2['train_params']))
        for i in range(min_params):
            if random.random() < 0.5:
                # Swap headway
                child1['train_params'][i]['headway'], child2['train_params'][i]['headway'] = \
                    child2['train_params'][i]['headway'], child1['train_params'][i]['headway']

            if random.random() < 0.5:
                # Swap initial station
                child1['train_params'][i]['initial_station'], child2['train_params'][i]['initial_station'] = \
                    child2['train_params'][i]['initial_station'], child1['train_params'][i]['initial_station']

            if random.random() < 0.5:
                # Swap dwell time factor
                child1['train_params'][i]['dwell_time_factor'], child2['train_params'][i]['dwell_time_factor'] = \
                    child2['train_params'][i]['dwell_time_factor'], child1['train_params'][i]['dwell_time_factor']

            if random.random() < 0.5:
                # Swap speed factor
                child1['train_params'][i]['speed_factor'], child2['train_params'][i]['speed_factor'] = \
                    child2['train_params'][i]['speed_factor'], child1['train_params'][i]['speed_factor']

        return child1, child2

    def _mutate(self, solution):
        """
        Mutate a solution.

        Args:
            solution (dict): Solution to mutate

        Returns:
            dict: Mutated solution
        """
        # Create a copy to avoid modifying the original
        mutated = copy.deepcopy(solution)
        constraints = self.config['constraints']

        # Mutate global parameters
        if random.random() < 0.3:
            # Adjust global headway within constraints
            mutated['global_headway'] = random.uniform(
                constraints['min_headway'],
                constraints['max_headway']
            )

        if random.random() < 0.3:
            # Adjust global speed factor
            current = mutated['global_speed_factor']
            # Change by up to ±10%
            change = random.uniform(0.9, 1.1)
            mutated['global_speed_factor'] = current * change
            # Keep within reasonable bounds
            mutated['global_speed_factor'] = max(0.8, min(1.2, mutated['global_speed_factor']))

        # Mutate number of trains
        if random.random() < 0.2:
            min_trains = constraints['min_trains']
            max_trains = constraints['max_trains'] if constraints['max_trains'] is not None else len(
                mutated['train_params']) * 2

            # Change number of trains by at most ±2
            delta = random.randint(-2, 2)
            new_num_trains = max(min_trains, min(max_trains, mutated['num_trains'] + delta))

            # Update train count
            mutated['num_trains'] = new_num_trains

            # Adjust train parameters array
            if len(mutated['train_params']) > new_num_trains:
                # Remove excess trains
                mutated['train_params'] = mutated['train_params'][:new_num_trains]
            elif len(mutated['train_params']) < new_num_trains:
                # Add new trains
                stations = len(self.simulator.line.stations)

                while len(mutated['train_params']) < new_num_trains:
                    mutated['train_params'].append({
                        'initial_station': random.randint(0, stations - 1),
                        'headway': random.uniform(constraints['min_headway'], constraints['max_headway']),
                        'dwell_time_factor': random.uniform(0.8, 1.2),
                        'speed_factor': random.uniform(0.9, 1.1)
                    })

        # Mutate individual train parameters
        for params in mutated['train_params']:
            # Mutate headway
            if random.random() < 0.2:
                params['headway'] = random.uniform(constraints['min_headway'], constraints['max_headway'])

            # Mutate initial station
            if random.random() < 0.2:
                params['initial_station'] = random.randint(0, len(self.simulator.line.stations) - 1)

            # Mutate dwell time factor
            if random.random() < 0.2:
                # Change by up to ±20%
                change = random.uniform(0.8, 1.2)
                params['dwell_time_factor'] *= change
                # Keep within reasonable bounds
                params['dwell_time_factor'] = max(0.6, min(1.4, params['dwell_time_factor']))

            # Mutate speed factor
            if random.random() < 0.2:
                # Change by up to ±10%
                change = random.uniform(0.9, 1.1)
                params['speed_factor'] *= change
                # Keep within reasonable bounds
                params['speed_factor'] = max(0.8, min(1.2, params['speed_factor']))

        return mutated

    def _get_neighbor(self, solution):
        """
        Generate a neighbor solution for hill climbing and simulated annealing.

        Args:
            solution (dict): Current solution

        Returns:
            dict: Neighbor solution
        """
        # For simplicity, we'll use mutation to generate a neighbor
        return self._mutate(solution)

    def _format_result(self):
        """
        Format the optimization result.

        Returns:
            dict: Formatted optimization result
        """
        return {
            'best_solution': self.best_solution,
            'best_score': self.best_score,
            'algorithm': self.config['algorithm'],
            'objective': self.config['objective'],
            'iterations': len(self.history),
            'history': self.history,
            'execution_time': self.optimization_time,
            'config': self.config
        }

    def apply_best_solution(self, simulator=None):
        """
        Apply the best solution found to a simulator.

        Args:
            simulator (optional): Simulator to apply the solution to.
                                  If None, uses the optimizer's simulator.

        Returns:
            The simulator with the applied solution
        """
        if simulator is None:
            simulator = self.simulator

        if self.best_solution is None:
            raise ValueError("No solution found yet. Run optimize() first.")

        # Apply solution to simulator
        self._apply_solution_to_simulator(simulator, self.best_solution)
        return simulator

    def save_results(self, filename=None):
        """
        Save optimization results to a file.

        Args:
            filename (str, optional): Filename to save to

        Returns:
            str: Filename where results were saved
        """
        if self.best_solution is None:
            raise ValueError("No optimization results to save")

        results = self._format_result()

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            algorithm = self.config['algorithm']
            objective = self.config['objective']
            filename = f"optimization_{algorithm}_{objective}_{timestamp}.json"

        with open(filename, 'w') as f:
            # Convert numpy values to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                return obj

            json.dump(convert_numpy(results), f, indent=2)

        return filename

    def __str__(self):
        algorithm = self.config['algorithm']
        objective = self.config['objective']
        if self.best_solution is None:
            return f"RailSystemOptimizer: {algorithm} for {objective} (not yet optimized)"
        else:
            return f"RailSystemOptimizer: {algorithm} for {objective}, best score: {self.best_score:.4f}"
