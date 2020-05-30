#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 23:15:50 2020

@author: Fred Valdez Ameneyro

#My generic Evolutionary Algorithm
"""
import numpy as np
import random as rd
import operator
import math
from inspect import signature


class Individual:
	def __init__(
		self, 
		creation_generation = None,
		parents_species = None,
		genotype = None, 
		phenotype = None,
		fitness_value = None,
		objective_values = None,             #used in MOEA
		n_dominators = None,                 #used in MOEA, pareto dominance
		n_dominated_solutions = None,        #used in MOEA, pareto dominance
		dominated_solutions = None,          #used in NSGA-II
		local_crowding_distance = None,      #used in NSGA-II
		non_domination_rank = None,          #used in NSGA-II
		comparison_operator = "fitness"):
		"""
		comparison_operator can be fitness, crowded_comparison_operator
		"""

		self.creation_generation = creation_generation,
		self.parents_speceis = parents_speceis,
		self.genotype = genotype
		self.phenotype = phenotype
		self.n_dominators = n_dominators
		self.n_dominated_solutions = n_dominated_solutions
		self.dominated_solutions = dominated_solutions
		self.objective_values = objective_values
		self.fitness_value = fitness_value
		self.local_crowding_distance = local_crowding_distance
		self.non_domination_rank = non_domination_rank
		self.comparison_operator = comparison_operator

	def __lt__(self, other): #less than

		if comparison_operator == "fitness":
			return self.fitness_value < other.fitness_value

		elif comparison_operator == "crowded_comparison_operator": #used in NSGA-II
			if self.non_domination_rank < other.non_domination_rank:
				return True
			elif self.non_domination_rank > other.non_domination_rank:
				return False
			else:
				return self.local_crowding_distance > other.local_crowding_distance
    
	def __eq__(self, other):
        
		if comparison_operator == "fitness":
			return self.fitness_value == other.fitness_value

		elif comparison_operator == "crowded_comparison_operator": #used in NSGA-II
			if self.non_domination_rank != other.non_domination_rank:
				return False
			else:
				return self.local_crowding_distance == other.local_crowding_distance

################################################################################
###################################   GP   #####################################
################################################################################

class GP_Toolbox:
	def __init__(
        self,
        terminals = ["constant", "input_index"], #if more are added, check terminals dependencies
        teminal_generation_method = "uniform",
        terminals_probabilities = (0.5, 0.5),
        operators_probabilities = None,
        operator_generation_method = "uniform",
        operations = ["add","sub","mul", "safe_divide_zero"],
        input_variable_count = 1,
        random_constant_lower_limit = -1,
        random_constant_upper_limit = 1):
		"""
		terminals options: random_constant, random_input_index
		teminal_generation_method can be uniform, by_probability
		operator_selection_method can be uniform, by_probability
		terminals_probabilities is only used if teminal_generation_method is by_probability
		it is a tuple of float numbers with length equal to terminals, total must equal 1
		operations options: and, sub, mul, safe_divide_zero, safe_divide_numerator, signed_if, sin, cos, and, or, if, not
		"""

		self.terminals = terminals
		self.random_constant_lower_limit = random_constant_lower_limit
		self.random_constant_upper_limit = random_constant_upper_limit
		self.teminal_generation_method = teminal_generation_method
		self.terminals_probabilities = terminals_probabilities
		self.input_variable_count = input_variable_count
		self.operators_probabilities = operators_probabilities
        self.operator_generation_method = operator_generation_method

		self.operations = []
		for operation in operations:
			if operation == "add": self.operations.append(operator.add)
			elif operation == "sub": self.operations.append(operator.sub)
			elif operation == "mul": self.operations.append(operator.mul)
			elif operation == "safe_divide_numerator": self.operations.append(self.safe_divide_numerator)
			elif operation == "safe_divide_zero": self.operations.append(self.safe_divide_zero)
			elif operation == "signed_if": self.operations.append(self.signed_if)
			elif operation == "sin": self.operations.append(math.sin)
			elif operation == "cos": self.operations.append(math.cos)
			elif operation == "and": self.operations.append(operator.and_)
			elif operation == "or": self.operations.append(operator.or_)
			elif operation == "if": self.operations.append(self.boolean_if)
			elif operation == "not": self.operations.append(operator.not_)

		
	#construction methods

	def generate_terminal(self, 
			terminal_type = None, 
			random_constant_lower_limit = None, 
			random_constant_upper_limit = None, 
			terminal_generation_method = None, 
			terminals_probabilities = None):
		"""
		terminal_type options: constant, input_index. Can be one or more depending on the generation method
		terminal_generation_method can be uniform, by_probability
		terminals_probabilities is only used if terminal_generation_method is by_probability
		it is a tuple of float numbers with length equal to terminals, total must equal 1
		returns content_type, content
		"""

	def get_arity(self, operator):
		sig = signature(operator)
        arity = len(sig.parameters)
		return arity

	def generate_individual(self, 
			max_depth, 
			method, 
			parent = None, 
			depth = 0,
			terminals = None,
			teminal_generation_method = None,
			terminals_probabilities = None,
			operators_probabilities = None,
			operator_generation_method = None,
			operations = None,
			input_variable_count = None,
			random_constant_lower_limit = None,
			random_constant_upper_limit = None
			):
		"""
		method can be full or grow
		"""
		if terminals is None: terminals = self.terminals
		if teminal_generation_method is None: teminal_generation_method = self.teminal_generation_method
        if terminals_probabilities is None: terminals_probabilities = self.terminals_probabilities
        if operators_probabilities is None: operators_probabilities = self.operators_probabilities
        if operator_generation_method is None: operator_generation_method = self.operator_generation_method
        if operations is None: operations = self.operations
        if input_variable_count is None: input_variable_count = self.input_variable_count
        if random_constant_lower_limit is None: random_constant_lower_limit = self.random_constant_lower_limit
        if random_constant_upper_limit is None: random_constant_upper_limit = self.random_constant_upper_limit

		if depth == max_depth - 1:
			return GP_Node(content_type_choices = terminals,
							content_type_probabilities = terminals_probabilities,
							content_type_generation_method = terminal_generation_method,
							input_variable_count = input_variable_count,
							random_constant_lower_limit = random_constant_lower_limit,
							random_constant_upper_limit = random_constant_upper_limit,
							parent = parent)
		else:

			if method == "full":
				node = GP_Node(content_choices = self.operations,
								conten_type = "operator",
								content_generation_method = "random_uniform",
								parent = parent)
				arity = get_arity(node.content)
				for _ in range(arity):
					child = self.generate_individual(max_depth = max_depth, 
														method = method, 
														parent = node, 
														depth = depth + 1,
														terminals = terminals,
														teminal_generation_method = teminal_generation_method,
														terminals_probabilities = terminals_probabilities,
														operators_probabilities = operators_probabilities,
														operator_generation_method = operator_generation_method,
														operations = operations,
														input_variable_count = input_variable_count,
														random_constant_lower_limit = random_constant_lower_limit,
														random_constant_upper_limit = random_constant_upper_limit
														):
					node.children.append(child)
				return node

			if method == "grow":
				if rd.choice([True, False]) or depth == 0:
					operator = self.generate_operator()
					arity = get_arity(operator)
					node = GP_Node(operator, parent = parent, content_type = "operator")
					for _ in range(arity):
						node.children.append(self.generate_individual(max_depth, method = "full", parent = node, depth = depth + 1))
					return node
				else:
					content_type, content = self.generate_terminal()
					return GP_Node(content, parent = parent, content_type = content_type)

		assert True, "Wrong method to generate individual"

	def generate_initial_population(self,
		n,
		method):



	#evaluations
	def evaluate(self, 
			node, 
			data):
		"""
		evaluates the tree with the given data
		"""
        if node.is_terminal():
        	if node.content == "data_index"
           		return data[node.content]
        	else:
          	  return node.content
   
        else:
        	assert node.content_type == "operator", "Non-terminal node has non-operation content!"
            arguments = [self.evaluate(child, data) for child in node.children]
            return node.content(*arguments)
        	
    #transformation
    def mutate(self, 
    		type = "subtree"):
    	"""
    	type can be subtree, single_node
    	"""
    	new_individual = parent.copy()
    	if type == "subtree":
    		pass


	#Operations
	def safe_divide_numerator(self, a, b):
	    """
	    Executes a/b. If b=0, returns a
	    """
	    if b == 0 : return a
	    else: return a/b

	def safe_divide_zero(self, a, b):
	    """
	    Executes a/b. If b=0, returns 0
	    """
	    if b == 0 : return 0
	    else: return a/b

	def signed_if(self, condition, a, b):
	    """
	    Returns a if condition is <= 0, b otherwise
	    """
	    if condition <= 0 : return a
	    else: return b

	def boolean_if(self, condition, a, b):
	    if condition: return a
	    else: return b		


class GP_Node:
	def __init__(self, 
		content_type = None,
		content = None,
		content_type_generation_method = None,
		content_generation_method = None,
		content_type_choices = None,
		content_choices = None,
		content_type_probabilities = None,
		content_probabilities = None,
		random_constant_lower_limit = -1, 
		random_constant_upper_limit = 1, 
		input_variable_count = None,
		parent = None):
		"""
		content_type can be [empty, "constant", "input_index", "operation"]
		generation_method can be [empty, "random_uniform", "by_probability"]
		"""

		self.parent = parent
		self.children = []

		if content is None:

			#set conten_type
			if content_type_generation_method is None:
				assert content_type is not None, "missing content_type"
				self.content_type = content_type

			elif content_type_generation_method == "by_probability":
				assert content_type_choices is not None, "missing choices"
				assert content_type_probabilities is not None, "missing probabilities"
				self.content_type = np.random.choice(content_type_choices, p = content_type_probabilities)
				
			elif content_type_generation_method == "random_uniform":
				assert content_type_choices is not None, "missing choices"
				self.content_type = rd.choice(content_type_choices)


			#set content
			if self.content_type == "input_index":
				assert input_variable_count is not None, "missing input_variable_count"

				if content_choices is None:
					content_choices = range(input_variable_count)

				if content_generation_method is None or content_generation_method == "random_uniform":
					self.content = rd.choice(content_choices)

				elif content_generation_method == "by_probability":
					assert content_probabilities is not None, "missing content_probabilities"
					self.content = np.random.choice(content_choices, p = content_probabilities)

				assert isinstance(self.content, int), "wrong input index"

			elif self.content_type == "constant":

				if content_generation_method is None or content_generation_method == "random_uniform":
					self.content = np.random.uniform(random_constant_lower_limit, random_constant_upper_limit)
			
				if content_generation_method == "by_probability"
					assert content_probabilities is not None, "missing content_probabilities"
					assert content_choices is not None, "missing content_choices"
					self.conten = np.random.choice(content_choices, p = content_probabilities)

			elif self.content_type = "operator":
				assert content_choices is not None, "missing content_choices"

				if content_generation_method is None or content_generation_method == "random_uniform":
					self.content = rd.choice(content_choices)

				if content_generation_method == "by_probability":
					self.content = np.random.choice(content_choices, p = content_type_probabilities)

		else:
			assert content_type is not None, "missing content_type"
			self.content = content
			self.content_type = content_type
		
		self.set_is_terminal()
		

    def set_is_terminal(self):
    	if self.content_type == "operator":
    		self.is_terminal = False
    	else:
    		self.is_terminal = True

    
    def is_root(self):
        return self.parent is None

    def get_subtree_nodes(self):
        """
        Returns a list with all the nodes of the subtree with this node as the root node, including himself
        """
        nodes = [self]
        i = 0
        while i < len(nodes):
            if not nodes[i].is_terminal():
                nodes.extend(nodes[i].children)
            i += 1
        return nodes

    def get_nodes_count(self):
        """
        Returns the number of nodes in this tree (including this node as the root) as an int
        """
        return len(self.subtree_nodes())

    def get_max_depth(self, depth = 0):
        """
        Returns the max depth of this tree as an int
        """
        new_depth = depth + 1
        if self.is_terminal():
            return new_depth
        else:
            return max([child.my_depth(new_depth) for child in self.children])
    
    def copy(self, parent=None):
        """
        Don't give arguments. Returns an unrelated new item with the same characteristics
        """       
        the_copy = GP_Node(self.content, parent = parent)
        if not self.is_terminal():
            for child in self.children:
               the_copy.children.append(child.copy(parent = the_copy))
        return the_copy

    def __eq__(self, other):
        if self.is_terminal and other.is_terminal:
        	if content_type == content_type:
            	return self.content == other.content
            else:
            	return False
        else:
            children_length = len(self.children)
            if children_length != len(other.children):
                return False
            else:
                for i in range(children_length):
                    if not self.__eq__(self.children[i], other.children[i]):
                        return False
                return True

    def __str__(self):
        if self.is_terminal():
            if isinstance(self.content, int):
                return "x" + str(self.content)
            else:
                return str(self.content)
        else:
            name_string = "(" + self.content.__name__
            for child in self.children:
                name_string += " " + str(child)
            name_string += ")"
            return name_string



################################################################################
###################################   CGP   ####################################
################################################################################



################################################################################
###################################   GA   #####################################
################################################################################


class GA_Toolbox:
	def __init__(
        self,
        genes = [0,1],
        initialisation_method = "uniform"
        ):
		"""
		
		"""
		self.genes = genes
		self.initialisation_method = initialisation_method


	def get_initial_population(self, 
			n = 100, 
			initialisation_method = None
			):
		if initialisation_method is None: initialisation_method = self.initialisation_method
		population = [self.generate_individual(initialisation_method) for _ in range(n)]
		return population

	def generate_individual(self):


################################################################################
###################################   EA   #####################################
################################################################################



class EA:
	def __init__(
            self,
            toolbox,
            experiment_name = None,
            algorithm = None,
            selection_method = "tournament",
            evolution_strategy = "1+1",
            input_variable_count = 1,
            initialisation_method = None
            ):
		"""
		algorithm can be NSGAII, SPEA2
		selection method can be tournament, random_uniform
		"""
		#assign
		self.toolbox = toolbox
		self.experiment_name = experiment_name
		self.algorithm = algorithm
		self.selection_method = selection_method
		self.evolution_strategy = evolution_strategy
		self.input_variable_count = input_variable_count

		#initialisation
		self.total_generations = 0
		self.population = []

		#defaults
		if self.algorithm == "NSGAII":
			#self.evolution_strategy 
			pass
		elif self.algorithm == "SPEA2":
			#self.evolution_strategy 
			pass

	def run_generations(
			n = 1, 
			evolution_strategy = None, 
			es_lambda = None):
		if evolution_strategy is None: evolution_strategy = self.evolution_strategy
		if es_lambda is None: es_lambda = self.es_lambda

		for generation in range(n):
			if evolution_strategy == "1+1":
				self.evaluate_population()
				self.get_offsprings()
				
			elif evolution_strategy == "1+l":
				assert es_lambda is not None, "wrong evolution strategy's parameters"
				pass
				#self.evaluate_population()
				#self.offsprings_generation()

			self.store_population()
			self.total_generations += 1

	def get_initial_population(self,
			n, 
			method
			):
		population = toolbox.generate_initial_population(n, method)
		self.population = population
		return population

	def select_individuals(self, 
			n, 
			population = None,
			selection_method = None, 
			tournament_size = None,
			population_is_sorted = False):
		"""
		selection method can be tournament, best
		"""
		if population is None: population = self.population
		if selection_method is None: selection_method = self.selection_method

		selected_individuals = []
		for individual_index in range(n):

			if selection_method == "tournament":
				assert tournament_size is not None, "wrong tournament selection parameters"
				competitors = [rd.choice(population) for _ in range(tournament_size)]
				if population_is_sorted:

				selected_individual = max(competitors)

		selected_individuals.append(selected_individual)

		return selected_individuals

	def store_population(self):
		pass

	def load_population(self):
		pass

	def fast_nondominated_sort(self,
			population = None):
		"""
		Originaly from the NSGA-II algorithm.
		Returns the population sorted
		"""
		if population is None: population = self.population

		current_front = []
		for individual_index, individual in enumerate(population):
			individual.n_dominators = 0           #pending: check if needs to be refreshed
			individual.dominated_solutions = []   #pending: check if needs to be refreshed
			for q_individual in population:
				if self.dominates(individual, q_individual):
					individual.dominated_solutions.append(q_individual)
				elif self.dominates(q_individual, individual):
					individual.n_dominators += 1
			if individual.n_dominators == 0:
				current_front.append(individual)
		front = 0
		while current_front != []:
			new_list = []
			for individual in current_front:
				individual.non_domination_rank = front
				for dominated_individual in individual.dominated_solutions:
					dominated_individual.n_dominators -= 1
					if dominated_individual.n_dominators == 0:
						new_list.append(dominated_individual)
			front += 1
			current_front = new_list

		return sorted(population)

	def set_local_crowding_distances(self,
			population = None):
		"""
		Originaly from the NSGA-II algorithm.
		"""
		if population is None: population = self.population
		for individual in population:
			individual.local_crowding_distance = 0

		for objective_index in range(len(population[0].objective_values)):
			sorted_population = sorted(population, key=lambda individual: individual.objective_values[objective_index])
			sorted_population[0].local_crowding_distance = np.inf
			sorted_population[-1].local_crowding_distance = np.inf
			for individual_index, individual in enumerate(sorted_population[1:-1]):
				individual.local_crowding_distance += (sorted_population[individual_index + 1] - sorted_population[individual_index - 1]) 


	def dominates(self,
			individual1, 
			individual2):
		"""
		Boolean, pareto dominance. Individual 1 dominates individual 2?
		"""
		assert isinstance(individual1.objective_values, list) and isinstance(individual2.objective_values, list), "Individuals have no objective values"
		assert len(individual1.objective_values) == len(individual2.objective_values), "Individuals got different number of objective values"
		for objective_index in range(len(individual1.objective_values)):
			if individual1.objective_values[objective_index] <= individual2.objective_values[objective_index]:
				return False
		return True














