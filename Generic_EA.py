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



################################################################################
###################################   GP   #####################################
################################################################################

class GP_Toolbox:
	def __init__(
        self,
        terminals = ["constant", "input_index"], #if more are added, check terminals dependencies
        teminal_generation_method = "uniform",
        terminals_probabilities = (0.5, 0.5),
        operations = ["add","sub","mul", "safe_divide_zero"],
        input_variable_count = 1):
		"""
		terminals options: random_constant, random_input_index
		teminal_generation_method can be uniform, by_probability
		terminals_probabilities is only used if teminal_generation_method is by_probability
		it is a tuple of float numbers with length equal to terminals, total must equal 1
		operations options: and, sub, mul, safe_divide_zero, safe_divide_numerator, signed_if, sin, cos, and, or, if, not
		"""

		self.terminals = []
		for terminal in terminals:
			if terminal == "random_constant":
				self.terminals.append(self.get_random_constant)
				self.random_constant_lower_limit = -1
				self.random_constant_upper_limit = 1
			if terminal == "random_input_index":
				self.terminals.append(self.get_random_input_index)
		self.teminal_generation_method = teminal_generation_method
		self.terminals_probabilities = terminals_probabilities

		self.functions = []
		for operation in operations:
			if operation == "add": operations.append(operator.add)
			elif operation == "sub": operations.append(operator.sub)
			elif operation == "mul": operations.append(operator.mul)
			elif operation == "safe_divide_numerator": operations.append(self.safe_divide_numerator)
			elif operation == "safe_divide_zero": operations.append(self.safe_divide_zero)
			elif operation == "signed_if": operations.append(self.signed_if)
			elif operation == "sin": operations.append(math.sin)
			elif operation == "cos": operations.append(math.cos)
			elif operation == "and": operations.append(operator.and_)
			elif operation == "or": operations.append(operator.or_)
			elif operation == "if": operations.append(self.boolean_if)
			elif operation == "not": operations.append(operator.not_)

		self.input_variable_count = input_variable_count
		
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
		if terminal_type is None:
			if terminal_generation_method == "uniform":
				terminal_type = self.terminals
			else:
				assert len(self.terminals) > 1, "Wrong terminal generation path"
				terminal_type = self.terminals[0]
		if terminal_generation_method is None: terminal_generation_method = self.terminal_generation_method
		if terminals_probabilities is None: terminals_probabilities = self.terminals_probabilities

		if terminal_generation_method == "uniform":
			options = []
			for t in terminal_type:
				if t == "constant": options.append(t)
				elif t == "input_index": options.extend([t for _ in range(self.input_variable_count)])
			choice = rd.choice(options)

		elif terminal_generation_method == "by_probability":
			temporal = np.random.uniform()
			index = 0
			accumulative_probability = terminals_probabilities[index]
			while temporal >= accumulative_probability:
				index += 1
				accumulative_probability += terminals_probabilities[index]
			choice = terminal_type[index]


		if choice == "constant":
			if random_constant_lower_limit is None: random_constant_lower_limit = self.random_constant_lower_limit
			if random_constant_upper_limit is None: random_constant_upper_limit = self.random_constant_upper_limit
			return "constant", np.random.uniform(random_constant_lower_limit, random_constant_upper_limit)

		elif choice == "input_index":
			return "input_index", rd.randint(0, self.input_variable_count - 1)

	def generate_automatic_terminal(self, terminals = None, teminal_generation_method = None, terminals_probabilities = None):

		else:
			if teminal_generation_method == "uniform":
				if terminals == ["random_constant", "random_input_index"]: #terminals dependent
					temporal = rd.randint(0, input_variable_count)
					if temporal == input_variable_count:
						content_type, content = self.generate_terminal(terminal_type = "random_constant")
						return content_type, content 
					else:
						content_type, content = self.generate_terminal(terminal_type = "random_input_index")
						return content_type, content

			if teminal_generation_method == "by_probability":
				temporal = np.random.uniform()
				index = 0
				accumulative_probability = terminals_probabilities[index]
				while temporal >= accumulative_probability:
					index += 1
					accumulative_probability += terminals_probabilities[index]
				content_type, content = self.generate_terminal(terminal_type = terminals[index])
				return content_type, content

	def generate_operator(self):
		return rd.choice(self.operations)

	def get_arity(self, operator):
		sig = signature(operator)
        arity = len(sig.parameters)
		return arity

	def generate_individual(self, max_depth, method, parent = None, depth = 0):
		"""
		method can be full or grow
		"""
		if depth == max_depth - 1:
			content_type, content = self.generate_automatic_terminal()
			return GP_Node(content, parent = parent, content_type = content_type)
		else:

			if method == "full":
				operator = self.generate_operator()
				arity = get_arity(operator)
				node = GP_Node(operator, parent = parent, content_type = "operator")
				for _ in range(arity):
					node.children.append(self.generate_individual(max_depth, method = "full", parent = node, depth = depth + 1))
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
					content_type, content = self.generate_automatic_terminal()
					return GP_Node(content, parent = parent, content_type = content_type)

		assert True, "Wrong method to generate individual"

	#evaluations
	def evaluate(self, node, data):
		"""
		evaluates the tree with the given data
		"""
        if not node.is_terminal():
        	assert node.content_type == "function", "Non-terminal node has non-operation content!"
            arguments = [self.evaluate(child, data) for child in node.children]
            return node.content(*arguments)
        elif node.content == "data_index"
            return data[node.content]
        else:
            return node.content

    def mutate(self, type = "subtree"):
    	"""
    	type can be subtree, single_node
    	"""
    	new_individual = parent.copy()
    	if type == "subtree":
    		pass


	#Operations
	def safe_divide_numerator(a, b):
	    """
	    Executes a/b. If b=0, returns a
	    """
	    if b == 0 : return a
	    else: return a/b

	def safe_divide_zero(a, b):
	    """
	    Executes a/b. If b=0, returns 0
	    """
	    if b == 0 : return 0
	    else: return a/b

	def signed_if(condition, a, b):
	    """
	    Returns a if condition is <= 0, b otherwise
	    """
	    if condition <= 0 : return a
	    else: return b

	def boolean_if(condition, a, b):
	    if condition: return a
	    else: return b



class GP_Node:
    def __init__(self, content, *children, parent = None, content_type = None):
    	"""
    	content_type can be constant, input_index, operation
    	"""
        self.content = content
        self.content_type = content_type
        self.parent = parent
        self.children = []
        for child in children:
            self.children.append(child)
    
    def is_terminal(self):
        return self.children == []
    
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
            return self.content == other.content
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
###################################   CGP   #####################################
################################################################################




################################################################################
###################################   EA   #####################################
################################################################################

class EA:
	def __init__(
            self,
            toolbox = GP_Toolbox(),
            experiment_name = None,
            algorithm = None,
            selection_method = "tournament",
            evolution_strategy = "1+1",
            input_variable_count = 1
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

	def run_generations(n = 1, evolution_strategy = None, es_lambda = None):
		if evolution_strategy is None: evolution_strategy = self.evolution_strategy
		if es_lambda is None: es_lambda = self.es_lambda

		for generation in range(n):
			if evolution_strategy == "1+1":
				pass
				#self.offsprings_generation()
				#self.evaluate_population()
			elif evolution_strategy == "1+l":
				pass
				#self.evaluate_population()
				#self.offsprings_generation()

			self.store_population()
			self.total_generations += 1

	def initialise_population(n = 100, method = "random"):
		pass
		#toolbox.initialise_population(n)

	def select_individual(self, n = 1, selection_method = None, tournament_size = 3):
		if selection_method is None: selection_method = self.selection_method 

	def store_population():
		pass

	def load_population():
		pass

	def get_best(n = 1, criteria = "fitness"):
		pass

	def select(n = 1, population = None, method = None):
		if method is None: method = self.selection_method
		if population is None: population = self.population
		
		pass

		#return selection

	def fast_nondominated_sort(population = None):
		"""
		Originaly from the NSGA-II algorithm.
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

	def set_local_crowding_distances(population = None):
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


	def dominates(individual1, individual2):
		"""
		Boolean, pareto dominance. Individual 1 dominates individual 2?
		"""
		assert isinstance(individual1.objective_values, list) and isinstance(individual2.objective_values, list), "Individuals have no objective values"
		assert len(individual1.objective_values) == len(individual2.objective_values), "Individuals got different number of objective values"
		for objective_index in range(len(individual1.objective_values)):
			if individual1.objective_values[objective_index] <= individual2.objective_values[objective_index]:
				return False
		return True





class Individual:
	def __init__(
		self, 
		generation_of_creation = None,
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

		self.generation_of_creation = generation_of_creation,
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









