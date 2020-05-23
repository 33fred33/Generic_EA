#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 23:15:50 2020

@author: Fred Valdez Ameneyro

#My generic Evolutionary Algorithm
Use:
"""
import numpy as np
import random as rd
import operator
import math


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
            ):
		"""
		algorithm can be NSGAII, SPEA2
		"""
		self.toolbox = toolbox
		self.experiment_name = experiment_name
		self.algorithm = algorithm

		self.total_generations = 0

	def run_generations(n = 1, evolution_strategy = "1+1", l = 0.5):
		for generation in range(n):
			if evolution_strategy == "1+1":
				self.evaluate_population()
				self.offsprings_generation()
			elif evolution_strategy == "1+l":
				self.evaluate_population()
				self.offsprings_generation()

			self.store_population()
			self.total_generations += n

	def initialise_population(n = 100, method):
		toolbox.initialise_population()

	def select individual(self, n = 1, selection_method = None, tournament_size = 3):
		if selection_method is None: selection_method = self.selection_method 

	def store_population():
		pass

	def load_population():
		pass

	def get_best(n = 1, criteria = "fitness"):
		pass






################################################################################
###################################   GP   #####################################
################################################################################

class GP_Toolbox:
	def __init__(
            self,
            terminals = ["random_constant", "random_input_index"],
            functions = ["add","sub","mul", "safe_divide_zero"],
            input_variable_count = 1):
	"""
	terminal options: random_constant, random_input_index
	function options: add, sub, mul, safe_divide_zero, safe_divide_numerator, signed_if, sin, cos, and, or, if, not
	"""

	self.terminal = []
	for terminal in terminals:

		if terminal == "random_constant":
			self.terminals.append(self.get_random_constant)
			self.random_constant_lower_limit = -1
			self.random_constant_upper_limit = 1
			print("Default warning (GP_Toolbox): Variables random_constant_lower_limit and random_constant_upper_limit\
			 to be used in the terminal random constant generation process are set to -1 and 1 as default.")

		if terminal == "random_input_index":
			self.terminals.append(self.get_random_input_index)

	self.functions = []
	for function in functions:

		if function == "add": functions.append(operator.add)
	    elif function == "sub": functions.append(operator.sub)
	    elif function == "mul": functions.append(operator.mul)
	    elif function == "safe_divide_numerator": functions.append(self.safe_divide_numerator)
	    elif function == "safe_divide_zero": functions.append(self.safe_divide_zero)
	    elif function == "signed_if": functions.append(self.signed_if)
	    elif function == "sin": functions.append(math.sin)
	    elif function == "cos": functions.append(math.cos)
	    elif function == "and": functions.append(operator.and_)
	    elif function == "or": functions.append(operator.or_)
	    elif function == "if": functions.append(self.boolean_if)
	    elif function == "not": functions.append(operator.not_)

	self.input_variable_count = input_variable_count
	



	def get_random_constant(lower_limit = self.random_constant_lower_limit, upper_limit = self.random_constant_upper_limit):
		return np.random.uniform(lower_limit, upper_limit)

	def get_random_input_index()
		if self.input_variable_count == 1:
			return 0
		else:
			return rd.randint(0, self.input_variable_count - 1)


	#Functions
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
    def __init__(self, content, *children, parent = None,):
        self.content = content
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
        Don't give arguments
        Returns an unrelated new item with the same characteristics
        """       
        the_copy = Node(self.content, parent = parent)
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








