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

class GP_Toolbox:
	def __init__(
            self,
            terminals = ,
            functions = ,
            fitness_function = ,
            input_variable_count = 1):

	self.terminal = []
	for terminal in terminals:

		if terminal == "random constant":
			self.terminals.append(self.get_random_constant)
			self.random_constant_lower_limit = -1
			self.random_constant_upper_limit = 1
			print("Default warning (GP_Toolbox): Variables random_constant_lower_limit and random_constant_upper_limit\
			 to be used in the terminal random constant generation process are now set to -1 and 1.")

		if terminal == "random input index":
			self.terminals.append(self.get_random_input_index)

	self.functions = []
	for function in functions:

		if function == "add":
	        functions.append(operator.add)
	    elif function == "sub":
	        functions.append(operator.sub)
	    elif function == "mul":
	        functions.append(operator.mul)
	    elif function == "safe_divide_numerator":
	        functions.append(self.safe_divide_numerator)
	    elif function == "signed_if":
	        functions.append(self.signed_if)
	    elif function == "sin":
	        functions.append(math.sin)
	    elif function == "cos":
	        functions.append(math.cos)
	    elif function == "and":
	        functions.append(operator.and_)
	    elif function == "or":
	        functions.append(operator.or_)
	    elif function == "if":
	        functions.append(self.boolean_if)
	    elif function == "not":
	        functions.append(operator.not_)
	    elif function == "and":
	        functions.append(self.boolean_and)
	    elif function == "or":
	        functions.append(self.boolean_or)

	self.input_variable_count = input_variable_count
	



	def get_random_constant(lower_limit = self.random_constant_lower_limit, upper_limit = self.random_constant_upper_limit):
		return np.random.uniform(lower_limit, upper_limit)

	def get_random_input_index()
		if self.input_variable_count == 1:
			return 0
		else:
			rd.randint(0, self.input_variable_count - 1)


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

	def boolean_and(a, b):
	    return a and b

	def boolean_or(a, b):
	    return a or b

	def boolean_if(condition, a, b):
	    if condition: return a
	    else: return b



class EA:
	def __init__(
            self,
            toolbox = GP_Toolbox(),
            population_size = 100,
            generations = 100,
            evaluation_function,
            tournament_size = 3,
            experiment_name = None,
            selection_method = "tournament",
            ensemble_type = "baseline"):

	def initialise_population():
		toolbox.initialise_population()