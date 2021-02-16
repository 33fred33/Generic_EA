#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 21:41:50 2021
@author: Fred Valdez Ameneyro
Evolutionary Algorithm's library
"""
import operator as op
from inspect import signature
import random as rd


#Global variables
generation = 0

#############################################
# Generic Classes ###########################
#############################################

class Individual:
    def __init__(self
            ,representation):
        self.representation = representation
        self.appearence_gen = generation
        self.evaluated = False
        self.evaluation = None

class Breeder:
    def __init__(self):
        pass

def get_arity(operator):
		"""
		Returns the arity of the method, operator or funtion as an int
		:param operator: is a method, operator or funtion
		"""
		sig = signature(operator)
		arity = len(sig.parameters)
		return arity

#############################################
# Cartessian Genetic Programming ############
#############################################

class CGP_Breeder(Breeder):
    def __init__(self
            ,n_inputs
            ,n_outputs
            ,levels_back
            ,n_rows
            ,n_columns
            ,*functions):
        """
        The outputs are generated as the last indexes in the graph
        """
        super().__init__()

        #Assignation
        self.functions = list(functions)
        self.n_inputs = n_inputs
        assert n_outputs >= 1
        self.n_outputs = n_outputs
        assert levels_back >= 1
        self.levels_back = levels_back
        assert n_rows >= 1
        self.n_rows = n_rows
        assert n_columns >= 1
        self.n_columns = n_columns

        #Inference
        self.n_functions = len(functions)
        self.function_arity = [get_arity(f) for f in functions]
        self.n_function_nodes = n_rows * n_columns
        self.max_lenght = self.n_function_nodes + n_outputs
        self.function_set = [i for i in range(self.n_functions)]
        self.inputs_set = [i for i in range(n_inputs)]
        self.output_indexes = [i for i in range(self.n_function_nodes, self.max_lenght)]
        
        #self.connections_set holds the set of indexes available to connect to each column in the graph
        self.connections_set = []
        for column in range(n_columns):
            if column > 0:
                lower_limit = n_inputs + n_rows * (column - levels_back)
                upper_limit = n_inputs + n_rows * column
                temp = [i for i in range(lower_limit, upper_limit) if i >= 0]
            else:
                temp = []
            column_connections = list(set(self.inputs_set+temp))
            self.connections_set.append(column_connections)

    def create_random(self):
        """Create a random individual"""
        genotype = {}
        for node_index in range(self.n_function_nodes):
            column = int(node_index/self.n_rows)
            output_index = node_index + self.n_inputs
            f_index = rd.choice(self.function_set)
            inputs = [rd.choice(self.connections_set[column]) for _ in range(self.function_arity[f_index])]
            print("inputs",inputs)
            print("f_index",f_index)
            node = CGP_Node(f_index
                    ,output_index
                    ,column
                    ,*inputs)
            genotype[output_index] = node
        graph = CGP_Graph(genotype = genotype
                ,n_inputs = self.n_inputs
                ,n_outputs = self.n_outputs)
        ind = Individual(representation = graph)
        return ind


class CGP_Node:
    def __init__(self
            ,function_index
            ,output_index
            ,column
            ,*inputs):
        self.function_index = function_index
        self.output_index = output_index
        self.column = column
        self.inputs = list(inputs)
    
    def __str__(self):
        input_label = " i"
        for i in self.inputs:
            input_label += " " + str(i)
        label = "  N" + str(self.output_index) + " f" + str(self.function_index) + input_label
        return label


class CGP_Graph:
    """
    The genotype is a list of nodes,
    each with indexes pointing to their inputs and functions
    """
    def __init__(self
            ,genotype
            ,n_inputs
            ,n_outputs):
        self.genotype = genotype
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
    
    def evaluate(self):
        max_graph_lenght = len(self.genotype)
        to_evaluate = [False for _ in range(max_graph_lenght)]
        node_output = [None for _ in range(max_graph_lenght + self.inputs)]
        graph_input = [None for _ in range(self.n_inputs)]
        graph_output = [None for _ in range(self.n_outputs)]
        nodes_used = [None for _ in range(max_graph_lenght)]
        output_gene = [None for _ in range(self.n_inputs)]
        node = [None for _ in range(max_graph_lenght)]

        for p in range(self.n_outputs):
            to_evaluate[output_gene[p]] = True

        for p in range(max_graph_lenght-1, 0, -1):
            if to_evaluate:
                pass
    
    def __str__(self):
        label = "G"
        column = 0
        for node in self.genotype.values():
            if node.column != column:
                label += "\n"
                column += 1
            label += "," + str(node)
        return label



breeder = CGP_Breeder(10,1,2,4,10,op.add,op.sub,op.mul)
i1 = breeder.create_random()
print(i1.representation)