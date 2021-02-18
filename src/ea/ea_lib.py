#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 21:41:50 2021
@author: Fred Valdez Ameneyro
Evolutionary Algorithm's library
"""
import operator as op
import random as rd
from collections import defaultdict
import numpy as np
import src.ea.utilities as ut

#############################################
# Generic Classes ###########################
#############################################

class Individual:
    def __init__(self
            ,representation
            ,generation = None):
        self.representation = representation
        self.appearence_gen = generation
        self.evaluated = False
        self.evaluation = None
        self.comparable_values = {}

    def evaluate(self, data):
        evaluation = self.representation.evaluate(data = data)
        self.evaluation = evaluation
        self.evaluated = True

    def __eq__(self, other):
        for i in range(len(self.comparable_values)):
            if self.comparable_values[i] != other.comparable_values[i]:
                return False
        return True
    
    def __lt__(self, other):
        for i in range(len(self.comparable_values)):
            if self.comparable_values[i] < other.comparable_values[i]:
                return True
            else:
                return False

class Representation:
    def __init__(self):
        pass
		
#############################################
# Cartessian Genetic Programming ############
#############################################

class CGP_Representation(Representation):
    def __init__(self
            ,n_inputs
            ,n_outputs
            ,levels_back
            ,n_rows
            ,n_columns
            ,allow_input_to_output = False
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
        self.allow_input_to_output = allow_input_to_output

        #Inference
        self.n_functions = len(functions)
        self.function_arity = [ut.get_arity(f) for f in functions]
        self.n_function_nodes = n_rows * n_columns
        self.max_lenght = self.n_function_nodes + n_outputs
        self.function_set = [i for i in range(self.n_functions)]
        self.inputs_set = [i for i in range(self.n_inputs)]
        self.all_connections = [i for i in range(self.max_lenght + self.n_inputs - 1)]
        if self.allow_input_to_output:
            self.output_connection_set = [i for i in range(self.max_lenght + self.n_inputs - 1)]
        else:
            self.output_connection_set = [i for i in range(self.n_inputs, self.max_lenght + self.n_inputs - 1)]
        
        #self.connections_set holds the set of indexes available to connect to each column in the graph
        self.connections_set = {}
        for column in range(n_columns):
            if column > 0:
                lower_limit = self.n_inputs + n_rows * (column - levels_back)
                upper_limit = self.n_inputs + n_rows * column
                temp = [i for i in range(lower_limit, upper_limit) if i >= 0]
            else:
                temp = []
            if column - levels_back < 0:
                temp += self.inputs_set
            column_connections = list(set(temp))
            self.connections_set[column] = column_connections

    def create_random(self):
        """Create a random individual"""
        genotype = {}
        for node_index in range(self.n_function_nodes):
            column = int(node_index/self.n_rows)
            output_index = node_index + self.n_inputs
            f_index = rd.choice(self.function_set)
            inputs = [rd.choice(self.connections_set[column]) for _ in range(self.function_arity[f_index])]
            input_dict = {i:inputs[i] for i in range(len(inputs))}
            node = CGP_Node(function_index = f_index
                    ,function = self.functions[f_index]
                    ,output_index = output_index
                    ,column = column
                    ,input_dict = input_dict)
            genotype[output_index] = node
        output_gene = rd.sample(self.output_connection_set,k=self.n_outputs)
        graph = CGP_Graph(genotype = genotype
                ,n_inputs = self.n_inputs
                ,output_gene = output_gene)
        return graph
    
    def point_mutation(self, graph):
        pass




class CGP_Node:
    def __init__(self
            ,function_index
            ,function
            ,output_index
            ,column
            ,input_dict):
        self.function_index = function_index
        self.function = function
        self.output_index = output_index
        self.column = column
        self.inputs = input_dict
        self.active = False

    def get_string(self):
        """Returns the graph as a string"""
        label = [self.function_index]
        label += self.inputs.values()
        return label

    def __str__(self):
        input_label = " i"
        for i in self.inputs.values():
            input_label += " " + str(i)
        if self.active:
            label = "(A)"
        else: label = "(_)"
        label += str(self.output_index) + " f" + str(self.function_index) + input_label
        return label


class CGP_Graph:
    """
    The genotype is a list of nodes,
    each with indexes pointing to their inputs and functions
    """
    def __init__(self
            ,genotype
            ,n_inputs
            ,output_gene):
        self.genotype = genotype
        self.n_inputs = n_inputs
        self.output_gene = output_gene
        self.n_outputs = len(output_gene)
        self.max_graph_lenght = len(self.genotype)
        self.n_available_connections = self.max_graph_lenght + self.n_inputs
        self.find_actives()

    def find_actives(self):
        to_evaluate = defaultdict(lambda:False)

        for p in range(self.n_outputs):
            to_evaluate[self.output_gene[p]] = True

        for p in reversed(range(self.n_inputs,self.n_available_connections)):
            if to_evaluate[p]:
                for i in self.genotype[p].inputs.values():
                    to_evaluate[i] = True
                self.genotype[p].active = True 
        
        self.active_genotype = {k:v for k,v in self.genotype.items() if v.active}

    def evaluate(self, data = None, show_collector=False):
        """
        As in Julian MIller's CGP tutorial:
        https://www.youtube.com/watch?v=qb2R0rL4OHQ&t=625s
        Missing speed test with "while"
        """
        output = {}
        for i,data_row in enumerate(data):
            output_collector = {}
            for p in range(self.n_inputs):
                output_collector[p] = data_row[p]
            for p in range(self.n_inputs, self.n_available_connections):
                if self.genotype[p].active:
                    output_collector[p] = self.genotype[p].function(*[output_collector[c] for c in self.genotype[p].inputs.values()])
                    #inputs = {}
                    #for key,connection in self.genotype[p].inputs.items():
                    #    inputs[key] = output_collector[connection]
                    #output_collector[p] = self.genotype[p].function(*inputs)
            output[i] = [output_collector[i] for i in self.output_gene]

            #debugger
            if show_collector:
                print("output_collector")
                for k,v in output_collector.items():
                    if k >= self.n_inputs:
                        print(k,v,"  ",self.genotype[k])
                    else:
                        print(k,v)
        
        return output

    def __str__(self):
        label = "Graph:\n"
        column = 0
        for node in self.genotype.values():
            if node.column != column:
                label += "\n"
                column += 1
            label += "  " + str(node)
        label += "\nOutput gene:"
        for output_gen in self.output_gene:
            label += " " + str(output_gen)
        return label

    def __eq__(self, other):
        return self.active_genotype == other.active_genotype




