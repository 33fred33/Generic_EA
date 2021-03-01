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
        self.semantics_all = {} 
        self.evaluations = {}

    def update_evaluation(self,objective,value):
        self.evaluations[objective.name] = value
    
    def update_semantics_all(self,semantics_all):
        self.semantics_all = {k:v for k,v in semantics_all.items()}

class Representation:
    def __init__(self):
        pass

class Objective:
    def __init__(self, name, best = None, worst = None, to_max = True):
        self.name = name
        self.to_max = to_max
        self.best = best
        self.worst = worst

def sort_population(population, objectives):
    """
    Inputs
    - population (list of "Individual" instances)
    - objectives: (list of Objective instances) used to compare the individuals
        list is assumed to be sorted by priority level
    Returns
    - (list of "Individual" instances) sorted population
    """
    for objective in reversed(objectives):
        population=sorted(population
            ,key=lambda ind: ind.evaluations[objective.name]
            ,reverse = objective.to_max)
    return population

def tournament_selection(population, tournament_size, objectives):
    """
    Inputs
    - population (list of "Individual" instances)
    - tournament_size: (int) individuals to be contemplated
    - objectives: (list of Objective instances) used to compare the individuals
        list is assumed to be sorted by priority level
    Returns
    - (Individual instance) winner of the tournament
    """
    population_size = len(population)
    assert population_size > tournament_size
    competitors = rd.sample(population,k=tournament_size)
    winner = sort_population(population=competitors, objectives = objectives)[0]
    return winner

def tournament_selection_index(population_size, tournament_size):
    """
    Tournament selection that assumes a sorted population
    Inputs
    - population_size: (int) size of the population to sample from
    - tournament_size: (int) individuals to be contemplated
    Returns
    - (int) index of the winner of the tournament
    """
    assert population_size > tournament_size
    competitors_indexes = rd.sample(range(population_size),k=tournament_size)
    winner_index = min(competitors_indexes)
    return winner_index

def fast_nondominated_sort(population, objectives):
    pass
    """
    Population sorting method proposed in the NSGA-II paper
    The objectives must be conflicting
    Creates the objective "fronts"
    Creates an evaluation of "fronts" in the Individual instances of the population
    Inputs
    - population: (list of Individual instances)
    - objectives: (list of Objective instances)
    Returns
    - (list of Individual instances) sorted population
    """

    """
    assert len(objectives) > 1
    self.fronts = defaultdict(lambda:[])
    for p in population:
        p.dominated_solutions = []
        p.domination_counter = 0
        for q in population:
            if self._dominates(p,q):
                p.dominated_solutions.append(q)
            elif self._dominates(q,p):
                p.domination_counter = p.domination_counter + 1
        if p.domination_counter == 0:
            p.rank = 1
            self.fronts[1].append(p)
    
    front_index = 1
    while self.fronts[front_index] != []:
        temporal_front = []
        for p in self.fronts[front_index]:
            for q in p.dominated_solutions:
                q.domination_counter = q.domination_counter - 1
                if q.domination_counter == 0:
                    q.rank = front_index + 1
                    temporal_front.append(q)
        front_index = front_index + 1
        self.fronts[front_index] = temporal_front
    """


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
            ,inputs_available_to_all_columns = False
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
        self.inputs_available_to_all_columns = inputs_available_to_all_columns

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
        self.connections_set = {col:self.get_available_connections(col) for col in range(self.n_columns)}

    def get_available_connections(self, column):
        """
        Returns a list with all the indexes of the connections available for the current column
        They always include the input indexes
        """
        if column > 0:
            lower_limit = self.n_inputs + self.n_rows * (column - self.levels_back)
            upper_limit = self.n_inputs + self.n_rows * column
            temp = [i for i in range(lower_limit, upper_limit) if i >= 0]
        else:
            temp = []
        
        if self.inputs_available_to_all_columns:
            temp += self.inputs_set
        else:
            if column - self.levels_back < 0:
                temp += self.inputs_set
        return list(set(temp))
        
    def create_random(self, seed = None):
        """Create a random individual"""
        if seed is not None:
            rd.seed(seed)
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
    
    def point_mutation(self, graph, rate):
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
    The output of the graph is a dictionary, with:
        key: the index of the output gene
        value: the output value
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
        self.active_genotype = {}
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

    def evaluate(self, data_row=None, show_collector=False):
        """
        Returns a dictionary:
            key: the index of the row of the data
            value: another dictionary with:
                key: the index of the output gene
                value: the output of the graph for that output gene
        As in Julian MIller's CGP tutorial:
        https://www.youtube.com/watch?v=qb2R0rL4OHQ&t=625s
        """
        output_collector = {}
        for p in range(self.n_inputs):
            output_collector[p] = data_row[p]
        #for p in range(self.n_inputs, self.n_available_connections):
        #    if self.genotype[p].active:
        #        output_collector[p] = self.genotype[p].function(*[output_collector[c] for c in self.genotype[p].inputs.values()])
        for p,v in self.active_genotype.items():
            output_collector[p] = self.genotype[p].function(*[output_collector[c] for c in self.genotype[p].inputs.values()])

        #debugger
        if show_collector:
            print("output_collector")
            for k,v in output_collector.items():
                if k >= self.n_inputs:
                    print(k,v,"  ",self.genotype[k])
                else:
                    print(k,v)
        return {k:output_collector[i] for k,i in enumerate(self.output_gene)}
        #return output

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




