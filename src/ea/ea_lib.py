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
        """
        Inputs
        - name (string) 
        - to_max (bool)
        - best
        - worst
        """
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

def get_nsgaii_objectives(front_name = "front", crowding_distance_name = "cd"):
    fo = Objective(name=front_name, to_max = False, best=1)
    cdo = Objective(name=crowding_distance_name, to_max = True, best=0)
    return [fo, cdo]

def _dominates(p, q, objectives):
    """
    Pareto dominance
    Inputs
    - p, q: (Individual instances)
    - objectives: (list of Objective instances)
    Returns
    - (bool) p dominates q?
    """
    for objective in objectives:
        p_val = p.evaluations[objective.name]
        q_val = q.evaluations[objective.name]
        if objective.to_max:
            if p_val < q_val:
                return False
        else:
            if p_val > q_val:
                return False
    return True

def _set_ranks(population, conflicting_objectives, front_objective):
    """
    Ranks the population according to their pareto dominance front
    The front is stored as an evaluation in every Individual instance
    Inputs
    - population: (list of Individual instances)
    - objectives: (list of Objective instances)
    - front_objective: (Objective instance) 
    """

    assert len(conflicting_objectives) > 1

    #initialisation
    fronts = defaultdict(lambda:[])

    #front 1
    for p in population:
        p.dominated_solutions = []
        p.domination_counter = 0
        for q in population:
            if _dominates(p,q, conflicting_objectives):
                p.dominated_solutions.append(q)
            elif _dominates(q,p, conflicting_objectives):
                p.domination_counter = p.domination_counter + 1
        if p.domination_counter == 0:
            p.evaluations[front_objective.name] = 1
            fronts[1].append(p)
    
    #rest of the fronts
    front_index = 1
    while fronts[front_index] != []:
        temporal_front = []
        for p in fronts[front_index]:
            for q in p.dominated_solutions:
                q.domination_counter = q.domination_counter - 1
                if q.domination_counter == 0:
                    q.evaluations[front_objective.name] = front_index + 1
                    temporal_front.append(q)
        front_index = front_index + 1
        fronts[front_index] = temporal_front
    
def _set_crowding_distances_by_front(population
        ,conflicting_objectives
        ,front_objective
        ,cd_objective):
    """
    Ranks the population according to their crowding distance
    The crowding distance is stored as an evaluation in every Individual instance
    Inputs
    - population: (list of Individual instances)
    - conflicting_objectives: (list of Objective instances)
    - front_objective: (Objective instance) 
    - cd_objective: (Objective instance)
    """
    assert len(conflicting_objectives) > 1

    #Initialisation
    for ind in population:
        ind.evaluations[cd_objective.name] = 0
    
    front = 1
    pop_set = [ind for ind in population if ind.evaluations[front_objective.name] == front]
    while pop_set != []:
        for objective in conflicting_objectives:
            sorted_population = sort_population(population = pop_set, objectives = [objective])
            best_individual = sorted_population[0]
            worst_individual = sorted_population[-1]
            best_value = best_individual.evaluations[objective.name]
            worst_value = worst_individual.evaluations[objective.name]
            gap = abs(best_value - worst_value)
            best_individual.evaluations[cd_objective.name] = np.inf
            worst_individual.evaluations[cd_objective.name] = np.inf
            if gap != 0:
                for idx, ind in enumerate(sorted_population[1:-1]):
                    ind.evaluations[cd_objective.name] = ind.evaluations[cd_objective.name] + abs((sorted_population[idx + 2].evaluations[objective.name] - sorted_population[idx].evaluations[objective.name])/gap)            
        front += 1
        pop_set = [ind for ind in population if ind.evaluations[front_objective.name] == front]

def fast_nondominated_sort(population
        ,conflicting_objectives
        ,nsgaii_objectives):
    """
    Population sorting method proposed in the NSGA-II paper
    The objectives must be conflicting
    Inputs
    - population: (list of Individual instances)
    - conflicting_objectives: (list of Objective instances)
    - nsgaii_objectives: (list of Objective instances). They can be obtained
        with the "get_nsgaii_objectives" method
    Returns
    - (list of Individual instances) sorted population
    """
    front_objective = nsgaii_objectives[0]
    cd_objective = nsgaii_objectives[1]

    _set_ranks(population = population
        ,conflicting_objectives = conflicting_objectives
        ,front_objective = front_objective)

    _set_crowding_distances_by_front(population = population
        ,conflicting_objectives = conflicting_objectives
        ,front_objective = front_objective
        ,cd_objective = cd_objective)

    sorted_population = sort_population(population = population
        ,objectives = [front_objective, cd_objective])
        
    return sorted_population

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
        graph.find_actives()
        return graph

    def mutate_function_gene(self, function_gene):
        pass

    def mutate_input_gene(self, node):
        pass

    def get_random_output_gene(self, value_to_avoid):
        """
        Fast method to sample
        """
        options = rd.sample(self.output_connection_set, k=2)
        if options[0] != value_to_avoid:
            new_gene = options[0]
        else:
            new_gene = options[1]
        return new_gene

    def get_random_function_index(self, value_to_avoid):
        """
        Fast method to sample
        """
        assert len(self.function_set) > 1
        options = rd.sample(self.function_set, k=2)
        if options[0] != value_to_avoid:
            new_gene = options[0]
        else:
            new_gene = options[1]
        return new_gene

    def point_mutation(self, graph, percentage):
        """
        Inputs
        - graph (CGP_Graph instance)
        - percentage (float)
        Returns
        - (CGP_Graph instance) The mutated graph
        """

        #Create a copy of the original graph, to change it later
        new_graph = graph.copy()

        #Calculate useful variables
        nodes_list = list(graph.genotype.values())
        n_function_genes = graph.max_lenght + sum([n.function_arity for n in graph.genotype.values()])
        n_genes = n_function_genes + graph.n_outputs
        mutations = int(n_genes * percentage / 100)

        #Iterate mutations times
        for _ in range(mutations):
            int_to_mutate = rd.randint(0, n_genes-1)
            mutate_output_index = int_to_mutate + 1 - n_function_genes
            
            #if the random number falls in the output gene
            if mutate_output_index >= 0:
                new_gene = self.get_random_output_gene(value_to_avoid = new_graph.output_gene[mutate_output_index])
                new_graph.output_gene[mutate_output_index] = new_gene
            else:
                node = rd.choice(nodes_list)

                #Randomly select if the mutation will affect the function or the inputs
                int_mutation = rd.randint(0, node.function_arity)
                if int_mutation == node.function_arity:
                    print("mutate_function") ######HERE
                else:
                    print("mutate input", int_mutation)
        return new_graph


    def probabilistic_mutation(self, graph, probability):
        pass

class CGP_Node:
    def __init__(self
            ,function_index
            ,function
            ,output_index
            ,column
            ,input_dict):
        """
        Inputs
        function_index: (int) reference to the function 
            in the functions variable in the CGP_Representation class
        function: (function)
        output_index: (int) address of this node's output
        column: (int) column of this node in the graph
        input_dict: (dict with
            key: (int) index of the input in the function
            value: (int) the address of other node's output)
        """

        #Assignation
        self.function_index = function_index
        self.function = function
        self.output_index = output_index
        self.column = column
        self.inputs = input_dict

        #Initialisation
        self.active = False
        self.function_arity = ut.get_arity(function)

    def copy(self):
        the_copy = CGP_Node(function_index = self.function_index
            ,function = self.function
            ,output_index = self.output_index
            ,column = self.column
            ,input_dict = {k:v for k,v in self.inputs.items()})
        return the_copy

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

    def __eq__(self, other):
        return self.function_index == other.function_index and self.inputs == other.inputs

class CGP_Graph:
    def __init__(self
            ,genotype
            ,n_inputs
            ,output_gene):
        """
        Inputs
        genotype: (dictionary with:
            key: (int) node index
            value: (CGP_Node instance))
        n_inputs: (int) number of inputs of the graph
        output_gene: (list of ints) each int is a reference to a node in the genotype
        """
        self.genotype = genotype
        self.n_inputs = n_inputs
        self.output_gene = output_gene

        self.n_outputs = len(output_gene)
        self.max_lenght = len(self.genotype)
        self.n_available_connections = self.max_lenght + self.n_inputs
        self.active_genotype = {}

    def copy(self):
        the_copy = CGP_Graph(genotype = {k:v.copy() for k,v in self.genotype.items()}
            ,n_inputs = self.n_inputs
            ,output_gene = [i for i in self.output_gene])
        return the_copy

    def find_actives(self):
        """
        Updates the self.active_genotype variable with the
        self.active_genotype is the active sample of self.genotype
        """
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
        if self.active_genotype == {}:
            self.find_actives()
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




