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
import matplotlib.pyplot as plt
import os
import errno
import csv
import pickle
import statistics as stat
import math
import pandas as pd


#############################################
# Generic Classes ###########################
#############################################

class Individual:
    def __init__(self
            ,representation
            ,created_in_gen
            ,parent_index = None
            ,parent = None):
        """
        Inputs
        - representation: (instance from class) asssigned individual characteristics
        - generation: (int) generation of creation of the individual
        """
        self.representation = representation
        self.semantics_all = {} 
        self.evaluations = {}
        self.created_in_gen = created_in_gen
        self.parent_index = parent_index
        self.parent = parent

        #For logs
        self.semantic_distance_from_parent = None
        self.damaged_semantics_from_parent = None
        self.improved_semantics_from_parent = None
        self.semantic_change_balance = None
        self.active_nodes_diff_from_parent = None

    def update_evaluation(self,objective,value):
        self.evaluations[objective.name] = value
    
    def update_semantics_all(self,semantics_all):
        self.semantics_all = {k:v for k,v in semantics_all.items()}

    def __Str__(self):
        label = "Ind gen " + str(self.generation) + " R: " + str(self.representation)
        return label
 
class Representation:
    def __init__(self):
        pass

class Objective:
    def __init__(self, name, best = None, worst = None, to_max = True, eval_function = None):
        """
        Inputs
        - name (string) 
        - to_max (bool)
        - best
        - worst
        - eval_function (function) function to be called when evaluating the objective
        """
        self.name = name
        self.to_max = to_max
        self.best = best
        self.worst = worst
        self.eval_function = eval_function


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
  
def inds_same_by_evals(ind1, ind2, objectives):
    """
    Inputs
    - ind1: (Individual instance)
    - ind2: (Individual instance)
    - objectives: (list of Objective instances)
    Returns
    - (bool) Are the evaluations of ind1 and ind2 the same for this objectives?
    """
    for obj in objectives:
        if ind1.evaluations[obj.name] != ind2.evaluations[obj.name]:
            return False
    return True
    
def get_unique_inds_by_evals(population, objectives):
    """
    For plotting purposes
    Inputs
    - population: (list of Individual instances)
    - objectives: (list of Objective instances)
    Returns
    - (list of list of floats) one list per objective
    - (list of ints) the amount of copies found for each ind
    """
    filtered_population = []
    counts = []
    for idx1,ind1 in enumerate(population):
        add = True
        for ind2 in population[:idx1]:
            if inds_same_by_evals(ind1, ind2, objectives):
                add = False
                break
        if add:
            filtered_population.append(ind1)
            count = 1
            for ind2 in population[idx1+1:]:
                if inds_same_by_evals(ind1, ind2, objectives):
                    count += 1
            counts.append(count)

    evals = []
    for obj in objectives:
        x = [i.evaluations[obj.name] for i in filtered_population]
        evals.append(x)

    return evals, counts

#Semantics
def semantic_distance(ind1, ind2, semantic_indexes):
    """
    Inputs
    - ind1, ind2 (Individual instances)
    - semantic_indexes (list of ints)
    Returns
    - (float) Average of absolute difference in semantics
    """
    return sum([abs(ind1.semantics_all[k]-ind2.semantics_all[k]) for k in semantic_indexes]) / len(semantic_indexes)

def get_semantic_peculiarity_objective(name = "semantic_peculiarity"):
    """
    Inputs:
    - name (string): name of the objective
    Returns:
    - (Objective instance) name = name
    """
    obj = Objective(name=name, to_max = True, best=1, worst=0)
    return obj

def set_semantic_peculiarity_by_front(population
        ,conflicting_objectives
        ,front_objective
        ,cd_objective):
    pass

def semantic_peculiarity(population, output_vector, semantic_indexes, sp_objective, b=math.sqrt(2)):
    """
    Updates the semantic peculiarity in every individual as one of
    their evaluations, with key = sp_objective.name
    Inputs:
    - population (list of Individual instances)
    - output_vector (list of labels) the full set of outputs of the fitness cases
    - semantic_indexes (list of ints): list of indexes that define the semantic set
    - sp_objective (Objective instance)
    Returns: 
    - (dict with fitness_case_index:semantic_relevance)
    """
    s = len(semantic_indexes)
    n = len(population)
    represented_fs = defaultdict(lambda:[])
    r = {}
    p = {}
    for f in semantic_indexes:
        c = 0
        for ind_idx, ind in enumerate(population):
            if output_vector[f]==ind.semantics_all[f]:
                represented_fs[ind_idx].append(f)
                c += 1
        r[f] = (1 - (c/n))**b
        #r[f] = 1-math.sqrt(1-(x-1)**2)
    for ind_idx, ind in enumerate(population):
        semantic_peculiarity = sum([r[f] for f in represented_fs[ind_idx]])/s
        #semantic_peculiarity = max([r[f] for f in represented_fs[ind_idx]])
        ind.evaluations[sp_objective.name] = semantic_peculiarity
        p[ind_idx] = semantic_peculiarity
    return p, r, represented_fs

def max_semantic_peculiarity(population, output_vector, semantic_indexes, sp_objective):
    """
    Updates the semantic peculiarity in every individual as one of
    their evaluations, with key = sp_objective.name
    Inputs:
    - population (list of Individual instances)
    - output_vector (list of labels) the full set of outputs of the fitness cases
    - semantic_indexes (list of ints): list of indexes that define the semantic set
    - sp_objective (Objective instance)
    Returns: 
    - (dict with fitness_case_index:semantic_relevance)
    """
    s = len(semantic_indexes)
    n = len(population)
    represented_fs = defaultdict(lambda:[])
    r = {}
    p = {}
    for f in semantic_indexes:
        c = 0
        for ind_idx, ind in enumerate(population):
            if output_vector[f]==ind.semantics_all[f]:
                represented_fs[ind_idx].append(f)
                c += 1
        r[f] = (1 - (c/n))**b
        #r[f] = 1-math.sqrt(1-(x-1)**2)
    for ind_idx, ind in enumerate(population):
        #semantic_peculiarity = sum([r[f] for f in represented_fs[ind_idx]])/s
        semantic_peculiarity = max([r[f] for f in represented_fs[ind_idx]])
        ind.evaluations[sp_objective.name] = semantic_peculiarity
        p[ind_idx] = semantic_peculiarity
    return p, r, represented_fs

#MOEA exclusive methods
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

def objective_space_euclidean_distance(ind1, ind2, objectives):
    pd = 0
    for obj in objectives:
        d = abs(ind1.evaluations[obj.name] - ind2.evaluations[obj.name])
        pd = math.sqrt(pd**2 + d**2)
    return pd

def hyperarea(population, objectives, front_objective):
    """
    Inputs
    population: (list of Individual instances) Individuals part of the pareto set
    objectives: (list of Objective instances (len=2))
    Returns
    (float) The hyperarea
    """
    assert len(objectives) == 2

    #Variables
    obj0 = objectives[0]
    obj1 = objectives[1]
    hyperarea = 0

    #Filter population to get pareto front only and sort
    set_ranks(population, objectives, front_objective)
    front_pop = [i for i in population if i.evaluations[front_objective.name]==1]
    sorted_pop = sort_population(front_pop, [obj0])

    #Iteration
    for ind_idx, ind in enumerate(reversed(sorted_pop)):
        if ind_idx == 0:
            last_value = ind.evaluations[obj0.name]
            hyperarea += (abs(last_value - obj0.worst)
                        *abs(ind.evaluations[obj1.name] - obj1.worst))
        else:
            new_value = ind.evaluations[obj0.name]
            if (new_value != last_value):
                hyperarea += (abs(new_value - last_value)
                        *abs(ind.evaluations[obj1.name] - obj1.worst))
                last_value = new_value
    return hyperarea

def plot_pareto(population, objectives, cluster_type = "size", front_objective=None, path = None, name = None):
    unique_evals, counts = get_unique_inds_by_evals(population, objectives)
    percentages = [c*100/sum(counts) for c in counts]
    
    if cluster_type == "size":
        sizes = [p * 15 for p in percentages]
        plt.scatter(unique_evals[0], unique_evals[1], alpha=0.5,facecolors="none", color="blue",s=sizes, edgecolor="blue")
        plt.scatter(unique_evals[0], unique_evals[1], alpha=0.5, color="blue", marker="x", s=sizes)
        plt.title("Known pareto front")
        plt.grid(alpha=0.5, linestyle="--")
        if path is not None:
            path = ut.verify_path(path)
            plt.savefig(path + name + ".png")
        else:
            plt.show()
    
    elif cluster_type == "color":
        plt.scatter(unique_evals[0], unique_evals[1], c=percentages, cmap='viridis', alpha=0.7, marker="x")
        plt.title("Pareto front distribution")
        plt.colorbar(label="% of population clustered")
        plt.grid(alpha=0.5, linestyle="--")
        if path is not None:
            path = ut.verify_path(path)
            plt.savefig(path + name + ".png")
        else:
            plt.show()
    
    else:
        print("Wrong cluster_type")
    plt.close('all')

#NSGAII exclusive methods
def get_nsgaii_objectives(front_name = "front", crowding_distance_name = "cd"):
    fo = Objective(name=front_name, to_max = False, best=1)
    cdo = Objective(name=crowding_distance_name, to_max = True,  worst=0)
    return [fo, cdo]

def get_front_n(population, front_objective, n=1):
    return [i for i in population if i.evaluations[front_objective.name]==n]

def set_ranks(population, conflicting_objectives, front_objective):
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
    
def set_crowding_distances_by_front(population
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

def fast_nondominated_sort(population, conflicting_objectives, nsgaii_objectives):
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

    set_ranks(population = population
        ,conflicting_objectives = conflicting_objectives
        ,front_objective = front_objective)

    set_crowding_distances_by_front(population = population
        ,conflicting_objectives = conflicting_objectives
        ,front_objective = front_objective
        ,cd_objective = cd_objective)

    sorted_population = sort_population(population = population
        ,objectives = [front_objective, cd_objective])
        
    return sorted_population

#SPEA2 exclusive methods
def get_spea2_objective(spea2_fitness_name = "spea2_fitness"):
    spea2_fitness_objective = Objective(name=spea2_fitness_name, to_max = False, best=0)
    return spea2_fitness_objective

def spea2_sort(population, conflicting_objectives, spea2_objective):
    pop_size = len(population)
    k_neighbor_idx = int(math.sqrt(pop_size))
    strengths = {i:0 for i in range(pop_size)}
    dominators = {i:[] for i in range(pop_size)}
    distances = {j:{i:np.inf for i in range(pop_size)} for j in range(pop_size)}
    for ind_idx, ind in enumerate(population[:-1]):
        for temp_idx, ind2 in enumerate(population[ind_idx+1:]):
            ind2_idx = temp_idx + ind_idx + 1
            d = objective_space_euclidean_distance(ind, ind2, conflicting_objectives)
            distances[ind_idx][ind2_idx] = d
            distances[ind2_idx][ind_idx] = d
            if _dominates(ind, ind2, conflicting_objectives):
                strengths[ind_idx] += 1
                dominators[ind2_idx] += [ind_idx]
            elif _dominates(ind2, ind, conflicting_objectives):
                strengths[ind2_idx] += 1
                dominators[ind_idx] += [ind2_idx]
                

    for ind_idx, ind in enumerate(population):
        k_neighbor_d = sorted(distances[ind_idx].values())[k_neighbor_idx]
        density = 1/(k_neighbor_d+2)
        spea2_fitness = sum([strengths[d] for d in dominators[ind_idx]]) + density
        ind.evaluations[spea2_objective.name] = spea2_fitness

    sorted_population = sort_population(population = population
        ,objectives = [spea2_objective])
    
    return sorted_population

#Logs and data
def get_cgp_log(population, representation, current_gen, clustering_objectives):
    """
    Inputs
    - population: (list of Individual instances)}
    - representation: (CGP_Representation instance)
    - current_gen: (int)
    - clustering_objectives: (list of Objective instances)
    Returns
    - (list of strings) header of the individual level logs
    - (list of lists of numbers) individual level logs
    - (list of strings) header of the generation level logs
    - (list of lists of numbers) generation level logs
    """
    pop_size = len(population)
    new_ind_pop = [ind for ind in population if ind.created_in_gen == current_gen]
    new_ind_pop_size = len(new_ind_pop)
    model = population[0]
    evaluation_keys = model.evaluations.keys()
    eval_names = model.evaluations.keys()
    n_evals = len(eval_names)
    f_idxs = [i for i in range(representation.n_functions)]
    logs = []
    _, cluster_counts = get_unique_inds_by_evals(population, clustering_objectives)

    #Variables to gather generational info
    nodes_evaluated = 0
    total_age = 0
    total_actives = 0
    total_skips = 0
    total_funcs_count = [0 for _ in f_idxs]
    total_used_inputs_count = [0 for _ in range(model.representation.n_inputs)]
    total_evals = [0 for _ in range(n_evals)]
    list_semantic_distance_from_parent = []
    list_damaged_semantics_from_parent = []
    list_improved_semantics_from_parent = []
    list_semantic_change_balance = []
    list_active_nodes_diff_from_parent = []


    #Define the header
    header = ["Index"
            ,"Generation"
            ,"Age"
            ,"Graph_n_inputs"
            ,"Graph_n_outputs"
            ,"Graph_max_lenght"
            ,"Graph_actives"
            ,"Ind_eval_skip"
            ,"Semantic_distance_from_parent"
            ,"Damaged_semantics_from_parent"
            ,"Improved_semantics_from_parent"
            ,"Semantic_change_balance"
            ,"Active_nodes_diff_from_parent"
            #,"Graph"
            ]
    header += ["f"+str(f_idx)+"_count" for f_idx in f_idxs]
    header += ["i"+str(i)+"_count" for i in range(representation.n_inputs)]
    header += [name for name in eval_names]
    header += ["parent_"+name for name in eval_names]

    gen_header = ["Gen"
                ,"Pop_size"
                ,"Eval_nodes"
                ,"Avg_age"
                ,"Avg_active_nodes"
                ,"Eval_skips"
                ,"Eval_skips_ptg"
                ,"Avg_semantic_distance_from_parent"
                ,"Std_semantic_distance_from_parent"
                ,"Avg_damaged_semantics_from_parent"
                ,"Std_damaged_semantics_from_parent"
                ,"Avg_improved_semantics_from_parent"
                ,"Std_improved_semantics_from_parent"
                ,"Avg_semantic_change_balance"
                ,"Std_semantic_change_balance"
                ,"Avg_active_nodes_diff_from_parent" #Number of active nodes only
                ,"Std_active_nodes_diff_from_parent"
                ,"Fitness_evals"
                ,"Semantic_different_5"
                ,"Semantic_different_10"
                ,"Semantic_different_15"
                ,"Semantic_different_20"
                ,"Semantic_different_25"
                ,"Semantic_different_30"
                ,"Semantic_different_35"
                ,"Semantic_different_40"
                ,"Semantic_different_45"
                ,"Semantic_different_50"
                ,"N_clusters"
                ,"Max_cluster_size_rate"
                ,"Clustered_pop_rate"
                ,"Avg_cluster_size_rate"]
    gen_header += ["f"+str(f_idx)+"_presence" for f_idx in f_idxs]
    gen_header += ["i"+str(i)+"_presence" for i in range(representation.n_inputs)]
    gen_header += ["Avg_" + name for name in eval_names]
    
    for idx, ind in enumerate(population):
        graph = ind.representation
        graph_actives = len(graph.active_genotype)

        #Gathered data for gen_logs
        total_age += current_gen - ind.created_in_gen
        total_actives += graph_actives
        if ind.created_in_gen == current_gen:
            list_semantic_distance_from_parent.append(ind.semantic_distance_from_parent)
            list_damaged_semantics_from_parent.append(ind.damaged_semantics_from_parent)
            list_improved_semantics_from_parent.append(ind.improved_semantics_from_parent)
            list_semantic_change_balance.append(ind.semantic_change_balance)
            list_active_nodes_diff_from_parent.append(ind.active_nodes_diff_from_parent)

        #Ind_eval_skip
        if graph.evaluation_skipped:
            evaluation_skipped =  1
            if ind.created_in_gen == current_gen:
                total_skips += 1
        else: 
            evaluation_skipped =  0
            if ind.created_in_gen == current_gen:
                nodes_evaluated += graph_actives

        #Node level data
        funcs_count = [0 for _ in f_idxs]
        used_inputs_count = [0 for _ in range(graph.n_inputs)]
        for node in graph.active_genotype.values():
            funcs_count[node.function_index] += 1
            for input_idx in node.inputs.values():
                if input_idx < graph.n_inputs:
                    used_inputs_count[input_idx] += 1
        evals = [ind.evaluations[name] for name in eval_names]
        if ind.parent is not None:
            parent_evals = [ind.parent.evaluations[name] for name in eval_names]
        else:
            parent_evals = [None for name in eval_names]
        total_evals = [total_evals[i] + evals[i] for i in range(n_evals)]
        total_funcs_count = [total_funcs_count[i] + funcs_count[i] for i in f_idxs]
        total_used_inputs_count = [total_used_inputs_count[i] + used_inputs_count[i] for i in range(graph.n_inputs)]

        #Individual level logs
        ind_row = [idx
            ,current_gen
            ,current_gen - ind.created_in_gen
            ,graph.n_inputs
            ,graph.n_outputs
            ,graph.max_lenght
            ,graph_actives
            ,evaluation_skipped
            ,ind.semantic_distance_from_parent
            ,ind.damaged_semantics_from_parent
            ,ind.improved_semantics_from_parent
            ,ind.semantic_change_balance
            ,ind.active_nodes_diff_from_parent
            #,str(ind.representation)
            ]
        ind_row += funcs_count
        ind_row += used_inputs_count
        ind_row += evals
        ind_row += parent_evals
        logs.append(ind_row)

    #No new individuals in current gen (possible if a subpopulation is passed to this method)
    if new_ind_pop_size == 0:
        list_semantic_distance_from_parent = [0]
        list_damaged_semantics_from_parent = [0]
        list_improved_semantics_from_parent = [0]
        list_semantic_change_balance = [0]
        list_active_nodes_diff_from_parent = [0]
        new_ind_pop_size = 1
        Semantic_different_5 = "N/A"
        Semantic_different_10 = "N/A"
        Semantic_different_15 = "N/A"
        Semantic_different_20 = "N/A"
        Semantic_different_25 = "N/A"
        Semantic_different_30 = "N/A"
        Semantic_different_35 = "N/A"
        Semantic_different_40 = "N/A"
        Semantic_different_45 = "N/A"
        Semantic_different_50 = "N/A"
    else:
        list_semantic_distance_from_parent = [x for x in list_semantic_distance_from_parent if x is not None]
        list_damaged_semantics_from_parent = [x for x in list_damaged_semantics_from_parent if x is not None]
        list_improved_semantics_from_parent = [x for x in list_improved_semantics_from_parent if x is not None]
        list_semantic_change_balance = [x for x in list_semantic_change_balance if x is not None]
        list_active_nodes_diff_from_parent = [x for x in list_active_nodes_diff_from_parent if x is not None]
        Semantic_different_5 = sum([1 for sd in list_semantic_distance_from_parent if sd>=0.05])/new_ind_pop_size
        Semantic_different_10 = sum([1 for sd in list_semantic_distance_from_parent if sd>=0.1])/new_ind_pop_size
        Semantic_different_15 = sum([1 for sd in list_semantic_distance_from_parent if sd>=0.15])/new_ind_pop_size
        Semantic_different_20 = sum([1 for sd in list_semantic_distance_from_parent if sd>=0.2])/new_ind_pop_size
        Semantic_different_25 = sum([1 for sd in list_semantic_distance_from_parent if sd>=0.25])/new_ind_pop_size
        Semantic_different_30 = sum([1 for sd in list_semantic_distance_from_parent if sd>=0.3])/new_ind_pop_size
        Semantic_different_35 = sum([1 for sd in list_semantic_distance_from_parent if sd>=0.35])/new_ind_pop_size
        Semantic_different_40 = sum([1 for sd in list_semantic_distance_from_parent if sd>=0.4])/new_ind_pop_size
        Semantic_different_45 = sum([1 for sd in list_semantic_distance_from_parent if sd>=0.45])/new_ind_pop_size
        Semantic_different_50 = sum([1 for sd in list_semantic_distance_from_parent if sd>=0.5])/new_ind_pop_size

    if cluster_counts != []:
        N_clusters = 0
        Max_cluster_size_rate = 0
        Clustered_pop_rate = 0
        Avg_cluster_size_rate = 0
    else:
        relevant_clusters = [i for i in cluster_counts if i>1]
        N_clusters = len(relevant_clusters)
        Max_cluster_size_rate = max(relevant_clusters)/pop_size
        Clustered_pop_rate = sum(relevant_clusters)/pop_size
        if len(relevant_clusters) > 0:
            Avg_cluster_size_rate = stat.mean(relevant_clusters)/pop_size

    if len(list_semantic_distance_from_parent) > 1:
        stdev_list_semantic_distance_from_parent = stat.stdev(list_semantic_distance_from_parent)
    else:
        stdev_list_semantic_distance_from_parent = 0
    if len(list_damaged_semantics_from_parent) > 1:
        stdev_list_damaged_semantics_from_parent = stat.stdev(list_damaged_semantics_from_parent)
    else:
        stdev_list_damaged_semantics_from_parent = 0
    if len(list_improved_semantics_from_parent) > 1:
        stdev_list_improved_semantics_from_parent = stat.stdev(list_improved_semantics_from_parent)
    else:
        stdev_list_improved_semantics_from_parent = 0
    if len(list_semantic_change_balance) > 1:
        stdev_list_semantic_change_balance = stat.stdev(list_semantic_change_balance)
    else:
        stdev_list_semantic_change_balance = 0
    if len(list_active_nodes_diff_from_parent) > 1:
        stdev_list_active_nodes_diff_from_parent = stat.stdev(list_active_nodes_diff_from_parent)
    else:
        stdev_list_active_nodes_diff_from_parent = 0

    gen_row = [current_gen
            ,pop_size
            ,nodes_evaluated
            ,total_age/pop_size
            ,total_actives/pop_size
            ,total_skips
            ,total_skips*100/new_ind_pop_size
            ,stat.mean(list_semantic_distance_from_parent)
            ,stdev_list_semantic_distance_from_parent
            ,stat.mean(list_damaged_semantics_from_parent)
            ,stdev_list_damaged_semantics_from_parent
            ,stat.mean(list_improved_semantics_from_parent)
            ,stdev_list_improved_semantics_from_parent
            ,stat.mean(list_semantic_change_balance)
            ,stdev_list_semantic_change_balance
            ,stat.mean(list_active_nodes_diff_from_parent)
            ,stdev_list_active_nodes_diff_from_parent
            ,new_ind_pop_size-total_skips
            ,Semantic_different_5
            ,Semantic_different_10
            ,Semantic_different_15
            ,Semantic_different_20
            ,Semantic_different_25
            ,Semantic_different_30
            ,Semantic_different_35
            ,Semantic_different_40
            ,Semantic_different_45
            ,Semantic_different_50
            ,N_clusters
            ,Max_cluster_size_rate
            ,Clustered_pop_rate
            ,Avg_cluster_size_rate
            ]
    gen_row += [total_funcs_count[i]*100/sum(total_funcs_count) for i in f_idxs]
    gen_row += [total_used_inputs_count[i]*100/sum(total_used_inputs_count) for i in range(graph.n_inputs)]
    gen_row += [v/pop_size for v in total_evals]
    #gen_row += [v/pop_size for v in total_evals]
    

    
    return header, logs, gen_header, gen_row

def individual_log(ind, representation, current_gen):
    """
    Inputs
    - ind (Individual instance)
    Returns
    - (list of lists of strings) log
    - (list of strings) header
    """
    f_idxs = [i for i in range(representation.n_functions)]
    eval_names = ind.evaluations.keys()
    header = ["Index"
            ,"Generation"
            ,"Age"
            ,"Graph_n_inputs"
            ,"Graph_n_outputs"
            ,"Graph_max_lenght"
            ,"Graph_actives"
            ,"Active_rate"
            ,"Ind_eval_skip"
            ,"semantic_distance_from_parent"
            ,"damaged_semantics_from_parent"
            ,"improved_semantics_from_parent"
            ,"semantic_change_balance"
            ,"active_nodes_diff_from_parent"
            #,"Semantics"
            ]
    header += ["f"+str(f_idx)+"_count" for f_idx in f_idxs]
    header += ["i"+str(i)+"_count" for i in range(representation.n_inputs)]
    header += [name for name in eval_names]

    graph = ind.representation
    graph_actives = len(graph.active_genotype)

    #Ind_eval_skip
    if graph.evaluation_skipped:
        evaluation_skipped =  1
    else: 
        evaluation_skipped =  0

    #Node level data
    funcs_count = [0 for _ in f_idxs]
    used_inputs_count = [0 for _ in range(graph.n_inputs)]
    for node in graph.active_genotype.values():
        funcs_count[node.function_index] += 1
        for input_idx in node.inputs.values():
            if input_idx < graph.n_inputs:
                used_inputs_count[input_idx] += 1
    evals = [ind.evaluations[name] for name in eval_names]
    total_funcs_count = [total_funcs_count[i] + funcs_count[i] for i in f_idxs]
    total_used_inputs_count = [total_used_inputs_count[i] + used_inputs_count[i] for i in range(graph.n_inputs)]

    #Individual level logs
    ind_row = [current_gen
        ,current_gen
        ,current_gen - ind.created_in_gen
        ,graph.n_inputs
        ,graph.n_outputs
        ,graph.max_lenght
        ,graph_actives
        ,graph_actives/graph.max_lenght
        ,evaluation_skipped
        ,ind.semantic_distance_from_parent
        ,ind.damaged_semantics_from_parent
        ,ind.improved_semantics_from_parent
        ,ind.semantic_change_balance
        ,ind.active_nodes_diff_from_parent
        #,ind.semantics_all.items()
        ]
    ind_row += funcs_count
    ind_row += used_inputs_count
    ind_row += evals
    
    return ind_row, header

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
            ,functions
            ,allow_input_to_output = False
            ,inputs_available_to_all_columns = False
            ):
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

    def get_random_output_gene(self, value_to_avoid=None):
        """
        Inputs
        - value_to_avoid: (int) index of the referenced gene
        Returns
        - (int) distinct valid index of the referenced gene
        """
        if value_to_avoid is None:
            return rd.choice(self.output_connection_set)
        
        options = rd.sample(self.output_connection_set, k=2)
        if options[0] != value_to_avoid:
            new_gene = options[0]
        else:
            new_gene = options[1]
        return new_gene

    def get_random_function_index(self, value_to_avoid=None):
        """
        Inputs
        - value_to_avoid: (int) index of the function in self.function_set
        Returns
        - (int) distinct valid function index
        """
        if value_to_avoid is None:
            return rd.choice(self.function_set)

        options = rd.sample(self.function_set, k=2)
        if options[0] != value_to_avoid:
            new_gene = options[0]
        else:
            new_gene = options[1]
        return new_gene

    def get_random_input_index(self, column, value_to_avoid=None):
        """
        Inputs
        - column: (int) column of the gene in the CGP Representation
        - value_to_avoid: (int)
        Returns
        - (int) distinct valid input index
        """
        if value_to_avoid is None:
            return rd.choice(self.connections_set[column])

        options = rd.sample(self.connections_set[column], k=2)
        if options[0] != value_to_avoid:
            new_gene = options[0]
        else:
            new_gene = options[1]
        return new_gene
        
    def mutate_node(self, node):
        """
        Inputs
        - node (CGP_Node instance)
        Outputs
        (CGP node instance) mutated node
        """
        int_mutation = rd.randint(0, node.function_arity)

        #mutate function
        if int_mutation == node.function_arity:
            new_func_index = self.get_random_function_index(value_to_avoid = node.function_index)
            node.function_index = new_func_index
            node.function = self.functions[new_func_index]

        #mutate input index
        else:
            new_input_index = self.get_random_input_index(column = node.column
                ,value_to_avoid = node.inputs[int_mutation])
            node.inputs[int_mutation] = new_input_index
        
        return node

    def point_mutation(self, graph, percentage):
        """
        Inputs
        - graph (CGP_Graph instance)
        - percentage (float)
        Returns
        - (CGP_Graph instance) The mutated graph
        - (Boolean) Was the active genotype altered?
        """

        #Create a copy of the original graph, to change it later
        new_graph = graph.copy()

        #Calculate useful variables
        nodes_list = list(new_graph.genotype.values())
        n_function_genes = new_graph.max_lenght + sum([n.function_arity for n in new_graph.genotype.values()])
        n_genes = n_function_genes + new_graph.n_outputs
        mutations = int(n_genes * percentage / 100)
        altered_active_genotype = False

        #Iterate mutations times
        for _ in range(mutations):
            int_to_mutate = rd.randint(0, n_genes-1)
            mutate_output_index = int_to_mutate - n_function_genes
            
            #if the random number falls in the output gene, mutate output gene
            if mutate_output_index >= 0:
                #print("Mutated output", mutate_output_index, int_to_mutate, n_function_genes)
                new_gene = self.get_random_output_gene(value_to_avoid = new_graph.output_gene[mutate_output_index])
                new_graph.output_gene[mutate_output_index] = new_gene
                altered_active_genotype = True

            #mutation will affect a node:    
            else:
                node = rd.choice(nodes_list)
                #print(str(int_to_mutate), str(len(nodes_list)))
                #node = nodes_list[int_to_mutate]
                if graph.genotype[node.output_index].active:
                    altered_active_genotype = True
                node = self.mutate_node(node)
                
                #assign the new node:
                new_graph.genotype[node.output_index] = node
        
        if altered_active_genotype:
            new_graph.find_actives()

        return new_graph, altered_active_genotype

    def single_active_mutation(self, graph):
        """
        Inputs
        - graph (CGP_Graph instance)
        Returns
        - (CGP_Graph instance) The mutated graph
        """

        #Create a copy of the original graph, to change it later
        new_graph = graph.copy()

        #Node selection
        len(graph.genotype) 
        options = list(new_graph.genotype.values()) + [i for i,_ in enumerate(graph.output_gene)]
        rd.shuffle(options)

        for opt in options:

            #Mutate and replace in the graph
            if isinstance(opt, int):
                new_graph.output_gene[opt] = self.get_random_output_gene(value_to_avoid = graph.output_gene[opt])
                break
            else:
                mutated_node = self.mutate_node(opt)
                new_graph.genotype[opt.output_index] = mutated_node
                if opt.active:
                    break
        
        #Update active labels in the graph
        new_graph.find_actives()

        return new_graph

    def accummulating_mutation(self, graph, percentage):
        altered = False
        mutations = 0
        while not altered:
            prev_graph = graph
            graph, altered = self.point_mutation(graph, percentage)
            mutations += 1
        return graph, mutations
        
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
        the_copy.active = self.active
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
        Upon creation, finding the active genotype with find_actives() is needed.
        It is not included here for flexibility.
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
        self.evaluation_skipped = False

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

        #Initialize all nodes as inactive
        for node in self.genotype.values():
            node.active = False

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
            #print("p",p)
            #print("inputs", self.genotype[p].inputs.values())
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




