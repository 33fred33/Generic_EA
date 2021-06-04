import operator as op
import src.ea.ea_lib as ea
import src.ea.problem_data as pb
import src.ea.utilities as ut
import random as rd
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from datetime import date
import pandas as pd
import statistics as stat
import importlib
import argparse
import shutil
import sys


#param_arg = 'experiments.param_files.params'
param_arg = sys.argv[1]
prm = importlib.import_module(param_arg)
params_dict = vars(prm)

#Create output path
experiment_output_path = os.path.join(*prm.output_path,"")
experiment_output_path = ut.verify_path(experiment_output_path)

#Parameters file
params_file_path = os.path.join(experiment_output_path,'params.csv')
params_vars = vars(prm)
bad = lambda k: k.startswith("__") or isinstance(k, classmethod)
params_dict = {k:[str(v)] for k,v in params_vars.items() if not bad(k)}
params_df = pd.DataFrame(params_dict)
params_df.to_csv(params_file_path, index=False)

#Load dataset
dataset = pb.Dataset()
dataset.load_problem(name = prm.dataset_name)
dataset.split_data(train_rate = prm.train_test_rate)
dataset.print_dataset_details()

#Semantics
n_semantic_indexes = int(dataset.x_train.shape[0]*prm.semantic_size_rate)
semantic_indexes = set(rd.sample(list(range(dataset.x_train.shape[0])), k=n_semantic_indexes))

def evaluate_ind(ind, semantic_indexes, dataset, objectives, active_altered, test=False):

    #Data test or train
    if test: x, y = dataset.x_train, dataset.y_train
    else: x, y = dataset.x_test, dataset.y_test

    #Graph evaluated if active nodes or output nodes were altered:
    if active_altered:
        outputs = {}
        semantics = {}
        for i,data_row in enumerate(x):
            output_dict = ind.representation.evaluate(data_row = data_row)

            #Extracting the 0th index output gene (only one output gene)
            output = output_dict[0]

            #The raw output needs to be transformed
            transformed_output = ut.threshold_map(value = output
                    ,threshold = prm.numeric_output_mapping_threshold
                    ,output_up = dataset.labels[prm.label_index_above_threshold]
                    ,output_down = dataset.labels[prm.label_index_below_threshold])
            outputs[i] = transformed_output
            semantics[i] = output

        #Each objective has its own evaluation method
        for obj_idx, obj_name in enumerate(prm.objective_names):
            if obj_name == "accuracy_in_label":
                val = ut.accuracy_in_label(y = y, y_output = outputs, label = dataset.labels[prm.accuracy_label_index[obj_idx]])
            elif obj_name == "accuracy":
                val = ut.accuracy(y = y, y_output = outputs)
            elif obj_name == "active_nodes":
                val = len(ind.representation.active_genotype)
            ind.update_evaluation(objective = objectives[obj_idx], value = val)
    
        #Update individual values
        if not test:
            ind.update_semantics_all(semantics_all = semantics)
            ind.update_comparable_outputs(outputs = outputs)

    #If the individual has no changes when compared to the parent
    else:
        ind.representation.evaluation_skipped = True
        for objective in objectives:
            ind.update_evaluation(objective = objective, value = ind.parent.evaluations[objective.name])
        if not test:
            ind.update_semantics_all(semantics_all = ind.parent.semantics_all)
            ind.update_comparable_outputs(outputs = ind.parent.comparable_outputs)

def sort_pop_moea(population, objectives, nsgaii_objectives, spea2_objective, sp_obj, test=False):
    if prm.moea_sorting_method == "NSGAII":
        sorted_population = ea.fast_nondominated_sort(population, objectives, nsgaii_objectives)
    elif prm.moea_sorting_method == "SPEA2":
        sorted_population = ea.spea2_sort(population, objectives, spea2_objective)
    elif prm.moea_sorting_method == "NSGAII_SP":
        ea.semantic_peculiarity(population = population, output_vector = dataset.y_train, semantic_indexes = semantic_indexes, sp_objective = sp_obj, b = prm.semantic_peculiarity_b)
        ea.set_ranks(population = population, conflicting_objectives = objectives, front_objective = front_objective)
        sorted_population = ea.sort_population(population = population, objectives=[front_objective, sp_obj])
    return sorted_population

def create_offspring(parent_population, current_gen):

    #Selection
    parent_index = ea.tournament_selection_index(population_size = len(parent_population), tournament_size = prm.tournament_size)
    parent = parent_population[parent_index]

    #Operator
    if prm.cgp_operator == "point":
        new_graph, active_altered = cgp.point_mutation(graph = parent.representation, percentage = prm.point_mutation_percentage)
    elif prm.cgp_operator == "sam":
        active_altered = True
        new_graph = cgp.single_active_mutation(graph = parent.representation)
    elif prm.cgp_operator == "accum":
        active_altered = True
        new_graph, accum_count = cgp.accummulating_mutation(graph = parent.representation, percentage = prm.point_mutation_percentage)
    elif prm.cgp_operator == "sasam":
        original_parent = parent
        for attempt in range(prm.max_sasam_attempts):
            new_graph = cgp.single_active_mutation(graph = parent.representation)
            #Create offspring
            offspring = ea.Individual(representation = new_graph
                                ,created_in_gen = current_gen
                                ,parent_index = parent_index
                                ,parent = original_parent
                                ,semantic_indexes = semantic_indexes)

            #Evaluate offspring
            offspring.update_evaluation(objective = generation_objective, value = current_gen)
            evaluate_ind(offspring, semantic_indexes, dataset, objectives, True)
            if ind.semantic_distance_from_parent != 0:
                return offspring
            else:
                parent = offspring
        return offspring

    #Create offspring
    offspring = ea.Individual(representation = new_graph
                            ,created_in_gen = current_gen
                            ,parent_index = parent_index
                            ,parent = parent
                            ,semantic_indexes = semantic_indexes)

    #Evaluate offspring
    offspring.update_evaluation(objective = generation_objective, value = current_gen)
    evaluate_ind(offspring, semantic_indexes, dataset, objectives, active_altered)

    return offspring





for trial in range(prm.trials):
    print("Trial:", trial)

    #Initialization
    current_gen = 0
    gen_logs = pd.DataFrame()

    #CGP
    cgp = ea.CGP_Representation(n_inputs = dataset.x_train.shape[1]
        ,n_outputs = prm.n_outputs
        ,levels_back = prm.levels_back
        ,n_rows = prm.n_rows
        ,n_columns = prm.n_columns
        ,allow_input_to_output = prm.allow_input_to_output
        ,inputs_available_to_all_columns = prm.allow_input_to_output
        ,functions = [op.add,op.sub,op.mul,ut.safe_divide_numerator])

    #objectives
    objectives = []
    for obj_idx, obj_name in enumerate(prm.objective_names):
        if  obj_name == "accuracy_in_label":
            obj_name = obj_name + "_" + str(dataset.labels[prm.accuracy_label_index[obj_idx]])
        obj = ea.Objective(name = obj_name
                            ,to_max = prm.objective_to_max[obj_idx]
                            ,best = prm.objective_best[obj_idx]
                            ,worst = prm.objective_worst[obj_idx])
        objectives.append(obj)
    generation_objective = ea.Objective(name = "generation", to_max = True)
    nsgaii_objectives = ea.get_nsgaii_objectives()
    front_objective = nsgaii_objectives[0]
    cd_objective = nsgaii_objectives[1]
    spea2_objective = ea.get_spea2_objective()
    sp_obj = ea.get_semantic_peculiarity_objective()

    #Initial population
    graphs = [cgp.create_random(seed = rd.random()) for _ in range(prm.population_size)]
    parent_population = [ea.Individual(representation=graphs[i]
                        ,created_in_gen = 0
                        ,semantic_indexes=semantic_indexes) for i in range(prm.population_size)]
    for ind in parent_population:
        evaluate_ind(ind, semantic_indexes, dataset, objectives, True)
        ind.update_evaluation(objective = generation_objective, value = current_gen)
    parent_population = sort_pop_moea(parent_population, objectives, nsgaii_objectives, spea2_objective, sp_obj)
    offspring_population = [create_offspring(parent_population, current_gen) for i in range(prm.population_size)]


    ## Main loop
    stop_criteria_value = 0
    while(True):
    #for _ in range(prm.generations):

        #Population management.
        population = parent_population + offspring_population
        sorted_population = sort_pop_moea(population, objectives, nsgaii_objectives, spea2_objective, sp_obj)

        #Logs
        current_gen_logs = ea.moea_population_log(sorted_population, objectives)
        gen_logs = gen_logs.append(current_gen_logs, ignore_index = True)
        #ea.plot_pareto(population, objectives, "size", path = experiment_output_path, name = f"plt_size_g{current_gen}")

        #Offspring generation
        parent_population = sorted_population[:prm.population_size]

        current_gen = current_gen + 1
        ea.raise_ages(population)
        offspring_population = [create_offspring(parent_population, current_gen) for i in range(prm.population_size)]

        #Stop criteria fitness_evaluations, node_evaluations, generations 
        if prm.stopping_criteria == "generations":
            stop_criteria_value = current_gen
            print(prm.stopping_criteria, ": ", str(stop_criteria_value), "of", str(prm.stop_value))
            if stop_criteria_value >= prm.stop_value:
                break
        elif prm.stopping_criteria == "fitness_evaluations":
            new_evals = gen_logs.iloc[-1,list(gen_logs.columns).index("Fitness_evals")]
            stop_criteria_value += new_evals
            print(prm.stopping_criteria, ": ", str(stop_criteria_value), "of", str(prm.stop_value))
            if stop_criteria_value >= prm.stop_value:
                break
        else:
            print("Wrong stop critera")
            break

    #Final logs
    #print(gen_logs.head())
    gen_logs.to_csv(path_or_buf=f"{experiment_output_path}{prm.gen_logs_name}{trial}.csv")
    ea.plot_pareto(population, objectives, "size", path = f"{experiment_output_path}", name = f"plt_g{current_gen}_t{trial}")

    #Test evaluation
    population = parent_population + offspring_population
    for ind in population:
        evaluate_ind(ind, semantic_indexes, dataset, objectives, True, True)
    population = sort_pop_moea(population, objectives, nsgaii_objectives, spea2_objective, sp_obj)

    #Logs
    test_logs = ea.moea_population_log(population, objectives)
    test_logs.to_csv(path_or_buf=f"{experiment_output_path}{prm.test_logs_name}{trial}.csv")
    ea.plot_pareto(population, objectives, "size", path = experiment_output_path, name =  f"plt_test_t{trial}")

final_logs = {}
final_test_logs = {}
for trial in range(prm.trials):
    final_logs[trial] = pd.read_csv(f"{experiment_output_path}{prm.gen_logs_name}{trial}.csv")
    final_test_logs[trial] = pd.read_csv(f"{experiment_output_path}{prm.test_logs_name}{trial}.csv")

mean_columns = ["Hyperarea","Avg_active_nodes","Sd_q10","Sd_q25","Sd_q50","Sd_q75"
                ,"Avg_hamming_distance_from_parent","Hd_q10","Hd_q25","Hd_q50","Hd_q75"
                ,"Pareto front size", "Front_Clustered_pop_rate", "Front_Mean_cluster_size_rate"
                ,"Front_Unique_objective_vectors", "Clustered_pop_rate"]
mean_dict = {}
for mean_column in mean_columns:
    column_mean_collector = []
    column_std_collector = []
    for gen in range(len(final_logs[0])):
        gen_collector = []
        for trial in range(prm.trials):
            gen_collector.append(final_logs[trial].iloc[gen,list(final_logs[trial].columns).index(mean_column)])
        column_mean_collector.append(stat.mean(gen_collector))
        column_std_collector.append(stat.stdev(gen_collector))
    mean_dict[mean_column+"_mean"] = column_mean_collector
    mean_dict[mean_column+"_stdev"] = column_std_collector

means_df = pd.DataFrame(mean_dict)
means_df.to_csv(path_or_buf=f"{experiment_output_path}global_train_results.csv")

test_mean_dict = {}
for mean_column in mean_columns:
    column_collector = []
    for trial in range(prm.trials):
        column_collector.append(float(final_test_logs[trial].iloc[0,list(final_test_logs[trial].columns).index(mean_column)]))
    test_mean_dict[mean_column+"_mean"] = [stat.mean(column_collector)]
    test_mean_dict[mean_column+"_stdev"] = [stat.stdev(column_collector)]

test_means_df = pd.DataFrame(test_mean_dict)
test_means_df.to_csv(path_or_buf=f"{experiment_output_path}global_test_results.csv")