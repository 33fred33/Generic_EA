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

prm = importlib.import_module('experiments.param_files.params')

################################################################
#### Vanilla CGP ###############################################
################################################################

"""
#Parameters
rd.seed(1)
population_size = 2000
problem_name = "spect"
levels_back = 2
n_rows = 4
n_columns = 9
allow_input_to_output = False
functions = [op.add,op.sub,op.mul,ut.safe_divide_one]

#Definition
dataset = pb.Dataset()
dataset.load_problem(name = problem_name)
cgp = ea.CGP_Representation(
            dataset.x_train.shape[1]
            ,1
            ,levels_back
            ,n_rows 
            ,n_columns
            ,allow_input_to_output
            ,*functions)


#Execution
generation = 0
start_t = time.time()
graphs = [cgp.create_random(seed = rd.random()) for _ in range(population_size)]
init_time = time.time()
print(str(init_time-start_t))
population = [ea.Individual(r,generation) for r in graphs]
for ind in population:
    outputs = ind.evaluate(data = dataset.x_train)
    ind.comparable_values[0] = ut.accuracy(y=dataset.y_train, y_output=ind.outputs)
eval_t = time.time()
print(str(eval_t-start_t))
#population = sorted(population)
for i,ind in enumerate(population):
    if i<5:
        print(ind.comparable_values)
"""

################################################################
#### MOEA CGP ###############################################
################################################################

#EXPERIMENT 1: 

"""
## CGP params
levels_back = 200
n_rows = 1
n_columns = 200
n_outputs = 1
allow_input_to_output = True
inputs_available_to_all_columns = True
functions = [op.add,op.sub,op.mul,ut.safe_divide_one]
functions_as_string = "[op.add,op.sub,op.mul,ut.safe_divide_one]"

## MOEA params
trials = 30
seed = 0
node_max_evals = 100000000
population_size = 200
tournament_size = 5
mutation_percentage = 9

moea_sorting_method = "SPEA2"
#moea_sorting_methods: NSGAII, SPEA2

problem_name = "spect"
#problem_names = ion, spect, yst_m3, yst_mit

#Objectives
objectives = [
    ea.Objective(name="acc0", to_max = True, best=1, worst=0),
    ea.Objective(name="acc1", to_max = True, best=1, worst=0)
    ]
generation_objective = ea.Objective(name="generation", to_max = True)
nsgaii_objectives = ea.get_nsgaii_objectives()
front_objective = nsgaii_objectives[0]
cd_objective = nsgaii_objectives[1]
spea2_objective = ea.get_spea2_objective()

## Experiment parameters
rd.seed(seed)
now = datetime.now()
time_string = now.strftime('%Y_%m_%d-%H_%M_%S')
output_path = os.path.join("outputs","NSGAII_cgp-" + problem_name + "-" + time_string, "")

#Instantiation
dataset = pb.Dataset()
dataset.load_problem(name = problem_name)
data_rows = dataset.x_train.shape[0]
cgp = ea.CGP_Representation(
            dataset.x_train.shape[1]
            ,n_outputs
            ,levels_back
            ,n_rows 
            ,n_columns
            ,allow_input_to_output
            ,inputs_available_to_all_columns
            ,*functions)
labels = list(set(dataset.y_train))

#Save parameters for reference
param_logs = [["rd.seed" , seed]
            ,["node_max_evals", node_max_evals]
            ,["population_size ", population_size]
            ,["tournament_size ", tournament_size]
            ,["mutation_percentage ", mutation_percentage]
            ,["levels_back ", levels_back]
            ,["n_rows ", n_rows]
            ,["n_columns ", n_columns]
            ,["n_outputs ", n_outputs]
            ,["allow_input_to_output ", allow_input_to_output]
            ,["inputs_available_to_all_columns ", inputs_available_to_all_columns]
            ,["functions ", functions_as_string]
            ,["moea_sorting_method", moea_sorting_method]]
ut.logs_to_file(param_logs, "param_logs", output_path)

def evaluate_ind(ind):
    outputs = {}
    for i,data_row in enumerate(dataset.x_train):
        output_dict = ind.representation.evaluate(data_row = data_row)
        #Extracting the 0th index output gene
        output = output_dict[0]
        #The raw output needs to be transformed
        #transformed_output = ut.custom_round(output)
        transformed_output = ut.threshold_map(value = output,threshold = 0.5, output_up = labels[1], output_down = labels[0])
        outputs[i] = transformed_output
    #Each objective has its own evaluation method
    acc0 = ut.accuracy_in_label(y = dataset.y_train, y_output = outputs, label = 0)
    acc1 = ut.accuracy_in_label(y = dataset.y_train, y_output = outputs, label = 1)
    ind.update_evaluation(objective = objectives[0], value = acc0)
    ind.update_evaluation(objective = objectives[1], value = acc1)
    ind.update_semantics_all(semantics_all = outputs)


def sort_pop_moea(population):
    if moea_sorting_method == "NSGAII":
        s_population = ea.fast_nondominated_sort(population = population, conflicting_objectives = objectives, nsgaii_objectives = nsgaii_objectives)
    elif moea_sorting_method == "SPEA2":
        s_population = ea.spea2_sort(population = population, conflicting_objectives = objectives, spea2_objective = spea2_objective)
    else:
        print("Wrong sorting method")
    return s_population



#Generation
def run_gen(population, current_gen):

    start_t = time.time()

    #Sort the population.
    sorted_population = sort_pop_moea(population)

    #Logs (for previous gen)
    hyperarea = ea.hyperarea(population, objectives)
    header, logs, g_header, g_logs = ea.get_cgp_log(sorted_population, cgp, current_gen)
    g_logs += [hyperarea]
    g_header += ["Hyperarea"]

    #Elitism
    parent_population = sorted_population[:population_size]

    #Offspring generation
    offspring_population = []
    for i in range(population_size):
        parent_index = ea.tournament_selection_index(population_size = len(parent_population), tournament_size = tournament_size)
        parent = parent_population[parent_index]
        new_graph, active_altered = cgp.point_mutation(graph = parent.representation, percentage = mutation_percentage)
        offspring = ea.Individual(representation = new_graph, created_in_gen = current_gen)

        #If the active graph was not altered, the individual does not need to be evaluated again:
        if active_altered:
            evaluate_ind(offspring)
        else:
            offspring.representation.evaluation_skipped = True
            for objective in objectives:
                offspring.update_evaluation(objective = objective, value = parent.evaluations[objective.name])
            offspring.update_semantics_all(semantics_all = parent.semantics_all)
        offspring_population.append(offspring)

    #Update the gen_of_creation of the offsprings
    for offspring in offspring_population:
        offspring.update_evaluation(objective = generation_objective, value = current_gen)

    #Formation of the population for the next gen
    population = offspring_population + parent_population

    return population, header, logs, g_header, g_logs



#Execution
for exp_idx in range(trials):
    print("run:", str(exp_idx+1))
    #path = output_path + "run" + str(exp_idx) + "/"
    path = os.path.join(output_path + "run" + str(exp_idx), "")
    
    
    #Initial generation
    generation = 0
    individual_level_logs = []
    gen_level_logs = []
    nodes_evaluated = 0

    #Random initial population. Specific initial conditions for the population can be specified here
    graphs = [cgp.create_random(seed = rd.random()) for _ in range(population_size)]

    #create instances of Individual to be grouped in the population
    parent_population = [ea.Individual(r, created_in_gen = generation) for r in graphs]

    #Evaluate and sort the population according to non-domination
    for ind in parent_population:
        evaluate_ind(ind)
    first_sorted_population = sort_pop_moea(parent_population)

    #Create the offsprings of the initial generation
    population = parent_population
    for i in range(population_size):

        #Binary tournament selection is used in the initial generation only according to NSGA-II. The offspring is evaluated and added to the population
        parent_index = ea.tournament_selection_index(population_size = population_size, tournament_size = 2)
        parent = first_sorted_population[parent_index]
        new_graph, active_altered = cgp.point_mutation(graph = parent.representation, percentage = mutation_percentage)
        offspring = ea.Individual(representation = new_graph, created_in_gen = generation)

        #If the active graph was not altered, the individual does not need to be evaluated again:
        if active_altered:
            evaluate_ind(offspring)
        else:
            offspring.representation.evaluation_skipped = True
            for objective in objectives:
                offspring.update_evaluation(objective = objective, value = parent.evaluations[objective.name])
            offspring.update_semantics_all(semantics_all = parent.semantics_all)
        population.append(offspring)

    for ind in population:
        ind.update_evaluation(objective = generation_objective, value = generation)



    #Main loop
    while True:
        generation += 1
        population, header, logs, g_header, g_logs = run_gen(population, generation)
        

        #Logs
        individual_level_logs += logs
        gen_level_logs += [g_logs]

        #Plots
        if generation%20==0:
            
            ea.plot_pareto(population, objectives, "size", path = path, name = "plt_size_g"+str(generation))
            ea.plot_pareto(population, objectives, "color", path = path, name = "plt_color_g"+str(generation))

            ut.logs_to_file(individual_level_logs, "Ind_logs", path)
            ut.logs_to_file(gen_level_logs, "Gen_logs", path)
        
        #stop_criteria:
        nodes_idx = g_header.index("Eval_nodes")
        nodes_evaluated += g_logs[nodes_idx] * data_rows
        print("progress: ", str(nodes_evaluated*100/node_max_evals), " nodes_evaluated: ", str(nodes_evaluated))
        if nodes_evaluated > node_max_evals:
            break



    #Final plots
    population = sort_pop_moea(population)

    ea.plot_pareto(population, objectives, "size", path = path, name = "plt_size_g"+str(generation))
    ea.plot_pareto(population, objectives, "color", path = path, name = "plt_color_g"+str(generation))

    individual_level_logs.insert(0, header)
    ut.logs_to_file(individual_level_logs, "Ind_logs", path)
    gen_level_logs.insert(0, g_header)
    ut.logs_to_file(gen_level_logs, "Gen_logs", path)
"""



################################################################
#### Semantics ###############################################
################################################################

#Semantics experiment
"""
#Methods
def evaluate_ind(ind, dataset, objective):
    outputs = {}
    for i,data_row in enumerate(dataset.x_train):
        output_dict = ind.representation.evaluate(data_row = data_row)
        #Extracting the 0th index output gene
        output = output_dict[0]
        #The raw output need to be transformed
        #transformed_output = ut.custom_round(output)
        mapped_output = ut.threshold_map(value = output,threshold = mapping_threshold, output_up = dataset.labels[1], output_down = dataset.labels[0])
        outputs[i] = mapped_output
    #Each objective has its own evaluation method
    ind.update_semantics_all(semantics_all = outputs)
    acc = ut.accuracy(y = dataset.y_train, y_output = outputs)
    ind.update_evaluation(objective = acc_obj, value = acc)


## Experiment parameters
generations = 5000
problem_name = "spect"
#problem_names = ion, spect, yst_m3, yst_mit
seed = 0
rd.seed(seed)
now = datetime.now()
time_string = now.strftime('%Y_%m_%d-%H_%M_%S')
mapping_threshold = 0
output_path = os.path.join("outputs","semantic_distances","point",problem_name + "-" + time_string, "") #define
output_path = ut.verify_path(output_path)

#Initialize the logs
sem_logs = pd.DataFrame()
final_log = pd.DataFrame()

#Loop
for i in range(4):

    ## CGP params
    levels_back = 200 + i*50
    n_rows = 1
    n_columns = 200 + i*50
    n_outputs = 1
    allow_input_to_output = True
    inputs_available_to_all_columns = True
    functions = [op.add,op.sub,op.mul,ut.safe_divide_one]
    functions_as_string = "[op.add,op.sub,op.mul,ut.safe_divide_one]"
    mutation_percentage = 9

    #Instantiation
    acc_obj = ea.Objective(name="acc", to_max = True, best=1, worst=0, eval_function = ut.accuracy)
    tpr_obj = ea.Objective(name="tpr", to_max = True, best=1, worst=0, eval_function = ut.accuracy_in_label)
    tnr_obj = ea.Objective(name="tnr", to_max = True, best=1, worst=0, eval_function = ut.accuracy_in_label)
    dataset = pb.Dataset()
    dataset.load_problem(name = problem_name)
    data_rows = dataset.x_train.shape[0]
    cgp = ea.CGP_Representation(
                dataset.x_train.shape[1]
                ,n_outputs
                ,levels_back
                ,n_rows 
                ,n_columns
                ,allow_input_to_output
                ,inputs_available_to_all_columns
                ,*functions)

    #Initial random ind
    graph = cgp.create_random(seed = rd.random())
    ind = ea.Individual(graph, created_in_gen = 0)
    evaluate_ind(ind, dataset, acc_obj)

    start_time = time.time()
    for gen in range(generations):

        ##Generate the new ind
        new_graph = cgp.single_active_mutation(ind.representation)
        altered = True
        #new_graph, muts = cgp.accummulating_mutation(ind.representation, 9)
        #new_graph, altered = cgp.point_mutation(ind.representation, mutation_percentage) #point
        new_ind = ea.Individual(new_graph, created_in_gen = 0)
        evaluate_ind(new_ind, dataset, acc_obj)

        #Logs
        row = {
            "max_nodes":cgp.n_function_nodes
            ,"prev_actives":len(ind.representation.active_genotype)
            ,"new_actives":len(new_ind.representation.active_genotype)
            ,"actives_change":len(new_ind.representation.active_genotype) - len(ind.representation.active_genotype)
            ,"prev_actives_rate":len(ind.representation.active_genotype) / cgp.n_function_nodes
            ,"new_actives_rate":len(new_ind.representation.active_genotype) / cgp.n_function_nodes
            ,"actives_change_rate":(len(new_ind.representation.active_genotype) - len(ind.representation.active_genotype))/cgp.n_function_nodes
            ,"semantic_distance":ea.semantic_distance(ind, new_ind)
            ,"altered":altered #point
            }
        sem_logs = sem_logs.append(row, ignore_index = True)

        #Reset variables for the loop to work
        ind = new_ind

    total_time = time.time() - start_time

    #Extract the subsets from the logs
    current_logs = sem_logs[sem_logs["max_nodes"]==cgp.n_function_nodes]
    sd_when_altered = list(current_logs.loc[current_logs["altered"]==True,"semantic_distance"]) #point

    #Calculations
    avg_sd = stat.mean(list(current_logs["semantic_distance"]))
    stdev_sd = stat.stdev(list(current_logs["semantic_distance"]))
    avg_actives = stat.mean(list(current_logs["prev_actives"]))
    altered_rate = sum(list(current_logs["altered"]))/generations
    if len(sd_when_altered) > 0:
        avg_sd_when_altered = stat.mean(sd_when_altered)
        stdev_sd_when_altered = stat.stdev(sd_when_altered)
    else:
        avg_sd_when_altered = 0
        stdev_sd_when_altered = 0
    f_l = {"avg_sd":avg_sd
            ,"stdev_sd":stdev_sd
            ,"avg_actives":avg_actives
            ,"generations":generations
            ,"total_time":total_time
            ,"avg_gen_time":total_time/generations
            ,"rd.seed":seed
            ,"levels_back ":levels_back
            ,"n_rows ":n_rows
            ,"n_columns ":n_columns
            ,"n_outputs ":n_outputs
            ,"n_inputs":cgp.n_inputs
            ,"functions ":functions_as_string
            ,"fitness_cases":len(dataset.y_train)
            ,"mutation_percentage ": mutation_percentage
            ,"altered_rate":altered_rate
            ,"avg_sd_when_altered":avg_sd_when_altered
            ,"stdev_sd_when_altered":stdev_sd_when_altered
            }
    final_log = final_log.append(f_l, ignore_index = True)
    print(str(f_l))

sem_logs.to_csv(os.path.join(output_path, "Sem_logs"))
final_log.to_csv(os.path.join(output_path, "Exp_logs"))
"""

#Semantic distances experiment

individual_level_logs = []
gen_level_logs = []

#Methods
def evaluate_ind(ind, semantic_indexes, dataset, objectives, current_gen, active_altered):
    ind.update_evaluation(objective = generation_objective, value = current_gen)

    #Graph evaluated if active nodes or output nodes were altered:
    if active_altered:
        outputs = {}
        for i,data_row in enumerate(dataset.x_train):
            output_dict = ind.representation.evaluate(data_row = data_row)
            #Extracting the 0th index output gene (only one output gene)
            output = output_dict[0]
            #The raw output needs to be transformed
            transformed_output = ut.threshold_map(value = output
                    ,threshold = prm.numeric_output_mapping_threshold
                    ,output_up = dataset.labels[prm.label_index_above_threshold]
                    ,output_down = dataset.labels[prm.label_index_below_threshold])
            outputs[i] = transformed_output

        #Each objective has its own evaluation method
        for obj_idx, obj_name in enumerate(prm.objective_names):
            if obj_name == "accuracy_in_label":
                val = ut.accuracy_in_label(y = dataset.y_train, y_output = outputs, label = dataset.labels[prm.accuracy_label_index[obj_idx]])
            elif obj_name == "accuracy":
                val = ut.accuracy(y = dataset.y_train, y_output = outputs)
            elif obj_name == "active_nodes":
                val = len(ind.representation.active_genotype)
            ind.update_evaluation(objective = objectives[obj_idx], value = val)
        ind.update_evaluation(objective = generation_objective, value = current_gen)
    
        #Update individual values
        ind.update_semantics_all(semantics_all = outputs)
        if ind.parent is not None:
            ind.semantic_distance_from_parent = ea.semantic_distance(ind1 = ind
                                                ,ind2 = ind.parent
                                                ,semantic_indexes = semantic_indexes)
            ind.damaged_semantics_from_parent = 0
            ind.improved_semantics_from_parent = 0
            for y_idx in semantic_indexes:
                if ind.semantics_all[y_idx] != ind.parent.semantics_all[y_idx]:
                    if ind.semantics_all[y_idx] == dataset.y_train[y_idx]:
                        ind.improved_semantics_from_parent = ind.improved_semantics_from_parent + 1
                    elif ind.parent.semantics_all[y_idx] == dataset.y_train[y_idx]:
                        ind.damaged_semantics_from_parent = ind.damaged_semantics_from_parent + 1
            ind.semantic_change_balance = ind.improved_semantics_from_parent - ind.damaged_semantics_from_parent
            ind.active_nodes_diff_from_parent = abs(len(ind.representation.active_genotype) - len(ind.parent.representation.active_genotype))

    else:
        ind.representation.evaluation_skipped = True
        for objective in objectives:
            ind.update_evaluation(objective = objective, value = ind.parent.evaluations[objective.name])
        ind.update_semantics_all(semantics_all = ind.parent.semantics_all)
        ind.damaged_semantics_from_parent = 0
        ind.improved_semantics_from_parent = 0
        ind.semantic_change_balance = 0
        ind.semantic_distance_from_parent = 0
        ind.active_nodes_diff_from_parent = 0
    

def sort_pop_moea(population, objectives, nsgaii_objectives, spea2_objective, sp_obj):
    if prm.moea_sorting_method == "NSGAII":
        sorted_population = ea.fast_nondominated_sort(population, objectives, nsgaii_objectives)
    elif prm.moea_sorting_method == "SPEA2":
        sorted_population = ea.spea2_sort(population, objectives, spea2_objective)
    elif prm.moea_sorting_method == "NSGAII_SP":
        ea.semantic_peculiarity(population = population, output_vector = dataset.y_train, semantic_indexes = semantic_indexes, sp_objective = sp_obj, b = prm.semantic_peculiarity_b)
        ea.set_ranks(population = population, conflicting_objectives = objectives, front_objective = front_objective)
        sorted_population = ea.sort_population(population = population, objectives=[front_objective, sp_obj])
    return sorted_population


def create_offspring(parent_population):

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

    offspring = ea.Individual(representation = new_graph
                            ,created_in_gen = current_gen
                            ,parent_index = parent_index
                            ,parent = parent)

    #If the active graph was not altered, the individual does not need to be evaluated again:
    evaluate_ind(offspring, semantic_indexes, dataset, objectives, current_gen, active_altered)
    return offspring



for trial in range(prm.trials):

    #Setup
    current_gen = 0
    output_path = os.path.join(*prm.output_path)
    output_path = os.path.join(output_path, str(trial), "")
    path = ut.verify_path(output_path)
    dataset = pb.Dataset()
    dataset.load_problem(name = prm.dataset_name)
    dataset.split_data(train_rate = prm.train_test_rate)
    dataset.print_dataset_details()

    #Semantics
    n_semantic_indexes = int(dataset.x_train.shape[0]*prm.semantic_size_rate)
    semantic_indexes = set(rd.sample(list(range(dataset.x_train.shape[0])), k=n_semantic_indexes))

    #CGP
    cgp = ea.CGP_Representation(n_inputs = dataset.x_train.shape[1]
        ,n_outputs = prm.n_outputs
        ,levels_back = prm.levels_back
        ,n_rows = prm.n_rows
        ,n_columns = prm.n_columns
        ,allow_input_to_output = prm.allow_input_to_output
        ,inputs_available_to_all_columns = prm.allow_input_to_output
        ,functions = [op.add,op.sub,op.mul,ut.safe_divide_one])

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
    parent_population = [ea.Individual(graphs[i], created_in_gen = 0) for i in range(prm.population_size)]
    for ind in parent_population:
        evaluate_ind(ind, semantic_indexes, dataset, objectives, current_gen, True)
    parent_population = sort_pop_moea(parent_population, objectives, nsgaii_objectives, spea2_objective, sp_obj)
    offspring_population = [create_offspring(parent_population) for i in range(prm.population_size)]


    ## Main loop
    previous_hyperarea = 0
    for _ in range(prm.generations):
        #Population management.
        population = parent_population + offspring_population
        sorted_population = sort_pop_moea(population, objectives, nsgaii_objectives, spea2_objective, sp_obj)

        #Individual_logs
        hyperarea = ea.hyperarea(sorted_population, objectives, front_objective)
        header, logs, g_header, g_logs = ea.get_cgp_log(sorted_population, cgp, current_gen)
        g_logs += [hyperarea]
        g_header += ["Hyperarea"]
        individual_level_logs += logs
        gen_level_logs += [g_logs]
        if current_gen%10==0:
            ea.plot_pareto(sorted_population, objectives, "size", path = path, name = "plt_size_g"+str(current_gen))
            ea.plot_pareto(sorted_population, objectives, "color", path = path, name = "plt_color_g"+str(current_gen))
            ut.logs_to_file(individual_level_logs, "Ind_logs", path)
            ut.logs_to_file(gen_level_logs, "Gen_logs", path)

        if previous_hyperarea > hyperarea:
            print("Damaged hyperarea")
            break
        else:
            previous_hyperarea = hyperarea

        #Offspring generation
        previous_population = [i for i in sorted_population] #debugging purposes
        parent_population = sorted_population[:prm.population_size]
        offspring_population = [create_offspring(parent_population) for i in range(prm.population_size)]

        current_gen = current_gen + 1

    #Final logs
    individual_level_logs.insert(0, header)
    ut.logs_to_file(individual_level_logs, "Ind_logs", path)
    gen_level_logs.insert(0, g_header)
    ut.logs_to_file(gen_level_logs, "Gen_logs", path)