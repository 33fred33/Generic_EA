from datetime import datetime

####################################
# Experiment setup #################
####################################


trials = 20
random_seed = 0
dataset_name = "ion"
    #Options: ion, spect, yst_m3, yst_mit, even_parity
train_test_rate = 0.5 
    #Domain: in (0,1)
semantic_size_rate = 1
    #Domain: in [0,1]
semantic_keep_label_rate = True
semantic_peculiarity_b = 1

####################################
# CGP ##############################
####################################

n_outputs = 1
n_rows = 1
n_columns = 200
levels_back = n_columns
allow_input_to_output = True
inputs_available_to_all_columns = True
functions = ["sum","sub","mul","safe_divide_one"]
    #Options: sum, sub, mul, safe_divide_one, safe_divide_one, 
point_mutation_percentage = 6
    #Domain: in (0,100]
numeric_output_mapping_threshold = 0.5
label_index_above_threshold = 1
label_index_below_threshold = 0
cgp_operator = "accum"
    #Options: point, sam, accum,


####################################
# EA ###############################
####################################

population_size = 200
tournament_size = 7
moea_sorting_method = "NSGAII"
    #Options: NSGAII, SPEA2, NSGAII_SP
stopping_criteria = "generations"
    #Options: fitness_evaluations, node_evaluations, generations 
generations = 100
node_max_evals = 100000000
fitness_max_evals = 100000
objective_names = ["accuracy_in_label", "accuracy_in_label"]
    #Options: accuracy, accuracy_in_label, active_nodes
objective_to_max = [True, True]
    #Conditions: must be the same length as the objective_names
objective_best = [1, 1]
    #Conditions: must be the same length as the objective_names
    #If unknown, complete with None
objective_worst = [0, 0]
    #Conditions: must be the same length as the objective_names
    #If unknown, complete with None
accuracy_label_index = [0,1] 
    #Relevant if any objective is named "accuracy_in_label".
    #Options: label indexes (order of appearence)
    #Conditions: must be the same length as the objective_names


####################################
# Logs #############################
####################################

experiment_name = dataset_name + "_" + cgp_operator + "_" + moea_sorting_method
timestamp_format = ("%Y_%m_%d-%H_%M_%S")
now = datetime.now()
time_string = now.strftime(timestamp_format)
output_path = ["outputs",experiment_name+"_"+time_string]