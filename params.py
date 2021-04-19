

####################################
# Experiment setup #################
####################################

experiment_name = "Semantic_distance"
    #Options: 
trials = 30
random_seed = 0
dataset_name = "ion"
    #Options: ion, spect, yst_m3, yst_mit, even_parity%n
train_test_rate = 0.5 
    #Domain: in (0,1)
semantic_size_rate = 1
    #Domain: in [0,1]

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
point_mutation_percentage = 9
    #Domain: in (0,100]
numeric_output_mapping_threshold = 0


####################################
# EA ###############################
####################################

population_size = 300
tournament_size = 5
moea_sorting_method = "SPEA2"
    #Options: NSGAII
stopping_criteria = "fitness_evaluations"
    #Options: fitness_evaluations, node_evaluations, generations 
generations = 200
node_max_evals = 100000000
fitness_max_evals = 100000
objective_names = ["accuracy_in_label", "accuracy_in_label"]
    #Options: accuracy, accuracy_in_label
objective_to_max = [True, True]
accuracy_labels = [""] #Requires dataset knowledge, relevant for objective with name "accuracy_in_label"
     #Options: labels of the dataset


####################################
# Logs #############################
####################################

timestamp_format = ("%Y_%m_%d-%H_%M_%S")
output_path = ["output",experiment_name]