import operator as op
import src.ea.ea_lib as ea
import src.ea.problem_data as pb
import src.ea.utilities as ut

################################################################
#### Vanilla CGP ###############################################
################################################################

#Parameters
seed = 1
population_size = 4
problem_name = "spect"
levels_back = 2
n_rows = 4
n_columns = 9
allow_input_to_output = False
functions = [op.add,op.sub,op.mul,ut.safe_divide_numerator]

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
graphs = [cgp.create_random() for _ in range(population_size)]
population = [ea.Individual(r,generation) for r in graphs]
for ind in population:
    ind.evaluate(data = dataset.x_train)
    ind.comparable_values[0] = ut.accuracy(dataset.y_train
            ,ind.evaluation
            ,ut.custom_round)
population = sorted(population)
for ind in population:
    print(ind.comparable_values)
