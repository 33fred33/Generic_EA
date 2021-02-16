import operator as op
import src.ea.ea_lib as ea


breeder = ea.CGP_Breeder(function_set = [op.add, op.sub, op.mul, op.div])