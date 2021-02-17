import operator as op
import src.ea.ea_lib as ea

################################################################
#### Vanilla CGP ###############################################
################################################################

breeder = ea.CGP_Breeder(1,2,4,9,False,op.add,op.sub,op.mul)
i1 = breeder.create_random()
print(i1.representation)
i1.representation.evaluate()