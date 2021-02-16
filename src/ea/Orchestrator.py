#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 23:15:50 2020

@author: Fred Valdez Ameneyro

#Tester and orchestrator for my generic Evolutionary Algorithm
"""

import Generic_EA as EA

GP = EA.GP_Toolbox()
individual = GP.generate_individual(max_depth = 4, method = "full", )
print(individual)
pop = GP.generate_initial_population(n = 10, max_depth = 3)
for ind in pop: print(ind)
x = 1
evaluations = [GP.evaluate(ind, x) for ind in pop]
for ev in evaluations:
	print("evaluations", ev)


