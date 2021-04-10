import numpy as np
import os
import random as rd

class Dataset:
    def __init__(self
            ,x_train = None
            ,x_test = None
            ,y_train = None
            ,y_test = None):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.labels = None

    def load_problem(self,name, test_train_rate=0.5, keep_labels_ratio = True, seed = None):
        """
        Available names: ion, spect
        """
        path = os.path.join("src","ea","data","")
        if name == "ion":
            data = np.genfromtxt(path + "ionosphere.dat"
                    ,skip_header=1
                    ,dtype=None
                    ,delimiter = ",")
            n_data = np.array([[d for d in row] for row in data])
            self.x_train = n_data[:,:-1]
            self.x_train = np.array([[float(v.decode('UTF-8')) for v in row] for row in self.x_train])
            self.y_train = n_data[:,-1]
            self.y_train = np.array([v.decode('UTF-8') for v in self.y_train])
            self.y_train = np.array([1 if v=="g" else 0 for v in self.y_train])
            
        elif name == "spect":
            data = np.genfromtxt(path + "spect.dat"
                    ,skip_header=1
                    ,dtype=None
                    ,delimiter = ",")
            self.x_train = data[:,:-1]
            self.y_train = data[:,-1]

        elif name == "yst_m3":
            data = np.genfromtxt(path + "yst_m3.dat"
                    ,dtype=None
                    ,delimiter = ",")
            n_data = np.array([[d for d in row] for row in data])
            self.x_train = n_data[:,1:-1]
            self.x_train = np.array([[float(v.decode('UTF-8')) for v in row] for row in self.x_train])
            self.y_train = n_data[:,-1]
            self.y_train = np.array([v.decode('UTF-8') for v in self.y_train])
            self.y_train = np.array([1 if v=="ME3" else 0 for v in self.y_train])
            

        elif name == "yst_mit":
            data = np.genfromtxt(path + "yst_mit.dat"
                    ,dtype=None
                    ,delimiter = ",")
            n_data = np.array([[d for d in row] for row in data])
            self.x_train = n_data[:,1:-1]
            self.x_train = np.array([[float(v.decode('UTF-8')) for v in row] for row in self.x_train])
            self.y_train = n_data[:,-1]
            self.y_train = np.array([v.decode('UTF-8') for v in self.y_train])
            self.y_train = np.array([1 if v=="MIT" else 0 for v in self.y_train])
         

        else:
            print("Unknown problem name")



        self.labels = list(set(self.y_train))

        if keep_labels_ratio:
            pass
        
