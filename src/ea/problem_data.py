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

    def even_parity(self):
        """
        Returns even parity output
        """
        y = []
        for row in self.x_train:
            if sum(row) % 2 == 0: 
                y.append(1)
            else:
                y.append(0)
        self.y_train = np.array(y)

    def load_problem(self,name):
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
         
        elif name[:11] == "even_parity":
            num = int(name[11:])
            data = []
            lim = 2**num
            for i,col in enumerate(range(int(num))):
                sw = 2**i
                row = []
                val = 0
                for j in range(lim):
                    if j%sw == 0 and j!=0:
                        if val==0:
                            val=1
                        else: 
                            val=0
                    row.append(val)
                data.insert(0,row)
            self.x_train = np.array(data).T
            self.even_parity()

        else:
            print("Unknown problem name")



        self.labels = list(set(self.y_train))

    def split_data(self, train_rate = 0.5, keep_labels_ratio = True, seed = None):
        pass
        
