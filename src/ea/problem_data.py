import numpy as np

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

    def load_problem(self,name):
        """
        Available names: ion, spect
        """
        if name == "ion":
            data = np.genfromtxt("src\ea\data\ionosphere.dat"
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
            data = np.genfromtxt("src\ea\data\spect.dat"
                    ,skip_header=1
                    ,dtype=None
                    ,delimiter = ",")
            self.x_train = data[:,:-1]
            self.y_train = data[:,-1]

        else:
            print("Unknown problem name")
