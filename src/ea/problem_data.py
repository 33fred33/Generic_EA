import numpy as np
import os
import random as rd
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self
            ,x_train = []
            ,x_test = []
            ,y_train = []
            ,y_test = []):
        self.x_train = np.array(x_train)
        self.x_test = np.array(x_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        self.labels = []
        self.train_label_count={}
        self.test_label_count={}

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
        self._update_details()
        
    def _update_details(self):
        self.labels = list(set(self.y_train))
        self.train_label_count = {label:0 for label in self.labels}
        self.test_label_count = {label:0 for label in self.labels}
        for train_label in self.y_train:
            self.train_label_count[train_label] += 1
        if len(self.y_test) > 0:
            for test_label in self.y_test:
                self.test_label_count[test_label] += 1

    def print_dataset_details(self):
        print("train_shape", str(self.x_train.shape))
        print("test_shape", str(self.x_test.shape))
        print("train_label_count", self.train_label_count)
        print("test_label_count", self.test_label_count)

    def split_data(self, train_rate = 0.5, keep_labels_ratio = True, seed = None):
        """
        Splits into training and testing
        Shuffles the data
        Inputs:
        Updates: x_train, x_test, y_train, y_test
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_train, self.y_train, test_size = train_rate, random_state=seed, stratify=self.y_train, shuffle = True)
        self._update_details()
        """
        if seed is not None:
            rd.seed(seed)
        if keep_labels_ratio:
            new_xtn = []
            new_ytn = []
            new_xtt = []
            new_ytt = []
            for label in self.labels:
                temp_x = [x for i,x in enumerate(self.x_train) if self.y_train[i]==label]
                temp_y = [y for i,y in enumerate(self.y_train) if self.y_train[i]==label]
                temp_zip = list(zip(temp_x, temp_y))
                rd.shuffle(temp_zip)
                temp_x, temp_y = zip(*temp_zip)
                train_size = int(len(temp_x)*train_rate)
                new_xtn += temp_x[:train_size]
                new_xtt += temp_x[train_size:]
                new_ytn += temp_y[:train_size]
                new_ytt += temp_y[train_size:]
            self.x_train = np.array(new_xtn)
            self.x_test = np.array(new_xtt)
            self.y_train = np.array(new_ytn)
            self.y_test = np.array(new_ytt)
        """