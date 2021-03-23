from inspect import signature
import numpy as np
import os
import errno
import csv

#Utility functions

def get_arity(operator):
    """
    Returns the arity of the method, operator or funtion as an int
    :param operator: is a method, operator or funtion
    """
    sig = signature(operator)
    arity = len(sig.parameters)
    return arity

def custom_round(num, dec=0):
    if isinstance(num, np.integer) or isinstance(num, int): return num
    num = str(num)[:str(num).index('.')+dec+2]
    if num[-1]>='5':
        return float(num[:-2-(not dec)]+str(int(num[-2-(not dec)])+1))
    return int(float(num[:-1]))

def threshold_map(value, threshold, output_up, output_down):
    """
    Inputs
    - value: (float)
    - threshold: (float)
    - output_up: (python object)
    - output_down: (python object)
    """
    if value > threshold:
        return output_up
    else:
        return output_down

def accuracy(y, y_output):
    n = len(y)
    bools = [y_output[i] == y[i] for i in range(n)]
    return sum(bools)/n

def accuracy_in_label(y,y_output,label):
    n = len([p for p in y if p==label])
    bools = [y_output[i] == p for i,p in enumerate(y) if p == label]
    return sum(bools)/n

def verify_path(tpath):
    if tpath is None:
        return ""
    else:
        #if tpath[-1] != "/":
        #    tpath += "/"

        if not os.path.exists(os.path.dirname(tpath)):
            try:
                os.makedirs(os.path.dirname(tpath))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        return tpath

def logs_to_file(logs, name, path = None):
    """
    logs is a dictionary with a key that can be split into two
    """
    path = verify_path(path)
    with open(path + name + ".csv", mode='w') as logs_file:
        logs_writer = csv.writer(logs_file, delimiter=',', lineterminator = '\n')
        for row in logs:
            str_row = [str(v) for v in row]
            logs_writer.writerow(str_row)


#Operations
def safe_divide_numerator(a, b):
    """
    Executes a/b. If b=0, returns a
    """
    if b == 0 : return a
    else: return a/b

def safe_divide_zero(a, b):
    """
    Executes a/b. If b=0, returns 0
    """
    if b == 0 : return 0
    else: return a/b

def safe_divide_one(a, b):
    """
    Executes a/b. If b=0, returns 0
    """
    if b == 0 : return 1
    else: return a/b

def signed_if(condition, a, b):
    """
    Returns a if condition is <= 0, b otherwise
    """
    if condition <= 0 : return a
    else: return b

def boolean_if(condition, a, b):
    if condition: return a
    else: return b
