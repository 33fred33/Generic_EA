from inspect import signature
import numpy as np

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

def accuracy(y, y_output):
    n = len(y)
    bools = [y_output[i] == y[i] for i in range(n)]
    return sum(bools)/n

def accuracy_in_label(y,y_output,label):
    n = len(y)
    bools = [y_output[i] == y[i] for i in range(n) if y[i] == label]
    return sum(bools)/n


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
