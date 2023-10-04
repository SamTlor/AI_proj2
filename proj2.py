# https://www.youtube.com/watch?v=7VV_fUe6ziw

import pandas as pd
import numpy as np
import random, time

# A: epsilon < 10^-5
# B: epsilon < 40
# C: epsilon < 700



def norm(dataset):
    for column_index in range(dataset.shape[1] - 1):
        max_val = dataset[column_index].max()
        min_val = dataset[column_index].min()

        dataset[column_index] = (dataset[column_index] - min_val) / (max_val - min_val)

def train(dataset, error_threshold):
    limit = 5000
    alpha = 0.01                     #hard activation function
    patterns = dataset.shape[0]
    w = [random.uniform(-0.5, 0.5) for _ in range(dataset.shape[1])]
    print(w)
    
    
    #while i is less than 5000 or the total error is not accurate enough
    i = 0
    error = patterns
    while i < limit or error > error_threshold:  #work on total error
        error = 0
        for row in range(patterns):                         #work on training and testing
            x = dataset.iloc[row].values
            
            scaled_x = np.dot(w, x)
            if scaled_x > 0:
                scaled_x = 1
            else:
                scaled_x = 0
                
            #for total error. sum[(out - desired)^2]. (-1)^2 = 1 and 1^2 = 1 so 
            #just use += when they're different. forehead
            if x[2] != scaled_x:
                error += 1
                
            delta_weight = alpha * (x[2] - scaled_x)
            delta_weighted_x = x * delta_weight
            w = np.array(w) + np.array(delta_weighted_x)
        i += 1
    
    
    # get accuracy, confusion matrices and rates
    return w



if __name__ == "__main__":
    #read the datasets
    a = pd.read_csv("groupA.csv", header = None)
    b = pd.read_csv("groupB.csv", header = None)
    c = pd.read_csv("groupC.csv", header = None)
    
    
    
    #convert to numeric
    for col in a.columns:
        a[col] = a[col].apply(pd.to_numeric, errors = 'coerce')
        b[col] = b[col].apply(pd.to_numeric, errors = 'coerce')
        c[col] = c[col].apply(pd.to_numeric, errors = 'coerce')
        
        

    #normalize the data
    norm(a)
    norm(b)
    norm(c)
    
    
    
    #hard activation function
        #split the data 75 for training and 25 for testing
        #split the data 25 for training and 75 for testing
        #answer questions about the two steps above
    #soft activation function
        #split the data 75 for training and 25 for testing
        #split the data 25 for training and 75 for testing
        #answer questions about the two steps above
        
    
    start = time.time()
    print(train(a, 0.00001))
    end = time.time()
    
    print(end - start)
    