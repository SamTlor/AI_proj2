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

def loop(dataset, error_threshold):
    limit = 5000
    alpha = 0.3
    patterns = dataset.shape[0]
    w = [random.uniform(-0.5, 0.5) for _ in range(dataset.shape[1])]
    
    
    
    #while i is less than 5000 or the total error is not accurate enough
    i = 0
    while i < limit or total_error() > error_threshold:     #work on total error
        #for every line in the csv
        for row in range(patterns):                         #work on training and testing
            x = dataset.iloc[row].values
            
            scaled_x = np.dot(w, x)
            if scaled_x > 0:
                scaled_x = 1
            else:
                scaled_x = 0
                
            delta_weight = alpha * (x[2] - scaled_x)
            delta_weighted_x = x * delta_weight
            w = np.array(w) + np.array(delta_weighted_x)
        i += 1
                
                
                
    # get accuracy, confusion matrices and rates
    return w



def total_error(test, correct):                                 #from slide 11
    comparison = test.compare(correct, align_axis = 2)
    return comparison['Value'].count()
    


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
    print(loop(a, 0.00001))
    end = time.time()
    
    print(end - start)
    
    # test = pd.read_csv("test.csv", header = None)
    # print(loop(test))