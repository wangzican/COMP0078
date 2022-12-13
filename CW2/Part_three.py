import numpy as np
import matplotlib.pyplot as plt

def sample_dataset(m,n,algorithm):   
    if algorithm == winnow:
        # Winnow need to be sampled from 1 or 0 
        x = np.random.choice([1,0],(m,n))
    else:
        # Other algorithms need to be sampled from -1 or 1
        x = np.random.choice([1,-1],(m,n))
    label = x[:,0]
    return x,label


def perceptron(x,y):
    # Algorithm for perceptron
    w = np.zeros(x.shape[1])
    # store the mistake is not necessary
    # mistake = 0
    for i in range(x.shape[0]):
        y_predict = np.sign(np.dot(w,x[i]))
        if (y_predict*y[i] <= 0):
            w = w+y[i]*x[i]
            # mistake += 1
    
    return w

def winnow(x,y):
    # Algorithm for winnow
    # Init the weight to all ones
    w = np.ones(x.shape[1])
    n = x.shape[1]
    for i in range(x.shape[0]):
        y_predict = 0
        # Make the prediction
        if (np.dot(w,x[i])>=n):
            y_predict = 1
        power = y[i] - y_predict
        # If the prediction is incorrect, update the weight
        if (y_predict!=y[i]):
            w *= (2. ** ((power)*x[i]))
        
    return w

def least_squares(x,y):
    # Algorithm for least_squares
    # return np.linalg.pinv(np.dot(x.T, x)).dot(x.T).dot(y)
    return np.dot(np.linalg.pinv(x), y)

def one_nn_with_prediction(x, y, x_test):
    # Algorithm for 1-NN
    # Unlike the other methods, 1nn would give the prediction directly
    all_predictions = []
    for i in range(x_test.shape[0]):
        # Calculate all the possible distance
        dist = np.linalg.norm(x - x_test[i], axis = 1)
        # Find the index for the nearest one
        index = np.argmin(dist)
        all_predictions.append(y[index])
    return np.array(all_predictions)

def sample_complexity(algo, max_n, test_size, epoches):
    complexity = np.zeros(max_n)
    if algo == one_nn_with_prediction:
        for n in range(max_n):
            m = 0
            current_error = 1.
            # Until the generalisation error is no more than 10 percent, break loop
            while (current_error>0.1):
                m += 1
                all_errors = []
                for epoch in range(epoches):
                    # Generate dataset for training and testing
                    x_train, y_train = sample_dataset(m,n+1,algo) 
                    x_test, y_test = sample_dataset(test_size,n+1,algo)
                    # 1nn function return the prediction directly
                    y_predict = algo(x_train,y_train,x_test)
                    # Calculate all the mistake for current epoch
                    M = np.count_nonzero(y_predict != y_test)
                    all_errors.append((M / test_size))
                current_error = np.mean(all_errors)
            complexity[n] = m
    else:
        # The first loop need to tranverse all the possible dimension from 1
        for n in range(max_n):
            m = 0
            current_error = 1.
            # Until the generalisation error is no more than 10 percent, break loop
            while (current_error>0.1):
                m += 1
                all_errors = []
                for epoch in range(epoches):
                    # Generate dataset for training and testing
                    x_train, y_train = sample_dataset(m,n+1,algo) 
                    x_test, y_test = sample_dataset(test_size,n+1,algo)
                    w = algo(x_train,y_train)
                    if algo == winnow:
                        # Make prediction for winnow is different
                        y_predict = np.where(np.dot(x_test,w) < (n+1), 0, 1)
                    else:
                        y_predict = np.sign(np.dot(x_test,w))
                    # Calculate all the mistake for current epoch
                    M = np.count_nonzero(y_predict != y_test)
                    all_errors.append((M / test_size))
                current_error = np.mean(all_errors)
            complexity[n] = m
    
    return complexity
                    

"""
================Part three question (a)==================
"""

def sample_complexity_perceptron():
    # Init Setting 
    max_n = 100
    epoches = 5
    test_size = 1000
    algo = perceptron
    # Store all the result for each epoch in one matrix
    all_complexity = np.zeros((epoches, max_n))
    for i in range(epoches):
        all_complexity[i, :] = sample_complexity(algo,max_n,test_size,epoches)
    
    # Calculate the avg and std for each epoch
    avg = np.mean(all_complexity, axis = 0)
    std = np.std(all_complexity, axis = 0)
    
    # Increase linearly
    fit_result = np.polyfit(range(1, 101), avg, 1)
    print(fit_result[0],fit_result[1])
    # Plot the graph
    plt.figure(figsize = (8, 6))
    plt.errorbar(range(1, 101), avg, yerr=std, capsize=5, label='Perceptron')
    plt.xlabel("dimension of each data, n")
    plt.ylabel("number of samples, m")
    plt.savefig('Perceptron', dpi=500)
    plt.show()

def sample_complexity_winnow():
    # Init Setting 
    max_n = 100
    epoches = 5
    test_size = 1000
    algo = winnow
    # Store all the result for each epoch in one matrix
    all_complexity = np.zeros((epoches, max_n))
    for i in range(epoches):
        all_complexity[i, :] = sample_complexity(algo,max_n,test_size,epoches)
    
    # Calculate the avg and std for each epoch
    avg = np.mean(all_complexity, axis = 0)
    std = np.std(all_complexity, axis = 0)
    
    # Increase logarithmically
    fit_result = np.polyfit(np.log(range(1, 101)), avg, 1)
    print(fit_result[0],fit_result[1])
    # Plot the graph
    plt.figure(figsize = (8, 6))
    plt.errorbar(range(1, 101), avg, yerr=std, capsize=5, label='Winnow')
    plt.xlabel("dimension of each data, n")
    plt.ylabel("number of samples, m")
    plt.savefig('Winnow', dpi = 500)
    plt.show()

def sample_complexity_least_squares():
    # Init Setting 
    max_n = 100
    epoches = 5
    test_size = 1000
    algo = least_squares
    # Store all the result for each epoch in one matrix
    all_complexity = np.zeros((epoches, max_n))
    for i in range(epoches):
        all_complexity[i, :] = sample_complexity(algo,max_n,test_size,epoches)
    
    # Calculate the avg and std for each epoch
    avg = np.mean(all_complexity, axis = 0)
    std = np.std(all_complexity, axis = 0)
    
    # Increase linearly
    fit_result = np.polyfit(range(1, 101), avg, 1)
    print(fit_result[0],fit_result[1])
    # Plot the graph
    plt.figure(figsize = (8, 6))
    plt.errorbar(range(1, 101), avg, yerr=std, capsize=5, label='Least_squares')
    plt.xlabel("dimension of each data, n")
    plt.ylabel("number of samples, m")
    plt.savefig('Least_squares', dpi = 500)
    plt.show()

def sample_complexity_1nn():
    # Init Setting 
    # 1nn caluculation is extremely slow, so change the max_n to 15
    max_n = 15
    epoches = 5
    test_size = 1000
    algo = one_nn_with_prediction
    # Store all the result for each epoch in one matrix
    all_complexity = np.zeros((epoches, max_n))
    for i in range(epoches):
        all_complexity[i, :] = sample_complexity(algo,max_n,test_size,epoches)
    
    # Calculate the avg and std for each epoch
    avg = np.mean(all_complexity, axis = 0)
    std = np.std(all_complexity, axis = 0)
    
    # Increase exponentially
    fit_result = np.polyfit(range(1, 16), np.log(avg), 1)
    print(fit_result[0],fit_result[1])
    # Plot the graph
    plt.figure(figsize = (8, 6))
    plt.errorbar(range(1, 16), avg, yerr=std ,capsize=5, label = '1NN')
    plt.xlabel("dimension of each data, n")
    plt.ylabel("number of samples, m")
    plt.savefig('1NN', dpi=500)
    plt.show()
        
if __name__ == "__main__":
    # x, y = sample_dataset(10,10)
    # x_test, y_test = sample_dataset(30,10)
    # sample_complexity_perceptron()
    # sample_complexity_least_squares()
    sample_complexity_winnow()
    # sample_complexity_1nn()
