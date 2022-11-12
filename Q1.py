from cProfile import label
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os,sys
import heapq


training = np.array([[1,3],[2,2],[3,0],[4,5]])

# helper functions
def mse(y, y_prime):
    """
    This function computes the mean squared error between y_prime and y

    y: an array of sampled 'outcome'
    y_prime: an array of calculated outcome
    """
    return ((y-y_prime)**2).mean(axis=0)

def mse_poly(y, x, w, dim=0):
    """
    This function computes the mean squared error between the polyfit outcome and y, given the bases dimension,
    for x with a single attribute

    x: an array of attribute value
    y: an array of sampled 'outcome'
    w: an array of weights of each power of x
    dim: int, the dimension (degree of polynomial + 1)

    e.g.    x = [x1,x2]
            y = [y1,y2]
            dim = 3
            w = [1,2,3]
            mse = 1/2 * { 
                    [(1 + 2*(x1) + 3*(x1)^2) - y1]^2 + 
                    [(1 + 2*(x2) + 3*(x2)^2) - y2]^2
                        }
    """
    # dimension not sepecified -> use calc_multi
    if dim == 0:
        y_prime = calc_multi(x,w)
    else:
        # if dimension specified
        y_prime = calc_poly(x,w,dim)
    
    # check shape of y and y':
    if(np.array(y).size == np.array(y_prime).size):
        return ((y-y_prime)**2).mean(axis=0)
    else:
        return 0

def calc_poly(x,w,dim):
    """
    calculate, with given dimension, the value of y' = w0 + w1 * x + w2 * x^2 + ...
    
    x: an array of attribute value
    dim: int, dimension
    w: an array of weights of each power of x

    e.g. 
        x = [x1,x2,x3]
        dim = 3
        w = [w0,w1,w2]
    """
    # check if w matches the dimension
    if(np.array(w).size != dim):
        print("w: ", w, "dim = ", dim, "dimension does not match")
        return 0

    # continue if dimension is correct
    y_prime = 0

    # calculate for each dimension
    for i in range (0,dim):
        y_prime += pow(x,i) * w[i]
    return y_prime

def calc_multi(x,w):
    """
    calculate, with multiple attributes dimension one, the value of y' = w0 + w1 * x1 + w2 * x2 + ...
    
    x: an array of attribute value
    w: an array of weights of each attribute of x

    e.g. 
        x = [[x11,x12,x13],
             [x21,x22,x23],
             [x31,x32,x33]]
        w = [w0,w1,w2]
    """
    if x.ndim == 1:
        x = x.reshape(x.size, 1)

    # sometimes we need to add an bias column
    if x.shape[1] != w.shape[0]:
        x = np.hstack((np.ones((x.shape[0], 1)),x))

    # calculating y'
    y_prime = np.dot(x,w)

    return y_prime

def calc_sin(x,w,dim):
    """
    calculate, with given dimension, the value of y' = w0 + w1 * sin(pi x) + w2 * sin(2pi x) + ...
    
    x: an array of attribute value
    dim: int, dimension
    w: an array of weights of each power of x

    e.g. 
        x = [x1,x2,x3]
        dim = 3
        w = [w0,w1,w2]
    """
    # check if w matches the dimension
    if(np.array(w).size != dim):
        print("w: ", w, "dim = ", dim, "dimension does not match")
        return 0

    # continue if dimension is correct
    y_prime = 0
    for i in range (0,dim):
        y_prime += sin(x,i+1) * w[i]
    return y_prime

def mse_sin(y,x,w,dim):
    """
    This function computes the mean squared error between the sinfit outcome and y, given the bases dimension,
    for x with a single attribute

    x: an array of attribute value
    y: an array of sampled 'outcome'
    w: an array of weights of each power of x
    dim: int, the dimension (degree of polynomial + 1)

    e.g.    x = [x1,x2]
            y = [y1,y2]
            dim = 3
            w = [1,2,3]
    """
    y_prime = calc_sin(x,w,dim)

    # check shape of y and y':
    if(np.array(y).size == np.array(y_prime).size):
        return ((y-y_prime)**2).mean(axis=0)
    else:
        return 0

def gd(shape, data, rate, max_iteration, small = 0.1):
    print("gd: ")
    y = np.transpose(data)[1].reshape(4,1)

    start = np.random.random()
    x = start
    for _ in range(max_iteration):
        diff = rate*np.gradient(x)
        if np.abs(diff)<small:
            break    


def lrn(data, n):
    """
    Fit the data given with base dimension n:

    data: combination(array) of x and y value as coordinates
    n: dimension, degree + 1

    e.g. data = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
         n = 2
    """
    # prepares data
    x_train = np.transpose(data)[0].reshape(4,1)
    y = np.transpose(data)[1].reshape(4,1)

    # adding bias term
    bias = np.full((4,1), 1)
    x = bias

    # add each dimension
    for i in range(1, n):
        x_more = np.power(x_train, i)
        x = np.concatenate((x, x_more), axis = 1)

    # using w = (x^T x)^-1 x^T y
    w = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(x), x)), np.transpose(x)), y)
    return w.reshape(n,1)

def lrn_separate(x: np.array, y: np.array, n):
    """
    Fit the x, y given with base dimension n:

    data: combination(array) of x and y value as coordinates
    n: dimension, degree + 1

    e.g. x = [x1,x2,x3]
         y = [y1,y2,y3]
         n = 2
    """

    # prepares data, fill in the dimension
    x = np.array(x)
    y = np.array(y)
    if(x.ndim == 1):
        x = x.reshape(x.size,1)
        y =  y.reshape(y.size,1)

    #adding bias term of ones, with the same size as number of samples
    bias = np.full((x.shape[0], 1), 1)
    x_train = bias

    for i in range(1, n):
        x_more = np.power(x, i)
        x_train = np.concatenate((x_train, x_more), axis = 1)

    # using the equation
    w = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(x_train), x_train)), np.transpose(x_train)), y)
    #print("k = ", n, ": ", w)
    return w.reshape(w.size,1)

def lrn_sin(x: np.array, y: np.array, n):
    """
    Fit the x, y given with base dimension n as sin(n pi x)

    data: combination(array) of x and y value as coordinates
    n: dimension

    e.g. x = [x1,x2,x3]
         y = [y1,y2,y3]
         n = 2
    """
    # prepares data
    x = np.array(x)
    x = x.reshape(x.size,1)
    y = np.array(y)
    y =  y.reshape(y.size,1)

    # first term of sin(pi x) to set dimension
    x_train = sin(x,1)

    for i in range(1, n):
        x_more = sin(x,i+1)
        x_train = np.concatenate((x_train, x_more), axis = 1)

    # using the equation
    w = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(x_train), x_train)), np.transpose(x_train)), y)
    #print("k = ", n, ": ", w)
    return w.reshape(n,1)

def sin(x, k):
    """
    Calculate sin(k pi x)
    """
    return np.sin(k*np.pi*x)

def sin_square(x, sigma, k = 2):
    """
    calculate the function g(x) = sin^2(k * Pi * X) + epsilon(normal noise)
    mean = 0;
    sigma = int, standard deviation
    x: array
    k; integer

    """
    return pow((np.sin(k*np.pi*x)),2) + np.random.normal(0, pow(sigma,2))

def initialize_data(size:int, sigma:int):
    """
    Initialize the sin function data with noise

    size: int
    sigma: int
    """
    x = []
    y = []
    for i in range (0,size):
        random = np.random.rand()
        x.append(random)
        y.append(sin_square(random, sigma))
    return x,y

def train_test_split(dataset, ratio):
    """
    Split the data randomly to two sets with a given ratio

    dataset: dataframe
    ratio: float = train/whole
    """
    # turn dataframe to list for faster computation
    test = dataset.values.tolist()
    train = list()
    # get expected size of train_Set
    train_size = int(ratio * len(dataset))

    # pop from the list, the rest are tests
    while len(train) < train_size:
        index = np.random.randint(0,len(test))
        train.append(test.pop(index))
    
    # convert back to dataframes
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    # add the column names back
    train.columns = dataset.columns
    test.columns = dataset.columns
    return train, test

def n_fold(dataset, n):
    """
    Returns n sets randomly poped from a dataframe that are equally sized,
    for cross-validation

    returns:
    sets: [[training, validation],
           [training, validation],
           [training, validation]]
    """
    sets = []
    validations = []
    
    rest = dataset
    # get the validation set
    for i in range(0,n-1):
        validation, rest = train_test_split(rest, 1/(n-i))
        validation = np.array(validation)
        validations.append(validation)
    validations.append(np.array(rest))

    # get the corresponding training set
    for index in range(0, len(validations)):
        validation = validations[index]
        whole = validations.copy()
        whole.pop(index)
        training = np.vstack(whole)
        sets.append([pd.DataFrame(training), pd.DataFrame(validation)])

    return sets

def gaussian(x1, x2, sigma):
    """
    calculate the gaussian kernel for one set of data
    """
    return np.exp( - np.linalg.norm(x1 - x2)**2 / (2 * sigma**2))

def calc_gaussian(x, xt, a, sigma):
    """
    Returns the y' of x1 and xt when given x, x_test, a* and sigma
    x: [[x1],
        [x2],
        [x3]]
    """
    # get shapes
    rows = xt.shape[0]
    y_prime = np.zeros(shape = (rows,1))
    # for each x_test
    index = 0
    for tests in xt:
        # for each x
        train_index = 0
        for trains in x:
            #print(a[index])
            y_prime[index] += a[train_index] * gaussian(trains, tests, sigma)
            train_index += 1
        index += 1
    return y_prime

def gaussian_kernel(x, sigma):
    """
    Returns the gaussian kernel of x1 and xt when given only x and sigma
    x: [[x1],
        [x2],
        [x3]]
    """
    # get shapes
    rows = x.shape[0]

    kernel = np.zeros(shape = (rows,rows))
    # for each x1
    i = 0
    for ki in x:
        #for each x2
        j = 0
        for kj in x:
            kernel[i,j] = gaussian(ki,kj,sigma)
            j+=1
        i+=1
    return kernel

def kernel_ridge(x, y, sigma, gamma):
    """
    Performs kernel ridge regression 
    """
    # prepares kernel and I
    kernel = gaussian_kernel(x,sigma)
    I = np.identity(kernel.shape[0])

    # apply function
    a_star = np.dot(np.linalg.pinv(kernel + kernel.shape[0] * gamma * I),  y)
    return a_star

def pH(n = 100):
    """
    prepares the hypothsis pH
    """    
    x = np.random.uniform(0, 1, (2, n))
    y = np.random.randint(0, 2, (n, 1))
    return x, y

def ph(n):
    """
    samples h from pH and generate ph(x, y)
    p(heads) = 0.8, p(tails) = 0.2 for each x
    when head, y = h3(x,y)
    when tail, y = random {0,1}
    n: size
    """
    x, y = pH()

    # randomly generate x of size n
    x_samples = np.random.uniform(0, 1, (2, n)).T
    y_samples = np.array([])
    for xs in x_samples:
        coin = np.random.choice(['head', 'tail'], p=[0.8, 0.2])
        ys = knn(x.T, y, xs.reshape(1,2)) if coin == 'head' else np.random.randint(0,2)
        y_samples = np.append(y_samples, ys)

    return x_samples, y_samples


def eucledian(p1,p2):
    """
    returns the eucledian distance between two points
    """
    dist = np.linalg.norm(p1-p2, 2, axis = 1)
    return dist

def knn(xtrain, ytrain, xtest, k=3):
    """
    predict the given point
    xtrain: [[x11,x12],
             [x21,x22],
             [x31,x32]]
    ytrain: [0,1,1]
    xtest: [[xt11, xt12],
            [xt21,xt22],
            xt31,xt32]
    """
    predict = np.array([])

    # Loop for each test data point
    for i in range(0,xtest.shape[0]):
        #if i % 1000 == 0:
        #    print(i)

        # sorting a list wrt distance and get the first k elements
        dist = eucledian(xtrain, xtest[i])
        index = np.argsort(dist)[0:k]

        average = np.mean(ytrain[index])

        y_prime = 0
        if average > 0.5:
            y_prime = 1
        elif average < 0.5:
            y_prime = 0
        else: 
            y_prime = np.random.randint(0,2)

        predict = np.append(predict, y_prime)
    return predict

# Section 1.1
def plot_graph():
    """
        Q1
    """
    #turn data into lists of x values and y values
    transposed_data = np.transpose(training)
    # plotting the values
    plt.plot(transposed_data[0], transposed_data[1], 'ro')


    # training the data with different dimensions
    x = np.linspace(1,5,50)
    w1 = lrn(training,1)
    w2 = lrn(training, 2)
    w3 = lrn(training, 3)
    w4 = lrn(training, 4)

    y1 = calc_poly(x, w1, 1)
    y2 = calc_poly(x, w2, 2)
    y3 = calc_poly(x, w3, 3)
    y4 = calc_poly(x, w4, 4)

    # plotting the fitted curves
    print("Q1 (a): ")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.plot(x, y1, color ="red", label = 'k = 1')
    plt.plot(x, y2, color ="blue", label = 'k = 2')
    plt.plot(x, y3, color ="black", label = 'k = 3')
    plt.plot(x, y4, color ="green", label = 'k = 4')
    plt.legend()
    plt.title("Fitting with different dimensions")
    plt.show()

    print("Q1 (b): ")
    print("k = 1: ", w1.flatten())
    print("k = 2: ", w2.flatten())
    print("k = 3: ", w3.flatten())
    print("k = 4: ", w4.flatten())
    print("\n")

    # output the mse of the respective function
    print("Q1 (c): ")
    print("k = 1: ", mse_poly(transposed_data[1], transposed_data[0], w1, 1))
    print("k = 2: ", mse_poly(transposed_data[1], transposed_data[0], w2, 2))
    print("k = 3: ", mse_poly(transposed_data[1], transposed_data[0], w3, 3))
    print("k = 4: ", mse_poly(transposed_data[1], transposed_data[0], w4, 4))

def draw_sin_with_noise(figure = None):
    """
    Q2, (a), i
    """
    if figure == None:
        figure = plt.figure()

    # ploting the points with noise
    x, y = initialize_data(30, 0.07)
    plt.plot(x, y, 'ro', figure = figure, label = 'samples')

    # plotting the sin graph
    samples = 100
    x = np.linspace(0,1,samples)
    y = []
    for i in range (0,samples):
        y.append(sin_square(x[i],0)) # no noise

    plt.plot(x, y, figure = figure, label = 'actual function')
    plt.legend()
    plt.title("Sample and actual function with sin")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    print("Q2 (a) i: ")
    plt.show()

def fitting_sin():
    """
    Q2, (a), ii
    """
    # generate points
    x, y = initialize_data(30, 0.07)
    plt.plot(x, y, 'ro')

    x_plot = np.linspace(0,1,100)

    w2 = lrn_separate(x,y,2)
    w5 = lrn_separate(x,y,5)
    w10 = lrn_separate(x,y,10)
    w14 = lrn_separate(x,y,14)
    w18 = lrn_separate(x,y,18)

    y2 = calc_poly(x_plot,w2,2)
    y5 = calc_poly(x_plot,w5,5)
    y10 = calc_poly(x_plot,w10,10)
    y14 = calc_poly(x_plot,w14,14)
    y18 = calc_poly(x_plot,w18,18)

    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.plot(x_plot, y2, color ="red", label = '2')
    plt.plot(x_plot, y5, color ="blue", label = '5')
    plt.plot(x_plot, y10, color ="black", label = '10')
    plt.plot(x_plot, y14, color ="green", label = '14')
    plt.plot(x_plot, y18, color ="pink", label = '18')
    plt.title("fitting the sin curve with different dimensions")
    
    plt.ylim(-1,2)
    plt.legend()
    print("Q2 (a) ii: ")
    plt.show()

def sin_ln_mse():
    """
        Q2, (b)
    """
    # generate points
    x, y = initialize_data(30, 0.07)

    # computing the mse in a loop
    dim = np.arange(1,19)
    err = []
    for i in dim:
        w = lrn_separate(x,y,i)
        err.append(np.log(mse_poly(y,np.array(x),w,i)))
    
    plt.plot(dim, err, color ="red")
    plt.xlabel("dimension")
    plt.ylabel("ln(MSE)")
    plt.title("natural log of MSE wrt Dimension")
    print("Q2 (b): ")
    plt.show()

def thousand_points():
    """
    Q2, (c)
    """
    # generate the test and training data
    x_train, y_train = initialize_data(30, 0.07)
    x_test, y_test = initialize_data(1000, 0.07)

    dim = np.arange(1,19)
    err = []
    for i in dim:
        w = lrn_separate(x_train,y_train,i)
        err.append(np.log(mse_poly(y_test,np.array(x_test),w,i)))
    
    #plotting test error
    plt.plot(dim, err, color ="red")
    plt.xlabel("dimension")
    plt.ylabel("ln(MSE)")
    plt.title("natural log of general MSE wrt Dimension")
    print("Q2 (c): ")
    plt.show()

def hundred_iter():
    """
    Q2, (d)
    """
    # a container for all iterations of train and test error
    all_train_err = []
    all_test_err = []
    for iter in range(0,100):
        #print("Iteration ", iter)
        x_train, y_train = initialize_data(30, 0.07)
        x_test, y_test = initialize_data(1000, 0.07)
        dim = np.arange(1,19)
        train_err = []
        test_err = []
        for i in dim:
            # calculating the train/test error for each dimension
            w = lrn_separate(x_train,y_train,i)
            train_err.append(np.log(mse_poly(y_train,np.array(x_train),w,i)))
            test_err.append(np.log(mse_poly(y_test,np.array(x_test),w,i)))

        # stacking the new row of errors on bottom
        if(iter == 0):
            all_train_err = train_err
            all_test_err = test_err
        else:
            all_train_err = np.vstack((all_train_err, train_err))
            all_test_err = np.vstack((all_test_err, test_err))

    # calculating the mean for each column
    mean_train_err = np.mean(all_train_err, axis = 0)
    mean_test_err = np.mean(all_test_err, axis = 0)

    plt.plot(dim, mean_train_err, color ="red")
    plt.plot(dim, mean_test_err, color ="blue")
    plt.xlabel("dimension")
    plt.ylabel("MSE")
    plt.title("natural log of averaged general mse in 100 trials wrt dimension")
    print("Q2 (d): ")
    plt.show()

def sin_basis():
    """
    Q3
    """
    # generate points
    x, y = initialize_data(30, 0.07)

    # computing the mse in a loop
    dim = np.arange(1,19)
    err = []
    for i in dim:
        w = lrn_sin(x,y,i)
        err.append(np.log(mse_sin(y,np.array(x),w,i)))
    
    plt.plot(dim, err, color ="red")
    plt.xlabel("dimension")
    plt.ylabel("ln(MSE)")
    plt.title("natural log of training MSE wrt Dimension sin(k pi x)")
    print("Q3 (b): ")
    plt.show()

    # generate the test and training data
    x_train, y_train = initialize_data(30, 0.07)
    x_test, y_test = initialize_data(1000, 0.07)
    # clear err
    err = []
    for i in dim:
        w = lrn_sin(x_train,y_train,i)
        err.append(np.log(mse_sin(y_test,np.array(x_test),w,i)))
    
    #plotting test error
    plt.plot(dim, err, color ="red")
    plt.xlabel("dimension")
    plt.ylabel("ln(MSE)")
    plt.title("natural log of general MSE wrt Dimension sin(k pi x)")
    print("Q3 (c): ")
    plt.show()

    # a container for all iterations of train and test error
    all_train_err = []
    all_test_err = []
    for iter in range(0,100):
        #print("Iteration ", iter)
        x_train, y_train = initialize_data(30, 0.07)
        x_test, y_test = initialize_data(1000, 0.07)
        dim = np.arange(1,19)
        train_err = []
        test_err = []
        for i in dim:
            # calculating the train/test error for each dimension
            w = lrn_sin(x_train,y_train,i)
            train_err.append(np.log(mse_sin(y_train,np.array(x_train),w,i)))
            test_err.append(np.log(mse_sin(y_test,np.array(x_test),w,i)))

        # stacking the new row of errors on bottom
        if(iter == 0):
            all_train_err = train_err
            all_test_err = test_err
        else:
            all_train_err = np.vstack((all_train_err, train_err))
            all_test_err = np.vstack((all_test_err, test_err))

    # calculating the mean for each column
    mean_train_err = np.mean(all_train_err, axis = 0)
    mean_test_err = np.mean(all_test_err, axis = 0)

    plt.plot(dim, mean_train_err, color ="red")
    plt.plot(dim, mean_test_err, color ="blue")
    plt.xlabel("dimension")
    plt.ylabel("MSE")
    plt.title("natural log of averaged general mse in 100 trials wrt dimension sin(k pi x)")
    print("Q3 (d): ")
    plt.show()
    
# Section 1.2
def naive_regression(mute = False):
    """
    Q4, (a)
    """
    data = pd.read_csv('Boston-filtered.csv')
    iteration = 20
    total_training_mse = []
    total_testing_mse = []
    for i in range(0,iteration):
        #prepare the data
        train, test = train_test_split(data, 2/3)
        y_train = np.array(train.iloc[:, 12:13])
        y_test = np.array(test.iloc[:, 12:13])

        # prepare using ones
        x_test = np.ones(test.iloc[:, 0:12].shape[0])
        x_train = np.ones(train.iloc[:, 0:12].shape[0])

        #fitting
        w = lrn_separate(x_train, y_train, 1)
        
        training_mse = mse_poly(y_train, x_train, w)
        testing_mse = mse_poly(y_test, x_test, w)

        total_training_mse.append(training_mse[0])
        total_testing_mse.append(testing_mse[0])

    mean_training_mse = np.mean(total_training_mse)
    mean_testing_mse = np.mean(total_testing_mse)
    
    # for question 5d, where this function will be called, but no print needed
    if not mute:
        print("Q4, (a): ")
        print("training loss is: ", mean_training_mse)
        print("testing loss is: ", mean_testing_mse)
        print("Q4, (b): w is the mean of the training y values.")
    return total_training_mse, total_testing_mse

def single_attribute(mute = False):
    """
    Q4, (c)
    """
    data = pd.read_csv('Boston-filtered.csv')
    iteration = 20
    total_training_mse = []
    total_testing_mse = []

    if not mute:
        print("Q4, (c): ")
    # loop for each attribute
    for attributes in range(0,12):

        single_attribute_training_mse = []
        single_attribute_testing_mse = []

        # loop for 20 iterations
        for i in range(0,iteration):
            train, test = train_test_split(data, 2/3)
            y_train = np.array(train.iloc[:, 12:13])
            y_test = np.array(test.iloc[:, 12:13])

            # prepare using one attribute
            x_test = np.array(test.iloc[:, attributes:attributes + 1])
            x_train = np.array(train.iloc[:, attributes:attributes + 1])

            # dim = 2 means using the ones and the actual value
            w = lrn_separate(x_train, y_train, 2)
            
            training_mse = mse_poly(y_train, x_train, w)
            testing_mse = mse_poly(y_test, x_test, w)

            single_attribute_training_mse.append(training_mse[0])
            single_attribute_testing_mse.append(testing_mse[0])

        mean_training_mse = np.mean(single_attribute_training_mse)
        mean_testing_mse = np.mean(single_attribute_testing_mse)

        total_training_mse.append(single_attribute_training_mse)
        total_testing_mse.append(single_attribute_testing_mse)
        if not mute:
            print("for attribute ", train.columns[attributes])
            print("training loss is: ", mean_training_mse)
            print("testing loss is: ", mean_testing_mse)
    return total_training_mse, total_testing_mse
        
def all_attributes(mute = False):
    """
    Q4, (d)
    """
    data = pd.read_csv('Boston-filtered.csv')
    iteration = 20
    total_training_mse = []
    total_testing_mse = []
    for i in range(0,iteration):
        #prepare the data
        train, test = train_test_split(data, 2/3)
        x_train = np.array(train.iloc[:, 0:12])
        y_train = np.array(train.iloc[:, 12:13])
        x_test = np.array(test.iloc[:, 0:12])
        y_test = np.array(test.iloc[:, 12:13])
        
        # dim = 2 means using the ones and the actual value
        w = lrn_separate(x_train, y_train, 2)
        
        training_mse = mse_poly(y_train, x_train, w)
        testing_mse = mse_poly(y_test, x_test, w)

        total_training_mse.append(training_mse[0])
        total_testing_mse.append(testing_mse[0])

    mean_training_mse = np.mean(total_training_mse)
    mean_testing_mse = np.mean(total_testing_mse)
    
    if not mute:
        print("Q4, (d): ")
        print("training loss is: ", mean_training_mse)
        print("testing loss is: ", mean_testing_mse)
    return total_training_mse, total_testing_mse

# Section 1.3
def five_fold(mute = False):
    """
    Q5, (a), (b)
    """
    if not mute:
        print("Q5, (a)")
    # load data
    data = pd.read_csv('Boston-filtered.csv')
    train, test = train_test_split(data, 2/3)
    x_whole_train = np.array(train.iloc[:, 0:12])
    y_whole_train = np.array(train.iloc[:, 12:13])
    x_test = np.array(test.iloc[:, 0:12])
    y_test = np.array(test.iloc[:, 12:13])
    # cross validation
    sets = n_fold(train, 5)

    # initialize gamma and sigma
    g_power = np.arange(-40,-25,1)
    s_power = np.arange(7,13.5, 0.5)
    gamma = pow(2., g_power)
    sigma = pow(2., s_power)

    best_loss = np.inf
    best_g = 0
    best_sig = 0
    # create an array for all the mean loss to sit in
    mean_loss_matrix = np.zeros((len(gamma), len(sigma)))

    # for all combinations of gamma and sigma
    i = 0
    for g_value in gamma:
        j = 0
        for sig in sigma:
            mean_loss = 0
            for (training, validation) in sets:
                x_train = np.array(training.iloc[:, 0:12])
                y_train = np.array(training.iloc[:, 12:13])
                x_validate = np.array(validation.iloc[:, 0:12])
                y_validate = np.array(validation.iloc[:, 12:13])
                # fit to kernel regressions
                a = kernel_ridge(x_train, y_train, sig, g_value)

                # compute y'
                y_prime = calc_gaussian(x_train, x_validate, a, sig)
                
                mean_loss += mse(y_prime, y_validate)
            mean_loss /= 5
            if not mute:
                print("gamma: ", g_value)
                print("sigma: ", sig)
                print("mean mse: ", mean_loss)
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_g = i
                best_sig = j
            mean_loss_matrix[i,j] = mean_loss
            j += 1
        i += 1
    
    # Q5, (a)
    if not mute:
        print("the best set is: gamma: 2^", g_power[best_g], 
              " sigma: 2^", s_power[best_sig], 
              " loss: ", best_loss)
    

    # Q5, (b)
    if not mute: 
        print("Q5, (b)")
        matplotlib.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.set_size_inches(30,30)

        plot_gamma, plot_sigma = np.meshgrid(g_power, s_power)
        # Plot the surface.
        surf = ax.plot_surface(plot_gamma, plot_sigma, mean_loss_matrix.T, cmap=matplotlib.cm.coolwarm, antialiased = True)
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=10)
        # Labels and title
        ax.set_xlabel("\nGamma",linespacing=4)
        ax.set_ylabel("\nSigma",linespacing=4)
        ax.set_zlabel("\nValidation loss",linespacing=4)
        ax.set_title("Validation loss agains gamma and sigma")
        ax.view_init(20, 20)
        plt.show()
    
    # Q5, (c)
    # Use the best parameter on the whole training set
    # fit to kernel regressions
    a = kernel_ridge(x_whole_train, y_whole_train, sigma[best_sig], gamma[best_g])

    # compute y'
    y_prime_train = calc_gaussian(x_whole_train, x_whole_train, a, sigma[best_sig])
    y_prime_test = calc_gaussian(x_whole_train, x_test, a, sigma[best_sig])
    
    training_mse = mse(y_prime_train, y_whole_train)
    testing_mse = mse(y_prime_test, y_test)

    if not mute:
        print("Q5, (c)")
        print("The loss with optimal gamma and sigma is: (train: ", training_mse, ") (test: ", testing_mse, ")")
    return training_mse, testing_mse

def comparing_mse():
    """
    Q5, (d)
    """
    print("Q5, (d)")
    # computer Q4, a,c,d
    naive_train, naive_test = naive_regression(mute = True)
    single_train, single_test = single_attribute(mute = True)
    all_train, all_test = all_attributes(mute = True)

    print("\nnaive regression:")
    print("train: ", np.mean(naive_train), " +- ", np.std(naive_train))
    print("test: ", np.mean(naive_test), " += ", np.std(naive_test))

    # there are 12 single attribute loss arrays for both train and test
    index = 1
    for train, test in zip(single_train, single_test):
        print("\nfor attribute ", index, ": ")
        print("train: ", np.mean(train), " +- ", np.std(train))
        print("test: ", np.mean(test), " += ", np.std(test))
        index += 1

    print("\nall attributes: ")
    print("train: ", np.mean(all_train), " +- ", np.std(all_train))
    print("test: ", np.mean(all_test), " +- ", np.std(all_test))

    # Computer Q5, a,c
    print("\ncalculating for kernel: ...")
    kernel_train = []
    kernel_test = []
    iteration = 20
    for i in range(0,iteration):
        train, test = five_fold(mute = True)
        kernel_train.append(train)
        kernel_test.append(test)
    print("\nkernel ridge:")
    print("train: ", np.mean(kernel_train), " +- ", np.std(kernel_train))
    print("test: ", np.mean(kernel_test), " +- ", np.std(kernel_test))

# Section 2.1
def knn_visualize():
    """
    Q6
    """
    # generating x and y randomly
    x, y = pH()

    # generate grid for all points
    precision = 100
    x_region = np.linspace(0,1, precision)
    x1, x2 = np.meshgrid(x_region, x_region)
    all_points = np.array([x1.ravel(), x2.ravel()]).T

    # transpose x to get x1, x2 pairs
    y_prime = knn(x.T, y, all_points)
    y_prime = y_prime.reshape(x1.shape)

    plt.contourf(x1, x2, y_prime, cmap = 'Blues_r')
    plt.scatter(x[0], x[1], c=y, cmap = 'binary')
    print("Q6: ")
    plt.show()

def knn_general_err():
    """
    Q7
    """
    print("Q7 (a): ")
    all_error = []
    # for each k
    for k in range(1,50):
        print("k = ", k)
        losses = 0
        iteration_size = 100
        # for each iteration
        for iteration in range(0,iteration_size):
            train_size = 4000
            test_size = 1000

            # get the train and test set from distribution ph
            x, y = ph(train_size)
            x_test, y_test = ph(test_size)

            # get the y' and calculate missclassification
            y_prime = knn(x, y, x_test, k)
            loss = np.linalg.norm(y_prime-y_test, 1)/test_size
            losses += loss

        # get mean loss
        losses /= iteration_size
        all_error.append(losses)
        print("loss = ", losses)
    print(all_error)
    
    plt.plot(np.arange(1,50), all_error)
    plt.xlabel("loss")
    plt.ylabel("k")
    plt.title("k against loss")
    plt.show()

def knn_optimal_k():
    """
    Q8
    """
    print("Q8 (a): ")
    best_ks = []
    # generate m
    m = np.arange(0,41,5)
    m[0] = 1
    m = 100*m

    for ms in m:
        print("m: ", ms)
        best_k_all = 0
        iteration_size = 100
        # for each iteration
        for iteration in range(0,iteration_size):
            # for each k
            best_loss = np.inf
            best_k = 0
            for k in range(1,50):
                train_size = ms
                test_size = 1000

                # get the train and test set from distribution ph
                x, y = ph(train_size)
                x_test, y_test = ph(test_size)

                # get the y' and calculate missclassification
                y_prime = knn(x, y, x_test, k)
                loss = np.linalg.norm(y_prime-y_test, 1)/test_size

                # update best loss
                if loss < best_loss:
                    best_loss = loss
                    best_k = k

            best_k_all += best_k
        # average the total loss from each iteration
        best_k_all /= iteration
        print("best k: ", best_k_all)
        best_ks.append(best_k_all)

    
    plt.plot(m, best_ks)
    plt.xlabel("k")
    plt.ylabel("m")
    plt.title("m against k")
    plt.show()


knn_optimal_k()