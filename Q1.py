from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


training = np.array([[1,3],[2,2],[3,0],[4,5]])

# helper functions
def mse(y, x, w, dim=0):
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


def knr(data, n):
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
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.transpose(x)), y)
    return w.reshape(n,1)

def knr_separate(x: np.array, y: np.array, n):
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
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x_train), x_train)), np.transpose(x_train)), y)
    #print("k = ", n, ": ", w)
    return w.reshape(w.size,1)

def knr_sin(x: np.array, y: np.array, n):
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
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x_train), x_train)), np.transpose(x_train)), y)
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
    ratio: float
    """
    # turn datafram to list for faster computation
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
    w1 = knr(training,1)
    w2 = knr(training, 2)
    w3 = knr(training, 3)
    w4 = knr(training, 4)

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
    print("k = 1: ", mse(transposed_data[1], transposed_data[0], w1, 1))
    print("k = 2: ", mse(transposed_data[1], transposed_data[0], w2, 2))
    print("k = 3: ", mse(transposed_data[1], transposed_data[0], w3, 3))
    print("k = 4: ", mse(transposed_data[1], transposed_data[0], w4, 4))

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

    w2 = knr_separate(x,y,2)
    w5 = knr_separate(x,y,5)
    w10 = knr_separate(x,y,10)
    w14 = knr_separate(x,y,14)
    w18 = knr_separate(x,y,18)

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
        w = knr_separate(x,y,i)
        err.append(np.log(mse(y,np.array(x),w,i)))
    
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
        w = knr_separate(x_train,y_train,i)
        err.append(np.log(mse(y_test,np.array(x_test),w,i)))
    
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
            w = knr_separate(x_train,y_train,i)
            train_err.append(np.log(mse(y_train,np.array(x_train),w,i)))
            test_err.append(np.log(mse(y_test,np.array(x_test),w,i)))

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
        w = knr_sin(x,y,i)
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
        w = knr_sin(x_train,y_train,i)
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
            w = knr_sin(x_train,y_train,i)
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
def naive_regression():
    """
    Q4, (a)
    """
    data = pd.read_csv('Boston-filtered.csv')
    iteration = 20
    total_training_mse = 0
    total_testing_mse = 0
    for i in range(0,iteration):
        #prepare the data
        train, test = train_test_split(data, 0.33)
        y_train = np.array(train.iloc[:, 12:13])
        y_test = np.array(test.iloc[:, 12:13])

        # prepare using ones
        x_test = np.ones(test.iloc[:, 0:12].shape[0])
        x_train = np.ones(train.iloc[:, 0:12].shape[0])

        #fitting
        w = knr_separate(x_train, y_train, 1)
        
        training_mse = mse(y_train, x_train, w)
        testing_mse = mse(y_test, x_test, w)

        total_training_mse += training_mse
        total_testing_mse += testing_mse

    mean_training_mse = total_training_mse/iteration
    mean_testing_mse = total_testing_mse/iteration
    
    print("Q4, (a): ")
    print("training loss is: ", mean_training_mse)
    print("testing loss is: ", mean_testing_mse)
    print("Q4, (b): w is the mean of the training y values.")

def single_attribute():
    """
    Q4, (c)
    """
    data = pd.read_csv('Boston-filtered.csv')
    iteration = 20
    total_training_mse = 0
    total_testing_mse = 0

    print("Q4, (c): ")
    # loop for each attribute
    for attributes in range(0,12):
        # loop for 20 iterations
        for i in range(0,iteration):
            train, test = train_test_split(data, 0.33)
            y_train = np.array(train.iloc[:, 12:13])
            y_test = np.array(test.iloc[:, 12:13])

            # prepare using one attribute
            x_test = np.array(test.iloc[:, attributes:attributes + 1])
            x_train = np.array(train.iloc[:, attributes:attributes + 1])

            # dim = 2 means using the ones and the actual value
            w = knr_separate(x_train, y_train, 2)
            
            training_mse = mse(y_train, x_train, w)
            testing_mse = mse(y_test, x_test, w)

            total_training_mse += training_mse
            total_testing_mse += testing_mse

        mean_training_mse = total_training_mse/iteration
        mean_testing_mse = total_testing_mse/iteration

        print("for attribute ", train.columns[attributes])
        print("training loss is: ", mean_training_mse)
        print("testing loss is: ", mean_testing_mse)
        
def all_attributes():
    """
    Q4, (d)
    """
    data = pd.read_csv('Boston-filtered.csv')
    iteration = 20
    total_training_mse = 0
    total_testing_mse = 0
    for i in range(0,iteration):
        #prepare the data
        train, test = train_test_split(data, 0.33)
        x_train = np.array(train.iloc[:, 0:12])
        y_train = np.array(train.iloc[:, 12:13])
        x_test = np.array(test.iloc[:, 0:12])
        y_test = np.array(test.iloc[:, 12:13])
        
        # dim = 2 means using the ones and the actual value
        w = knr_separate(x_train, y_train, 2)
        
        training_mse = mse(y_train, x_train, w)
        testing_mse = mse(y_test, x_test, w)

        total_training_mse += training_mse
        total_testing_mse += testing_mse

    mean_training_mse = total_training_mse/iteration
    mean_testing_mse = total_testing_mse/iteration
    
    print("Q4, (d): ")
    print("training loss is: ", mean_training_mse)
    print("testing loss is: ", mean_testing_mse)


# Section 1.3