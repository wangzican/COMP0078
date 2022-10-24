from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


training = np.array([[1,3],[2,2],[3,0],[4,5]])

def mse(y, x, w, dim):
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
    y_prime = calc(x,w,dim)

    # check shape of y and y':
    if(np.array(y).size == np.array(y_prime).size):
        return ((y-y_prime)**2).mean(axis=0)
    else:
        return 0

def calc(x,w,dim):
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
    for i in range (0,dim):
        y_prime += pow(x,i) * w[i]
    return y_prime

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
    x_train = np.transpose(data)[0].reshape(4,1)
    bias = np.full((4,1), 1)
    x = bias

    for i in range(1, n):
        x_more = np.power(x_train, i)
        x = np.concatenate((x, x_more), axis = 1)


    y = np.transpose(data)[1].reshape(4,1)
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
    x = np.array(x)
    x = x.reshape(x.size,1)
    y = np.array(y)
    y =  y.reshape(y.size,1)

    bias = np.full((x.size, 1), 1)
    x_train = bias

    for i in range(1, n):
        x_more = np.power(x, i)
        x_train = np.concatenate((x_train, x_more), axis = 1)

    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x_train), x_train)), np.transpose(x_train)), y)
    #print("k = ", n, ": ", w)
    return w.reshape(n,1)


def sin_square(x, sigma):
    """
    calculate the function g(x) = sin^2(2 * Pi * X) + epsilon(normal noise)
    mean = 0;
    sigma = int, standard deviation
    x: array

    """
    return pow((np.sin(2*np.pi*x)),2) + np.random.normal(0, pow(sigma,2))

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

    y1 = calc(x, w1, 1)
    y2 = calc(x, w2, 2)
    y3 = calc(x, w3, 3)
    y4 = calc(x, w4, 4)

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

    y2 = calc(x_plot,w2,2)
    y5 = calc(x_plot,w5,5)
    y10 = calc(x_plot,w10,10)
    y14 = calc(x_plot,w14,14)
    y18 = calc(x_plot,w18,18)

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
        print("Iteration ", iter)
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
    plt.title("natural log of averaged general mse in 1000 trials wrt dimension")
    print("Q2 (d): ")
    plt.show()
