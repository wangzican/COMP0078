import numpy as np
import matplotlib.pyplot as plt


training = np.array([[1,3],[2,2],[3,0],[4,5]])

def k1r(data):
    x = np.transpose(data)[0].reshape(4,1)
    bias = np.full((4,1), 1)
    x = np.concatenate((bias,x), axis=1)

    y = np.transpose(data)[1].reshape(4,1)
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.transpose(x)), y)
    print("k = 1: ", w)
    return w

def mse(y, x, w, poly):
    y_prime = 0
    for i in range (0,poly+1):
        y_prime += pow(x,i) * w[i]
    return ((y-y_prime)**2).mean(axis=0)


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
    
    x_train = np.transpose(data)[0].reshape(4,1)
    bias = np.full((4,1), 1)
    x = np.concatenate((bias,x_train), axis=1)

    for i in range(2, n+1):
        print(i)
        x_more = np.power(x_train, i)
        x = np.concatenate((x, x_more), axis = 1)
        print(x)


    y = np.transpose(data)[1].reshape(4,1)
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.transpose(x)), y)
    print("k = 1: ", w)
    return w

def plot_graph():
    #turn data into list of x values and y values
    transposed_data = np.transpose(training)
    plt.plot(transposed_data[0], transposed_data[1], 'ro')

    w1 = k1r(training)
    x = np.linspace(1,5,50)
    y1 = x*w1[1] + w1[0]

    w2 = knr(training, 2)
    y2 = pow(x,2) * w2[2] + x*w2[1] + w2[0]

    w3 = knr(training, 3)
    y3 = pow(x,3) * w3[3] + pow(x,2) * w3[2] + x*w3[1] + w3[0]

    w4 = knr(training, 4)
    y4 = pow(x,4) * w4[4] +  pow(x,3) * w4[3] + pow(x,2) * w4[2] + x*w4[1] + w4[0]

    print(mse(transposed_data[1], transposed_data[0], w1, 1))
    print(mse(transposed_data[1], transposed_data[0], w2, 2))
    print(mse(transposed_data[1], transposed_data[0], w3, 3))
    print(mse(transposed_data[1], transposed_data[0], w4, 4))

    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.plot(x, y1, color ="red")
    plt.plot(x, y2, color ="blue")
    plt.plot(x, y3, color ="black")
    #plt.plot(x, y4, color ="green")
    plt.show()

plot_graph()