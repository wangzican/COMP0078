import numpy as np

def sample_wo_replacement(data, classes, size, return_index=True):
    # prevent repeated labels
    classes = set(classes)

    all_index = np.array([])
    all_sampled_data = np.array([])
    for class_label in classes:
        # get the indices with the correct label
        index = np.where(data[:,0] == class_label)[0]
        if index.size == 0:
            print("No label with ", class_label)
            continue
        # select a random index for the class
        selected_index = np.random.choice(index, size, replace=False)
        sampled_data = np.take(data, selected_index, axis=0)

        # start with no data
        if all_sampled_data.size == 0:
            all_sampled_data = sampled_data
            all_index = selected_index
        else:
            all_sampled_data = np.vstack((all_sampled_data, sampled_data))
            all_index = np.hstack((all_index, selected_index))

    if return_index:
        return all_index
        
    data = np.delete(data, all_index, axis=0)

    rest_data = data
    return all_sampled_data, rest_data

def eucledian(p1,p2):
    """
    returns the eucledian distance between two points
    """
    dist = np.linalg.norm(p1-p2, 2, axis = 1)
    return dist

def knn(graph, k=3):
    """
    return the weight matrix of a graph with k-nn connection
    """
    number_of_points = graph.shape[0]
    weight_matrix = np.zeros((number_of_points, number_of_points))

    # Loop for each data point
    for i in range(0, number_of_points):
        #if i % 1000 == 0:
        #    print(i)

        # sorting a list wrt distance and get the first k elements
        dist = eucledian(graph, graph[i])
        # get the first k+1 ones, so that all k neighbors except the first one with itself
        index = np.argsort(dist)[1:k+1]

        weight_matrix[i, index] = 1
        for j in range(index.size):
            if weight_matrix[index[j], i] != 1:
                weight_matrix[index[j], i] = 1

    return weight_matrix

def Laplacian_matrix(graph, k=3):
    """
    Return the Laplacian matrix
    """
    weight = knn(graph, k)
    diagonal = np.diag(np.sum(weight, axis=1))
    Laplacian = diagonal - weight
    return Laplacian

def Laplacian_kernel(L, labeled_index):
    """
    Retrun the kernel for Laplacian interpolation
    """
    inverse = np.linalg.pinv(L)
    # print(np.meshgrid(labeled_index, labeled_index))
    kernel_matrix = np.zeros((labeled_index.shape[0], labeled_index.shape[0]))

    for i in range(labeled_index.shape[0]):
        kernel_matrix[i] = inverse[labeled_index[i], labeled_index]

    return kernel_matrix

def e(index, length):
    """
    Return one hot vector representation
    """
    e = np.zeros(length)
    e[index] = 1
    return e

def LaplacianInterpolation(data, labeled_index):
    """
    For two classes, 1 and 3
    v := argmin(u) u.T L u : ui = yi; for i in labeled data
    u.T L u = sum(i<j) wij(ui - uj)**2
    """
    unlabeled_index = np.delete(np.arange(data.shape[0]), labeled_index)
    # splitting the data, start with the other y unlabeled
    x_data = data[:, 1:]
    y_true_label = data[:, 0]
    y_predict = np.zeros(y_true_label.shape[0])
    y_predict[labeled_index] = y_true_label[labeled_index]
    # turn classes into +-1
    y_predict[y_predict == 1] = -1
    y_predict[y_predict == 3] = 1

    # calculate the weight matrix
    weights = knn(x_data)

    max_iteration = data.shape[0] * 100
    # optimize each label random walk
    for iteration in range(max_iteration):
        # get free nodes
        index = np.random.choice(unlabeled_index)
        y_predict[index] = weights[index].dot(y_predict)/np.sum(weights[index])

    y_predict[y_predict > 0] = 3
    y_predict[y_predict <= 0] = 1
    wrong_prediction = (y_predict - y_true_label)
    wrong_prediction[wrong_prediction != 0] = 1
    errors = np.sum(wrong_prediction)
    return errors

def LaplacianKernelInterpolation(data, labeled_index):
    """
    For two classes 1, 3
    """
    # splitting the data
    x_data = data[:, 1:]
    y_true_label = data[:, 0]
    # getting the labeled y
    y_labeled = y_true_label[labeled_index]
    y_labeled[y_labeled == 1] = -1
    y_labeled[y_labeled == 3] = 1

    # calculate the weight matrix
    Lap = Laplacian_matrix(x_data)

    kernel = Laplacian_kernel(Lap, labeled_index)
    kernel_inv = np.linalg.pinv(kernel)
    a_star = np.dot(kernel_inv, y_labeled)

    V = 0
    for i in range(labeled_index.shape[0]):
        V += a_star[i] * np.dot(e(labeled_index[i], data.shape[0]), np.linalg.pinv(Lap)) 

    V[V > 0] = 3
    V[V <= 0] = 1

    wrong_prediction = (V - y_true_label)
    wrong_prediction[wrong_prediction != 0] = 1
    errors = np.sum(wrong_prediction)
    return errors

def Part_2():
    data1 = np.loadtxt('Data\dtrain13_50.dat')
    data2 = np.loadtxt('Data\dtrain13_100.dat')
    data3 = np.loadtxt('Data\dtrain13_200.dat')
    data4 = np.loadtxt('Data\dtrain13_400.dat')
    Datasets = [data1, data2, data3, data4]
    L = [1, 2, 4, 8, 16]

    index = 0
    for dataset in Datasets:
        index += 1
        for l in L:
            iteration = 20
            total_errLI = []
            total_errLKI = []
            for i in range(iteration):
                sampled_index = sample_wo_replacement(dataset, [1, 3], l)
                sampled_index = np.sort(sampled_index)
                # split the data into features and labels
                errLI = LaplacianInterpolation(dataset, sampled_index)
                errLKI = LaplacianKernelInterpolation(dataset, sampled_index)
                total_errLI.append(errLI)
                total_errLKI.append(errLKI)
            mean_errLI = np.mean(total_errLI)
            std_errLI = np.std(total_errLI)
            mean_errLKI = np.mean(total_errLKI)
            std_errLKI = np.std(total_errLKI)
            print("LI: dataset ", index, "L = ", l, "error = ", mean_errLI, " +- ", std_errLI)
            print("LKI: dataset ", index, "L = ", l, "error = ", mean_errLKI, " +- ", std_errLKI)

if __name__ == "__main__":
    Part_2()