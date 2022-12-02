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
        data = np.delete(data, selected_index, axis=0)

    if return_index:
        return all_index

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
        index = np.argsort(dist)[0:k+1]
        index = index[1:]
        weight_matrix[i,index] += 1

    return weight_matrix

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


def LaplacianKernelInterpolation(dataset, L):
    return

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
            total_errors = []
            for i in range(iteration):
                sampled_index = sample_wo_replacement(dataset, [1, 3], l)
                # split the data into features and labels
                errors = LaplacianInterpolation(dataset, sampled_index)
                total_errors.append(errors)
            mean_errors = np.mean(total_errors)
            std_errors = np.std(total_errors)
            print("dataset ", index, "L = ", l, "error = ", mean_errors, " +- ", std_errors)

if __name__ == "__main__":
    Part_2()