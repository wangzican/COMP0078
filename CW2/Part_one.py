import numpy as np
import math
import itertools
import pandas as pd
from matplotlib import pyplot as plt

def train_test_split(dataset, ratio, split_label = True):
    """
    Split the data randomly to two sets with a given ratio

    dataset: nparray
    ratio: float = train/whole
    """
    # turn dataframe to list for faster computation
    test = dataset.tolist()
    train = list()
    # get expected size of train_Set
    train_size = int(ratio * len(dataset))

    # pop from the list, the rest are tests
    while len(train) < train_size:
        index = np.random.randint(0,len(test))
        train.append(test.pop(index))
    
    train = np.array(train)
    test = np.array(test)
    # sometimes dont want to split the data and label
    if not split_label:
        return train, test

    x_train = np.array(train[:,1:])
    y_train = np.array(train[:,0])
    x_test = np.array(test[:,1:])
    y_test = np.array(test[:,0])


    return x_train, y_train, x_test, y_test

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
        validation, rest = train_test_split(rest, 1/(n-i), split_label=False)
        validation = np.array(validation)
        validations.append(validation)
    validations.append(np.array(rest))

    # get the corresponding training set
    for index in range(0, len(validations)):
        validation = validations[index]
        whole = validations.copy()
        whole.pop(index)
        training = np.vstack(whole)
        sets.append([training, validation])

    return sets

def visualize(digits):
    """
    Show the digit and its corresponding figure
    digits: array of digits, with label
    """
    size = (16, 16)
    number = digits.shape[0]
    i = 0
    for digit in digits:
        i += 1
        plt.subplot(math.ceil(number/5), 5, i)
        twod = digit[1:].reshape(size)
        plt.imshow(twod, cmap=plt.get_cmap('gray'))
        plt.title(digit[0])
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

def poly_kernel(xi, xj, degree):
    return xi.dot(xj.T) ** degree

def gaussian_kernel(xi, xj, c):
    """
    Calculate the gaussian kernel matrix
    """
    xi_norm = np.expand_dims(np.sum(xi**2, axis =-1), axis = 1)
    xj_norm = np.expand_dims(np.sum(xj**2, axis =-1), axis = 0)
    
    return np.exp(-c*(xi_norm + xj_norm - 2 * (xi.dot(xj.T))))

class onevrest():
    def __init__(self, X, Y, degree, number_of_class=10, kernel_function=poly_kernel) -> None:
        self.number_of_features = X.shape[1]
        self.number_of_data = X.shape[0]
        self.X = X
        self.Y = Y
        self.degree = degree
        self.number_of_class = number_of_class
        self.kernel_function = kernel_function
        # weights for each 1vrest classifier, initialize to zeros
        self.weights = np.zeros((number_of_class, self.number_of_data))

    def fit(self, learning_rate = 1, max_epoch=20, threshold = 0.1):
        """
        Turning a 2 class calssifier into a multiclass one with a polynomial kernel
        """
        # kernel matrix, each (K(xi, xt)) would be each column of the matrix 
        kernel = self.kernel_function(self.X, self.X, self.degree)

        # getting "online" data in random order
        indices = np.arange(self.number_of_data)
        np.random.shuffle(indices)

        for epoch in range(max_epoch):
            errors = 0
            mean_errors = 0
            previous_mean_erros = 0
            # in one epoch do:
            for i in indices:
                # y_hat = argmax(k)((wt|k)(xt))
                # The weights that are not looped (that comes after xt) on are zeros, 
                # which has the same effect of sum(i=0->t) 
                y_hat = int(np.argmax(self.weights.dot(kernel[i,:])))

                # changing the weights of each classifier
                if y_hat != self.Y[i]:
                    errors += 1
                    # update the weight for the classifiers that are responsiblee for missclassifing
                    # i.e. the wrong one should be reduced, the ground truth one should be increased
                    # the rest stays unchanged
                    self.weights[int(self.Y[i]), i] += learning_rate
                    self.weights[y_hat, i] -= learning_rate

            mean_errors = errors/self.number_of_data
            #print("epoch ", epoch, ", errors: ", errors)
            # break out if converges
            if epoch >= 2 and np.abs(previous_mean_erros - mean_errors) < threshold:
                break
            previous_mean_erros = mean_errors
        return self.weights, mean_errors

    def predict(self, x_test):
        """
        predicting with the fitted weights
        """
        kernel = self.kernel_function(self.X, x_test, self.degree)
        y_hat = np.argmax(self.weights.dot(kernel))
        return y_hat

    def test(self, x_test, y_test, detail=False, count_each=False):
        """
        get the mean error for the whole test set
        """
        kernel = self.kernel_function(self.X, x_test, self.degree).T
        error = 0
        number_of_tests = x_test.shape[0]
        # placeholders
        confusion_mtr = np.zeros((self.number_of_class, self.number_of_class))
        number_of_sample_for_each_class = np.zeros(self.number_of_class)
        errors_for_each = np.zeros(number_of_tests)

        for i in range(number_of_tests):
            y_hat = np.argmax(self.weights.dot(kernel[i]))
            # counting the number of samples of each class
            if detail:
                number_of_sample_for_each_class[int(y_test[i])] += 1
            if y_hat != y_test[i]:
                error += 1
                # counting for the confusion matrix
                if detail:
                    confusion_mtr[int(y_test[i]), y_hat] += 1
                if count_each:
                    errors_for_each[i] += 1

        mean_error = error/number_of_tests
        #print(number_of_sample_for_each_class)
        # normalize the mistakes in confusion matrix
        
        if detail:
            for i in range(self.number_of_class):
                confusion_mtr[i] = confusion_mtr[i]/number_of_sample_for_each_class[i]
            return mean_error, confusion_mtr
        if count_each:
            return mean_error, errors_for_each
        return mean_error

class onevone():
    def __init__(self, X, Y, degree, number_of_class=10) -> None:
        self.number_of_features = X.shape[1]
        self.number_of_data = X.shape[0]
        self.X = X
        self.Y = Y
        self.degree = degree
        self.number_of_class = number_of_class

        # a list that returns the value that a classifier is voting for, given the index of the classifier
        self.classifiers = list(itertools.combinations(range(self.number_of_class), 2))
        number_of_classifiers = len(self.classifiers)
        # weights for each 1v1 classifier (n(n-1)/2), initialize to zeros
        self.weights = np.zeros((number_of_classifiers, self.number_of_data))

    def vote(self, confidence):

        votes = np.zeros(self.number_of_class)

        # turn confidence to votes
        confidence[confidence>0] = 1
        confidence[confidence<=0] = 0

        # counting the votes for all voters
        for i in range(len(confidence)):
            votes[self.classifiers[i][int(confidence[i])]] += 1

        # return the class with most votes
        return np.argmax(votes)

    def fit(self, learning_rate = 1, max_epoch=20, threshold = 0.1):
        """
        Turning a 2 class calssifier into a multiclass one with a polynomial kernel
        """
        # kernel matrix, each (K(xi, xt)) would be each column of the matrix 
        kernel = poly_kernel(self.X, self.X, self.degree)

        # getting "online" data in random order
        indices = np.arange(self.number_of_data)
        np.random.shuffle(indices)

        for epoch in range(max_epoch):
            errors = 0
            mean_errors = 0
            previous_mean_erros = 0
            # in one epoch do:
            for i in indices:
                # confidence for each classifier will be calculated like the 1vrest 
                confidence = self.weights.dot(kernel[i,:])
                y_hat = self.vote(confidence)
                
                # changing the weights of each classifier
                if y_hat != self.Y[i]:
                    errors += 1

                # updating all the wrong 1v1 classsifiers
                for index in range(len(self.classifiers)):
                    classes = self.classifiers[index]
                    # if the negative class is classified as positive
                    if self.Y[i] == classes[0] and confidence[index] > 0:
                        self.weights[index, i] -= learning_rate
                    if self.Y[i] == classes[1] and confidence[index] <= 0:
                        self.weights[index, i] += learning_rate
                    

            mean_errors = errors/self.number_of_data
            #print("epoch ", epoch, ", errors: ", errors)
            # break out if converges
            if epoch >= 2 and np.abs(previous_mean_erros - mean_errors) < threshold:
                break
            previous_mean_erros = mean_errors
        return self.weights, mean_errors

    def predict(self, x_test):
        """
        predicting with the fitted weights
        """
        kernel = poly_kernel(self.X, x_test, self.degree)
        confidence = self.weights.dot(kernel)
        y_hat = self.vote(confidence)
        return y_hat

    def test(self, x_test, y_test):
        """
        get the mean error for the whole test set
        """
        # run prediction for all the tests and get the wrong ones
        error = 0
        number_of_tests = x_test.shape[0]
        for i in range(number_of_tests):
            kernel = poly_kernel(self.X, x_test[i], self.degree)
            confidence = self.weights.dot(kernel)
            y_hat = self.vote(confidence)
            if y_hat != y_test[i]:
                error += 1
        mean_error = error/number_of_tests

        return mean_error


# functions for the actual questions, Part 1

def P1Q1(digit_data):
    """
    Part 1
    """
    all_train_loss = []
    all_test_loss = []
    runs = 20
    # iterate through the degrees
    for d in range(0,7):
        degree = d+1
        print("degree: ", degree)
        train_list = []
        test_list = []
        # for each degree, iterate
        for run in range(runs):
            # split the data randomly into train and test portions
            x_train, y_train, x_test, y_test = train_test_split(digit_data, 0.8)
            # define the multiclass classifier
            model = onevrest(x_train, y_train, degree=degree)
            # fitting, also gets the train loss
            weights, loss = model.fit()
            # getting the test loss
            test_loss = model.test(x_test, y_test)

            # print("degree: ", degree, " iter: ", run, " train loss: ", loss)
            # print("test loss: ", test_loss)
            train_list.append(loss)
            test_list.append(test_loss)
        # calculate the mean and standard deviation of train and test errors
        train_mean, train_std = np.mean(train_list), np.std(train_list)
        test_mean, test_std = np.mean(test_list), np.std(test_list)
        all_train_loss.append((train_mean, train_std))
        all_test_loss.append((test_mean, test_std))

    print("Q1: train and test error rate for each degree")
    for i in range(0,7):
        degree = i+1
        print("degree: ", degree)
        print("train: ", all_train_loss[i][0], " +- ",  all_train_loss[i][1])
        print("test: ", all_test_loss[i][0], " +- ",  all_test_loss[i][1])
    print(pd.DataFrame(all_train_loss))
    print(pd.DataFrame(all_test_loss))

def P1Q2(digit_data):
    """
    Question 2
    """
    # initialize list of d and test errors
    optimal_ds = []
    test_errors = []
    total_confusion = []
    runs = 20
    max_degree = 7
    # iterate 20 times
    for run in range(runs):
        train, test = train_test_split(digit_data, 0.8, split_label=False)
        sets = n_fold(train, 5)
        #print(test.shape)
        # iterate through the degrees
        mean_validation_loss = np.zeros(max_degree)
        for d in range(max_degree):
            degree = d+1
            print("run ", run, "degree ", degree)
            validation_error = 0
            for (training, validation) in sets:
                #print(training.shape)
                # split the data into features and labeels 
                x_train = np.array(training)[:,1:]
                y_train = np.array(training)[:,0]
                x_validate = np.array(validation)[:,1:]
                y_validate = np.array(validation)[:,0]
                # define the multiclass classifier
                model = onevrest(x_train, y_train, degree=degree)
                # fitting
                _, _ = model.fit()
                # getting the test loss
                validation_loss = model.test(x_validate, y_validate)
                validation_error += validation_loss

            mean_validation_loss[d] = validation_error
        # add the optimal d value to the list
        optimal_d = np.argmin(mean_validation_loss) + 1
        optimal_ds.append(optimal_d)

        # training using the optimal d
        x_full_train = np.array(train)[:,1:]
        y_full_train = np.array(train)[:,0]
        x_test = np.array(test)[:,1:]
        y_test = np.array(test)[:,0]
        model = onevrest(x_full_train, y_full_train, degree=optimal_d)
        # fitting
        _, _ = model.fit()
        # getting the test loss
        print("test after validation")
        test_loss, confusion_mtr = model.test(x_test, y_test, detail=True)
        test_errors.append(test_loss)
        # getting the list of confusion matrix 
        total_confusion.append(confusion_mtr)
    
    # calculate mean d and test error
    test_mean = np.mean(test_errors)
    test_std = np.std(test_errors)
    d_mean = np.mean(optimal_ds)
    d_std = np.std(optimal_ds)

    # calculate the mean and std of confusion matrix elements
    print("Q2: test errors and optimal ds for each run: ")
    print(test_errors)
    print(optimal_ds)
    print("test: ", test_mean, " +- ", test_std)
    print("d: ", d_mean, " +- ", d_std)
    return np.mean(total_confusion, axis=0), np.std(total_confusion, axis=0)

def P1Q3(mean, std):
    """
    Question 3 given the result from Q2
    """
    mean = mean.tolist()
    std = std.tolist()
    # comebine the mean and std matrix into one
    confusion_mtr = [[(round(mu, 5), round(sigma, 5)) for mu, sigma in zip(mu_line, sigma_line)] for mu_line, sigma_line in zip(mean, std)]
    print(pd.DataFrame(confusion_mtr))

def P1Q4(digit_data):
    """
    Question 4
    """
    iterations = 50
    all_errors = np.zeros(digit_data.shape[0])
    for i in range(iterations):
        optimal_degree = 2
        # split the data randomly into train and test portions
        x_train, y_train, x_test, y_test = train_test_split(digit_data, 0.8)
        # define the multiclass classifier
        model = onevrest(x_train, y_train, degree=optimal_degree)
        # fitting, also gets the train loss
        _, _ = model.fit()

        x_full = digit_data[:,1:]
        y_full = digit_data[:,0]
        # testing will the whole dataset
        _, each_error = model.test(x_full, y_full, count_each=True)
        all_errors += each_error
        print("iteration: ", i)

    # getting the most predicted wrong digits
    worst_five_index = np.argsort(all_errors)[-5:]
    worst_five = []
    for i in range(worst_five_index.size):
        worst_five.append(digit_data[worst_five_index[i]])
    
    # show the digits
    visualize(np.array(worst_five))

def P1Q51(digit_data):
    all_train_loss = []
    all_test_loss = []
    runs = 20
    # define paramter for gaussian kernel
    c = np.arange(1,8)
    # iterate through the degrees
    for d in range(0,7):
        degree = float(2)**(-c[d])
        print("c: ", degree)
        train_list = []
        test_list = []
        # for each degree, iterate
        for run in range(runs):
            # split the data randomly into train and test portions
            x_train, y_train, x_test, y_test = train_test_split(digit_data, 0.8)
            # define the multiclass classifier, with gaussian kernel this time
            model = onevrest(x_train, y_train, degree=degree, kernel_function=gaussian_kernel)
            # fitting, also gets the train loss
            weights, loss = model.fit()
            # getting the test loss
            test_loss = model.test(x_test, y_test)

            # print("degree: ", degree, " iter: ", run, " train loss: ", loss)
            # print("test loss: ", test_loss)
            train_list.append(loss)
            test_list.append(test_loss)
        # calculate the mean and standard deviation of train and test errors
        train_mean, train_std = np.mean(train_list), np.std(train_list)
        test_mean, test_std = np.mean(test_list), np.std(test_list)
        all_train_loss.append((train_mean, train_std))
        all_test_loss.append((test_mean, test_std))

    
    print("Q6-1: train and test error rate for each c value in gaussian kernel")
    for i in range(0,7):
        degree = float(2)**(-c[d])
        print("c: ", degree)
        print("train: ", all_train_loss[i][0], " +- ",  all_train_loss[i][1])
        print("test: ", all_test_loss[i][0], " +- ",  all_test_loss[i][1])
    print(pd.DataFrame(all_train_loss))
    print(pd.DataFrame(all_test_loss))

def P1Q52(digit_data):
    """
    Question 5 repeating q2
    """
    # initialize list of d and test errors
    optimal_ds = []
    test_errors = []
    runs = 20
    max_degree = 7
    # iterate 20 times
    for run in range(runs):
        train, test = train_test_split(digit_data, 0.8, split_label=False)
        sets = n_fold(train, 5)
        #print(test.shape)
        # iterate through the degrees
        mean_validation_loss = np.zeros(max_degree)
        c = np.arange(1,8)
        for d in range(max_degree):
            degree = float(2)**(-c[d])
            print("run ", run, "c: ", degree)
            validation_error = 0
            for (training, validation) in sets:
                #print(training.shape)
                # split the data into features and labeels 
                x_train = np.array(training)[:,1:]
                y_train = np.array(training)[:,0]
                x_validate = np.array(validation)[:,1:]
                y_validate = np.array(validation)[:,0]
                # define the multiclass classifier
                model = onevrest(x_train, y_train, degree=degree, kernel_function=gaussian_kernel)
                # fitting
                _, _ = model.fit()
                # getting the test loss
                validation_loss = model.test(x_validate, y_validate)
                validation_error += validation_loss

            mean_validation_loss[d] = validation_error
        # add the optimal d value to the list
        optimal_d = np.argmin(mean_validation_loss) + 1
        optimal_ds.append(optimal_d)

        # training using the optimal d
        x_full_train = np.array(train)[:,1:]
        y_full_train = np.array(train)[:,0]
        x_test = np.array(test)[:,1:]
        y_test = np.array(test)[:,0]
        model = onevrest(x_full_train, y_full_train, degree=optimal_d, kernel_function=gaussian_kernel)
        # fitting
        _, _ = model.fit()
        # getting the test loss
        print("test after validation")
        test_loss = model.test(x_test, y_test, detail=False)
        test_errors.append(test_loss)
    
    # calculate mean d and test error
    test_mean = np.mean(test_errors)
    test_std = np.std(test_errors)
    d_mean = np.mean(optimal_ds)
    d_std = np.std(optimal_ds)

    print("Q5-2: test errors and optimal ds for each run, with gaussian kernel: ")
    print(test_errors)
    print(optimal_ds)
    print("test: ", test_mean, " +- ", test_std)
    print("d: ", d_mean, " +- ", d_std)

def P1Q61(digit_data):
    """
    Part 1 Question 6
    """
    all_train_loss = []
    all_test_loss = []
    runs = 20
    # iterate through the degrees
    for d in range(0,7):
        degree = d+1
        print("degree: ", degree)
        train_list = []
        test_list = []
        # for each degree, iterate
        for run in range(runs):
            # split the data randomly into train and test portions
            x_train, y_train, x_test, y_test = train_test_split(digit_data, 0.8)
            # define the multiclass classifier
            model = onevone(x_train, y_train, degree=degree)
            # fitting, also gets the train loss
            weights, loss = model.fit()
            # getting the test loss
            test_loss = model.test(x_test, y_test)

            # print("degree: ", degree, " iter: ", run, " train loss: ", loss)
            # print("test loss: ", test_loss)
            train_list.append(loss)
            test_list.append(test_loss)
        # calculate the mean and standard deviation of train and test errors
        train_mean, train_std = np.mean(train_list), np.std(train_list)
        test_mean, test_std = np.mean(test_list), np.std(test_list)
        all_train_loss.append((train_mean, train_std))
        all_test_loss.append((test_mean, test_std))

    print("Q6-1: train and test error rate for each degree, with one vs one multiclass classifier")
    for i in range(0,7):
        degree = i+1
        print("degree: ", degree)
        print("train: ", all_train_loss[i][0], " +- ",  all_train_loss[i][1])
        print("test: ", all_test_loss[i][0], " +- ",  all_test_loss[i][1])
    print(pd.DataFrame(all_train_loss))
    print(pd.DataFrame(all_test_loss))

def P1Q62(digit_data):
    """
    Question 6 
    """
    # initialize list of d and test errors
    optimal_ds = []
    test_errors = []
    runs = 20
    max_degree = 7
    # iterate 20 times
    for run in range(runs):
        train, test = train_test_split(digit_data, 0.8, split_label=False)
        sets = n_fold(train, 5)
        #print(test.shape)
        # iterate through the degrees
        mean_validation_loss = np.zeros(max_degree)
        for d in range(max_degree):
            degree = d+1
            print("run ", run, "degree ", degree)
            validation_error = 0
            for (training, validation) in sets:
                #print(training.shape)
                # split the data into features and labeels 
                x_train = np.array(training)[:,1:]
                y_train = np.array(training)[:,0]
                x_validate = np.array(validation)[:,1:]
                y_validate = np.array(validation)[:,0]
                # define the multiclass classifier
                model = onevone(x_train, y_train, degree=degree)
                # fitting
                _, _ = model.fit()
                # getting the test loss
                validation_loss = model.test(x_validate, y_validate)
                validation_error += validation_loss

            mean_validation_loss[d] = validation_error
        # add the optimal d value to the list
        optimal_d = np.argmin(mean_validation_loss) + 1
        optimal_ds.append(optimal_d)

        # training using the optimal d
        x_full_train = np.array(train)[:,1:]
        y_full_train = np.array(train)[:,0]
        x_test = np.array(test)[:,1:]
        y_test = np.array(test)[:,0]
        model = onevone(x_full_train, y_full_train, degree=optimal_d)
        # fitting
        _, _ = model.fit()
        # getting the test loss
        print("test after validation")
        test_loss = model.test(x_test, y_test)
        test_errors.append(test_loss)
        
    # calculate mean d and test error
    test_mean = np.mean(test_errors)
    test_std = np.std(test_errors)
    d_mean = np.mean(optimal_ds)
    d_std = np.std(optimal_ds)

    # calculate the mean and std of confusion matrix elements
    print("Q6-2: test errors and optimal ds for each run, with one vs one classifier: ")
    print(test_errors)
    print(optimal_ds)
    print("test: ", test_mean, " +- ", test_std)
    print("d: ", d_mean, " +- ", d_std)

if __name__ == "__main__":
    data = np.loadtxt('Data\zipcombo.dat')
    P1Q52(data)