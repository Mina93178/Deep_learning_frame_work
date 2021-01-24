import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import layers , forward_model , Losses , backward_model
from dataset import dataset
import math
from optimization import momentum ,ADAM
from Metric import evaluation_metrics
from matplotlib import style
import matplotlib; matplotlib.use("TkAgg")
#style.use('fivethirtyeight')
#fig = plt.figure()
#ax1 = fig.add_subplot(1, 1, 1)
fig,ax1=plt.subplots()
#from plotting import animate
class training_model:
    '''
    " The class concerned the training using mini_batch gradient descent with optimizations like adam and momnetum GD "
    '''
    def __init__(self,minibatches, activations,alpha=0.0007 , no_of_iterations=25, print_cost = True, lambd = 0 ,momentum=0,beta=0.9,ADAM=1):
        '''
        :param minibatches : list of mini_batches after dividing the training set to mini_batches
        :param activations : the type of activation function in each layer
        :param alpha: learning rate
        :param no_of_iterations: number of iterations
        :param print_cost: boolean variable , put it with ( True ) value if you want to print the cost every 10
                    iterations
        :param lambd: regularization parameter
        :param momentum : boolean : put it with ( True ) value if you want to apply the momentum Gradient descent
        :param beta : momentum parameter
        :param ADAM : boolean : put it with a ( True ) value if you want to apply ADAM optimization
        :return : The Trained Parameters for a certain model trained on a certain dataset
        '''
        self.input = minibatches
        self.activation_functions=activations
        self.Beta=beta
        self.learning_rate = alpha
        self.no_of_iterations = no_of_iterations
        self.print_cost=print_cost
        self.regularization_parameter=lambd
        self.momentum_or_no=momentum
        self.adam_or_not=ADAM


    def update_parameters(self,parameters, grads, learning_rate):
        '''
          "The function which is used to update the weights and biases of the model with Vanilla Gradient descent ."
                :param parameters: weights and biases of the past iteration ( before updating )
                :param grads: the Gradients of the weights and biases , the output of the backward propagation
                :param learning_rate: The learning rate
                :return: parameters : updated weights and biases after completing one iteration of the training
        '''
        L = len(parameters) // 2  # number of layers in the neural network
        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

        return parameters
    #def cost_animation(self,cost):
     #   cost = []
      #  self.append[cost]
    #def change_plot(self,y):
      #  self.fig.set_ydata(np.random.rand(100))
       # plt.pause(1)
       # plt.show()
    def animate(i):
        '''
         "A function which is used to draw a live plotting of the Cost function during the training process "
        '''
        # print('hi')
        graph_data = open('costs.txt', 'r').read()
        lines = graph_data.split('\n')
        xs = []
        ys = []
        # plt.title("Learning rate =" + str(self.learning_rate))
        for line in lines:
            if len(line) > 1:
                x, y = line.split(',')
                xs.append(float(x))
                ys.append(float(y))
        ax1.clear()
        ax1.set_xlim(-2, 26)
        ax1.set_ylim(0, 1.6)
        plt.ylabel('cost')
        plt.xlabel('iterations')
        ax1.plot(xs, ys)
    def train(self):
        '''
                " This function considered as the integration of all the past Modules together to start training any model
                  the deep learning engineer will have the option to choose :
                  1- the activation function of each layer
                  2- the loss type
                  3- the number of iterations
                 this function will plot live graph for the training cost and finally will print the the accuracy resulted
                  from the test set training using the parameters resulted from the training set training .
               :return: The Trained parameters
               '''
        adam_flag=1
        #m = self.input.shape[1]  # number of examples
        layers_dimensions = [self.input[0][0].shape[0], 128, 10]
        temp_layers = layers.layers(layers_dimensions)
        # temp_forward = forward_model.forward_model(layers_dimensions)
        # Initialize parameters dictionary.
        parameters = temp_layers.layers_init()
        temp = parameters
        if (self.momentum_or_no):
            velocity=momentum(parameters).velocity_preparation()
        if(self.adam_or_not):
            exponentially_weighted_parameter,RMS_parameter= ADAM(parameters).adam_preparation()
        #print (parameters)
        #print (parameters)
        #print("weights:")
        #print(parameters["W3"].shape)

        # Loop (gradient descent)
        cost_file = open("costs.txt", 'a+')
        cost_file.truncate(0)
        cost_file.close()
        for i in range(0,self.no_of_iterations):
            #if (self.momentum_or_no):
             #   velocity = momentum(parameters).velocity_preparation()
            #if (self.adam_or_not):
             #   exponentially_weighted_parameter, RMS_parameter = ADAM(parameters).adam_preparation()
            for j in range(len(self.input)):
                train_Y = data_set.labels_to_onehot(self.input[i][1])
                train_X = self.input[i][0]
               # scaler = StandardScaler()
               # train_X=scaler.fit_transform(train_X)
                #print(train_X.shape)
                no_of_training_examples=self.input[i][1].shape[1]

                predictions, packet_of_packets = forward_model.forward_model().forward_model(train_X, parameters,self.activation_functions)

                # print("DONE")

                # Cost function
                if self.regularization_parameter == 0:
                    #print(predictions)
                    cost = Losses.multiclass_loss(train_Y, predictions).cost()
                # print(cost)
                else:
                    cost = Losses.regularization().compute_cost_with_regularization(predictions, train_Y, parameters,self.regularization_parameter,"multiclass")

                # Backward propagation.
                #assert (self.regularization_parameter == 0 )  # it is possible to use both L2 regularization and dropout,
            # but this assignment will only explore one at a time
                grads = backward_model.model_backward_general(predictions, train_Y, packet_of_packets, "multiclass",self.regularization_parameter,self.activation_functions).model_backward()
            # Update parameters.
                if(self.momentum_or_no):
                    parameters, velocity = momentum(parameters).update_with_momentum(velocity, self.learning_rate,
                                                                                     self.Beta, grads)

                elif(self.adam_or_not):
                    parameters,exponentially_weighted_parameter,RMS_parameter=ADAM(parameters).update_with_adam(exponentially_weighted_parameter,RMS_parameter,self.learning_rate,parameters,grads,i)

                else:
                    parameters = self.update_parameters(parameters, grads, self.learning_rate)
            # plot the cost
            #costs.append(cost_avg)
            cost_file = open("costs.txt", 'a+')
            cost_file.write(f"{i},{cost} \n")
            cost_file.close()
            plt.ion()
            plt.show()
            plt.draw()
            plt.pause(1)
            print(f"cost after epoch{i}: {cost}")
            #print(grads)
           # plt.figure()
           # plt.plot(costs)
           # plt.ylabel('cost')
           # plt.xlabel('iterations')
           # plt.title("Learning rate =" + str(self.learning_rate))
        #plt.show()

        return parameters

        # self.change_plot(costs)

            # Print the loss every 10000 iterations
            # if self.print_cost and i % 10 == 0:
            #   print("Cost after iteration {}: {}".format(i, cost))
            # if self.print_cost and i % 1000 == 0:
            #   costs.append(cost)




def random_mini_batches(X, Y, mini_batch_size=1024):
    '''
    " this function will divide the whole training examples to small modules each one called ( mini_batch )
    :param X : The whole Training Set AKA all the training examples
    :param Y : The whole Training examples' labels
    :param mini_batch_size: The wanted mini_batch size for the training examples
    :return : a list of mini batches after division
    '''

    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, int(m / mini_batch_size) * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, int(m / mini_batch_size) * mini_batch_size:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

if __name__ == "__main__":

    data_set = dataset('mnist', r'C:\Users\FacultyStudent\PycharmProjects\final_NN')
    train_X, train_Y, test_X, test_Y = data_set.get_dataset()
    training_minibatches = random_mini_batches(train_X.T, train_Y,mini_batch_size=2000)
    ani = animation.FuncAnimation(fig, training_model.animate, interval=1000)
    # test_X_minibatches = random_mini_batches(test_X.T,test_Y)
    # print(test_Y.shape)
    # print(test_Y)
    # print(train_X.shape)
    # print(test_X.shape)
    # print(train_Y.shape)
    # print(test_Y.shape)

    ''''''
    # for i in range(len(training_minibatches)):
    # train_Y = data_set.labels_to_onehot(training_minibatches[i][1])
    # train_X=training_minibatches[i][0]
    # parameters=1
    # print(train_X.shape)
    # print(train_Y.shape)
    # global parameters
    activation_functions = ["NONE", "sigmoid", "sigmoid"]

    parameters = training_model(training_minibatches,activation_functions).train()
    print("on training set : ")
    print(evaluation_metrics(train_Y, train_X.T, parameters,activation_functions).Accuracy(train_X.shape[0]))
    print("on test set : ")
    print(evaluation_metrics(test_Y, test_X.T, parameters,activation_functions).Accuracy(test_X.shape[0]))
    # print(evaluation_metrics(test_Y,test_X.T,parameters).Accuracy(test_X.shape[0]))
    # print("On the test set:")
    # predictions_test = predict(test_X.T, test_Y, parameters)
    # print(f"cost at iteration={np.mean(cost)}")
    # costs_minibatch.append(np.mean(cost))
    # parameters = training_model(X_train, Y_train).train()
    # print(f"On the training set:{i}")
    # predictions_train = predict(X_train, Y_train, parameters)
    # predictions_train = predict(train_X, train_Y, parameters)
    # print("On the test set:")
    # predictions_test = predict(test_X.T, test_Y, parameters)
    # predictions_test = predict(X_test, Y_test, parameters)

    # print("On the training set:")
    # predictions_train = predict(X_train, Y_train, parameters)
    # predictions_train = predict(train_X, train_Y, parameters)
    # plt.figure()
    # plt.plot(costs_minibatch)
    # plt.ylabel('cost')
    # plt.xlabel('iterations (x1,000)')
    # plt.title("Learning rate =" + str(self.learning_rate))
    # plt.show()
    '''
    plt.title("Model without regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, test_X.T), train_X, train_Y)
    '''

