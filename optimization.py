import numpy as np
class momentum:
    def __init__(self,parameters):
        self.parameters=parameters


    def velocity_preparation(self):
        weights = len(self.parameters) // 2  # number of layers in the neural networks
        v = {}
        # Initialize velocity
        for w in range(weights):
            v["dW" + str(w + 1)] = np.zeros((self.parameters["W" + str(w + 1)].shape[0], self.parameters["W" + str(w + 1)].shape[1]))
            v["db" + str(w + 1)] = np.zeros((self.parameters["b" + str(w + 1)].shape[0], self.parameters["b" + str(w + 1)].shape[1]))
        return v


    def update_with_momentum(self,velocity,learning_rate,exponentially_weighted_average_parameter,gradients):
        L = len(self.parameters) // 2  # number of layers in the neural networks
        for l in range(L):
            # compute velocities
             velocity["dW" + str(l + 1)] = exponentially_weighted_average_parameter * velocity["dW" + str(l + 1)] + (1 - exponentially_weighted_average_parameter) *  gradients['dW' + str(l + 1)]
             velocity["db" + str(l + 1)] = exponentially_weighted_average_parameter *  velocity["db" + str(l + 1)] + (1 - exponentially_weighted_average_parameter) * gradients['db' + str(l + 1)]
            # update parameters
             self.parameters["W" + str(l + 1)] = self.parameters["W" + str(l + 1)] -  learning_rate *  velocity["dW" + str(l + 1)]
             self.parameters["b" + str(l + 1)] = self.parameters["b" + str(l + 1)] -  learning_rate *  velocity["db" + str(l + 1)]

        return self.parameters, velocity




class ADAM:
    def __init__(self,parameters):
        self.parameters=parameters


    def adam_preparation(self):
        L = len(self.parameters) // 2  # number of layers in the neural networks
        EWA = {}
        RMS = {}

        # Initialize v, s. Input: "parameters". Outputs: "v, s".
        for l in range(L):
            # exponentially weighted average parameters
            EWA["dW" + str(l + 1)] = np.zeros((self.parameters["W" + str(l + 1)].shape[0], self.parameters["W" + str(l + 1)].shape[1]))
            EWA["db" + str(l + 1)] = np.zeros((self.parameters["b" + str(l + 1)].shape[0], self.parameters["b" + str(l + 1)].shape[1]))
            # RMS prop average parameters
            RMS["dW" + str(l + 1)] = np.zeros((self.parameters["W" + str(l + 1)].shape[0], self.parameters["W" + str(l + 1)].shape[1]))
            RMS["db" + str(l + 1)] = np.zeros((self.parameters["b" + str(l + 1)].shape[0], self.parameters["b" + str(l + 1)].shape[1]))

        return EWA, RMS

    def update_with_adam(self,EWA,RMS,learning_rate,parameters,gradients,epoch_num,frist_beta=0.9,second_beta=0.99,epsilon=1e-8,):

        L = len(parameters) // 2  # number of layers in the neural networks
        EWA_corrected = {}  # Initializing first moment estimate, python dictionary
        RMS_corrected = {}  # Initializing second moment estimate, python dictionary

        # Perform Adam update on all parameters
        for l in range(L):

            EWA["dW" + str(l + 1)] = frist_beta * EWA["dW" + str(l + 1)] + (1 - frist_beta) * gradients['dW' + str(l + 1)]
            EWA["db" + str(l + 1)] = frist_beta * EWA["db" + str(l + 1)] + (1 - frist_beta) * gradients['db' + str(l + 1)]
            #EWA_corrected["dW" + str(l + 1)] = EWA["dW" + str(l + 1)] / ((1 - frist_beta ** epoch_num)+epsilon)
            #EWA_corrected["db" + str(l + 1)] = EWA["db" + str(l + 1)] / ((1 - frist_beta ** epoch_num)+epsilon)
            RMS["dW" + str(l + 1)] = second_beta * RMS["dW" + str(l + 1)] + (1 - second_beta) * np.square(gradients['dW' + str(l + 1)])
            RMS["db" + str(l + 1)] = second_beta * RMS["db" + str(l + 1)] + (1 - second_beta) * np.square(gradients['db' + str(l + 1)])
            #RMS_corrected["dW" + str(l + 1)] = RMS["dW" + str(l + 1)] / ((1 - second_beta ** epoch_num)+epsilon)
            #RMS_corrected["db" + str(l + 1)] = RMS["db" + str(l + 1)] / ((1 - second_beta ** epoch_num)+epsilon)
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * EWA["dW" + str(l + 1)] / (np.sqrt(RMS["dW" + str(l + 1)]) + epsilon)
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * EWA["db" + str(l + 1)] / (np.sqrt(RMS["db" + str(l + 1)]) + epsilon)


        return parameters, EWA,RMS
