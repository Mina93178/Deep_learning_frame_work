import numpy as np

class initialization:
    '''
    <spanclass="font-weight-bold comment">
    " This is the initialization class which will initialize the weights and biases for the layers of any model . "
    </span>
    '''
    def __init__(self,dimension_layers):
        self.dimension_layers = dimension_layers
        '''
        <spanclass="font-weight-bold comment">
        " The Constructor of this class takes the a list contains the number of hidden units in each layer ."
        
        :type dimension_layers: List . 
        :param dimension_layers: list of the number of hidden units in each hidden layer .
        
        </span>
        '''


    def initialize_parameters(self): # [6,5,9,1]

        '''
         <spanclass="font-weight-bold comment">
         " This function will initialize a random values for weights and zeros for biases using the
                            dimension_layers list . "

         :type parameters: dictionary .
         :param parameters: this dictionary contains the weights and biases with the same dimension as
                                each layers' dimension .
        </span>
        '''
        parameters = {}
        #np.random.seed(3)
        for layer in range(1, len(self.dimension_layers)):
            #parameters['W' + str(l)] = np.random.normal(0,1,(self.dimension_layers[l], self.dimension_layers[l - 1])) * 0.01
            parameters['W' + str(layer)] = np.random.randn(self.dimension_layers[layer], self.dimension_layers[layer - 1]) *0.1
            parameters['b' + str(layer)] = np.zeros((self.dimension_layers[layer], 1))
            assert (parameters['W' + str(layer)].shape == (self.dimension_layers[layer], self.dimension_layers[layer - 1]))
            assert (parameters['b' + str(layer)].shape == (self.dimension_layers[layer], 1))

        return parameters



'''
init=initialization([5,4,3])
parameters=init.initialize_parameters()
print(parameters)
for i in range(1,len(parameters)):
    print(parameters["W" + str(i)].shape)
    print(parameters["b" + str(i)].shape)
    print("########################")
'''
