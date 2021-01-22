import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import pandas as pd
from forward_model import forward_model

class evaluation_metrics:
    def __init__(self, labels, input,parameters):
        self.input = input
        self.Y_true = labels
        self.parameters=parameters
#Accuracy = (TP+TN)/(TP+FP+FN+TN)
    def confusionMatrix(self):
        y_hat=[]
        predictions,packet_of_packets=forward_model().forward_model(self.input,self.parameters)
        predictions=predictions.T
       # print(predictions.shape)
       # print(predictions)
       # print(predictions.shape)
        for i in range(predictions.shape[0]):
            max=np.argmax(predictions[i])
            y_hat.append(max)
       # print(len(y_hat))
       # print(y_hat)
       # print(self.Y_true.shape)
       # print(self.Y_true)
        #print(y_hat)
        classes = set(self.Y_true[0])
       # print("sklearn accraucy : ")
        #print(accuracy_score(self.Y_true[0],y_hat))
        number_of_classes = len(classes)
        conf_matrix = pd.DataFrame(
            np.zeros((number_of_classes, number_of_classes), dtype=int),
            index=classes,
            columns=classes)
        for i, j in zip(self.Y_true[0], y_hat):
            conf_matrix.loc[i, j] += 1
        return conf_matrix.values, conf_matrix

    def TP(self):
        values, cm = self.confusionMatrix()
        return np.diag(cm)

    def FP(self):
        values, cm = self.confusionMatrix()
        return np.sum(cm, axis=0) - self.TP()

    def FN(self):
        values, cm = self.confusionMatrix()
        return np.sum(cm, axis=1) - self.TP()

    def Accuracy(self, data_size):
        return np.sum(self.TP()/data_size)

    def Precision(self):
        return np.mean(self.TP() / (self.TP() + self.FP()))

    def Recall(self):
        return np.mean(self.TP() / (self.TP() + self.FN()))

    def F1_score(self):
        if self.TP() > 0:
            return 2 * ((self.Precision() * self.Recall()) / (self.Precision() + self.Recall()))
        else:
            return 0

'''
Y_hat = np.array([0, 1, 2, 3, 4, 2, 3, 3, 2, 1, 2, 3, 1, 5, 1, 1, 9, 8, 7, 6])
Y_true = np.array([0, 1, 3, 3, 2, 5, 5, 3, 2, 1, 4, 3, 2, 1, 1, 2, 9, 8, 7, 6])
ev = evaluation_metrics(Y_true, Y_hat)
values, cm = ev.confusionMatrix()
print (values)
print (confusion_matrix(Y_true, Y_hat))
print (ev.Precision())
print (precision_score(Y_true, Y_hat, average='macro'))
print (ev.Recall())
print (recall_score(Y_true, Y_hat, average='macro'))
print (ev.Accuracy(20.))
print (accuracy_score(Y_true, Y_hat))
print (ev.F1_score())
print (f1_score(Y_true, Y_hat, average='macro'))
'''

