import util
import numpy as np
import sys
import random

PRINT = True

random.seed(42)
np.random.seed(42)

class BaggingClassifier:
    """
    Bagging classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, ratio, num_classifiers):

        self.ratio = ratio
        self.num_classifiers = num_classifiers
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.num_classifiers)]

    def train( self, trainingData, trainingLabels):
        """
        The training loop samples from the data "num_classifiers" time. Size of each sample is
        specified by "ratio". So len(sample)/len(trainingData) should equal ratio. 
        """

        self.features = trainingData[0].keys()
        
        sample_size = int(self.ratio * len(trainingData))
        for i in range(self.num_classifiers):
            random_integers = np.random.randint(len(trainingData), size = sample_size)
            sampled_data = []
            sampled_labels = []
            for choosen_element in random_integers:
                sampled_data.append(trainingData[choosen_element])
                sampled_labels.append(trainingLabels[choosen_element])

            self.classifiers[i].train(sampled_data, sampled_labels, None)






    def classify( self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. This is done by taking a polling over the weak classifiers already trained.
        See the assignment description for details.

        Recall that a datum is a util.counter.

        The function should return a list of labels where each label should be one of legaLabels.
        """

       
        prediction = []
        for i in range(len(data)):
            guesses = []
            for j in range(self.num_classifiers):
                guesses.append(self.classifiers[j].classify([data[i]])[0])

            prediction.append(int(np.sign(sum(guesses))))

        return prediction
