# Author: Ioannis Matzakos | Date: 19/12/2019

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from Utilities import Log

# configure logger
log = Log.setup_logger("supervised_learning")


class SupervisedLearning:
    """
    Partition the data into a training set and a test set (60-40% or 70-30% or 80%-20%) (Try having a validation set)
    Algorithms:
        1. Naive Bayes (NB)
            i. Gaussian
            ii. Multinomial
            iii. Bernouli
        2. Support Vector Machines (SVM)
        3. Decision Trees (DT)
            i. ID3
            ii. C4.5
            iii. C5.0
            iv. CART
        4. Ensemble Methods:
            i. Random Forests (RF)
            ii. AdaBoost (AB)
        5. K-Nearest Neighbor (KNN)
        6. Neural Networks
        7. Deep Neural Networks
        8. Linear Discriminant Analysis (LDA)
        8. Quadratic Discriminant Analysis (QDA)
    """
    '''
    def __init__(self):
        """Default constructor that initializes the class variables with empty pandas dataframes, which is
        the expected type of the actual input of the class.
        """
        self.training_set = pd.DataFrame({'text': []})
        self.training_labels = pd.DataFrame({'labels': []})
        self.testing_set = pd.DataFrame({'text': []})
        self.testing_labels = pd.DataFrame({'labels': []})
    '''
    def __init__(self, train_set, train_labels, test_set, test_labels):
        """Constructor that assigns the pandas dataframes which contain the training and test data with their labels
        to the respective class variables.
        """
        self.training_set = train_set
        self.training_labels = train_labels
        self.testing_set = test_set
        self.testing_labels = test_labels

    def setTrainingSet(self, x_train):
        self.training_set = x_train

    def getTrainingSet(self):
        return self.training_set

    def setTrainingLabels(self, training_labels):
        self.training_labels = training_labels

    def getTrainingLabels(self):
        return self.training_labels

    def setTestSet(self, x_test):
        self.test_set = x_test

    def getTestSet(self):
        return self.test_set

    def setTestLabels(self, test_labels):
        self.test_labels = test_labels

    def getTestLabels(self):
        return self.test_labels

    def training(self, classifier):
        train_set = self.getTrainingSet()
        train_labels = self.getTrainingLabels()
        classifier.fit(train_set, train_labels)
        log.info("Training phase completed successfully.")
        return classifier

    def testing(self, classifier):
        log.info("\n\n\t\t\t\t---------- SUPERVISED LEARNING: TESTING PHASE ----------\n")
        test = self.getTestSet()
        predictions = classifier.predict(test)
        log.info("Testing phase completed successfully.")
        return predictions

    def bernouliNaiveBayes(self):
        log.info("Bernouli Naive Bayes: Start Training")
        bernouli_nb = BernoulliNB(alpha=.01)
        log.debug(bernouli_nb)
        self.training(bernouli_nb)
        log.info("Bernouli Naive Bayes: End Training")

    def multinomialNaiveBayes(self):
        log.info("Multinomial Naive Bayes: Start Training")
        multinomial_nb = MultinomialNB(alpha=.01)
        log.debug(multinomial_nb)
        self.training(multinomial_nb)
        log.info("Multinomial Naive Bayes: End Training")

    def randomForests(self):
        log.info("Random Forests: Start Training")
        random_forests = RandomForestClassifier()
        log.debug(random_forests)
        self.training(random_forests)
        log.info("Random Forests: End Training")

    def executeTraining(self):
        log.info("\n\n\t\t\t\t---------- SUPERVISED LEARNING: TRAINING PHASE ----------\n")
        self.bernouliNaiveBayes()
        self.multinomialNaiveBayes()
        self.randomForests()
