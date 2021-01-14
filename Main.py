# Author: Ioannis Matzakos | Date: 22/07/2019

from Utilities import Log
from DataPreprocessing import DataPreprocessing
from SupervisedLearning import SupervisedLearning

import time

# configure logger
log = Log.setup_logger("main")

log.info("Propaganda Recognition\n\n\t\t\t\t---------- PROPJECT: PROPAGANDA RECOGNITION ----------"
         "\n\t\t\t\t------------------------------------------------------\n")

log.info("Start of Execution\n\n\t\t\t\t------------ DATA MINING PIPELINE: START ------------\n")

# start calculating execution time
start_time = time.time()

# load training data
file = "data/train.csv"

# Data Preprocessing
dp = DataPreprocessing()
dp.setFile(file)
data = dp.readFile()
dp.preliminaryAnalysis(data)
text = dp.getColumn(data, "text")
labels = dp.getColumn(data, 'label')
text = dp.cleansing(text)
bag_of_words = dp.createCorpus(text, data)
log.info(f"Dataset after preprocessing: \n{text}")
# log.debug(f"document term matrix: {dp.documentTermMatrix(data, text)}")
# Feature Selection & Feature Engineering


# Splitting the data
training_set, test_set, training_labels, test_labels = dp.splitDataset(bag_of_words, labels, 0.7, 0.3)

# Supervised Learning
sl = SupervisedLearning(training_set, training_labels, test_set, test_labels)
sl.executeTraining()
sl.testing()

# Semi-Supervised Learning


# Unsupervised Learning


# Performance Evaluation


# Information Visualization


# stop calculating execution time
end_time = time.time()
total_time = end_time - start_time
log.info(f"Execution Time: {total_time} seconds")
'''
milliseconds = total_time * 1000
seconds = (milliseconds / 1000) % 60
minutes = ((milliseconds / (1000*60)) % 60)
hours = ((milliseconds / (1000*60*60)) % 24)
log.info(f"Execution Time: {hours}:{minutes}:{seconds}:{milliseconds}")
'''
log.info("End of Execution\n\n\t\t\t\t------------ DATA MINING PIPELINE: END ------------\n")


