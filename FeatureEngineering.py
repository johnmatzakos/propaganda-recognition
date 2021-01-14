# Author: Ioannis Matzakos | Date: 24/02/2020

from Utilities import Log
import pandas as pd
import numpy as np
import re

# configure logger
log = Log.setup_logger("feature_engineering")


class FeatureEngineering:
    def getDocumentTermMatrix(self, column_names, data):
        """Creates a document term matrix using as column names the dataframe
        derived from the tokenization stage of the data preprocessing phase.
        :param column_names: list of columns
        :param data: the dataset of articles in a pandas dataframe
        :return: a pandas dataframe as a document term matrix
        """
        # create an empty pandas dataframe, set the column names
        # and initialize the dtm with zero frequencies for every word.
        zero_data = np.zeros(shape=(len(data), len(column_names)))
        dtm = pd.DataFrame(zero_data, columns=column_names)
        log.debug(f"column names of document term matrix: {dtm.columns}")
        log.debug(f"Initialized DTM: {dtm}")

        log.debug(f"column_names type: {type(column_names)}")
        log.debug(f"data type: {type(data)}")
        log.debug(f"dtm type: {type(dtm)}")

        # for each word in each article match with a word, a column, of the dtm
        # and for each occurrence increase the frequency of this word
        have_equal_len = len(data) == len(dtm)
        log.debug(f"have_equal_size = {have_equal_len}")

        have_equal_size = data.size == dtm.size
        log.debug(f"have_equal_size = {have_equal_size}")

        if have_equal_len:
            i=0
            j=0
            k=0
            for article in data.iterrows():
                words = self.extractWordListOfString(article)
                # log.debug(f"article: {i}")
                i = i + 1
                for column in dtm.columns:
                    for word in words:
                        if word == column:
                            # dtm[column][article] = dtm[column][article] + 1
                            dtm[word][column] = dtm[word][column] + 1

                            # log.debug(f"word no: {k} | word: {word}")
                            k = k + 1
                    # log.debug(f"column: {j} | column name: {column}")
                    j + j + 1
        return dtm

    def getWordFrequency(self, data, dtm):
        """Calculates the frequency of each word in every document.
        :param data: news articles
        :type: pandas dataframe
        :return: word frequency for each document
        :rtype: pandas dataframe
        """

    def extractWordListOfString(self, string):
        """Extracts words from a string, hence splits a string into its words.
        :param string:
        :return: list of strings
        """
        # log.debug(f"Original article: {string}")
        # extract words from string
        # result = string.split(" ")
        string = str(string)
        result = re.findall(r'\s?(\s*\S+)', string.rstrip())
        log.debug(f"The list of words is : {str(result)}")
        # log.debug(f"result var type: {type(result)}")
        '''
        result = tuple(result)
        log.debug(f"The tuple of words is : {str(result)}")
        log.debug(f"result var type: {type(result)}")
        '''
        return list(result)
