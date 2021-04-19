# Author: Ioannis Matzakos | Date: 08/07/2019

from Utilities import Log
import pandas as pd
import nltk                                                     # Natural Language Tool Kit
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import PorterStemmer                             # Stemmer
from nltk.corpus import stopwords                               # Stop word corpus
from textblob import Word, TextBlob                             # For lemmatizer and n-grams
import numpy as np                                              # For creating a matrix
from scipy import sparse                                        # For creating a sparse matrix
from sklearn.model_selection import train_test_split            # For splitting the dataset into training and testing
from sklearn.feature_extraction.text import CountVectorizer     # For Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer     # For TF-IDF
from FeatureEngineering import FeatureEngineering as FeatEng    # used for creating a dtm


# configure logger
log = Log.setup_logger("data_preprocessing")


class DataPreprocessing:
    """
    Data Preprocessing: A necessary phase before training a machine learning model. Includes a preliminary analysis of
    the data, data cleansing from redundant characters, tokenization, stemming, stopword removal and the creation of a
    corpus in the form of a term document matrix (tdm) or a document term matrix (dtm). Additionally, in creating a tdm
    or a dtm the text data will be transformed using the text representation methods term frequency (tf) and term
    frequency-invert document frequency (tf-idf) before proceeding with feature selection.

    :param file: The path where the file (dataset) is located.
    :type file: string
    :returns: the dataset in the form of a term document matrix (tdm) or a document term matrix (dtm)
    :rtype: pandas dataframe
    """

    def __init__(self):
        self.file = ""

    def setFile(self, file):
        self.file = file

    def getFile(self):
        return self.file

    def readFile(self):
        """Reads data from file 'filename.csv' in a folder named "data".
        The filepath is passed through an object and called by the variable name.
        :return: data
        :rtype: pandas dataframe
        """
        log.info(f"Reading the data from: {self.file}")
        data = pd.read_csv(self.file)
        return data

    def preliminaryAnalysis(self, data):
        """Performs a preliminary analysis on imported data.
        :param data: a dataframe created in readFile() function
        :type: pandas dataframe
        :return: void
        """
        log.info("\n\n\t\t\t\t------------ PRELIMINARY ANALYSIS ------------\n")
        # Preview the first 5 lines of the loaded data
        log.info("Preview the first 5 lines of the loaded data: ")
        print(data.head())
        # label meanings
        log.info("Labels:\t0: reliable\t1: unreliable")
        # Display the names of the columns of the data
        log.info(f"Data Columns: {data.columns.tolist()}\n")
        log.info(f"\nData Types: \n{data.dtypes} of {type(data)}\n")
        # Display some basic statistics about the dataset
        log.info(f"\nData Description: \n{data.describe()}\n")

    def getColumn(self, data, column):
        """Separates the given column from the dataset in order to be processed separately and save it into a new file.
        :param data: a dataframe created in readFile() function
        :type: pandas dataframe
        :param column: represents the name of the column you want to separate from the dataset for processing purposes
        :return text: the text column of the dataset
        :rtype: pandas dataframe
        """
        text = data[[column]]
        log.debug(f"text variable type: {type(text)}")
        log.debug(f"text data type: {text.dtypes}")
        text.to_csv(r'data/' + column + '.csv')
        log.info(f"The {column} of the articles saved in csv file successfully.")
        return text

    def removeURL(self, text):
        """Removes URLs from the data.
        :param text: news articles
        :type: pandas dataframe
        :return text: text without any URLs
        :rtype: pandas dataframe
        """
        text = text.replace(to_replace=r'http\S+', value='', regex=True)
        # text = text.replace(to_replace=r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\\=]*)', value='', regex=True)
        text = text.replace(to_replace=r'[-a - zA - Z0 - 9 @: %._\+~  # =]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', value='', regex=True)
        log.info(f"\nURLs Removed: \n{text}\n")
        return text

    def removePunctuation(self, text):
        """Removes punctuation from the data.
        :param text: news articles
        :type: list
        :return text: text without punctuation
        :rtype: pandas dataframe
        """
        text = text.replace(to_replace=r"[^\w\s]", value='', regex=True)
        log.info(f"\nPunctuation Removed: \n{text}\n")
        return text

    def deCapitalize(self, text):
        """Transforms all text to lower case
        :param text: news articles
        :type: pandas dataframe
        :return text: text in lower case
        :rtype: pandas dataframe
        """
        text = text['text'].str.lower()
        log.info(f"\nDe-Capitalized: \n{text}\n")
        return text

    def cleansing(self, text):
        """Performs data cleansing on the data using the functions: removeURL, removePunctuation, deCapitalize.
        Essentially removes unwanted or redundant elements from text.
        :param text: news articles
        :type: pandas dataframe
        :return text: cleaned text
        :rtype: pandas dataframe
        """
        log.info("\n\n\t\t\t\t------------ DATA CLEANSING ------------\n")
        # remove URLs
        text = self.removeURL(text)
        # remove punctuation
        text = self.removePunctuation(text)
        # de-capitalization
        text = self.deCapitalize(text)
        return text

    def wordTokenizer(self, text):
        """Tokenizes text word for word.
        :param text: news articles
        :type: pandas dataframe
        :return text: tokenized text
        :rtype: pandas dataframe
        """
        for i in range(len(text)):
            tokenized_text = nltk.word_tokenize(str(text[i]), 'english')
        log.info(f"\nTokenized Text Word For Word: \n{tokenized_text}\n")
        return tokenized_text

    def sentenceTokenizer(self, text):
        """Tokenizes text sentence by sentence.
        :param text: news articles
        :type: pandas dataframe
        :return text: tokenized text
        :rtype: pandas dataframe
        """
        for i in range(len(text)):
            tokenized_text = nltk.sent_tokenize(str(text[i]), 'english')
        log.info(f"\nTokenized Text Sentence By Sentence: \n{tokenized_text}\n")
        return tokenized_text

    def removeStopWords(self, text):
        """Removes stop words on the tokenized text data.
        Stop words are the commonly used words where their use in a text has trivial contribution to its meaning.
        :param text: news articles
        :type: pandas dataframe
        :return text: stemmed and tokenized text free of stop words
        :rtype: pandas dataframe
        """
        stop_words = set(stopwords.words('english'))
        meaningful_words = [w for w in text if not w in stop_words]
        log.info(f"\nStop words removed: \n{stop_words}\n")
        log.info(f"\nStop words count: {len(stop_words)}\n")
        log.info(f"\nMeaningful words: \n{meaningful_words}\n")
        log.info(f"\nMeaningful words count: {len(meaningful_words)}\n")
        return text

    def stemmer(self, tokenized_text):
        """Performs stemming on the tokenized text data.
        Stemming is the process where only the radical is kept from each word in order to
        minimize noise even more and ease the way ti calculate the overall importance of a word.
        :param text: news articles
        :type: pandas dataframe
        :return text: stemmed text
        :rtype: pandas dataframe
        """
        # stemmed_text = nltk.PorterStemmer()
        # return "".join([stemmed_text.stem(i) for i in tokenized_text.split()])
        stemming = PorterStemmer()
        stemmed_list = [stemming.stem(word) for word in tokenized_text]
        return stemmed_list

    def lemmatizer(self, tokenized_text):
        """Performs lemmatization on the tokenized text data.
        lemmatizing is the process of keeping only the radical of each word in the data based on
        grammar rules rather than regular expressions as the stemming process.
        :param text: news articles
        :type: pandas dataframe
        :return text: lemmatized text
        :rtype: pandas dataframe
        """
        tokenized_text = tokenized_text.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        log.info(f"Lemmatized text: {tokenized_text}")
        return tokenized_text

    def bagOfWords(self, text, data):
        """Creates a Bag of words vector. Each word from the dataset in unique (only 1 entry) in the bag, in order to
        calculate the frequency of each word. For every occurrence of a word in the dataset a count for that word
        is augmented. This count is also called term frequency.
        :param text: news articles
        :type: pandas dataframe
        :return: a bag of word corpus using term frequency
        :rtype: scipy learn sparce matrix/vector
        """
        log.debug(f"input bow parameter: {text}")
        log.debug(f"input bow size: {len(text)}")
        # bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1, 1), analyzer="word")

        bow = CountVectorizer(analyzer="word",
                              ngram_range=(1, 1),
                              stop_words='english',
                              tokenizer=callable)

        # token_pattern="[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+"

        log.debug(f"bow vectorizer: {bow}")
        text_bow = bow.fit_transform(text)
        
        log.info(f"document term matrix column names: {text}")
        # fe = FeatEng()
        # bow = fe.getDocumentTermMatrix(text, data)
        log.info(f"Bag of words is created: {bow}")
        log.debug(f"BOW Type: {type(bow)}")
        return bow

    def term_frequency(self, text):
        """Creates a vector based on the text representation method Term Frequency (tf).
        :param text: news articles
        :type: pandas dataframe
        :return: a bag of words corpus based on tf
        :rtype: scipy learn sparce matrix/vector
        """
        tf1 = (text[1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        tf1.columns = ['words', 'tf']
        return tf1

    def inverse_document_frequency(self, text):
        """Creates a vector based on the text representation method Inverse Document Frequency (idf).
        :param text: news articles
        :type: pandas dataframe
        :return: a bag of words corpus based on idf
        :rtype: scipy learn sparce matrix/vector
        """
        tf1 = self.term_frequency(text)
        for i, word in enumerate(tf1['words']):
            tf1.loc[i, 'idf'] = np.log(text.shape[0] / (len(text[text['tweet'].str.contains(word)])))
        return tf1

    def tf_idf(self, text):
        """Creates a vector based on the text representation method Term Frequency-Inverse Document Frequency (tf-idf).
        Each word is represented by the number of times in each document multiplied by the frequency each word appears
        in all the documents of the dataset.
        purpose: to show how important a word is to a document in a collection or corpus
        logic: The tf–idf value increases proportionally to the number of times a word appears
        in the document and is offset by the number of documents in the corpus that contain the word.
        :param text: news articles
        :type: pandas dataframe
        :return: a bag of words corpus based on tf-idf
        :rtype: scikit learn sparce matrix/vector
        """
        # tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word', stop_words='english', ngram_range=(1, 1))
        # tfidf_data = tfidf.fit_transform(text)
        # Initialize
        vectorizer = TfidfVectorizer()
        doc_vec = vectorizer.fit_transform(text)
        # Create dataFrame
        tfidf_data = pd.DataFrame(doc_vec.toarray().transpose(), index=vectorizer.get_feature_names())
        log.info(f"Tf-idf corpus is created: {tfidf_data}")
        log.debug(f"Tf-idf Type: {type(tfidf_data)}")
        return tfidf_data

    # temporary, not finished method
    def ngrams(self, text, n):
        """Creates a vector of n word combinations, the n-grams text representation method
        (e.g. bi-grams(2-grams), tri-grams(3-grams)).
        :param text: news articles
        :type: pandas dataframe
        :param n: number of tokens needed in each n-gram
        :type: integer
        :return: a corpus based on n-grams
        :rtype: list (???)
        """
        # Create a matrix
        for i in range(len(text)):
            ngram_data = TextBlob(text[i]).ngrams(n)
            matrix = np.array([ngram_data])
        log.debug(f"n-gram matrix: {matrix}")
        # Create compressed sparse row (CSR) matrix
        matrix_sparse = sparse.csr_matrix(matrix)
        log.debug(f"n-gram sparse matrix: {matrix_sparse}")
        log.info(f"{n}-grams corpus is created: {ngram_data}")
        log.debug(f"{n}-grams Type: {type(ngram_data)}")
        log.debug(f"Size: {len(ngram_data)}")
        return ngram_data

    # temporary, not finished method
    def documentTermMatrix(self, text, tokenized_text):
        """Creates a document term matrix (dtm), where each row is an article/document
        and each column is a word after performing data cleansing.
        :param text: news articles
        :type: pandas dataframe
        :return text: stemmed and tokenized text free of stop words
        :rtype: pandas dataframe (???)
        """
        dtm = pd.DataFrame()
        log.info(dtm)
        tokenized_text_list = tokenized_text.values.tolist()
        feng = FeatEng()
        feng.getWordFrequency(text, tokenized_text)
        # dtm = pd.DataFrame(columns=tokenized_text_list)
        # log.info(dtm)

    def createCorpus(self, text, data):
        """Creates a corpus using a text representation method.
        :param text: news articles
        :type: pandas dataframe
        :return text: stemmed and tokenized text free of stop words
        :rtype: pandas dataframe (???)
        """
        log.info("\n\n\t\t\t\t------------ CREATE A CORPUS ------------\n")
        # word tokenization
        tokenized_text = self.wordTokenizer(text)
        # stopword removal
        text['stem_meaningful'] = self.removeStopWords(tokenized_text)
        log.info(f"\nTokenized Text With Removed Stopwords: \n{text['stem_meaningful']}\n")
        # stemming
        text['stemmed_words'] = self.stemmer(text['stem_meaningful'])
        log.info(f"Tokenized Text After Stemming With Removed Stopwords : \n{text['stemmed_words']}")
        # bag of words
        log.info(f"type text stemmed words: {type(text['stemmed_words'])}")
        bag_of_words = self.bagOfWords(text['stemmed_words'], data)
        # tf-idf
        # tfidf = self.tf_idf(text['stemmed_words'])
        # n-grams
        # ngrams = self.ngrams(text['stemmed_words'], 2)
        return bag_of_words

    def createCorpusTEMP(self, text):
        """Creates a corpus using a text representation method.
        :param text: news articles
        :type: pandas dataframe
        :return text: stemmed and tokenized text free of stop words
        :rtype: pandas dataframe (???)
        """
        log.info("\n\n\t\t\t\t------------ CREATE A CORPUS ------------\n")
        # bag of words
        bag_of_words = self.bagOfWords(text)
        # tf-idf
        # tfidf = self.tf_idf(text)
        # n-grams
        # ngrams = self.ngrams(text['stemmed_words'], 2)
        return bag_of_words

    def splitDataset(self, corpus, labels, training_ratio, testing_ratio):
        """Splits the given dataset into a training set and a test set based on the given equivalent ratios.
        :param corpus: matrix of documents and their words
        :type: scipy sparse matrix of word frequencies in documents
        :param labels: reliable or not reliable
        :type: pandas dataframe of strings
        :param training_ratio: the number of articles that will be used to train the machine learning models
        :type: float
        :param testing_ratio: the number of articles that will be used to test the machine learning models
        :type: float
        :return text: news articles derived from the original dataset
        :rtype: pandas dataframe
        """
        log.info("\n\n\t\t\t\t---------- SPLITTING THE DATA ----------\n")
        dataset_size = corpus.size
        label_size = labels.size
        log.debug(f"Dataset Size: {dataset_size}")
        log.debug(f"Labels Size: {label_size}")
        log.info(f"Training Set Ratio: {training_ratio} -> {training_ratio*100}%")
        training_set_size = dataset_size * training_ratio
        log.info(f"The size of the training set with ratio {training_ratio*100}% will be {training_set_size}")
        log.info(f"Test Set Ratio: {testing_ratio} -> {testing_ratio*100}%")
        test_set_size = dataset_size * testing_ratio
        log.info(f"The size of the test set with ratio {testing_ratio*100}% will be {test_set_size}")
        # x = the data and y = the labels
        # corpus = pd.DataFrame(corpus.toarray())
        log.debug(f"corpus variable type: {type(corpus)}")
        log.debug(f"labels variable type: {type(labels)}")
        x_train, x_test, y_train, y_test = train_test_split(corpus, labels, test_size=testing_ratio, train_size=training_ratio)
        return x_train, x_test, y_train, y_test

        # x_train, x_test, y_train, y_test = dp.splitDataset(bag_of_words, labels, 0.7, 0.3)
