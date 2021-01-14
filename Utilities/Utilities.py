# Author: Ioannis Matzakos | Date: 20/12/2019

import csv


def writeInCSV(data):
    csvfile = open('data.csv', 'w')
    with csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
