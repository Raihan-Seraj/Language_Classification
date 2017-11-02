import csv
import numpy

def read_file(file_name):
    maximum = 0

    reader = csv.reader(open(file_name, newline=''), delimiter=',')
    next(reader,None) #skip header
    for row in reader:
        maximum = int(row[0])

    data = [None] * maximum
    reader = csv.reader(open(file_name, newline=''), delimiter=',')
    next(reader,None) #skip header
    for row in reader:
        data[int(row[0]) - 1] = row[1]
    return data

def read_file_with_instance_no(file_name):
    data = []
    reader = csv.reader(open(file_name, newline=''), delimiter=',')
    next(reader,None) #skip header
    for row in reader:
        data.append(row)
    return data
