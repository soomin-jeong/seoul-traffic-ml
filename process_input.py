# Importing Required modules

import csv
from datetime import datetime


def get_day_and_hour(str_date):
    # "2018-08-26 15:38:56.36" -> datetime
    datetime_obj = datetime.strptime(str_date[:19], "%Y-%m-%d %H:%M:%S")
    coded_day = datetime_obj.weekday()  # 0 Mon, 1 Tue, 2 Wed, ...
    coded_hour = datetime_obj.hour
    return coded_day, coded_hour

def make_training_set(file_name):
    training_set = []
    training_set_y = []

    with open(file_name, "rb") as file:
        reader = csv.reader(file)
        next(reader, None)  # skipping the header
        for row in reader:
            coded_day, coded_hour = get_day_and_hour(str_date=row[0])
            training_set.append([coded_day, coded_hour])
            training_set_y.append(row[3])

    return training_set, training_set_y

def make_test_set(file_name):
    testing_set = []
    testing_set_y = []
    with open(file_name, "rb") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            coded_day, coded_hour = get_day_and_hour(str_date=row[0])
            testing_set.append([coded_day, coded_hour])
            testing_set_y.append(row[3])

    return testing_set, testing_set_y