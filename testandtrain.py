import getopt
import json
import math
import numpy as np
import pickle
import random
import requests
import sys
import time
from sklearn import mixture
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

alpha = 2.5
thresh_hold_row_count = 0

prev_row_count = 0
current_row_count = 0
aws_base_url = 'http://wireless.uufjdwcjme.us-west-2.elasticbeanstalk.com/rest/test/'


class row:
    def __init__(self, lat, lon, rssi):
        self.lat = lat
        self.lon = lon
        self.rssi = rssi

    def processRowInstance(self):
        row_text = ""
        row_text += str(self.lat) + '`' + str(self.lon) + '`'
        i = 0
        temp = ""
        l_rssi = len(self.rssi)
        while i < l_rssi:
            if (i == l_rssi - 1):
                temp += str(self.rssi[i])
            else:
                temp += str(self.rssi[i]) + ','
            i += 1
        row_text += temp
        return row_text


def generateMeanForTraining(tower_cords, grids_res, min_lat, max_lat, min_long, max_long):
    i = 0
    mean = []
    grids_size = grids_res ** 2
    lat_diff = float(math.fabs(max_lat - min_lat)) / grids_res
    lon_diff = float(math.fabs(max_long - min_long)) / grids_res
    while i < grids_size:
        lat_cord = min_lat + (i / grids_res) * lat_diff
        lon_cord = min_long + (i % grids_res) * lon_diff
        j = 0
        cell_mean = []
        while j < len(tower_cords):
            dist = haversine(lat_cord, lon_cord, tower_cords[j][0], tower_cords[j][1]) * 1000
            if dist > 1:
                loss = 10 * 2.0 * math.log10(dist)
            else:
                loss = 0
            curr_power = 10 * math.log10(0.016) - loss
            cell_mean.append(curr_power)
            j += 1
        mean.append(cell_mean)
        i += 1
    return mean


def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def generateData(data_points_count, trans_count, distance_in_km=1.5):
    if trans_count >= int(math.sqrt(data_points_count)):
        print 'Transmitters occupying more area'

    return generateDataForOneIteration(data_points_count, trans_count, distance_in_km)
    # return powers


def generateRandomTowers(all_locations, trans_count):
    loc_count = len(all_locations)
    if trans_count > loc_count:
        print "Number of transmitters cannot be more than the total number of data points. Exiting the application."
        exit(0)

    tower_locations = random.sample(xrange(loc_count), trans_count)
    return tower_locations


def calculateDistance(i, j, k, l):
    return math.sqrt((math.pow((k - i), 2)) + math.pow((l - j), 2))


def generateDataForOneIteration(dim, trans_count, distance_in_km):
    all_locations = generateLocationVector(dim, distance_in_km)
    tower_indices = generateRandomTowers(all_locations, trans_count)

    # Populate the co-ordinates where towers are present so that u dont calculate path loss for those cells
    tower_cords = []
    i = 0
    while i < trans_count:
        tower_cords.append(all_locations[tower_indices[i]])
        i += 1

    powers = []
    i = 0
    min_lat = 0
    max_lat = 0
    min_long = 0
    max_long = 0
    initialized = False
    while i < dim:
        if initialized == False:
            min_lat = all_locations[i][0]
            max_lat = all_locations[i][0]
            min_long = all_locations[i][1]
            max_long = all_locations[i][1]
            initialized = True
        else:
            if all_locations[i][0] > max_lat and all_locations[i][1] > max_long:
                max_lat = all_locations[i][0]
                max_long = all_locations[i][1]
            elif all_locations[i][0] < min_lat and all_locations[i][1] < min_long:
                min_lat = all_locations[i][0]
                min_long = all_locations[i][1]
        j = 0
        temp = []
        while j < trans_count:
            dist = haversine(all_locations[i][0], all_locations[i][1], tower_cords[j][0], tower_cords[j][1]) * 1000
            if dist > 1:
                loss = 10 * alpha * math.log10(dist)
            else:
                loss = 0
            curr_power = 10 * math.log10(0.016) - loss
            if curr_power < -113:
                curr_power = -113
            temp.append(curr_power)
            j += 1
        powers.append(row(all_locations[i][0], all_locations[i][1], temp))
        i += 1
    return powers, min_lat, min_long, max_lat, max_long, tower_cords


def deleteAllRowsFromTable():
    global aws_base_url
    current_row_count = long(getRowCount())
    if current_row_count > 0:
        url = aws_base_url + 'processRequest?request=1'
        myResponse = requests.get(url)
        if (myResponse.ok):
            print "All rows from table deleted\n"


def populateDataBase(data_points_count, trans_count):
    global aws_base_url
    deleteAllRowsFromTable()
    url = aws_base_url + 'processRequest?request=2|'
    powers, min_lat, min_long, max_lat, max_long = generateData(data_points_count, trans_count)
    total_len = len(powers)
    my_randoms = random.sample(xrange(total_len), total_len)
    i = 0
    j = 0
    while i < total_len:
        wait_time = random.randint(0, 2) / 10
        time.sleep(wait_time)
        row_text = powers[my_randoms[j]].processRowInstance()
        temp = url + row_text
        myResponse = requests.get(temp)
        # print temp
        if not myResponse.ok:
            print "Insert failed for the row values " + row_text + "\nExiting the program\n"
            exit(0)
        i += 1
        j += 1


def readDataFromTables():
    global aws_base_url
    url = aws_base_url + 'getDataFromTables'
    myResponse = requests.get(url)
    loc = []
    rssi_data = []
    if myResponse.ok:
        json_obj_list = json.loads(myResponse.content)
        for json_obj in json_obj_list:
            rssi_list = []
            temp_loc = str(json_obj['lat']) + ',' + str(json_obj['lon'])
            loc.append(temp_loc)
            temp_rssi = json_obj['rssi_vector'].split(',')
            for rssi in temp_rssi:
                rssi_list.append(float(rssi))
            rssi_data.append(rssi_list)

    return loc, rssi_data


def trainModelSuperVisedLearning(loc_training_data, rssi_training_data):
    model = GaussianNB()
    model.fit(rssi_training_data, loc_training_data)
    print "Supervised learning Model trained"
    return model


def trainModelUnsupervisedLearning(rssi_training_data, resolution, tower_cords, min_lat, min_lon, max_lat, max_lon):
    number_of_clusters = resolution ** 2
    model = mixture.GMM(n_components=number_of_clusters, covariance_type='full')
    # model.means_ = generateMeanForTraining(tower_cords, resolution, min_lat, max_lat, min_lon, max_lon)
    model.fit(rssi_training_data)
    output = open('data.pkl', 'w')
    pickle.dump(model, output)
    output.close()
    return model


def testModelSuperVisedLearning(model, test_data, actual_loc_data):
    test_loc_data = model.predict(test_data)
    euclid_dist = []
    i = 0
    while i < len(test_loc_data):
        dist = calculateDistBetweenLocations(actual_loc_data[i], test_loc_data[i]) * 1000
        euclid_dist.append(dist)
        i += 1
    return euclid_dist


def testModelUnsupervisedLearning(model, test_data, actual_loc_data, lat_diff, lon_diff, min_lat, min_lon, res):
    predicted_loc_cluster = model.predict(test_data)
    euclid_dist = []
    j = 0
    while j < len(predicted_loc_cluster):
        predicted_loc = ''
        predicted_loc += str(min_lat + predicted_loc_cluster[j] / res * lat_diff) + ','
        predicted_loc += str(min_lon + predicted_loc_cluster[j] % res * lon_diff)
        dist = calculateDistBetweenLocations(actual_loc_data[j], predicted_loc) * 1000
        euclid_dist.append(dist)
        j += 1
    return euclid_dist


def getRowCount():
    global aws_base_url
    url = aws_base_url + 'getTableRowCount'
    myResponse = requests.get(url)
    return myResponse.content


def startAppInLearningmode():
    global model, prev_row_count, current_row_count, thresh_hold_row_count, resolution
    if thresh_hold_row_count == 0:
        print 'Refer to the program usage.'
        help_message()
        exit(0)
    # while True:
    current_row_count = long(getRowCount())
    if current_row_count == 0:
        print "there is no data in the table to learn. Exiting..."
        exit(0)
    if current_row_count - long(prev_row_count) > thresh_hold_row_count:
        prev_row_count = current_row_count
        trained_model = trainModelSuperVisedLearning()
        output = open('data.pkl', 'w')
        pickle.dump(trained_model, output)
        output.close()
        # time.sleep(3)


def startAppInClientMode():
    pkl_file = open('data.pkl', 'r')
    model = pickle.load(pkl_file)
    result = testModelForAllDataPoints(model)
    return result


def help_message():
    print "Command line for the application must follow the below format :"
    print "In Learning mode:"
    print "\ttestandtrain.py -l 1 -x <thresh_hold_row_count>"
    print "In client mode:"
    print "\ttestandtrain.py -c 1 -d <testing_data>"
    print "Populate data base:"
    print "\ttestandtrain.py -p 1 -t <trans_count> -r <grid_resolution>"
    print "Test all data points:"
    print "\ttestandtrain.py -e 1"


def main(argv):
    data_points_count = 0
    trans_count = 0
    test_data = ''
    res = 0
    in_learning_mode = False
    is_client_mode = False
    is_explore_mode = False
    populate_db = False
    try:
        opts, args = getopt.getopt(argv, 'c:t:d:r:l:x:e:p:k:')
    except getopt.GetoptError:
        help_message()

    for opt, arg in opts:
        if opt == '-l':
            in_learning_mode = True
        elif opt == '-c':
            is_client_mode = True
        elif opt == '-t':
            trans_count = int(arg)
        elif opt == '-d':
            test_data = arg
        elif opt == '-r':
            data_points_count = int(arg)
        elif opt == '-x':
            thresh_hold_row_count = int(arg)
        elif opt == '-e':
            is_explore_mode = True
        elif opt == '-p':
            populate_db = True
        elif opt == '-k':
            res = int(arg)

    if in_learning_mode == True:
        startAppInLearningmode()
    elif is_client_mode == True:
        if test_data == '':
            print 'Test data is empty. Exiting the program.'
            help_message()
            exit(0)
        result = startAppInClientMode(test_data)
        print result
    elif is_explore_mode == True:
        # generateAnalysisReports(data_points_count, trans_count, 6, res)
        # varyTransmitters(data_points_count)
        varyGridResolutions(data_points_count, trans_count)
        # varyTestingFraction(data_points_count, trans_count)
    elif populate_db == True:
        populateDataBase(data_points_count, trans_count)


def calculateDistBetweenLocations(loc1, loc2):
    x = float(loc1.split(',')[0])
    y = float(loc1.split(',')[1])
    t_x = float(loc2.split(',')[0])
    t_y = float(loc2.split(',')[1])
    dist = haversine(t_x, t_y, x, y)
    return dist


def testModelForAllDataPoints():
    pkl_file = open('data.pkl', 'r')
    model = pickle.load(pkl_file)
    actual_loc_data, rssi_data = readDataFromTables()
    row_count = len(actual_loc_data)
    i = 0
    loc_diff_dist = []
    while i < row_count:
        test_data = rssi_data[i]
        # test_data = Utils.remove_unwanted_white(test_data)
        if len(test_data) > 0:
            predicted_loc_data = testModelSuperVisedLearning(model, test_data, actual_loc_data)
            dist = calculateDistBetweenLocations(actual_loc_data[i], predicted_loc_data)
            loc_diff_dist.append(dist)
        i += 1
    print loc_diff_dist
    plt.plot(np.sort(loc_diff_dist), np.linspace(0, 1, len(loc_diff_dist), endpoint=False))


def generateLocationVector(number_of_data_points, distance_in_km):
    # convert radius to miles
    radius = distance_in_km * 0.621371 * 1000
    # convert radius to degree
    r = float(radius) / 111300
    # Center coordinates of StonyBrook as per Google maps
    x0 = 40.90
    y0 = -73.125
    # Choose number of Lat Long to be generated
    all_locations = []
    for i in range(1, number_of_data_points + 1):
        temp_loc = []
        u = float(random.uniform(0.0, 1.0))
        v = float(random.uniform(0.0, 1.0))

        w = r * math.sqrt(u)
        t = 2 * math.pi * v
        x = w * math.cos(t)
        y = w * math.sin(t)

        xLat = x + x0
        yLong = y + y0
        temp_loc.append(xLat)
        temp_loc.append(yLong)
        all_locations.append(temp_loc)
    return all_locations


def splitData(consolidate_powers, percent):
    total = len(consolidate_powers)
    test_len = int(percent * total)
    test_data = []
    train_data = []
    test_loc = []
    train_loc = []

    test_ind = random.sample(xrange(total), test_len)
    i = 0
    while i < total:
        if test_ind.__contains__(i):
            test_data.append(consolidate_powers[i].rssi)
            loc = str(consolidate_powers[i].lat) + "," + str(consolidate_powers[i].lon)
            test_loc.append(loc)
        else:
            train_data.append(consolidate_powers[i].rssi)
            loc = str(consolidate_powers[i].lat) + "," + str(consolidate_powers[i].lon)
            train_loc.append(loc)
        i += 1

    return test_data, train_data, test_loc, train_loc


# Vary Grid resolutions resulting in varying clusters count for unsupervised learning
def varyGridResolutions(data_points_count, trans_count):
    print 'Varying Grid resolution'
    powers, min_lat, min_lon, max_lat, max_lon, tower_cords = generateData(data_points_count, trans_count)
    testing_data, training_data, testing_data_loc, training_data_loc = splitData(powers, 0.1)
    grid_res = [10, 20, 30, 40, 50]
    # grid_res = [10]
    med_err = []
    for res in grid_res:
        lat_diff = float(math.fabs(max_lat - min_lat)) / res
        lon_diff = float(math.fabs(max_lon - min_lon)) / res
        model = trainModelUnsupervisedLearning(training_data, res, tower_cords, min_lat, min_lon, max_lat, max_lon)
        euclid_dist = testModelUnsupervisedLearning(model, testing_data, testing_data_loc, lat_diff, lon_diff, min_lat,
                                                    min_lon, res)
        # model = trainModelSuperVisedLearning(training_data_loc, training_data)
        # euclid_dist = testModelSuperVisedLearning(model, testing_data, testing_data_loc)
        med = median(euclid_dist)
        med_err.append(med)
        print med

    plt.xlabel('Grid Resolution', fontsize=12)
    plt.ylabel('Median error(meters)', fontsize=12)
    plt.plot(grid_res, med_err)
    plt.show()


# Vary number of transmitters to be considered for data generation
def varyTransmittersCount(data_points_count):
    print 'Varying Transmitters count'
    trans = [4, 6, 8, 10, 12, 15]
    res = 20
    med_err = []
    for trans_count in trans:
        powers, min_lat, min_lon, max_lat, max_lon = generateData(data_points_count, trans_count)
        lat_diff = float(math.fabs(max_lat - min_lat)) / res
        lon_diff = float(math.fabs(max_lon - min_lon)) / res
        testing_data, training_data, testing_data_loc, training_data_loc = splitData(powers, 0.1)
        model = trainModelUnsupervisedLearning(training_data, res)
        euclid_dist = testModelUnsupervisedLearning(model, testing_data, lat_diff, lon_diff)
        med = median(euclid_dist)
        med_err.append(med)

    plt.xlabel('Number of Transmitters', fontsize=12)
    plt.ylabel('Median error(meters)', fontsize=12)
    plt.plot(trans, med_err)
    plt.show()


# Vary the percentage of data to be used for training and testing
def varyTestingFraction(data_points_count, trans_count):
    print 'Varying Percent of testing training data split'
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5]
    res = 20
    med_err = []
    powers, min_lat, min_lon, max_lat, max_lon, tower_cords = generateData(data_points_count, trans_count)
    lat_diff = float(math.fabs(max_lat - min_lat)) / res
    lon_diff = float(math.fabs(max_lon - min_lon)) / res
    for fraction in fractions:
        testing_data, training_data, testing_data_loc, training_data_loc = splitData(powers, fraction)
        model = trainModelUnsupervisedLearning(training_data, res, tower_cords, min_lat, min_lon, max_lat, max_lon)
        euclid_dist = testModelUnsupervisedLearning(model, testing_data, testing_data_loc, lat_diff, lon_diff, min_lat,
                                                    min_lon, res)
        med = median(euclid_dist)
        med_err.append(med)
    plt.xlabel('Percentage of data points used as testing data', fontsize=12)
    plt.ylabel('Median error(meters)', fontsize=12)
    plt.plot(fractions, med_err)
    plt.show()


# Vary the radius of the area to be considered for data points generation
def varyRadiusForAnalysis(number_of_data_points, trans_count):
    dist = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    res = 20
    med_err = []
    for d in dist:
        powers, min_lat, min_lon, max_lat, max_lon = generateData(number_of_data_points, trans_count, d)
        lat_diff = float(math.fabs(max_lat - min_lat)) / res
        lon_diff = float(math.fabs(max_lon - min_lon)) / res
        testing_data, training_data, testing_data_loc, training_data_loc = splitData(powers, 0.1)
        model = trainModelUnsupervisedLearning(training_data, res)
        euclid_dist = testModelUnsupervisedLearning(model, testing_data, lat_diff, lon_diff)
        med = median(euclid_dist)
        med_err.append(med)

    plt.xlabel('Radius in KM', fontsize=12)
    plt.ylabel('Median error(meters)', fontsize=12)
    plt.plot(dist, med_err)
    plt.show()


# return median of a list
def median(lst):
    return np.median(np.array(lst))


if __name__ == '__main__':
    main(sys.argv[1:])
