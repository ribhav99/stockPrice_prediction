from train_test_split import my_train_test_split
import numpy as np
from tqdm import tqdm, trange
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pickle
import os

methods = ["decision_tree", "random_forest", "adaboost", "xgboost", "all"]
symbol = "MFC"
method = methods[4]
test = True

# 0 for all
# 1 for only High
# 2 for only Low
# 3 for only Close
# 4 for only Adj Close
pred_num = 1

x_train, x_test, y_train, y_test, _ = my_train_test_split(symbol=symbol)

# Adaboost only predicts one price so this is required
if pred_num == 1:
    y_train, y_test = y_train[:, 0], y_test[:, 0]
elif pred_num == 2:
    y_train, y_test = y_train[:, 1], y_test[:, 1]
elif pred_num == 3:
    y_train, y_test = y_train[:, 2], y_test[:, 2]
elif pred_num == 4:
    y_train, y_test = y_train[:, 3], y_test[:, 3]


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

if method == "decision_tree":

    if pred_num == 0:
        name = 'models/NonDeep/decisionTreeRegressor.sav'
    elif pred_num == 1:
        name = 'models/NonDeep/decisionTreeRegressorHigh.sav'
    elif pred_num == 2:
        name = 'models/NonDeep/decisionTreeRegressorLow.sav'
    elif pred_num == 3:
        name = 'models/NonDeep/decisionTreeRegressorClose.sav'
    elif pred_num == 4:
        name = 'models/NonDeep/decisionTreeRegressorAdjClose.sav'

    if not test:
        model = tree.DecisionTreeRegressor()
        model.fit(x_train, y_train)

        pickle.dump(model, open(
            name, 'wb'))

    else:
        model = pickle.load(
            open(name, 'rb'))
        predictions = model.predict(x_test)
        loss_array = (abs(predictions - y_test) / y_test) * 100

        if pred_num == 0:
            loss = 0
            for i in loss_array:
                ting = 0
                ting += sum(i)
                ting /= 4
                loss += ting

            loss /= len(loss_array)
        else:
            loss = sum(loss_array)/len(loss_array)

        print("Prediction is within {}% of actual value".format(loss))

    visualise = input("save fig? y/n: ")
    if visualise.lower().strip() == 'y' or visualise.lower().strip() == 'yes':
        fig = plt.figure()
        _ = tree.plot_tree(model, filled=True)
        fig.savefig('models/NonDeep/decisionTreeRegressor.png')

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

elif method == "random_forest":

    if pred_num == 0:
        name = 'models/NonDeep/randomForestRegressor.sav'
    elif pred_num == 1:
        name = 'models/NonDeep/randomForestRegressorHigh.sav'
    elif pred_num == 2:
        name = 'models/NonDeep/randomForestRegressorLow.sav'
    elif pred_num == 3:
        name = 'models/NonDeep/randomForestRegressorClose.sav'
    elif pred_num == 4:
        name = 'models/NonDeep/randomForestRegressorAdjClose.sav'

    if not test:
        model = RandomForestRegressor(n_estimators=230, n_jobs=os.cpu_count())
        model.fit(x_train, y_train)

        pickle.dump(model, open(
            name, 'wb'))

    else:
        model = pickle.load(
            open(name, 'rb'))
        predictions = model.predict(x_test)
        loss_array = (abs(predictions - y_test) / y_test) * 100

        if pred_num == 0:
            loss = 0
            for i in loss_array:
                ting = 0
                ting += sum(i)
                ting /= 4
                loss += ting

            loss /= len(loss_array)
        else:
            loss = sum(loss_array)/len(loss_array)

        print("Prediction is within {}% of actual value".format(loss))

    visualise = input("graph tree values? y/n: ")
    if visualise.lower().strip() == 'y' or visualise.lower().strip() == 'yes':
        loss_percentage = []
        num_trees = [x for x in range(1, 1000)]
        for i in trange(1, 1000):
            model = RandomForestRegressor(
                n_estimators=i, n_jobs=os.cpu_count())
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            loss_array = (abs(predictions - y_test) / y_test) * 100

            loss = 0
            for i in loss_array:
                ting = 0
                ting += sum(i)
                ting /= 4
                loss += ting

            loss /= len(loss_array)

            loss_percentage.append(loss)

        loss_percentage = np.array(loss_percentage)
        num_trees = np.array(num_trees)
        with open("loss_percentage_randomforest.txt", "w") as file:
            np.savetxt(file, loss_percentage)

        with open("num_trees_randomforest.txt", "w") as file:
            np.savetxt(file, num_trees)

        plt.plot(num_trees, loss_percentage)
        plt.show()

    # x = np.loadtxt("loss_percentage_randomforest.txt")
    # y = np.loadtxt("num_trees_randomforest.txt")
    # print(y[np.argmin(x)])

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

elif method == "adaboost":

    if pred_num == 0:
        name = 'models/NonDeep/adaBoostRegressor.sav'
    elif pred_num == 1:
        name = 'models/NonDeep/adaBoostRegressorHigh.sav'
    elif pred_num == 2:
        name = 'models/NonDeep/adaBoostRegressorLow.sav'
    elif pred_num == 3:
        name = 'models/NonDeep/adaBoostRegressorClose.sav'
    elif pred_num == 4:
        name = 'models/NonDeep/adaBoostRegressorAdjClose.sav'

    if not test:
        model = AdaBoostRegressor(n_estimators=935, random_state=6)
        model.fit(x_train, y_train)

        pickle.dump(model, open(
            name, 'wb'))

    else:
        model = pickle.load(
            open(name, 'rb'))
        predictions = model.predict(x_test)
        loss_array = (abs(predictions - y_test) / y_test) * 100

        loss = sum(loss_array)/len(loss_array)

        print("Prediction is within {}% of actual value".format(loss))

    visualise = input("graph tree values? y/n: ")

    if visualise.lower().strip() == 'y' or visualise.lower().strip() == 'yes':
        loss_percentage = []
        num_trees = [x for x in range(1, 1000)]
        for i in trange(1, 1000):
            model = AdaBoostRegressor(n_estimators=i)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            loss_array = (abs(predictions - y_test) / y_test) * 100

            loss = sum(loss_array)/len(loss_array)

            loss_percentage.append(loss)

        loss_percentage = np.array(loss_percentage)
        num_trees = np.array(num_trees)
        with open("loss_percentage_adaboost.txt", "w") as file:
            np.savetxt(file, loss_percentage)

        with open("num_trees_adaboost.txt", "w") as file:
            np.savetxt(file, num_trees)

        plt.plot(num_trees, loss_percentage)
        plt.show()

    # x = np.loadtxt("loss_percentage_adaboost.txt")
    # y = np.loadtxt("num_trees_adaboost.txt")
    # print(y[np.argmin(x)])

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


elif method == "xgboost":

    if pred_num == 0:
        name = 'models/NonDeep/xgBoostRegressor.sav'
    elif pred_num == 1:
        name = 'models/NonDeep/xgBoostRegressorHigh.sav'
    elif pred_num == 2:
        name = 'models/NonDeep/xgBoostRegressorLow.sav'
    elif pred_num == 3:
        name = 'models/NonDeep/xgBoostRegressorClose.sav'
    elif pred_num == 4:
        name = 'models/NonDeep/xgBoostRegressorAdjClose.sav'

    if not test:
        model = XGBRegressor(n_jobs=os.cpu_count(),
                             random_state=6)
        model.fit(x_train, y_train)

        pickle.dump(model, open(
            name, 'wb'))

    else:
        model = pickle.load(
            open(name, 'rb'))
        predictions = model.predict(x_test)
        loss_array = (abs(predictions - y_test) / y_test) * 100

        loss = sum(loss_array)/len(loss_array)

        print("Prediction is within {}% of actual value".format(loss))

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

elif method == "all":
    # use all and average results
    if pred_num == 0:
        xgboostname = 'models/NonDeep/xgBoostRegressor.sav'
        adaboostname = 'models/NonDeep/adaBoostRegressor.sav'
        randomforestname = 'models/NonDeep/randomForestRegressor.sav'
        treename = 'models/NonDeep/decisionTreeRegressor.sav'

    elif pred_num == 1:
        xgboostname = 'models/NonDeep/xgBoostRegressorHigh.sav'
        adaboostname = 'models/NonDeep/adaBoostRegressorHigh.sav'
        randomforestname = 'models/NonDeep/randomForestRegressorHigh.sav'
        treename = 'models/NonDeep/decisionTreeRegressorHigh.sav'

    elif pred_num == 2:
        xgboostname = 'models/NonDeep/xgBoostRegressorLow.sav'
        adaboostname = 'models/NonDeep/adaBoostRegressorLow.sav'
        randomforestname = 'models/NonDeep/randomForestRegressorLow.sav'
        treename = 'models/NonDeep/decisionTreeRegressorLow.sav'

    elif pred_num == 3:
        xgboostname = 'models/NonDeep/xgBoostRegressorClose.sav'
        adaboostname = 'models/NonDeep/adaBoostRegressorClose.sav'
        randomforestname = 'models/NonDeep/randomForestRegressorClose.sav'
        treename = 'models/NonDeep/decisionTreeRegressorClose.sav'

    elif pred_num == 4:
        xgboostname = 'models/NonDeep/xgBoostRegressorAdjClose.sav'
        adaboostname = 'models/NonDeep/adaBoostRegressorAdjClose.sav'
        randomforestname = 'models/NonDeep/randomForestRegressorAdjClose.sav'
        treename = 'models/NonDeep/decisionTreeRegressorAdjClose.sav'

    if not test:
        print("Train each model separately, then test here!")

    else:
        xgboost = pickle.load(
            open(xgboostname, 'rb'))
        xgboost_predictions = xgboost.predict(x_test)

        adaboost = pickle.load(
            open(adaboostname, 'rb'))
        adaboost_predictions = adaboost.predict(x_test)

        randomforest = pickle.load(
            open(randomforestname, 'rb'))
        randomforest_predictions = randomforest.predict(x_test)

        tree = pickle.load(
            open(treename, 'rb'))
        tree_predictions = tree.predict(x_test)

        predictions = xgboost_predictions + adaboost_predictions + \
            randomforest_predictions + tree_predictions
        predictions /= 4

        loss_array = (abs(predictions - y_test) / y_test) * 100

        loss = sum(loss_array)/len(loss_array)

        print("Prediction is within {}% of actual value".format(loss))
