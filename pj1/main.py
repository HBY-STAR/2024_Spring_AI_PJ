import argparse
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from fea import feature_extraction


class SVMModel:
    def __init__(self, kernel='rbf', C=1.0):
        self.model = SVC(kernel=kernel, C=C, probability=True)

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        return self.model.score(data, targets)


class LRModel:
    """
        Initialize Logistic Regression (from sklearn) model.

        Parameters:
        - C (float): Inverse of regularization strength; must be a positive float. Default is 1.0.
    """

    def __init__(self, C=1.0):
        self.model = LogisticRegression(C=C)

    """
        Train the Logistic Regression model.

        Parameters:
        - train_data (array-like): Training data.
        - train_targets (array-like): Target values for the training data.
    """

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    """
        Evaluate the performance of the Logistic Regression model.

        Parameters:
        - data (array-like): Data to be evaluated.
        - targets (array-like): True target values corresponding to the data.

        Returns:
        - float: Accuracy score of the model on the given data.
    """

    def evaluate(self, data, targets):
        return self.model.score(data, targets)


class LinearSVMModel:
    """
        Initialize Linear SVM (from sklearn) model.

        Parameters:
        - C (float): Inverse of regularization strength; must be a positive float. Default is 1.0.
    """

    def __init__(self, C=1.0):
        self.model = LinearSVC(C=C)

    """
        Train and Evaluate are the same.
    """

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        return self.model.score(data, targets)


def data_preprocess(args):
    if args.ent:
        diagrams = feature_extraction()[0]
    else:
        diagrams = np.load('./data/diagrams.npy')
    cast = pd.read_table('./data/SCOP40mini_sequence_minidatabase_19.cast')
    cast.columns.values[0] = 'protein'

    data_list = []
    target_list = []
    for task in range(1, 56):  # Assuming only one task for now
        task_col = cast.iloc[:, task]

        train_data = []
        train_targets = []
        test_data = []
        test_targets = []
        for item in range(0, len(task_col)):
            features = diagrams[item]
            # +train
            if task_col[item] == 1:
                train_data.append(features)
                train_targets.append(1)
            # -train
            elif task_col[item] == 2:
                train_data.append(features)
                train_targets.append(0)
            # +test
            elif task_col[item] == 3:
                test_data.append(features)
                test_targets.append(1)
            # -test
            elif task_col[item] == 4:
                test_data.append(features)
                test_targets.append(0)
            # error
            else:
                print("Error in data")
                continue
        data_list.append((train_data, test_data))
        target_list.append((train_targets, test_targets))

    return data_list, target_list


def main(args):
    data_list, target_list = data_preprocess(args)

    task_acc_train = []
    task_acc_test = []

    # Model Initialization based on input argument
    if args.model_type == 'svm':
        model = SVMModel(kernel=args.kernel, C=args.C)
    else:
        print("Attention: Kernel option is not supported")
        if args.model_type == 'linear_svm':
            model = LinearSVMModel(C=args.C)
        elif args.model_type == 'lr':
            model = LRModel(C=args.C)
        else:
            raise ValueError("Unsupported model type")

    start_time = time.time()

    for i in range(len(data_list)):
        train_data, test_data = data_list[i]
        train_targets, test_targets = target_list[i]

        # print(f"Processing dataset {i + 1}/{len(data_list)}")

        # Train the model
        model.train(train_data, train_targets)

        # Evaluate the model
        train_accuracy = model.evaluate(train_data, train_targets)
        test_accuracy = model.evaluate(test_data, test_targets)

        print(f"Dataset {i + 1}/{len(data_list)} - Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        task_acc_train.append(train_accuracy)
        task_acc_test.append(test_accuracy)

    print(args.model_type + ' - ' + args.kernel)

    print("Training accuracy:", sum(task_acc_train) / len(task_acc_train))
    print("Testing accuracy:", sum(task_acc_test) / len(task_acc_test))

    print("Time taken:", time.time() - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM Model Training and Evaluation")
    parser.add_argument('--model_type', type=str, default='svm', choices=['svm', 'linear_svm', 'lr'], help="Model type")
    parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid'],
                        help="Kernel type")
    parser.add_argument('--C', type=float, default=20, help="Regularization parameter")
    parser.add_argument('--ent', action='store_true',
                        help="Load data from a file using a feature engineering function feature_extraction() from "
                             "fea.py")
    args = parser.parse_args()
    main(args)
