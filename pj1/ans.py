import os
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from fea import feature_extraction

from Bio.PDB import PDBParser

class SVMModel:
    def __init__(self, kernel='rbf', C=1.0):
        self.model = SVC(kernel=kernel, C=C, probability=True)

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        return self.model.score(data, targets)

class LRModel:
    # todo
    def __init__(self, C=1.0):
        self.model = LogisticRegression(C=C, max_iter=10000)

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        return self.model.score(data, targets)

class LinearSVMModel:
    # todo
    def __init__(self, C=1.0):
        self.model = LinearSVC(C=C, max_iter=10000)

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        return self.model.score(data, targets)

def data_preprocess(args):
    # Load data
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

        # todo
        # Partition training and testing sets
        train_set = task_col.isin([1, 2])
        test_set = task_col.isin([3, 4])

        # Generate training and testing targets
        train_targets_all = np.ravel(label_binarize(task_col, classes=[1]))
        test_targets_all = np.ravel(label_binarize(task_col, classes=[3]))
        train_targets = train_targets_all[train_set]
        test_targets = test_targets_all[test_set]

        # Partition diagrams
        train_data = diagrams[train_set]
        test_data = diagrams[test_set]

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

    for i in range(len(data_list)):
        train_data, test_data = data_list[i]
        train_targets, test_targets = target_list[i]

        print(f"Processing dataset {i+1}/{len(data_list)}")

        # Train the model
        model.train(train_data, train_targets)

        # Evaluate the model
        train_accuracy = model.evaluate(train_data, train_targets)
        test_accuracy = model.evaluate(test_data, test_targets)

        print(f"Dataset {i+1}/{len(data_list)} - Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        task_acc_train.append(train_accuracy)
        task_acc_test.append(test_accuracy)


    print("Training accuracy:", sum(task_acc_train)/len(task_acc_train))
    print("Testing accuracy:", sum(task_acc_test)/len(task_acc_test))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM Model Training and Evaluation")
    parser.add_argument('--model_type', type=str, default='svm', choices=['svm', 'linear_svm', 'lr'], help="Model type")
    parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid'], help="Kernel type")
    parser.add_argument('--C', type=float, default=20, help="Regularization parameter")
    parser.add_argument('--ent', action='store_true', help="Load data from file")
    args = parser.parse_args()
    main(args)