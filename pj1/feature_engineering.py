import argparse
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC


class SVMModel:
    def __init__(self, kernel='rbf', C=1.0):
        self.model = SVC(kernel=kernel, C=C, probability=True)

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        return self.model.score(data, targets)


def data_preprocess(n_dimesion):
    data = np.load('./data/diagrams.npy')

    pca = PCA(n_components=n_dimesion)
    diagrams = pca.fit_transform(data)
    # diagrams = data[:, :n_dimesion]

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


def main(args, n_dimension):
    data_list, target_list = data_preprocess(n_dimension)

    task_acc_train = []
    task_acc_test = []

    # Model Initialization based on input argument
    model = SVMModel(kernel=args.kernel, C=args.C)

    start_time = time.time()

    for i in range(len(data_list)):
        train_data, test_data = data_list[i]
        train_targets, test_targets = target_list[i]

        # Train the model
        model.train(train_data, train_targets)

        # Evaluate the model
        train_accuracy = model.evaluate(train_data, train_targets)
        test_accuracy = model.evaluate(test_data, test_targets)

        task_acc_train.append(train_accuracy)
        task_acc_test.append(test_accuracy)

    print("Training accuracy:", sum(task_acc_train) / len(task_acc_train))
    print("Testing accuracy:", sum(task_acc_test) / len(task_acc_test))

    end_time = time.time()
    print("Time taken:", end_time - start_time)

    return sum(task_acc_train) / len(task_acc_train), sum(task_acc_test) / len(task_acc_test), end_time - start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM Model Training and Evaluation")
    parser.add_argument('--kernel', type=str, default='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid'],
                        help="Kernel type")
    parser.add_argument('--C', type=float, default=20, help="Regularization parameter")
    args = parser.parse_args()

    dimension = []
    train_acc = []
    test_acc = []
    run_time = []

    x = 101
    print(f"----------Dimension: {x}----------")
    dimension.append(x)
    train_accuracy, test_accuracy, time_taken = main(args, x)
    train_acc.append(train_accuracy)
    test_acc.append(test_accuracy)
    run_time.append(time_taken)
    print("---------------------------------")

    for i in range(1, 280, 20):
        print(f"----------Dimension: {i}----------")
        dimension.append(i)
        train_accuracy, test_accuracy, time_taken = main(args, i)
        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
        run_time.append(time_taken)
        print("---------------------------------")

    # Dimension: [1, 21, 41, 61, 81, 101, 121, 141, 161, 181, 201, 221, 241, 261]
    # Train Accuracy: [0.9021180886197333, 0.9024953662415559, 0.9024953662415559, 0.9024953662415559, 0.9024953662415559, 0.9024953662415559, 0.9024953662415559, 0.9024953662415559, 0.9024953662415559, 0.9024953662415559, 0.9024953662415559, 0.9024953662415559, 0.9024953662415559, 0.9024953662415559]
    # Test Accuracy: [0.9783672271651311, 0.9779068316873651, 0.9779068316873651, 0.9779068316873651, 0.9779068316873651, 0.9779068316873651, 0.9779068316873651, 0.9779068316873651, 0.9779068316873651, 0.9779068316873651, 0.9779068316873651, 0.9779068316873651, 0.9779068316873651, 0.9779068316873651]
    # Time taken: [5.0683324337005615, 6.666937828063965, 6.532778024673462, 6.83823037147522, 7.085592269897461, 40.27690362930298, 7.689087152481079, 7.9545512199401855, 8.088777542114258, 8.413077116012573, 37.477004528045654, 9.062399625778198, 9.338968515396118, 9.659477472305298]

    print("Dimension:", dimension)
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    print("Time taken:", run_time)

    # draw the plot
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(dimension, train_acc, color=color, label='Train Accuracy')
    ax1.plot(dimension, test_acc, color='tab:blue', label='Test Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Time taken (s)', color=color)
    ax2.plot(dimension, run_time, color=color, label='Time taken')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.show()

