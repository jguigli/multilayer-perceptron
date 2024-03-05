import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load

def export_set(X_train, X_test, y_train, y_test, columns):
    X_columns = columns.remove('Diagnosis')
    df = pd.DataFrame(X_train.csv, columns=X_columns)
    df.to_csv(f"../data_sets/X_train.csv", index=False)
    df = pd.DataFrame(X_test.csv, columns=X_columns)
    df.to_csv(f"../data_sets/X_test.csv", index=False)
    df = pd.DataFrame(y_train, columns=['Diagnosis'])
    df.to_csv(f"../data_sets/y_train.csv", index=False)
    df = pd.DataFrame(y_test, columns=['Diagnosis'])
    df.to_csv(f"../data_sets/y_test.csv", index=False)
    print(f"Exporting file : train and test datasets has been saved to ./data_sets/")


def ft_test_train(data, test_size = 0, train_size = 0):
    data = np.random.shuffle(data)
    X_data = data.drop("Diagnosis")
    y_data = data["Diagnosis"]

    if train_size:
        train_size = int(train_size * len(data))
        test_size = int((1 - train_size) * len(data))
    elif test_size:
        train_size = int((1 - test_size) * len(data))
        test_size = int(test_size * len(data))
    
    X_train = X_data[:train_size]
    X_test = X_data[train_size:]
    y_train = y_data[:train_size]
    y_test = y_data[train_size:]
    return X_train, X_test, y_train, y_test


def process_data():
    try:
        raw_data = load("./data/data.csv")
        columns_names = ['ID number', 'Diagnosis']
        raw_data.columns = columns_names
        X_train, X_test, y_train, y_test = ft_test_train(raw_data, 0.33)
        export_set(X_train, X_test, y_train, y_test, columns_names)
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    process_data()

if __name__ == "__main__":
    main()

