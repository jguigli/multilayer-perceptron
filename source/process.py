import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import load

def export_set(X_train, X_test, y_train, y_test, columns):
    X_columns = columns.remove('Diagnosis')

    df = pd.DataFrame(X_train, columns=X_columns)
    df.to_csv(f"../data/X_train.csv", index=False)
    df = pd.DataFrame(X_test, columns=X_columns)
    df.to_csv(f"../data/X_test.csv", index=False)
    df = pd.DataFrame(y_train, columns=['Diagnosis'])
    df.to_csv(f"../data/y_train.csv", index=False)
    df = pd.DataFrame(y_test, columns=['Diagnosis'])
    df.to_csv(f"../data/y_test.csv", index=False)
    print(f"Exporting file : train and test datasets has been saved to /data")


def ft_test_train(data, test_size = 0, train_size = 0):
    data_shuffled = data.sample(frac=1, random_state=42)
    y_data = data_shuffled["Diagnosis"]
    X_data = data_shuffled.drop("Diagnosis", axis=1)

    if test_size:
        test_size = int(test_size * len(data))
        train_size = int((1 - test_size) * len(data))
    elif train_size:
        train_size = int(train_size * len(data))
        test_size = int((1 - train_size) * len(data))
    
    X_train = X_data[:train_size]
    X_test = X_data[train_size:]
    y_train = y_data[:train_size]
    y_test = y_data[train_size:]
    return X_train, X_test, y_train, y_test


def process_data():
    try:
        raw_data = load("../data/data.csv")
        section_names = ['Mean', 'Standard error', 'Largest']
        real_valued_names = ['radius', 'texture', 'perimeter', 'area', 'smootheness', 'compactness', 'concavity', 'concave points', 'symmetry', 'fractal dimension']
        columns_names = ['ID number', 'Diagnosis']

        for section in section_names:
            for name in real_valued_names:
                real_name = section + " " + name
                columns_names.append(real_name)
        
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

