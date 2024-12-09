import pandas as pd
from utils import load

def export_set(X_train, X_test, y_train, y_test, columns):
    X_columns = columns.remove('Diagnosis')

    df = pd.DataFrame(X_train, columns=X_columns)
    df.to_csv(f"../data_sets/X_train.csv", index=False)
    df = pd.DataFrame(X_test, columns=X_columns)
    df.to_csv(f"../data_sets/X_validation.csv", index=False)
    df = pd.DataFrame(y_train, columns=['Diagnosis'])
    df.to_csv(f"../data_sets/y_train.csv", index=False)
    df = pd.DataFrame(y_test, columns=['Diagnosis'])
    df.to_csv(f"../data_sets/y_validation.csv", index=False)
    print(f"Exporting file : train and validation datasets has been saved to /data_sets")


def split_data_train_validation(data, validation_rate = 0.2):
    data_shuffled = data.sample(frac=1, random_state=42)
    y_data = data_shuffled["Diagnosis"]
    X_data = data_shuffled.drop(["ID number", "Diagnosis"], axis=1)

    validation_size = int(validation_rate * len(data))
    
    X_train = X_data[validation_size:]
    X_validation = X_data[:validation_size]
    y_train = y_data[validation_size:]
    y_validation = y_data[:validation_size]
    return X_train, X_validation, y_train, y_validation


def process_data():
    try:
        raw_data = load("../data_sets/data.csv")
        section_names = ['Mean', 'Standard error', 'Largest']
        real_valued_names = ['radius', 'texture', 'perimeter', 'area', 'smootheness', 'compactness', 'concavity', 'concave points', 'symmetry', 'fractal dimension']
        columns_names = ['ID number', 'Diagnosis']

        for section in section_names:
            for name in real_valued_names:
                real_name = section + " " + name
                columns_names.append(real_name)
        
        raw_data.columns = columns_names

        # df = pd.DataFrame(raw_data).drop(["ID number",'Mean radius', 'Mean texture', 'Mean perimeter', 'Mean area', 'Mean smootheness', 'Mean compactness', 'Mean concavity', 'Mean concave points', 'Mean symmetry', 'Mean fractal dimension'], axis=1)
        # df.to_csv(f"../data_sets/data_mean.csv", index=False)
        # df = pd.DataFrame(raw_data).drop(["ID number",'Standard error radius', 'Standard error texture', 'Standard error perimeter', 'Standard error area', 'Standard error smootheness', 'Standard error compactness', 'Standard error concavity', 'Standard error concave points', 'Standard error symmetry', 'Standard error fractal dimension'], axis=1)
        # df.to_csv(f"../data_sets/data_std.csv", index=False)
        # df = pd.DataFrame(raw_data).drop(["ID number",'Largest radius', 'Largest texture', 'Largest perimeter', 'Largest area', 'Largest smootheness', 'Largest compactness', 'Largest concavity', 'Largest concave points', 'Largest symmetry', 'Largest fractal dimension'], axis=1)
        # df.to_csv(f"../data_sets/data_largest.csv", index=False)

        X_train, X_validation, y_train, y_validation = split_data_train_validation(raw_data, 0.2)
        export_set(X_train, X_validation, y_train, y_validation, columns_names)
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    process_data()

if __name__ == "__main__":
    main()

