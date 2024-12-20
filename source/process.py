import pandas as pd

def export_set(X_train, y_train, X_validation, y_validation, columns):
    X_columns = columns.remove('Diagnosis')

    df = pd.DataFrame(X_train, columns=X_columns)
    df.to_csv(f"../data_sets/X_train.csv", index=False)
    df = pd.DataFrame(y_train, columns=['Diagnosis'])
    df.to_csv(f"../data_sets/y_train.csv", index=False)

    df = pd.DataFrame(X_validation, columns=X_columns)
    df.to_csv(f"../data_sets/X_validation.csv", index=False)
    df = pd.DataFrame(y_validation, columns=['Diagnosis'])
    df.to_csv(f"../data_sets/y_validation.csv", index=False)

    print(f"=> train and validation datasets has been saved to /data_sets <=")


def split_data_train_validation(data, validation_rate=0.2):
    data_shuffled = data.sample(frac=1)
    y_data = data_shuffled["Diagnosis"]
    X_data = data_shuffled.drop(["ID number", "Diagnosis"], axis=1)

    validation_size = int(validation_rate * len(data))
    
    X_validation = X_data[:validation_size]
    y_validation = y_data[:validation_size]
    
    X_train = X_data[validation_size:]
    y_train = y_data[validation_size:]

    return X_train, y_train, X_validation, y_validation


def process_data():
    try:
        raw_data = pd.read_csv("../data_sets/data.csv")
        section_names = ['Mean', 'Standard error', 'Largest']
        real_valued_names = ['radius', 'texture', 'perimeter', 'area', 'smootheness', 'compactness', 'concavity', 'concave points', 'symmetry', 'fractal dimension']
        columns_names = ['ID number', 'Diagnosis']

        for section in section_names:
            for name in real_valued_names:
                real_name = section + " " + name
                columns_names.append(real_name)
        
        raw_data.columns = columns_names

        print("Process dataset ...")
        X_train, y_train, X_validation, y_validation = split_data_train_validation(raw_data, 0.2)
        export_set(X_train, y_train, X_validation, y_validation, columns_names)
        
    except Exception as e:
        print(f"Error handling: {str(e)}")
        return

def main():
    process_data()

if __name__ == "__main__":
    main()

