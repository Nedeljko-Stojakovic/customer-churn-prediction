import pandas as pd


def load_data():
    """
    Load data from the dataset.xlsx file and features_description.xlsx file
    """
    data = pd.read_excel('data/dataset.xlsx')
    features_desc = pd.read_excel('data/features_description.xlsx')

    # Print data types of each column
    print("Data types of each column:")
    print(data.dtypes)

    # Print unique values in the Segment column
    print("\nUnique values in Segment column:")
    print(data['Segment'].unique())

    return data, features_desc
