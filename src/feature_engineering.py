import pandas as pd


def engineer_features(data):
    # Create a ratio feature
    data['USAGE_RATIO'] = data['USAGE_OUT_ONNET_DUR'] / (data['USAGE_OUT_OFFNET_DUR'] + 1)

    # One-hot encode the 'Segment' column
    data = pd.get_dummies(data, columns=['Segment'], prefix='Segment')

    # Fill NaN values with the mean of the column
    data = data.fillna(data.mean())

    return data