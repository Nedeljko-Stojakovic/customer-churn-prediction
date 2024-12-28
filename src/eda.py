import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def perform_eda(data):
    """
    Perform Exploratory Data Analysis on the dataset
    :param data:
    """
    print(data.info())
    print("\nSummary Statistics:")
    print(data.describe())

    # Create a copy of the data for correlation analysis
    data_numeric = data.copy()

    # Convert 'Segment' to numeric
    data_numeric['Segment'] = pd.Categorical(data_numeric['Segment']).codes

    # Select only numeric columns
    numeric_columns = data_numeric.select_dtypes(include=[np.number]).columns

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(data_numeric[numeric_columns].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.savefig('outputs/correlation_heatmap.png')
    plt.close()

    # Distribution of numerical features
    for column in numeric_columns:
        if column not in ['SUBSCRIBER_ID', 'CHURN']:
            plt.figure(figsize=(10, 6))
            sns.histplot(data[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.savefig(f'outputs/distribution_{column}.png')
            plt.close()

    # Segment distribution
    plt.figure(figsize=(10, 6))
    data['Segment'].value_counts().plot(kind='bar')
    plt.title('Distribution of Segments')
    plt.xlabel('Segment')
    plt.ylabel('Count')
    plt.savefig('outputs/segment_distribution.png')
    plt.close()

    # Churn rate by Segment
    plt.figure(figsize=(10, 6))
    churn_rate = data.groupby('Segment')['CHURN'].mean()
    churn_rate.plot(kind='bar')
    plt.title('Churn Rate by Segment')
    plt.xlabel('Segment')
    plt.ylabel('Churn Rate')
    plt.savefig('outputs/churn_rate_by_segment.png')
    plt.close()