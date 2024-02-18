import pandas as pd
from pycaret.classification import ClassificationExperiment

def load_data(filepath):
    """
    Loads diabetes data into a DataFrame from a string filepath.
    """
    df = pd.read_csv('/Users/aaron/Documents/Jupyter/data/prepped_churn_data.csv', index_col='customerID')
    return df


def make_predictions(df):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    """
    classifier = ClassificationExperiment()
    model = classifier.load_model('pycaret_model')
    predictions = classifier.predict_model(model, data=df)
    predictions.rename({'Label': 'Churn'}, axis=1, inplace=True)
    predictions['Churn'].replace({1: 'Churn', 0: 'No churn'},
                                            inplace=True)
    return predictions['Churn']


if __name__ == "__main__":
    df = load_data('/Users/aaron/Documents/Jupyter/data/prepped_churn_data.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)
