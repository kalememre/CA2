import pickle
import pandas as pd
import numpy as np



def get_labour_variables():
    with open('labour_variables.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

def test_data_predict():
    data = get_labour_variables()
    # Test data
    # Modify the code to retrieve Ireland and UK data
    year = 2020
    ireland_pred = pd.DataFrame({'TIME_PERIOD': [year], 'country': [data['ireland_encoded']]})
    uk_pred = pd.DataFrame({'TIME_PERIOD': [year], 'country': [data['uk_encoded']]})

    # Perform the predictions
    ireland_pred = data['model'].predict(ireland_pred)
    uk_pred = data['model'].predict(uk_pred)

    # Define the expected results (example values)
    ireland_expected = np.array([7.02956029])
    uk_expected = np.array([6.71902388])

    # Compare the predictions with the expected results
    np.testing.assert_allclose(ireland_pred, ireland_expected)
    np.testing.assert_allclose(uk_pred, uk_expected)