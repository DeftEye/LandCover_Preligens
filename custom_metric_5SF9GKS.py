"""
The metric of the challenge, the Kullback-Leibler divergence between two discrete distributions.
"""
import numpy as np
import pandas as pd


def custom_metric_function(dataframe_y_true, dataframe_y_pred):
    """
        The metric of the challenge, the Kullback-Leibler divergence between two discrete distributions.
    Args:
        dataframe_y_true: Pandas Dataframe
            Dataframe containing the true values of y.
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_y_true = pd.read_csv(CSV_1_FILE_PATH, index_col=0, sep=',')

        dataframe_y_pred: Pandas Dataframe
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_y_pred = pd.read_csv(CSV_2_FILE_PATH, index_col=0, sep=',')

    Returns
        score: Float
            The metric evaluated with the two dataframes: the mean Kullback-Leibler divergence between all
            sample pairs. This must not be NaN.
    """
    y_true, y_pred = np.asarray(dataframe_y_true, dtype=np.float64), np.asarray(dataframe_y_pred, dtype=np.float64)

    # Normalize to sum to 1 if it's not already
    y_true /= y_true.sum(1, keepdims=True)
    y_pred /= y_pred.sum(1, keepdims=True)
    # add a small constant for smoothness around 0
    y_true += 1e-7
    y_pred += 1e-7
    score = np.mean(np.sum(y_true * np.log(y_true / y_pred), 1))
    try:
        assert np.isfinite(score)
    except AssertionError as e:
        raise ValueError('score is NaN or infinite') from e
    return score


if __name__ == '__main__':

    CSV_FILE_Y_TRUE = '--------.csv'
    CSV_FILE_Y_PRED = '--------.csv'

    df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',')
    df_y_pred = pd.read_csv(CSV_FILE_Y_PRED, index_col=0, sep=',')

    df_y_pred = df_y_pred.loc[df_y_true.index]
    # remove ignored class columns "no_data" and "clouds" if present
    df_y_pred = df_y_pred.drop(['no_data', 'clouds'], axis=1, errors='ignore')
    df_y_true = df_y_true.drop(['no_data', 'clouds'], axis=1, errors='ignore')

    df_y_pred = df_y_pred.loc[:, df_y_true.columns]

    if np.any(np.allclose(df_y_pred.sum(1), 0.)):
        # if any row predicted sums to 0, raise an error
        raise ValueError("some row vectors in y_pred sum to 0")

    print(custom_metric_function(df_y_true, df_y_pred))
