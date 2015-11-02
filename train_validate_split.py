import pandas as pd

"""
This is a crude little script that performs the train/validate split
in two different ways:

    1. Suppose `n` is the number of days associated with
        the test set. Then, we take the last `n` days of data from
        the training set as our validation set.

    2. Get the week numbers over which the test data is taken. Then,
        we take the training data over these same week numbers but from the
        previous year as our validation data.

"""


print "Loading the data..."
df = pd.read_csv("data/train_normalized.csv.gz", compression='gzip')
df_test = pd.read_csv("data/test_normalized.csv.gz", compression='gzip')

print "Performing train/validate split on last n days..."
n_days_test = df_test['NowDayFromStart'].max() - df_test['NowDayFromStart'].min()
mask = df['NowDayFromStart'].max() - df['NowDayFromStart'] > n_days_test
df_train_1, df_validate_1 = df.loc[mask], df.loc[~mask]

print "Performing train/validate split on same week numbers of previous year"
mask_validate = (
    (df['NowWeek'] >= df_test['NowWeek'].min()) &
    (df['NowWeek'] <= df_test['NowWeek'].max()) &
    (df['NowYear'] == 2014)
)
mask_train = (
    (
        (df['NowYear'] == 2014) &
        (df['NowWeek'] < df_test['NowWeek'].min())
    ) |
    (df['NowYear'] < 2014)
)
df_train_2, df_validate_2 = df.loc[mask_train], df.loc[mask_validate]


print "Writing data to disk..."
df_train_1.to_csv('data/train_1.csv')
df_train_2.to_csv('data/train_2.csv')
df_validate_1.to_csv('data/validate_1.csv')
df_validate_2.to_csv('data/validate_2.csv')
