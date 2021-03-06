import numpy
import pandas

from math import sqrt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


RANDOM_STATE = 6578439


def split_date(dt):
    weekday = dt.weekday()
    return (
        weekday,
        1 if weekday >= 5 else 0,
        dt.day,
        dt.month,
        dt.year,
    )


def load_dataset(path):
    stores_df = pandas.read_csv('data/store.csv')
    stores_df = stores_df.fillna(-1)
    stores_df['StoreType'] = LabelEncoder().fit_transform(stores_df['StoreType'])
    stores_df['Assortment'] = LabelEncoder().fit_transform(stores_df['Assortment'])
    # Dropping yields a better performance than:
    # - Giving each month a boolean column
    # - Replace the string with a count of the months
    stores_df = stores_df.drop('PromoInterval', axis=1)

    annotated_df = pandas.read_csv(path, parse_dates=['Date'], dtype={'StateHoliday': object})
    # Dropping yields a better performance than:
    # - Label encoding
    annotated_df = annotated_df.drop('StateHoliday', axis=1)
    # Ugly but fast way to convert Date column to useful, seperate columns
    (
        annotated_df['DayOfWeek'],
        annotated_df['IsWeekend'],
        annotated_df['DayOfMonth'],
        annotated_df['Month'],
        annotated_df['Year']
    ) = zip(*annotated_df['Date'].map(split_date))
    annotated_df = annotated_df.drop('Date', axis=1)
    annotated_df = annotated_df.fillna(-1)

    # Merging dataset and stores
    return pandas.merge(annotated_df, stores_df, on='Store', how='inner', sort=False)


def root_mean_square_percentage(labels, predictions):
    if len(labels) != len(predictions):
        raise Exception("Labels and predictions must be of same length")
    # Remove pairs where label == 0
    labels, predictions = tuple(
        zip(*filter(lambda x: x[0] != 0, zip(labels, predictions)))
    )
    labels = numpy.array(labels, dtype=float)
    predictions = numpy.array(predictions, dtype=float)
    return sqrt(numpy.power((labels - predictions) / labels, 2.0).sum() / len(labels))


def main():
    print "Loading train set..."
    train_df = load_dataset('data/train.csv')
    data_columns = train_df.columns.difference(['Sales', 'Customers'])
    target_column = 'Sales'

    print "Splitting train and verification sets..."
    train_index, validation_index = train_test_split(
        train_df.index,
        test_size=0.1,
        random_state=RANDOM_STATE
    )

    print "Training random forest..."
    random_forest = RandomForestRegressor(
        n_jobs=-1,  # Auto selects number of cores
        random_state=RANDOM_STATE,
        max_features=1 / 3.0,
        n_estimators=100,
    ).fit(
        X=train_df.loc[train_index, data_columns],
        y=train_df.loc[train_index, target_column],
    )
    print "Feature importances:"
    pairs = zip(data_columns, random_forest.feature_importances_)
    pairs.sort(key=lambda x: -x[1])
    for column, importance in pairs:
        print " ", column, importance

    print "Verifying random forest..."
    predictions = random_forest.predict(
        X=train_df.loc[validation_index, data_columns],
    )
    print "Root mean square percentage:"
    print " ", root_mean_square_percentage(
        train_df.loc[validation_index, target_column],
        predictions
    )

    print "Retraining on the entire training set..."
    random_forest = RandomForestRegressor(
        n_jobs=-1,  # Auto selects number of cores
        random_state=RANDOM_STATE,
        max_features=1 / 3.0,
        n_estimators=100,
    ).fit(
        X=train_df[data_columns],
        y=train_df[target_column],
    )

    print "Loading test dataset..."
    unannotated_df = load_dataset('data/test.csv')

    print "Getting predictions..."
    predictions = random_forest.predict(
        X=unannotated_df[unannotated_df.columns.difference(['Id'])],
    )

    print "Writing predictions to file..."
    with open('submission.csv', 'wb') as f:
        f.write("Id,Sales\n")
        for i, prediction in enumerate(predictions):
            f.write("%d,%d\n" % (unannotated_df['Id'][i], int(round(prediction))))

    print "Done"


if __name__ == '__main__':
    main()
