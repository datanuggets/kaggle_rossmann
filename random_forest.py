import arrow
import pandas
import datetime
import numpy
import time

from math import sqrt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


RANDOM_STATE=6578439


def root_mean_square_percentage(labels, predictions):
	""" As defined by competition """
	if len(labels) != len(predictions):
		raise Exception("Labels and predictions must be of same length")
	# Filter pairs where label == 0
	labels, predictions = tuple(zip(*
		filter(lambda x: x[0] != 0, zip(labels, predictions))
	))
	labels = numpy.array(labels, dtype=float)
	predictions = numpy.array(predictions, dtype=float)
	return sqrt(numpy.power((labels - predictions) / labels, 2.0).sum() / len(labels))


if __name__ == '__main__':
	print "Loading annotated dataset..."
	annotated_df = pandas.read_csv(
		'data/train.csv',
		dtype={
			'StateHoliday': object,
			'Sales': float,
			'Customers': float,
		},
		parse_dates=['Date']
	)

	print "Preparing annotated dataset for sklearn usage..."
	annotated_df['StateHoliday'] = LabelEncoder().fit_transform(annotated_df['StateHoliday'])

	print "Enriching annotated dataset with extra features..."
	annotated_df['DayOfMonth'] = annotated_df['Date'].apply(lambda dt: dt.day)
	annotated_df['Month'] = annotated_df['Date'].apply(lambda dt: dt.month)
	annotated_df['Year'] = annotated_df['Date'].apply(lambda dt: dt.year)
	annotated_df['UnixTimestamp'] = annotated_df['Date'].apply(lambda dt: time.mktime(dt.timetuple()))
	annotated_df.drop('Date', axis=1, inplace=True)

	print "Splitting train and test sets..."
	train_df, test_df = train_test_split(
		annotated_df,
		test_size=0.10,
		random_state=RANDOM_STATE
	)

	print "Training random forest..."
	random_forest = RandomForestRegressor(
		n_jobs=-1,  # Auto selects number of cores
		random_state=RANDOM_STATE,
		max_features="log2",
		n_estimators=10,
	).fit(
		X=train_df[train_df.columns.difference(['Sales'])],
		y=train_df['Sales'],
	)
	print "Feature importances:"
	pairs = zip(train_df.columns.difference(['Sales']), random_forest.feature_importances_)
	pairs.sort(key=lambda x: -x[1])
	for column, importance in pairs:
		print " ", column, importance

	print "Testing random forest..."
	predictions = random_forest.predict(
		X=test_df[test_df.columns.difference(['Sales'])],
	)
	print "Root mean square percentage:"
	print " ", root_mean_square_percentage(test_df['Sales'], predictions)
