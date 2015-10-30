# Based on Ben Hamner script from Springleaf
# https://www.kaggle.com/benhamner/springleaf-marketing-response/random-forest-example

library(readr)
library(randomForest)
library(data.table)
library(lubridate)
library(forecast)

set.seed(123)

train <- read_csv("data/train.csv")
test  <- read_csv("data/test.csv")
store <- read_csv("data/store.csv")

# Add Year, Month, Week and Day to trainingset
train = as.data.table(train)
train$day = day(as.Date(train$Date,"%y/%m/%d"))
train$week = week(as.Date(train$Date,"%y/%m/%d"))
train$month = month(as.Date(train$Date,"%y/%m/%d"))
train$year = year(as.Date(train$Date,"%y/%m/%d"))

# Investigate the number of days stores are open
days_open = train[,list(count = sum(Open)), by = list(Store)]

hist(days_open$count)
plot(days_open$Store, days_open$count)

# Investigate the number of days stores are open
# write.csv(as.data.frame(table(days_open$count)), file='days_open.csv')

######## Funky timeseries stuff ########
i = 85
specific_store = train[Store == i]
# store = train[Store == i & Year > 2014]
plot(specific_store$Date, specific_store$Sales, type='line')

sales_timeseries = ts(specific_store$Sales, frequency=365, start=c(2015,1))
sales_timeseries_components = decompose(sales_timeseries)
plot(sales_timeseries_components)

sales_timeseries_forecast = HoltWinters(sales_timeseries)
plot(sales_timeseries_forecast)
sales_timeseries_forecast2 <- forecast.HoltWinters(sales_timeseries_forecast, h=45)
plot(sales_timeseries_forecast2)
sales_timeseries_forecast2$mean
######## End of funky timeseries stuff ########

# Random Forest model

train <- read_csv("data/train.csv")
test  <- read_csv("data/test.csv")
store <- read_csv("data/store.csv")

train <- merge(train,store)
test <- merge(test,store)

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0

# looking at only stores that were open in the train set
train <- train[ which(train$Open=='1'),]

# seperating out the elements of the date column for the train set
train$month <- as.integer(format(train$Date, "%m"))
train$year <- as.integer(format(train$Date, "%y"))
train$day <- as.integer(format(train$Date, "%d"))

# removing the date column from train (since elements are extracted) and also StateHoliday which has a lot of NAs
train <- train[,-c(3,8)]
train$StoreType.Assortment=paste(train$StoreType,train$Assortment)

# seperating out the elements of the date column for the test set
test$month <- as.integer(format(test$Date, "%m"))
test$year <- as.integer(format(test$Date, "%y"))
test$day <- as.integer(format(test$Date, "%d"))

# removing the date column from test (since elements are extracted) and also StateHoliday which has a lot of NAs
test <- test[,-c(4,7)]
test$StoreType.Assortment=paste(test$StoreType,test$Assortment)

feature.names <- names(train)[c(1,2,5:20)]

for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

clf <- randomForest(train[,feature.names],
                    log(train$Sales+1),
                    ntree=17,
                    sampsize=200000,
                    do.trace=TRUE)

pred <- exp(predict(clf, test)) -1
submission <- data.frame(Id=test$Id, Sales=pred)

write_csv(submission, "rf1.csv")
