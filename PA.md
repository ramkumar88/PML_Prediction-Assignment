Predicting exercise activity
========================================================
##### Author: Ramkumar Bommireddipalli
##### Published: July 24, 2014


Synopsis
========================================================
In this report our goal is to predict the manner in which a person exercises based on the acvitiy data

To perform the predictions, we obtained the test and training data from the groupware human activity recognition [http://groupware.les.inf.puc-rio.br/har]

```r
library(caret)
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```


```r
dataDir <- "data"
trainingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
trainingCSV <- file.path(dataDir,"pml-training.csv")
testingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
testingCSV <- file.path(dataDir,"pml-testing.csv")
if (!file.exists(dataDir)){
    dir.create(file.path(dataDir))
}
## read csv if raw data is not defined
if (!file.exists(trainingCSV)){
    download.file(trainingUrl, destfile = trainingCSV, method="curl")
}
## read csv if raw data is not defined
if (!file.exists(testingCSV)){
    download.file(testingUrl, destfile = testingCSV, method="curl")
}
set.seed(13343)
## Load the data from csv
trainingData <- read.csv(trainingCSV)
testingData <- read.csv(testingCSV)
```

Cleaning and Exploring Data
========================================================

```r
## Remove all columns with NA
trainingData <- trainingData[,colSums(is.na(trainingData))==0]
## Remove near zero variance data from training
trainingData <- trainingData[,-nearZeroVar(trainingData)]

## convert factors to indicator variables on training
trainingData.dummyvars <- dummyVars(classe ~ ., data=trainingData)
trainingData.forest <- as.data.frame(predict(trainingData.dummyvars, newdata=trainingData))
trainingData.forest <- cbind(trainingData$classe,trainingData.forest)
colnames(trainingData.forest)[1] <- "classe"

## Remove near zero variance data from the new indicator variables
trainingData <- trainingData[,-nearZeroVar(trainingData)]

## Split training to training and validation
inTrain <- createDataPartition(y=trainingData.forest$classe,p=0.70,list=FALSE)
validationData <- trainingData.forest[-inTrain,]
subTrainingData <- trainingData.forest[inTrain,]

## Get the most correlated variables
variableCorrelations <-  abs(cor(subTrainingData[,-1]))
## Clear out the diagnoal
diag(variableCorrelations) <- 0
## Get the most correlations
highestCorVariables <- which(variableCorrelations > 0.9,arr.ind=T)
## Get the unique list of variables with highest correlations
topCorVariables <- paste(unique(names(highestCorVariables[,1])),collapse = " + ")
## Build the formula for training
variableFormula <- paste("classe ~ ",topCorVariables)
variableFormula
```

```
## [1] "classe ~  pitch_belt + accel_belt_x + magnet_dumbbell_x + magnet_dumbbell_y + total_accel_belt + accel_belt_y + accel_belt_z + user_name.adelmo + roll_belt + gyros_arm_y + gyros_arm_x + gyros_dumbbell_z + gyros_forearm_z + gyros_dumbbell_x + user_name.pedro"
```

