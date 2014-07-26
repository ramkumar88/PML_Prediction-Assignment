Predicting exercise activity
========================================================
##### Author: Ramkumar Bommireddipalli
##### Published: July 25, 2014


Synopsis
========================================================
In this report our goal is to predict the manner in which a person exercises based on the acvitiy data.

To perform the predictions, we obtained the test and training data from the groupware human activity recognition [http://groupware.les.inf.puc-rio.br/har]. The random forest prediction model was selected to get highly accurate predictions.

Initial Setup
========================================================
The caret library is required for the method used to fit a training model and test predictions. The random forest method was selected to fit the model for highly accurate predictions.

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

Getting Data
========================================================
The  testing and training csv files were obtained from the groupware human activity dataset. The csv files are then loaded into training and testing data frames as show below.

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
## Load the data from csv
trainingData <- read.csv(trainingCSV)
testingData <- read.csv(testingCSV)
```


To better validate the trained modal, the training data was split further into a sub-training and validation data set.

```r
# set seed
set.seed(13343)

## Split training to training and validation
inTrain <- createDataPartition(y=trainingData$classe,p=0.70,list=FALSE)
validationData <- trainingData[-inTrain,]
subTrainingData <- trainingData[inTrain,]
```

Cleaning & Pre-processing Data
========================================================
Prior to training a model, it is important to pre-process the training data set to focus on the important variables that have the biggest effect on prediction. The initial training data had over 159 variables. The first pre-process step is to remove variables variables with NA values and with near zero variance. This will remove variables that do not contribute much to the prediction of the model.

```r
## Remove all columns with NA
subTrainingData <- subTrainingData[,colSums(is.na(subTrainingData))==0]
## Remove near zero variance data from training
subTrainingData <- subTrainingData[,-nearZeroVar(subTrainingData)]
```

The next step of the analysis is to transform the data to improve the training predictions. The following code looks for the different classes of variables in the training data set.

```r
variableTypes <- unique(lapply(subTrainingData, class))
variableTypes
```

```
## [[1]]
## [1] "integer"
## 
## [[2]]
## [1] "factor"
## 
## [[3]]
## [1] "numeric"
```

As we can see, there are some factor class variables in the training set. These need to be transformed to indicator variables to be considerd for training the model. The following code converts the factor variables to indicator variables.

```r
## convert factors to indicator variables on training
subTrainingData.dummyvars <- dummyVars(classe ~ ., data=subTrainingData)
subTrainingData.forest <- as.data.frame(predict(subTrainingData.dummyvars, newdata=subTrainingData))
subTrainingData.forest <- cbind(subTrainingData$classe,subTrainingData.forest)
colnames(subTrainingData.forest)[1] <- "classe"
subTrainingData <- subTrainingData.forest
```

After removing the NA, near zero variance variables and transforming the factor variables, there are 82 variables. This reduced the total variables by about 48.43%.


Feature selection
========================================================
The number of variables can be further reduced to the specific features that have the highest predictive value. This can be achieved by identifying the variables with a high co-variance. In this case a criteria of co-variance of > 0.8 was applied using the cor method.

```r
## Get the most correlated variables
variableCorrelations <-  abs(cor(subTrainingData[,-1]))
## Clear out the diagnoal
diag(variableCorrelations) <- 0
## Get the variables with high correlations > 0.80
highestCorVarNames<- which(variableCorrelations > 0.80,arr.ind=T)
## Get the unique list of variables
highestCorVarNames <- unique(names(highestCorVarNames[,1]))
```

After selecting the highly co-variate features, there are 28 variables. This is a total reduction of 82.39% from the original set of variables. These training variables are then extracted from the sub training data set to train the prediction model.

```r
## Subset the specific variables from training
bestTrainingVariables <- subTrainingData[ , which(names(subTrainingData) %in% highestCorVarNames)]
```


Training the model
========================================================
The best training variables are then to used to train a random forest model. The resulting model has close to 100% accuracy as shown in the details below.

```r
#fit the model
modelFit <- train(subTrainingData$classe~ .,method="rf",data=bestTrainingVariables,trControl=trainControl(method = 'cv'))

## see model details
modelFit
```

```
## Random Forest 
## 
## 13737 samples
##    28 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 12362, 12364, 12361, 12363, 12363, 12363, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.004        0.005   
##   20    1         1      0.004        0.005   
##   30    1         1      0.006        0.007   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

Validating the Model
========================================================
The next step is to predict the validation data and verify the accuracy of the model. Since the validation data also has some factor columns, there is some cleanup performed on the validation data.

```r
## Remove all columns with NA
validationData <- validationData[,colSums(is.na(validationData))==0]
## Remove near zero variance data from training
validationData <- validationData[,-nearZeroVar(validationData)]

## convert factors to indicator variables on training
validationData.dummyvars <- dummyVars(classe ~ ., data=validationData)
validationData.forest <- as.data.frame(predict(validationData.dummyvars, newdata=validationData))
validationData.forest <- cbind(validationData$classe,validationData.forest)
colnames(validationData.forest)[1] <- "classe"
validationData <- validationData.forest

## check for accuracy on validation data
validationPred <- predict(modelFit,validationData)
```

The following table shows the quality of predictions on the validation data set, the high values along the diagonal indicate that our model is highly accurate.

```r
## Show accuracy on validation data set
table(validationPred,validationData$classe)
```

```
##               
## validationPred    A    B    C    D    E
##              A 1674   17    1    1    1
##              B    0 1114   10    0    1
##              C    0    8 1013   37    1
##              D    0    0    2  926    4
##              E    0    0    0    0 1075
```


Testing the Model
========================================================
The final step is to predict the testing data using the model trained from before. Since the testing data also has some factor columns, there is some cleanup performed on the testing data.

```r
## Remove all columns with NA
testingData <- testingData[,colSums(is.na(testingData))==0]
## Remove near zero variance data from training
testingData <- testingData[,-nearZeroVar(testingData)]

## convert factors to indicator variables on training
testingData.dummyvars <- dummyVars( ~ ., data=testingData)
testingData.forest <- as.data.frame(predict(testingData.dummyvars, newdata=testingData))
testingData <- testingData.forest

## check for accuracy on validation data
testingPred <- predict(modelFit,testingData)

## Show predictions from the testing data set
testingPred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

The following method is used to generate the files with responses. These files were submitted and yielded 100% correct results.

```r
testResultDir <- "test_results"

## create test results directory
if (!file.exists(testResultDir)){
    dir.create(file.path(testResultDir))
}

## declare method to write each result to a file
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = file.path(testResultDir,paste0("problem_id_",i,".txt"))
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
## call method to write the reults
pml_write_files(testingPred)
```
