---
title: "Practical Machine Learning Coursera Peer Assessment"
output:
  html_document:
    highlight: pygments
    keep_md: yes
    theme: united
  pdf_document:
    highlight: zenburn
date: "11-Oct-19"
---

## Summary

This report uses machine learning algorithms to predict the manner in which users of exercise devices exercise. 


### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website [here:](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset). 

### Data 

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


### Set the work environment and knitr options


```r
rm(list=ls(all=TRUE)) #start with empty workspace
startTime <- Sys.time()
library(knitr)
opts_chunk$set(echo = TRUE, cache= TRUE, results = 'hold')
```

### Load libraries and Set Seed

Load all libraries used, and setting seed for reproducibility. *Results Hidden, Warnings FALSE and Messages FALSE*


```r
library(ElemStatLearn)
library(caret)
library(rpart)
library(randomForest)
library(RCurl)
set.seed(2014)
```

### Load and prepare the data and clean up the data




Load and prepare the data


```r
trainingLink <- getURL("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
pml_CSV  <- read.csv(text = trainingLink, header=TRUE, sep=",", na.strings=c("NA",""))
pml_CSV <- pml_CSV[,-1] # Remove the first column that represents a ID Row
```

### Data Sets Partitions Definitions

Create data partitions of training and validating data sets.


```r
inTrain = createDataPartition(pml_CSV$classe, p=0.60, list=FALSE)
training = pml_CSV[inTrain,]
validating = pml_CSV[-inTrain,]
# number of rows and columns of data in the training set
dim(training)
# number of rows and columns of data in the validating set
dim(validating)
```

```
## [1] 11776   159
## [1] 7846  159
```
## Data Exploration and Cleaning

Since we choose a random forest model and we have a data set with too many columns, first we check if we have many problems with columns without data. So, remove columns that have less than 60% of data entered.


```r
# Number of cols with less than 60% of data
sum((colSums(!is.na(training[,-ncol(training)])) < 0.6*nrow(training)))
```

[1] 100

```r
# apply our definition of remove columns that most doesn't have data, before its apply to the model.
Keep <- c((colSums(!is.na(training[,-ncol(training)])) >= 0.6*nrow(training)))
training   <-  training[,Keep]
validating <- validating[,Keep]
# number of rows and columns of data in the final training set
dim(training)
```

[1] 11776    59

```r
# number of rows and columns of data in the final validating set
dim(validating)
```

[1] 7846   59

## Modeling
In random forests, there is no need for cross-validation or a separate test set to get an unbiased estimate of the test set error. It is estimated internally, during the execution. So, we proceed with the training the model (Random Forest) with the training data set.


```r
model <- randomForest(classe~.,data=training)
print(model)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.18%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3348    0    0    0    0 0.000000000
## B    3 2276    0    0    0 0.001316367
## C    0    6 2046    2    0 0.003894839
## D    0    0    6 1923    1 0.003626943
## E    0    0    0    3 2162 0.001385681
```

### Model Evaluate
And proceed with the verification of variable importance measures as produced by random Forest:


```r
importance(model)
```

```
##                      MeanDecreaseGini
## user_name                  89.9838066
## raw_timestamp_part_1      944.3592642
## raw_timestamp_part_2       10.5709278
## cvtd_timestamp           1444.2381695
## new_window                  0.2115355
## num_window                527.5040554
## roll_belt                 534.4457545
## pitch_belt                274.9582972
## yaw_belt                  335.4510997
## total_accel_belt          104.4310640
## gyros_belt_x               38.8372560
## gyros_belt_y               48.8986671
## gyros_belt_z              116.3904437
## accel_belt_x               61.4676246
## accel_belt_y               63.4971696
## accel_belt_z              184.5359154
## magnet_belt_x             113.0755151
## magnet_belt_y             195.4956309
## magnet_belt_z             199.0961402
## roll_arm                  128.8274451
## pitch_arm                  56.4138371
## yaw_arm                    74.4500201
## total_accel_arm            28.6137214
## gyros_arm_x                42.8256567
## gyros_arm_y                46.0857273
## gyros_arm_z                19.4473446
## accel_arm_x                99.4387262
## accel_arm_y                49.0229689
## accel_arm_z                40.0509570
## magnet_arm_x              107.5643699
## magnet_arm_y               80.7814716
## magnet_arm_z               56.8804904
## roll_dumbbell             200.7490798
## pitch_dumbbell             78.2348469
## yaw_dumbbell              112.2073274
## total_accel_dumbbell      116.6781721
## gyros_dumbbell_x           43.0507306
## gyros_dumbbell_y          110.5035499
## gyros_dumbbell_z           24.6446127
## accel_dumbbell_x          124.1001418
## accel_dumbbell_y          185.6873624
## accel_dumbbell_z          130.5843879
## magnet_dumbbell_x         242.6537492
## magnet_dumbbell_y         301.9486969
## magnet_dumbbell_z         297.2671681
## roll_forearm              232.6033122
## pitch_forearm             297.2715845
## yaw_forearm                51.0542722
## total_accel_forearm        30.8366541
## gyros_forearm_x            25.6667959
## gyros_forearm_y            39.6384180
## gyros_forearm_z            25.1370627
## accel_forearm_x           136.1516560
## accel_forearm_y            45.1717342
## accel_forearm_z            96.4929645
## magnet_forearm_x           69.8675218
## magnet_forearm_y           78.6690536
## magnet_forearm_z           94.8622608
```

Now we evaluate our model results through confusion Matrix.


```r
confusionMatrix(predict(model,newdata=validating[,-ncol(validating)]),validating$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B    0 1518    5    0    0
##          C    0    0 1362    1    0
##          D    0    0    1 1285    2
##          E    0    0    0    0 1440
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9989          
##                  95% CI : (0.9978, 0.9995)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9985          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   0.9956   0.9992   0.9986
## Specificity            1.0000   0.9992   0.9998   0.9995   1.0000
## Pos Pred Value         1.0000   0.9967   0.9993   0.9977   1.0000
## Neg Pred Value         1.0000   1.0000   0.9991   0.9998   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1935   0.1736   0.1638   0.1835
## Detection Prevalence   0.2845   0.1941   0.1737   0.1642   0.1835
## Balanced Accuracy      1.0000   0.9996   0.9977   0.9994   0.9993
```

And confirmed the accuracy at validating data set by calculate it with the formula:


```r
accuracy <-c(as.numeric(predict(model,newdata=validating[,-ncol(validating)])==validating$classe))
accuracy <-sum(accuracy)*100/nrow(validating)
```

Model Accuracy as tested over Validation set = **99.9%**.  

### Model Test

Finally, we proceed with predicting the new values in the testing csv provided, first we apply the same data cleaning operations on it and coerce all columns of testing data set for the same class of previous data set. 

#### Getting Testing Dataset


```r
testingLink <- getURL("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
pml_CSV  <- read.csv(text = testingLink, header=TRUE, sep=",", na.strings=c("NA",""))
pml_CSV <- pml_CSV[,-1] # Remove the first column that represents a ID Row
pml_CSV <- pml_CSV[ , Keep] # Keep the same columns of testing dataset
pml_CSV <- pml_CSV[,-ncol(pml_CSV)] # Remove the problem ID
# Apply the Same Transformations and Coerce Testing Dataset
# Coerce testing dataset to same class and strucuture of training dataset 
testing <- rbind(training[100, -59] , pml_CSV) 
# Apply the ID Row to row.names and 100 for dummy row from testing dataset 
row.names(testing) <- c(100, 1:20)
```

#### Predicting with testing dataset


```r
predictions <- predict(model,newdata=testing[-1,])
print(predictions)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```




```r
endTime <- Sys.time()
```
The analysis was completed on Fri Oct 11 5:55:20 PM 2019  in 0 seconds.
