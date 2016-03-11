---
title: "Classification of five behaviors based on quantified self movement data"
author: "Yiyun"
date: "March 11, 2016"
output: html_document
---
###Introduction
#####Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

###Contents
####1. How to build the model
####2. How to use the cross-validation
####3. Estemated out of sample error
####4. Compare the prediction accuracy of different models
####5. Validation of the predicted results using test data

####Initiliazation: update your variables here

```{r, cache=TRUE}
library(lattice)
library(ggplot2)
library(caret)
library(rpart)
library(rattle)
library(MASS)
library(mlbench)
library(class)
library(randomForest)
library(klaR)
Training_Link <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
Test_Link <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
Traning_File_Name <- "pml-training.csv"
Test_File_Name <- "pml-testing.csv"
DataFolder <- "./Human Activity Project/" # folder where all your data are saved
```

####Data Acquisition: all data are saved under the DataFolder

```{r, cache=TRUE}
if (!file.exists(DataFolder)) dir.create(DataFolder)
setwd(DataFolder)
if (!file.exists(Traning_File_Name)) download.file(url = Training_Link, destfile = paste(DataFolder, Traning_File_Name, sep = ""))
if (!file.exists(Test_File_Name)) download.file(url = Test_Link, destfile = paste(DataFolder, Test_File_Name, sep = ""))
Training_Data <- read.csv(file = Traning_File_Name)
Test_Data <- read.csv(file = Test_File_Name)
```

####Preprocessing 1: remove non-features and transform data to numeric type

```{r, cache=TRUE}
Training_Data[, 7:159] <- sapply(Training_Data[, 7:159], as.numeric)
Test_Data[, 7:159] <- sapply(Test_Data[, 7:159], as.numeric)
Training_Data <- Training_Data[8:160]
Test_Data <- Test_Data[8:160]
```

####Preprocessing 2: remove NA features
```{r, cache=TRUE}
nas <- is.na(apply(Test_Data,2,sum))
Test_Data <- Test_Data[,!nas]
dim(Test_Data)
Training_Data <- Training_Data[,!nas]
```

####Create crossvalidation set in variable validation, the rest is training set in variable buildData
####70% of the training data will be used to train the model, 30% will be used for prediction and estimate
####the out of sample error. 

```{r, cache=TRUE}
inBuild <- createDataPartition(Training_Data$classe, p = 0.7, list = FALSE)
validation <- Training_Data[-inBuild[,1],]
buildData <- Training_Data[inBuild[,1],]
```

####Rank feature importance and select important features for model training
####In this case, there are less than 20 features 

```{r, cache=TRUE}
model <- train(classe~., data = buildData, method = "rpart")
importance <- varImp(model, scale = FALSE)
print(importance, top = 15)
plot(importance, top = 15)
```

```{r, cache=TRUE}
ImpVariables <- c("pitch_forearm",         
                  "roll_forearm",         
                  "roll_belt",             
                  "magnet_dumbbell_y",     
                  "accel_belt_z",          
                  "magnet_belt_y",          
                  "yaw_belt",               
                  "magnet_dumbbell_z",      
                  "total_accel_belt",       
                  "magnet_arm_x",           
                  "accel_arm_x",            
                  "roll_dumbbell",          
                  "accel_dumbbell_y",     
                  "magnet_dumbbell_x",
                  "total_accel_dumbbell",
                  "pitch_belt",
                  "accel_dumbbell_x",
                  "accel_forearm_x")

important_features <- buildData[, colnames(buildData) %in% c(ImpVariables, "classe")]
validation_features <- validation[, colnames(validation) %in% c(ImpVariables, "classe")]
```

####Use selected features to 1) train the the models and predict the outcome with validation dataset
#### Models used: 1) Recursive partitioning 2) linear discriminant analysis 3) random forest

1) Recursive partitioning 

```{r, cache=TRUE}
modFit_rpart <- train(classe~., method = "rpart", data = important_features)
fancyRpartPlot(modFit_rpart$finalModel)
pred_Rpart <- predict(modFit_rpart, validation_features)
Matrix_Rpart <- confusionMatrix(pred_Rpart, validation$classe)
```

2) linear discriminant analysis

```{r, cache=TRUE}
modFit_lda <- train(classe~., method = "lda", data = important_features, prox = TRUE)
pred_lda <- predict(modFit_lda, newdata = validation_features)
Matrix_lda <- confusionMatrix(pred_lda, validation$classe)
```

3) random forest

```{r, cache=TRUE}
modFit_rf <- train(classe~., method = "rf", data = important_features)
pred_rf <- predict(modFit_rf, newdata = validation_features)
Matrix_rf <- confusionMatrix(pred_rf, validation$classe)
```


####Now we compare the prediction accuracy of out of sample errors among the 3 different models below.
####It is clear that the random forest model has the lowest out of smaple errors and is selected as the
####optimal model to predict the test dataset

1) Recursive partitioning 
```{r, cache=TRUE}
Matrix_Rpart
```

2) Linear discriminant analysis
```{r, cache=TRUE}
Matrix_lda
```

3) random forest
```{r, cache=TRUE}
Matrix_rf
```


####Prediction outcome using the test dataset
```{r, cache=TRUE}
predict(modFit_rf, newdata = Test_Data)
```

###Conclusion
####After comparing four different types of models, our result shows random forest provides the most accurate outcome and reached 100% prediction rate in the test dataset. 

####Among 153 features collected, we ranked and chose the most important 18 features and fed them the training models. By doing this, I believe the out of sample error is reduced, as well as the computational intensity. 

####However, the random forest require a lot more time for training than the others and won't be ideal for a quick prediction. Our results are limited to four models due to the limited time and future work should focus on other models to test both accuracy and efficiency. 




