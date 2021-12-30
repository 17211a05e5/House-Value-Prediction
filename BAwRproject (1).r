#Load required libraries
library(corrplot)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(neuralnet)
library(gplots)
library(Amelia)
library(forecast)
library(Hmisc)


#read the Paris Housing dataset
housing.df <- read.csv("ParisHousingClass.csv")

#replacing category variable that contains 'Basic' with 0 and 'Luxury' with 1
housing.df$category <- factor(housing.df$category,levels = c("Basic", "Luxury"), labels = c(0, 1))

#performing EDA
str(housing.df)
summary(housing.df)
dim(housing.df)
sapply(housing.df,class)

#create a correlation plot
corrplot(round(cor(housing.df[,-18]),2))

rcorr(as.matrix(housing.df),type='spearman')

hist.data.frame(housing.df)

#create a heat map
heatmap.2(cor(housing.df[,-18]), Rowv = FALSE, Colv = FALSE, dendrogram = "none", 
          cellnote = round(cor(housing.df[,-18]),2), 
          notecol = "black", key = FALSE, trace = 'none', margins = c(10,10))

#create a missingness map
missmap(housing.df)


# partition data to training and validation datasets
set.seed(2)
train.index <- sample(c(1:dim(housing.df)[1]), dim(housing.df)[1]*0.8)

train.df <- housing.df[train.index, ]
valid.df <- housing.df[-train.index, ]

#price prediction using random forest
rf<-randomForest(train.df$price~.,data=train.df,ntree=500,mtry=4,nodesize=5,importance=TRUE)
varImpPlot(rf, type = 1)

rf.pred <- predict(rf, valid.df)
data.frame(actual =rf.pred[1:10], valid.df$price[1:10])
accuracy(rf.pred, valid.df$price)


#category prediction using logistic regression
logit.reg <- glm(category ~ ., data = train.df, family = "binomial") 
options(scipen=999)
summary(logit.reg)

logit.reg.pred <- predict(logit.reg, valid.df[,-18], type = "response")
data.frame(actual = valid.df$category[1:100], predicted = logit.reg.pred[1:100])

confusionMatrix(as.factor(round(logit.reg.pred,digits=0)),valid.df$category)


#category prediction using decision tree
default.ct <- rpart(category ~ ., data = train.df ,method = "class")
prp(default.ct, type = 1, extra = 2, under = TRUE, split.font = 1, varlen = -10)

default.ct.point.pred.train <- predict(default.ct,valid.df,type = "class")
data.frame(actual = valid.df$category[1:10],default.ct.point.pred.train[1:10])
length(default.ct$frame$var[default.ct$frame$var == "<leaf>"])

confusionMatrix(default.ct.point.pred.train, valid.df$category)


#category prediction using neural network
nn <- neuralnet(category~., data = train.df, linear.output = F, hidden = 3)
nn$weights
plot(nn, rep="best")

x<- valid.df[,-18]
p<-predict(nn,x)

confusionMatrix(as.factor(round(p[,2],digits=0)),valid.df$category)


