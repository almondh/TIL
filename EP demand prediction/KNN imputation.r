# load packages --------------------
install.packages("psych")
install.packages("DMwR")
library(psych)
library(dplyr)
library(DMwR)
library(zoo)

# 1. load data ---------------------
# 1-1. weather
weather <- read.csv("C:/Users/huiyeon/Desktop/Dacon/11회 data 과거 인천 기상예측/hourly_weather.csv")
colnames(weather) <- c("place", "Time", "Temp", "rain", "wind", "hum", "snow", "condition", "cloud")
#지점, 일시, 기온, 강수량, 풍속, 습도, 적설, 날씨, 전운량

# 1-2. Elctric demand
train <- read.csv("C:/Users/huiyeon/Desktop/Dacon/11회 data/train.csv")  # (16909, 1301)
test <- read.csv("C:/Users/huiyeon/Desktop/Dacon/11회 data/test.csv")  # (8760, 201)


# 2. EDA ---------------------------
# 2-1. weather
summary(weather)
describe(weather)
weather <- weather[, c("Time", "Temp", "hum")]


# 2-2. Electirc demand
# ex) X692
summary(train$X692)
describe(train$X692)

# 3. KNN-Imputation ----------------
# 3-1. weather
# Temp, hum
sum(is.na(weather))  # 38
k <- round(sqrt(nrow(weather)))  # 적정 K값
weather_pre <- knnImputation(weather, k=k)
sum(is.na(weather_pre))  # 0


# 3-2. Electric demand
"""
# ERROR
which(is.na(train$X692)== F, arr.ind = TRUE)[1]  # 13646
which(is.na(train$X1272)== F, arr.ind = TRUE)[1]  # 13646
which(is.na(train$X553)== F, arr.ind = TRUE)[1]  # 13646
# ex) X692, X1272
f1 <- which(is.na(train$X692)== F, arr.ind = TRUE)[1]  # first not null
f2 <- which(is.na(train$X1272)== F, arr.ind = TRUE)[1]  # first not null
f3 <- which(is.na(train$X553)== F, arr.ind = TRUE)[1]  # first not null

a <- data.frame(train$X692[f1:length(train$X692)],
                train$X1272[f2:length(train$X1272)],train$X553[f3:length(train$X553)])
colnames(a) <- c('X692', 'X1272', 'X553')  # 열 이름 지정
head(a)

sum(is.na(a))  # 622
k <- round(sqrt(nrow(a)))  # 적정 K값
a_pre <- knnImputation(a)
"""
# 4. To csv ------------------------
write.csv(weather_pre, "C:/Users/huiyeon/Desktop/Dacon/weather_knn.csv")
