# load packages -------------
install.packages("psych")
install.packages("DMwR")
library(psych)
library(DMwR)

# load data -----------------
weather <- read.csv("C:/Users/huiyeon/Desktop/Dacon/11회 data 과거 인천 기상예측/hourly_weather.csv")
colnames(weather) <- c("place", "Time", "Temp", "rain", "wind", "hum", "snow", "condition", "cloud")
#지점, 일시, 기온, 강수량, 풍속, 습도, 적설, 날씨, 전운량

# EDA -----------------------
summary(weather)
describe(weather)
weather <- weather[, c("Time", "Temp", "hum")]

# KNN-Imputation ------------
# Temp, hum
sum(is.na(weather))  # 38
k <- round(sqrt(nrow(weather)))  # 적정 K값
weather_pre <- knnImputation(weather, k=k)
sum(is.na(weather_pre))  # 0

# To csv
write.csv(weather_pre, "C:/Users/huiyeon/Desktop/Dacon/weather_knn.csv")
