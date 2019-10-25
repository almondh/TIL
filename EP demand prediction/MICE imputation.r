# load packages --------------------
install.packages("mice")
install.packages("corrplot")
install.packages("pool")
install.packages("VIM")
library(mice)
library(corrplot)
library(pool)
library(VIM)

# 1. load data ---------------------
train_med <- read.csv("C:/Users/huiyeon/Desktop/Dacon/train_removeNaN_1.csv")  # (16909, 1301)
weather_pre <- read.csv("C:/Users/huiyeon/Desktop/Dacon/weather_knn-All.csv")  # (17088, 7)
#test <- read.csv("C:/Users/huiyeon/Desktop/Dacon/11회 data/test.csv")  # (8760, 201)

# Total NA
train_med_count_na <- apply(train_med,2,function(x)sum(is.na(x)))
train_med_count_na[1:50]


# 2. EDA ---------------------------
# 2-1. train_med
summary(train_med)
describe(train_med)
str(train_med)
head(train_med$Time)

# 2-2. weather_pre
summary(weather_pre)
describe(weather_pre)
str(weather_pre)

# 상관분석
# X, Time만 제외
weather_cor <- cor(weather_pre[, !names(weather_pre) %in% c('X', 'Time')], method='pearson')

# 시각화
corrplot(weather_cor, method='shade', shade.col=NA, tl.col='black', tl.srt=45)


# 3. Merge -------------------------
# X변수 제거
weather_pre <- weather_pre[, !names(weather_pre) %in% c('X')]

# Time to TimeSeries
weather_pre$Time <- as.POSIXlt(weather_pre[,1],format="%Y.%m.%d %H:%M")
train_med$Time <- as.POSIXlt(train_med[,1],format="%Y-%m-%d %H:%M")

# Merge 전에 list로 변환
train_med <- as.list(train_med)
weather_pre <- as.list(weather_pre)

# Merge
data <- merge(train_med, weather_pre, by='Time', all.x=T)
summary(data)
class(data)  # data.frame

# 모두 null값인 컬럼 지우기
null_col <- c('X1', 'X22', 'X24', 'X3', 'X34', 'X4', 'X45', 'X64')
data <- data[, !names(data) %in% null_col]

# After merge, Total count na 
Total_count_na <- apply(data,2,function(x)sum(is.na(x)))
Total_count_na[(length(data)-100):length(data)]

# 4. Mice --------------------------
# 100개 변수만 뽑아서 실행해 보기
data_test <- colnames(data)[1:10]
data_test_w <- colnames(data)[(length(data)-4): length(data)]
colname_test <- cbind(data_test, data_test_w)
mice_test <- data[, names(data) %in% colname_test]

# NA pattern 확인
md.pattern(mice_test)

aggr_plot <- aggr(mice_test, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(mice_test), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))


# Mice
# able : cart
# Unable : norm.boot, lda, pmm, norm, norm.predict, logreg, polyreg, rf, midastouch
# m defalut=5
imp <- mice(mice_test, m=10, method='cart',
            seed=1234)  # dataset을 5개 생성

# m=5 일 때
# X692으로 모형적합
fit_X692 <- with(imp,step(lm(X692~Temp+hum+rain+wind+snow))) # analysis는 lm(), glm(), gam(), nbrm() 등  
# fit에는 m개의 분석결과가 들어간다 
# stepwise 결과 X692~Temp+Wind+snow
fit_X692 <- with(imp,lm(X692~Temp+wind))
pooled = pool(fit_X692)  # pooled는 m개의 분석결과의 평균
summary(pooled)

imp  # imputaion inf
imp$imp  # 실제 대치값
result_1 <- imp$imp$X692  # m=5일 경우

# m=10 일 때
# X692으로 모형적합
fit_X692 <- with(imp,step(lm(X692~Temp+hum+rain+wind+snow))) # analysis는 lm(), glm(), gam(), nbrm() 등  
# fit에는 m개의 분석결과가 들어간다 
# stepwise 결과 X692~Temp+Wind+snow
fit_X692 <- with(imp,lm(X692~Temp+wind))
pooled = pool(fit_X692)  # pooled는 m개의 분석결과의 평균
summary(pooled)

imp  # imputaion inf
imp$imp  # 실제 대치값
result_2 <- imp$imp$X692  # m=5일 경우

# m=5일때, 10일 때
data.frame(result_1, result_2)


# X1272으로 모형적합
fit_X1272 <- with(imp,lm(X1272~Temp+hum+wind+snow))
pooled = pool(fit_X1272)
summary(pooled)

# 분석모형 유의성 진단
round(summary(pool(fit_X692)),2)  # 유의하지 않음

###### 결과적으로 원자료 값이 없어서 모형적합도는 확인하지 않기로 결정 #####

# 모든 데이터로 mice 실행하기
