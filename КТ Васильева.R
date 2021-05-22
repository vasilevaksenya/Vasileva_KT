data(package = .packages(all.available = TRUE)) #выберем базу данных


install.packages('MASS')
library('MASS')
View(cats) #база состоит из 144 наблюдений об анатомических данных домашних котиков
#она включает в себя 3 столбца - пол кота, вес тела и вес сердца
#Построим нейронку, определяющую пол котика - male/famale

str(data) #посмотрим тип данных и замечаем, что столбец Sex имеет тип данных Factor,
#наша задача - преобразовать тип данных в int, так как с такой формой не работает пакет neuralnet
set.seed(20)
data <- cats
View(data)
data$Sex <- as.numeric(data$Sex)
str(data)

#строим нейронную сеть Neuralnet

max_data <- apply(data, 2, max)
min_data <- apply(data, 2, min)
head(data)

data_scaled <- scale(data, center = min_data, scale = max_data - min_data)
View(data_scaled)

index <- sample(1:nrow(data), round(0.80*nrow(data)))

train_data <- as.data.frame(data_scaled[index,])
test_data <- as.data.frame(data_scaled[-index,])

n <- colnames(data)
f <- as.formula(paste('Sex ~', paste(n[!n %in% 'Sex'], collapse = '+')))

library('neuralnet')
#Создадим нейронную сеть по тренировочным данным, в которых 3 скрытых слоя (9, 7 и 8)
n_net1 <- neuralnet(f, data = train_data, hidden = c(9, 7, 8), linear.output = F)
plot(n_net1)

predicted <- compute(n_net1, test_data[2:3])
print(predicted$net.result)


predicted$net.result <- sapply(predicted$net.result, round, digits = 0)

test1 <- table(test_data$Sex, predicted$net.result)

table(test_data$low)

sum(test1[1,])
sum(test1[2,])

# точность модели равна 0,62
Accuracy1 <- (test1[1,1] + test1[2, 2])/sum(test1)
Accuracy1

#строим нейронную сеть RSNNS
install.packages('RSNNS')
library(RSNNS)
set.seed(20)

data2 <- cats[sample(1:nrow(cats), length(1:nrow(cats))),
              1:ncol(cats)]
data2$Sex <- as.numeric(data2$Sex)

data2_Values <- data2[, 2:3]
data2_Target <- data2[, 1]

data2 <- splitForTrainingAndTest(data2_Values, data2_Target, ratio = 0.2)
data2 <- normTrainingAndTestSet(data2)

model <- mlp(data2$inputsTrain,
             data2$targetsTrain,
             size = 5,
             maxit = 50,
             inputsTest = data2$inputsTest,
             targetsTest = data2$targetsTest)

test2 <- confusionMatrix(data2$targetsTrain, encodeClassLabels(fitted.values(model),
                                                               method = "402040", l = 0.5, h = 0.51))
test2

sum(test2[1,])
sum(test2[2,])

Accuracy2 <- (test2[1, 1])/sum(test2)
Accuracy2
#Точность модели 0.32

#строим нейронную сеть Кохонена

install.packages("kohonen")
library('kohonen')
set.seed(20)

data3 <- cats[2:3]
data3_1 <- cats[1:1]
data3_1$Sex <- as.numeric(data3_1$Sex)
table(data3_1)

train <- sample(nrow(data3), 115)
X_train <- scale(data3[train,])
X_test <- scale(data3[-train,],
                center = attr(X_train, "scaled:center"),
                scale = attr(X_train, "scaled:center"))
train_data <- list(measurements = X_train,
                   data3_1 = data3_1[train,])
test_data <- list(measurements = X_test,
                  data3_1 = data3_1[-train,])

mygrid <- somgrid(5, 5, 'hexagonal')
som.data3 <- supersom(train_data, grid = mygrid)
som.predict <- predict(som.data3, newdata = test_data)


test3 <- table(data3_1[-train,], som.predict$predictions$data3_1)

sum(test3[1,])
sum(test3[2,])
test3

#Оценим точность модели
Accuracy3 <- (test3[1,1] + test3[2, 2])/sum(test3)
Accuracy3
#Точность модели 1.0

a <- Accuracy1*100
b <- Accuracy2*100
c <- Accuracy3*100


#Построим сводную таблицу с данными в процентах
end_matrix <- cbind(a, b, c)
rownames(end_matrix) <- c("Точность")
colnames(end_matrix) <- c("NeuralNet", "RSNNS", "Kohonen")
View(end_matrix)

#Итого библиотека Кохонена предложила лучшую нейронную сеть после обучения,
#точность в ней равна 100%, но наверное это что-то странное
#В любом случае, библиотека Neuralnet дает лучшие результаты, чем RSNNS