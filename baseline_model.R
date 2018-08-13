rm(list = ls())

library(keras)
library(tidyverse)
keras::k_clear_session()

mnist <- dataset_mnist()
c(c(x_test, y_test), c(x_train, y_train)) %<-% mnist # inverted train and test set to increase dificulty

x_train <- array_reshape(x_train/255, c(10000, 28, 28, 1))
x_test <- array_reshape(x_test/255, c(60000, 28, 28, 1))
y_train <- to_categorical(y_train)
y_test <- to_categorical(y_test)

model <- keras_model_sequential() %>%
  layer_separable_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_separable_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_separable_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 100, batch_size=128,
  validation_data = list(x_test, y_test)
)

results <- model %>% evaluate(x_test, y_test)
print(results)
