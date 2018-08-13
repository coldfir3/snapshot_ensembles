rm(list = ls())
library(keras)
library(tidyverse)
keras::k_clear_session()

dataset <- dataset_boston_housing()
c(c(x_train, y_train), c(x_test, y_test)) %<-% dataset
rm(dataset)
save(x_train, y_train, x_test, y_test, file = 'dataset.data')

mean <- apply(x_train, 2, mean)
std <- apply(x_train, 2, sd)
save(mean, std, file = 'scale.data')

x_train <- scale(x_train, center = mean, scale = std)
x_test <- scale(x_test, center = mean, scale = std)

model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = dim(x_train)[2]) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1) %>% 
  compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae")
  )

# cyclic_cosine_annealing
build_cca_scheduler <- function(a0, epochs, steps){
  function(epoch, lr)
    a0/2 * (cos(pi * ((epoch - 1) %% (steps))/(steps)) + 1)
}
build_cb_saver <- function(epochs, steps, filepath){
  function(epoch, logs)
    if(epoch %in% seq(0, epochs, by = steps)[-1])
      save_model_hdf5(model, paste0(filepath, floor(epoch/steps)))
  
}

epochs = 200
steps = 40
lr0 = 0.1

cb_cca_scheduler <- build_cca_scheduler(lr0, epochs, steps)
cb_saver <- build_cb_saver(epochs, steps, 'models/model')
  
cb <- list(
  callback_learning_rate_scheduler(cb_cca_scheduler),
  callback_lambda(on_epoch_end = cb_saver)
)

history <- model %>% fit(
  x_train, y_train,
  epochs = epochs+1, batch_size=128,
  validation_data = list(x_test, y_test),
  callbacks = cb
)

models <- dir('models') %>% grep(glob2rx('model*'), .,value = TRUE) %>% paste('models', ., sep = '/') %>% map(load_model_hdf5)
results <- models %>% map(evaluate, x_test, y_test)
w <- results %>% map_dbl('mean_absolute_error') %>% (function(x) (1/x)/sum(1/x))

pred_ens <- function(x, models, w){
  models %>% map(predict_proba, x) %>% map2(w, function(a, b) a*b) %>% reduce(`+`) %>% as.vector
}
y_pred <- pred_ens(x_test, models, w)

results %>% map_dbl('mean_absolute_error')
mean(abs(y_pred - y_test))

