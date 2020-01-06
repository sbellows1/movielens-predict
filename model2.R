library(dplyr)
library(caret)

load(file = 'rda/train.rda')

#Creating datasets
set.seed(1, sample.kind = 'Rounding')
test_index <- createDataPartition(y = train$rating, times = 1, p = 0.2, list = FALSE)
test0 <- train[test_index,]
train <- train[-test_index,]

##Making sure only users and movies with data in the training set appear in the test set.
test <- test0 %>% semi_join(train, by = 'userId') %>% semi_join(train, by = 'movieId')
removed <- anti_join(test0, test)
train <- rbind(train, removed)

#Create default prediction
grandmean <- mean(train$rating)

##Use to evaluate models
RMSE <- function(actual, predictions){
  se <- sum((predictions - actual)**2)
  ret <- sqrt(se/length(actual))
  ret
}

test$def_prediction <- seq(from = grandmean, to = grandmean, length.out = length(test$rating))

model1 <- RMSE(test$rating, test$def_prediction)

##The default (just guessing) RMSE is 1.04, which is a benchmark to beat.

###################################################################################################
##Statistical Modeling
###################################################################################################

user_bias <- train %>% group_by(userId) %>% summarize(u_bias = mean(rating - grandmean))

test$user_prediction <- left_join(test, user_bias, by = 'userId') %>% .$u_bias + grandmean

model2 <- RMSE(test$rating, test$user_prediction)

##By adjusting for user-specific effect, the RMSE on the test set drops to .93
##Now we must add a movie-specific effect while keeping the user effect in mind.
train <- left_join(train, user_bias, by = 'userId')
movie_bias <- train %>% group_by(movieId) %>% summarize(m_bias = mean(rating - grandmean - u_bias))

test <- left_join(test, movie_bias, by = 'movieId') %>% mutate(full_prediction = user_prediction + m_bias)

model3 <- RMSE(test$rating, test$full_prediction)

##Adjusting for the movie_specific effect drops the RMSE again to .88

###################################################################################################
##Regularization
###################################################################################################

lambda <- 3

user_bias_reg <- train %>% group_by(userId) %>% summarize(u_bias_r = sum(rating - grandmean)/(n() + lambda))

test$user_prediction_reg <- left_join(test, user_bias_reg, by = 'userId') %>% .$u_bias_r + grandmean

user_RMSE_reg <- RMSE(test$rating, test$user_prediction_reg)

train <- left_join(train, user_bias_reg, by = 'userId')
movie_bias_reg <- train %>% group_by(movieId) %>% summarize(m_bias_r = sum(rating - grandmean - u_bias)/(n() + lambda))
train <- left_join(train, movie_bias_reg, by = 'movieId')

test <- left_join(test, user_bias_reg, by = 'userId')
test <- left_join(test, movie_bias_reg, by = 'movieId') %>% mutate(full_prediction_reg = user_prediction_reg + m_bias_r)

model4 <- RMSE(test$rating, test$full_prediction_reg)

#Lambda of 3 actually decreases our RMSE. Let's optimize lambda.

lambdas <- seq(0, 10, .25)
rmses <- sapply(lambdas, function(x){
  u_b <- train %>% group_by(userId) %>% summarize(u_b = sum(rating - grandmean)/(n() + x))
  m_b <- train %>% left_join(u_b, by = 'userId') %>% group_by(movieId) %>% 
    summarize(m_b = sum(rating - grandmean - u_b)/(n() + x))
  predictions <- test %>% left_join(u_b, by = 'userId') %>% left_join(m_b, by = 'movieId') %>%
    mutate(predictions = grandmean + u_b + m_b) %>% .$predictions
  return(RMSE(predictions, test$rating))
})

lambda_df = data.frame(lambdas = lambdas, RMSE = rmses)

lambda_df %>% ggplot(aes(x = lambdas, y = RMSE)) + geom_point()

##According to our plot, regularization of approximately lambda 3 is the best performer.

###################################################################################################
##Using Factorization via Recommenderlab
###################################################################################################

library(tidyr)

pca_df <- train %>% select(userId, movieId, rating)

##Function to make sure train set only contains examples from the test set
df_match <- function(train, test){
  return(train %>% semi_join(test, by = 'userId') %>% semi_join(test, by = 'movieId'))
}

pca_df <- df_match(pca_df, test)

##Functions to transform dataframes to appropriate matrix format with rating or resid column
make_matrix_rating <- function(df){
  df %>% group_by_at(vars(-rating)) %>% mutate(row_id = 1:n()) %>% ungroup() %>% 
    spread(movieId, rating) %>% select(-row_id) %>% distinct(userId, .keep_all = TRUE) %>%
    select(-userId) %>% as.matrix()
}

make_matrix_resid <- function(df){
  df %>% group_by_at(vars(-resid)) %>% mutate(row_id = 1:n()) %>% ungroup() %>% 
    spread(movieId, resid) %>% select(-row_id) %>% distinct(userId, .keep_all = TRUE) %>%
    select(-userId) %>% as.matrix()
}


library(recommenderlab)

#Creat matrices
pca_df <- make_matrix_rating(pca_df)
pca_test <- test %>% select(userId, movieId, rating)
pca_test <- make_matrix_rating(pca_test)

##Recommender function
recommender <- function(train, test){
  cm <- colMeans(train, na.rm = TRUE)
  train <- sweep(train, 2, cm)
  rm <- rowMeans(train, na.rm = TRUE)
  train <- sweep(train, 1, rm)
  train[is.na(train)] <- 0
  
  train_rr <- as(train, 'realRatingMatrix')
  rec <- Recommender(train_rr, method = 'LIBMF')
  
  test <- sweep(test, 2, cm)
  test <- sweep(test, 1, rm)
  pre <- predict(rec, train_rr, type = 'ratingMatrix')
  pre_matrix <- as(getRatingMatrix(pre), 'matrix')
  
  RMSE(test[!is.na(test)], pre_matrix[!is.na(test)])
}

model5 <- recommender(pca_df, pca_test)

##This recommendation system gives an RMSE of .88, which is not terrible, 
##but does not beat our baseline model with about .86.

###################################################################################################
##Factorization with Residuals of Statistical Model
###################################################################################################

##This is the residuals from the statistical model.
train <- train %>% mutate(resid = rating - grandmean - u_bias_r - m_bias_r)
test <- test %>% mutate(resid = rating - grandmean - u_bias_r - m_bias_r)

resid_df <- train %>% select(userId, movieId, resid)
resid_df <- df_match(resid_df, test)

resid_df <- make_matrix_resid(resid_df)
resid_test <- test %>% select(userId, movieId, resid)
resid_test <- make_matrix_resid(resid_test)

model6 <- recommender(resid_df, resid_test)
##This has slightly dropped our RMSE to .86, but it is clear we are still overfitting.
##I will try fitting a funkSVD model that has a gamma term to regularize and reduce the overfit.

###################################################################################################
##FunkSVD on residuals model
###################################################################################################

train_svd <- df_match(train, test)
train_svd <- train_svd %>% select(userId, movieId, resid)
train_svd <- make_matrix_resid(train_svd)
test_svd <- test %>% select(userId, movieId, resid)
test_svd <- make_matrix_resid(test_svd)

##Function to fit funkSVD model
makesvd <- function(train, test, k = 10, gamma = .015, lambda = .001){
  
  fsvd <- funkSVD(train, k = k, gamma = gamma, lambda = lambda)
  r <- tcrossprod(fsvd$U, fsvd$V)
  acctrain <- RMSE(train[!is.na(train)], r[!is.na(train)])
  p <- predict(fsvd, test, verbose = TRUE)
  acctest <- RMSE(test[!is.na(test)], p[!is.na(test)])
  return(c(acctrain, acctest))
}

model7 <- makesvd(train_svd, test_svd, 10, .015, .001)

##This gives us our best RMSE yet of .84 along with a less overfit training RMSE of .77

###################################################################################################
##FunkSVD on normalized ratings
###################################################################################################

train_svd_rat <- df_match(train, test)
train_svd_rat <- train_svd_rat %>% select(userId, movieId, rating)
train_svd_rat <- make_matrix_rating(train_svd_rat)
test_svd_rat <- test %>% select(userId, movieId, rating)
test_svd_rat <- make_matrix_rating(test_svd_rat)

cm <- colMeans(train_svd_rat, na.rm = TRUE)
train_svd_rat <- sweep(train_svd_rat, 2, cm)
rm <- rowMeans(train_svd_rat, na.rm = TRUE)
train_svd_rat <- sweep(train_svd_rat, 1, rm)

test_svd_rat <- sweep(test_svd_rat, 2, cm)
test_svd_rat <- sweep(test_svd_rat, 1, rm)

model8 <- makesvd(train_svd_rat, test_svd_rat, 10, .015, .001)
##This performs worse than the model with the residuals

###################################################################################################
##Optimize on FunkSVD Residuals
###################################################################################################

k <- seq(5, 100, 5)
gamma <- seq(0, .1, .01)
lambda <- seq(.0005, .005, .0005)

k_opt <- sapply(k, makesvd, train = train_svd, test = test_svd, gamma = .015, lambda = .001)

k_df = data.frame(k = k, RMSE_train = k_opt[1,], RMSE_test = k_opt[2,])
k_df <- k_df %>% gather('TrainTest', 'RMSE', -k)
k_df$TrainTest <- sapply(k_df$TrainTest, function(x){strsplit(x, '_')}) %>%
  sapply(function(x){x[2]})
k_df %>% ggplot(aes(x = k, y = RMSE, color = TrainTest)) + geom_line() +
  ggtitle('Optimization of K Parameter') + labs(y = 'RMSE') + labs(colour = 'Dataset')

##Low values of K seem optimal. We will use the default value of 10 which is close to the optimum. This likely
##needs to be reoptimized on the larger dataset but will also take FOREVER

gamma_opt <- sapply(gamma, makesvd, train = train_svd, test = test_svd, k = 10, lambda = .003, column = 'resid')

g_df = data.frame(gamma = gamma, RMSE_train = gamma_opt[1,], RMSE_test = gamma_opt[2,])
g_df <- g_df %>% gather('TrainTest', 'RMSE', -gamma)
g_df$TrainTest <- sapply(g_df$TrainTest, function(x){strsplit(x, '_')}) %>%
  sapply(function(x){x[2]})
g_df %>% ggplot(aes(x = gamma, y = RMSE, color = TrainTest)) + geom_line() +
  ggtitle('Optimization of Gamma Parameter') + labs(y = 'RMSE') + labs(colour = 'Dataset')

##Once again the default gamma parameter of .015 appears to be close to optimal, so we will continue to use it.

lambda_opt <- sapply(lambda, makesvd, train = train_svd, test = test_svd, k = 10, gamma = .015, column = 'resid')

l_df = data.frame(lambda = lambda, RMSE_train = lambda_opt[1,], RMSE_test = lambda_opt[2,])
l_df <- l_df %>% gather('TrainTest', 'RMSE', -lambda)
l_df$TrainTest <- sapply(l_df$TrainTest, function(x){strsplit(x, '_')}) %>%
  sapply(function(x){x[2]})
l_df %>% ggplot(aes(x = lambda, y = RMSE, color = TrainTest)) + geom_line() +
  ggtitle('Optimization of Lambda Parameter') + labs(y = 'RMSE') + labs(colour = 'Dataset')

##Here we see that large lambda is optimal but also starts to show evidence of overfitting at the highest values.
##I will use a value of .003 as the learning rate where the curve flattens for the test set, but will also reoptimize
##gamma using this new lambda to try and reduce overfitting.

##With lambda .003, It seems that using a gamma of .05 slightly reduces the overfitting that occurs.
##As such, my final model will be with gamma .05, lambda .003, and k = 10.
##I will now run this model on the validation set to see if we have overfit.

###################################################################################################
##Validation Set
###################################################################################################

##Repeat process with validation dataset
load(file = 'rda/validation.rda')

val_train <- train %>% select(userId, movieId, resid)
validation <- validation %>% left_join(user_bias_reg, by = 'userId') %>% left_join(movie_bias_reg, by = 'movieId')
validation <- validation %>% mutate(resid = rating - grandmean - u_bias_r - m_bias_r)
val_test <- validation %>% select(userId, movieId, resid)

val_train <- df_match(val_train, val_test)
val_train <- make_matrix_resid(val_train)
val_test <- make_matrix_resid(val_test)

val_model <- makesvd(val_train, val_test, k = 10, gamma = .05, lambda = .003)
##The validation set achieved an RMSE of .806, which is slightly worse than the performance on the test set but 
##still a significant improvement over our current best model. I am happy with the model I have created!


