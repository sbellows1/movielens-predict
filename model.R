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

default_RMSE <- RMSE(test$rating, test$def_prediction)

##The default (just guessing) RMSE is 1.04, which is a benchmark to beat.

###################################################################################################
##Statistical Modeling
###################################################################################################

user_bias <- train %>% group_by(userId) %>% summarize(u_bias = mean(rating - grandmean))

test$user_prediction <- left_join(test, user_bias, by = 'userId') %>% .$u_bias + grandmean

user_RMSE <- RMSE(test$rating, test$user_prediction)

##By adjusting for user-specific effect, the RMSE on the test set drops to .93
##Now we must add a movie-specific effect while keeping the user effect in mind.
train <- left_join(train, user_bias, by = 'userId')
movie_bias <- train %>% group_by(movieId) %>% summarize(m_bias = mean(rating - grandmean - u_bias))

test <- left_join(test, movie_bias, by = 'movieId') %>% mutate(full_prediction = user_prediction + m_bias)

full_RMSE <- RMSE(test$rating, test$full_prediction)

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

full_RMSE_reg <- RMSE(test$rating, test$full_prediction_reg)

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

#organize data into matrix format
pca_df <- train %>% select(userId, movieId, rating)

##Here I remove any movies or users that are not in the test set in order for the test and train matrices
##to be the same shape.
pca_df1 <- pca_df %>% semi_join(test, by = 'userId') %>% semi_join(test, by = 'movieId')

##Spread the movieId column to be the header.
pca_df1 <- pca_df1 %>% group_by_at(vars(-rating)) %>% mutate(row_id = 1:n()) %>% ungroup() %>% 
  spread(movieId, rating) %>% select(-row_id)
pca_df1 <- pca_df1 %>% distinct(userId, .keep_all = TRUE) 
pca_matrix1 <- pca_df1 %>% select(-userId) %>% as.matrix()

#normalize data and set NAs = 0
cm1 <- colMeans(pca_matrix1, na.rm = TRUE)
pca_matrix1 <- sweep(pca_matrix1, 2, cm1)
rm1 <- rowMeans(pca_matrix1, na.rm = TRUE)
pca_matrix1 <- sweep(pca_matrix1, 1, rm1)
pca_matrix1[is.na(pca_matrix1)] <- 0

#Try with recommenderlab
library(recommenderlab)
##Initialize Recommender on the training set.
pca_matrix1 <- as(pca_matrix1, 'realRatingMatrix')
rec1 <- Recommender(pca_matrix1, method = 'LIBMF')

##Turn test into matrix format.
test_df <- test %>% select(userId, movieId, rating)
test_df <- test_df %>% group_by_at(vars(-rating)) %>% mutate(row_id = 1:n()) %>% ungroup() %>% 
  spread(movieId, rating) %>% select(-row_id)
test_matrix <- test_df %>% distinct(userId, .keep_all = TRUE) %>% select(-userId) %>% as.matrix()

#Normalize test by same values used on training data.

test_matrix <- sweep(test_matrix, 2, cm1)
test_matrix <- sweep(test_matrix, 1, rm1)
##test_matrix[is.na(test_matrix)] <- 0

pre1 <- predict(rec1, pca_matrix1, type = 'ratingMatrix')
pre1mat <- as(getRatingMatrix(pre1), 'matrix')

acc1 <- RMSE(test_matrix[!is.na(test_matrix)], pre1mat[!is.na(test_matrix)])

##This recommendation system gives an RMSE of .88, which is not terrible, 
##but does not beat our baseline model with about .86.

###################################################################################################
##Factorization with Residuals of Statistical Model
###################################################################################################

##This is the residuals from the statistical model.
train <- train %>% mutate(resid = rating - grandmean - u_bias_r - m_bias_r)

resid_df <- train %>% select(userId, movieId, resid)
resid_df <- resid_df %>% semi_join(test, by = 'userId') %>% semi_join(test, by = 'movieId')

##Spread the movieId column to be the header.
resid_df <- resid_df %>% group_by_at(vars(-resid)) %>% mutate(row_id = 1:n()) %>% ungroup() %>% 
  spread(movieId, resid) %>% select(-row_id)
resid_df <- resid_df %>% distinct(userId, .keep_all = TRUE) 
resid_m <- resid_df %>% select(-userId) %>% as.matrix()

#I tried normalized and non normalized. Normalization had no effect on the residuals model.
#resid_m[is.na(resid_m)] <- 0

resid_m_rr <- as(resid_m, 'realRatingMatrix')
rec3 <- Recommender(resid_m_rr, method = 'LIBMF')

##Turn test into matrix format.
test <- test %>% mutate(resid = rating - grandmean - u_bias_r - m_bias_r)
test_df2 <- test %>% select(userId, movieId, resid)
test_df2 <- test_df2 %>% group_by_at(vars(-resid)) %>% mutate(row_id = 1:n()) %>% ungroup() %>% 
  spread(movieId, resid) %>% select(-row_id)
test_matrix2 <- test_df2 %>% distinct(userId, .keep_all = TRUE) %>% select(-userId) %>% as.matrix()

pre3 <- predict(rec1, resid_m_rr, type = 'ratingMatrix')
pre3mat <- as(getRatingMatrix(pre3), 'matrix')

acc3 <- RMSE(test_matrix2[!is.na(test_matrix2)], pre3mat[!is.na(test_matrix2)])

##This has slightly dropped our RMSE to .86, but it is clear we are still overfitting.
##I will try fitting a funkSVD model that has a gamma term to regularize and reduce the overfit.

###################################################################################################
##FunkSVD on residuals model
###################################################################################################

fsvd <- funkSVD(resid_m)
r <- tcrossprod(fsvd$U, fsvd$V)
acctrain <- RMSE(resid_m[!is.na(resid_m)], r[!is.na(resid_m)])
##This gives us an RMSE of .76 on the training set, which is much more realistic. Let us see the test set.
p <- predict(fsvd, test_matrix2)
acctest <- RMSE(test_matrix2[!is.na(test_matrix2)], p[!is.na(test_matrix2)])
##This gives us our best RMSE yet of .84

###################################################################################################
##FunkSVD on normalized ratings
###################################################################################################

fsvd_m <- pca_df1 %>% select(-userId) %>% as.matrix()
cm1 <- colMeans(fsvd_m, na.rm = TRUE)
fsvd_m <- sweep(fsvd_m, 2, cm1)
rm1 <- rowMeans(fsvd_m, na.rm = TRUE)
fsvd_m <- sweep(fsvd_m, 1, rm1)

fsvd2 <- funkSVD(fsvd_m)
r2 <- tcrossprod(fsvd2$U, fsvd2$V)
acctrain2 <- RMSE(fsvd_m[!is.na(fsvd_m)], r2[!is.na(fsvd_m)])
#Training RMSE
p2 <- predict(fsvd2, test_matrix)
acctest2 <- RMSE(test_matrix[!is.na(test_matrix)], p2[!is.na(test_matrix)])
##This performs worse than the model with the residuals

###################################################################################################
##Optimize on FunkSVD Residuals
###################################################################################################

makesvd <- function(k, gamma, lambda){
  fsvd <- funkSVD(resid_m, k = k, gamma = gamma, lambda = lambda, verbose = TRUE)
  r <- tcrossprod(fsvd$U, fsvd$V)
  acctrain <- RMSE(resid_m[!is.na(resid_m)], r[!is.na(resid_m)])
  p <- predict(fsvd, test_matrix2, verbose = TRUE)
  acctest <- RMSE(test_matrix2[!is.na(test_matrix2)], p[!is.na(test_matrix2)])
  return(c(acctrain, acctest))
}

k <- seq(5, 100, 5)
gamma <- seq(0, .1, .01)
lambda <- seq(.0005, .005, .0005)

k_opt <- sapply(k, makesvd, gamma = .015, lambda = .001)

k_df = data.frame(k = k, RMSE_train = k_opt[1,], RMSE_test = k_opt[2,])
k_df <- k_df %>% gather('TrainTest', 'RMSE', -k)
k_df$TrainTest <- sapply(k_df$TrainTest, function(x){strsplit(x, '_')}) %>%
  sapply(function(x){x[2]})
k_df %>% ggplot(aes(x = k, y = RMSE, color = TrainTest)) + geom_line() +
  ggtitle('Optimization of K Parameter') + labs(y = 'RMSE') + labs(colour = 'Dataset')

##Low values of K seem optimal. We will use the default value of 10 which is close to the optimum. This likely
##needs to be reoptimized on the larger dataset but will also take FOREVER

gamma_opt <- sapply(gamma, makesvd, k = 10, lambda = .003)

g_df = data.frame(gamma = gamma, RMSE_train = gamma_opt[1,], RMSE_test = gamma_opt[2,])
g_df <- g_df %>% gather('TrainTest', 'RMSE', -gamma)
g_df$TrainTest <- sapply(g_df$TrainTest, function(x){strsplit(x, '_')}) %>%
  sapply(function(x){x[2]})
g_df %>% ggplot(aes(x = gamma, y = RMSE, color = TrainTest)) + geom_line() +
  ggtitle('Optimization of Gamma Parameter') + labs(y = 'RMSE') + labs(colour = 'Dataset')

##Once again the default gamma parameter of .015 appears to be close to optimal, so we will continue to use it.

lambda_opt <- sapply(lambda, makesvd, k = 10, gamma = .015)

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

resid_df <- train %>% select(userId, movieId, resid)
resid_df <- resid_df %>% semi_join(validation, by = 'userId') %>% semi_join(validation, by = 'movieId')

##Spread the movieId column to be the header.
resid_df <- resid_df %>% group_by_at(vars(-resid)) %>% mutate(row_id = 1:n()) %>% ungroup() %>% 
  spread(movieId, resid) %>% select(-row_id)
resid_df <- resid_df %>% distinct(userId, .keep_all = TRUE) 
resid_m <- resid_df %>% select(-userId) %>% as.matrix()

fsvd_val <- funkSVD(resid_m, gamma = .05, lambda = .003)
r_val <- tcrossprod(fsvd_val$U, fsvd_val$V)
acctrain_val <- RMSE(resid_m[!is.na(resid_m)], r_val[!is.na(resid_m)])

#Transform the validation set into residuals.
validation <- validation %>% left_join(user_bias_reg, by = 'userId') %>% left_join(movie_bias_reg, by = 'movieId')
validation <- validation %>% mutate(resid = rating - grandmean - u_bias_r - m_bias_r)

#Create validation matrix
val_df <- validation %>% select(userId, movieId, resid)
val_df <- val_df %>% group_by_at(vars(-resid)) %>% mutate(row_id = 1:n()) %>% ungroup() %>% 
  spread(movieId, resid) %>% select(-row_id)
val_matrix <- val_df %>% distinct(userId, .keep_all = TRUE) %>% select(-userId) %>% as.matrix()

p_val <- predict(fsvd_val, val_matrix)
acctest_val <- RMSE(val_matrix[!is.na(val_matrix)], p_val[!is.na(val_matrix)])

##The validation set achieved an RMSE of .806, which is slightly worse than the performance on the test set but 
##still a significant improvement over our current best model. I am happy with the model I have created!


