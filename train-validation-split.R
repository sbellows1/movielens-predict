library('data.table')
library('tidyverse')
library('caret')

smallratingspath <- 'data/ml-latest-small/ratings.csv'
smallmoviespath <- 'data/ml-latest-small/movies.csv'
largeratingspath <- 'data/ml-latest/ratings.csv'
largemoviespath <- 'data/ml-latest/movies.csv'

##The path can be changed out to use the smaller or larger dataset.
ratings <- fread(text = gsub('::', '/t', readLines(smallratingspath)), 
                 col.names = c('userId', 'movieId', 'rating', 'timestamp'))

##Again, you can choose whether to use the smaller or larger dataset.
movies <- read.csv(smallmoviespath)
colnames(movies) <- c('movieId', 'title', 'genres')
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = 'movieId')
set.seed(1, sample.kind = 'Rounding')

##Useful function to split out dataset.
traintest <- function(train){
  test_index <- createDataPartition(y = train$rating, times = 1, p = 0.2, list = FALSE)
  train <- movielens[-test_index,]
  test <- movielens[test_index,]

  ##Only include values in test set that are in train set
  validation <- test %>% semi_join(train, by = 'movieId') %>% semi_join(train, by = 'userId')
  
  #Get removed rows and re add them to train set
  removed <- anti_join(test, validation)
  train <- rbind(train, removed)
  return (list(train, validation))
}

datasets <- traintest(movielens)
train <- datasets[[1]]
validation <- datasets[[2]]

setwd('rda')
save(train, file = 'train.rda')
save(validation, file = 'validation.rda')
rm(train, validation)
setwd('..')