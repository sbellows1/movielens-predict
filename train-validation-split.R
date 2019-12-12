##A large caveat. The majority of this code is not my code but was provided by edX. I have edited some of the code as
##certain lines were not functioning the way I wanted, but most of the code is not mine.

library('data.table')
library('tidyverse')

ratings <- fread(text = gsub('::', '/t', readLines('data/ml-latest-small/ratings.csv')), 
                 col.names = c('userId', 'movieId', 'rating', 'timestamp'))

movies <- read.csv('data/ml-latest-small/movies.csv')
colnames(movies) <- c('movieId', 'title', 'genres')
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = 'movieId')
set.seed(1, sample.kind = 'Rounding')
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.2, list = FALSE)
train <- movielens[-test_index,]
test <- movielens[test_index,]

validation <- test %>% semi_join(train, by = 'movieId') %>% semi_join(train, by = 'userId')

removed <- anti_join(train, validation)
train <- rbind(train, removed)

rm(test_index, movielens, removed, movies, ratings, test)

setwd('rda')
save(train, file = 'train.rda')
save(validation, file = 'validation.rda')
rm(train, validation)