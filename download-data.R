url <- 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'

urllarge <- 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'

##Large URL has full dataset, can be exchanged once all code is bug free

download.file(url, 'data/data.zip')
unzip('data/data.zip', exdir = 'data')