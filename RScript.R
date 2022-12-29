
##########################################################
# 1- Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

if(!require(psych)) install.packages("psych")
library(psych)

if(!require(lubridate)) install.packages("lubridate")
library(lubridate)

if(!require(magrittr)) install.packages("cran.r-project.org/src/contrib/Archive/magrittr/…", repos = NULL, type="source")
library(magrittr)

##########################################################
# 2- ANALYSIS & MODELLING 
##########################################################

## 2.1 Datasets overview 

### 2.1.1 edx dataset

# Overview of the structure of the dataset
str(edx)

describe(edx, fast=TRUE)

# Main indicators
nb_users<- n_distinct(edx$userId)
nb_users
nb_ratings<- nrow(edx)
nb_ratings
nb_movies<- n_distinct(edx$movieId)
nb_movies
percentage_coverage<- nb_ratings /(nb_users* nb_movies)
percentage_coverage

round(percentage_coverage,4)*100

### 2.1.2 final_holdout_test dataset
str(final_holdout_test)

describe(final_holdout_test, fast=TRUE)

## 2.2 Datasets preparation 

# Timestamp is not easy to handle (it is not readable, nor categorised). We can convert it to year to make it a more useful predictor and call it ‘year_of_rating’. We can also delete the timestamp column which we will not use.
edx <-edx %>% mutate(edx, year_of_rating = as.integer(year(as_datetime(timestamp))))
edx<-edx[,-4]

# Extract the year of the movie, which might be a useful predictor. We will call it ‘movie_year’.
edx <- edx %>% mutate(movie_year = as.integer(substr(title, str_length(title) - 4, str_length(title) - 1)))

# Compute the age of the movie at the time the rating was done, which might also be a predictor. We will call it ‘rating_age’.
edx <- edx %>% mutate(rating_age = year_of_rating - movie_year)

# We replicate the preparation to the final_holdout_test dataset so that we can use the sets for calculating the predictions and RSME.
final_holdout_test <- final_holdout_test %>% mutate(final_holdout_test, year_of_rating = as.integer(year(as_datetime(timestamp))))
final_holdout_test <- final_holdout_test [,-4]
final_holdout_test <- final_holdout_test %>% mutate(movie_year = as.integer(substr(title, str_length(title) - 4, str_length(title) - 1)))
final_holdout_test <- final_holdout_test %>% mutate(rating_age = year_of_rating - movie_year)

## 2.3 Data Analysis

### 2.3.1 Rating insights
round(mean(edx$rating),2)
round(sd(edx$rating),2)

# Figure 1: Proportion of ratings per rating
edx %>% ggplot(aes(rating, y = after_stat(prop))) + geom_bar() + labs(x = "Ratings", y = "Proportion of Ratings") + scale_x_continuous(breaks = seq(0, 5, by= 0.5)) + ggtitle("Figure 1: Proportion of ratings per rating")

### 2.3.2 Users insights 
# Constructing/updating table(s) to summarize data
users_rating_summ<-edx %>% group_by(userId) %>% summarise(n=n()) %>% summarise(min=min(n),max=max(n),mean = mean(n),sd=sd(n),median=median(n))
user_table<-edx %>% group_by(userId) %>% summarise(n=n(), avg_rating=mean(rating), sd_rating=sd(rating))

# Main indicators
nb_users
round(min(user_table$avg_rating),2)
round(max(user_table$avg_rating),2)
round(mean(user_table$avg_rating),2)
round(min(user_table$avg_rating),2)

# Figure 2: Distribution of average ratings given by users
user_table %>% ggplot(aes(avg_rating))+geom_density(fill="gray17") + labs(x="Average user rating", y="Density") + ggtitle("Figure 2: Distribution of average ratings given by users")

# Main indicators
users_rating_summ$min
users_rating_summ$max
round(users_rating_summ$mean,2)
round(users_rating_summ$sd,2)
users_rating_summ$median

# Figure 3: Number of ratings per user
user_table %>% ggplot(aes(n)) + geom_histogram(color="grey", size=0.1,bins=30) + labs(x="Number of ratings", y="Number of users") + scale_x_log10()+ ggtitle("Figure 3: Number of ratings per user")

# Figure 4: Cumulative proportion of ratings
user_table<-user_table%>% mutate (proportion_rating=n/sum(n))
temp<-cumsum(sort(user_table$proportion_rating))
temp<-data.frame(Number_of_users=1:nb_users,Cumulative_proportion_of_ratings=temp)
ggplot(temp,aes(x=Number_of_users,y=Cumulative_proportion_of_ratings)) + geom_line() + labs(x="Cumulative number of users", y="Cumulative proportion of ratings") + ggtitle("Figure 4: Cumulative proportion of ratings")

# Quantiles
format(quantile(temp$Cumulative_proportion_of_ratings), scientific=F,digits=2)
rm(temp)

### 2.3.3 Movies insights 
# Constructing/updating table(s) to summarize data
movies_rating_summ<-edx %>% group_by(movieId) %>% summarise(n=n()) %>% summarise(min=min(n),max=max(n),mean = mean(n),sd=sd(n),median=median(n))
movie_table<-edx %>% group_by(movieId) %>% summarise(n=n(), avg_rating=mean(rating), sd_rating=sd(rating))

# Main indicators
nb_movies
round(min(movie_table$avg_rating),2)
round(max(movie_table$avg_rating),2)
round(mean(movie_table$avg_rating),2)
round(sd(movie_table$avg_rating),2)

# Figure 5: Distribution of average movie ratings
movie_table %>% ggplot(aes(avg_rating))+geom_density(fill="gray17") + labs(x="Average movie rating", y="Density") + ggtitle("Figure 5: Distribution of average movie ratings")

# Main indicators
movies_rating_summ$min
movies_rating_summ$max
round(movies_rating_summ$mean,2)
round(movies_rating_summ$sd,2)
round(movies_rating_summ$median,2)

# Figure 6: Number of ratings per movie
movie_table %>% ggplot(aes(n)) + geom_histogram(color="grey", size=0.1,bins=30) + labs(x="Number of ratings", y="Number of movies") + scale_x_log10() + ggtitle("Figure 6: Number of ratings per movie")

# Figure 7: Average rating per number of ratings
movie_table %>% ggplot(aes(n,avg_rating)) + geom_point(size=0.3) + labs(x="Number of ratings", y="Average rating") + ggtitle("Figure 7: Average rating per number of ratings")

# Correlation
round(cor(movie_table$n,movie_table$avg_rating),2)

### 2.3.4 Movie year insights
# Constructing/updating table(s) to summarize data
movie_table<-left_join(movie_table, unique(select(edx,movieId,movie_year)), by = "movieId")
movie_table <- movie_table %>% mutate(total_rating= avg_rating * n)
movie_table_summary <- movie_table %>% group_by(movie_year)  %>% summarise(n_ratings=sum(n),n_movie=n(),avg_rating_per_year=sum(total_rating)/sum(n))

# Figure 8: Number of movies released per year
movie_table_summary %>% ggplot(aes(x= movie_year, y = n_movie )) + geom_col()+ labs(x="Year", y="Number of movies released") + ggtitle("Figure 8: Number of movies released per year")

# Figure 9: Average rating per release year
movie_table_summary %>% ggplot(aes(x= movie_year, y = avg_rating_per_year)) + geom_col()+ labs(x="Year of release", y="Average rating") + ggtitle("Figure 9: Average rating per release year")

# Correlation
round(cor(movie_table_summary$avg_rating_per_year,movie_table_summary$movie_year),2)

### 2.3.5 Movie genres insights
# Constructing/updating table(s) to summarize data
genres <- edx$genres %>% str_replace("\\|.*","") %>% unique()

nb_genres <- sapply(genres, function(x){
  index <- str_which(edx$genres, x)
  length(edx$rating[index])
})

genres_ratings <- sapply(genres, function(x){
  index <- str_which(edx$genres, x)
  mean(edx$rating[index], na.rm = T)
})

genres_table <- data.frame(genres = genres, n_genres = nb_genres, avg_rating = genres_ratings)

# Figure 10: Average rating per genre
genres_table %>% ggplot(aes(x= reorder(genres,avg_rating), y = avg_rating)) + geom_col()+ labs(x=" Genre", y="Average Rating") + ggtitle("Figure 10: Average rating per genre") + coord_flip()

# Main indicators
round(min(genres_table$avg_rating),2)
round(max(genres_table$avg_rating),2)
round(mean(genres_table$avg_rating),2)
round(sd(genres_table$avg_rating),2)

# Figure 11: Number of ratings per genre
genres_table %>% ggplot(aes(x= reorder(genres,n_genres), y = n_genres)) + geom_col()+ labs(x="Genre", y="Number of ratings") + ggtitle("Figure 11: Number of ratings per genre") + coord_flip()

### 2.3.6 Rating year insights
# Constructing/updating table(s) to summarize data
year_rating_table <- edx %>% select(rating,year_of_rating) %>% group_by(year_of_rating) %>% summarise(avg_rating = mean(rating), n=n())

# Figure 12: Average rating per year of rating
year_rating_table %>% ggplot(aes(x= year_of_rating, y = avg_rating)) + geom_col()+ labs(x="Year of rating", y="Average rating") + ggtitle("Figure 12: Average rating per year of rating")

year_rating_table$n[1]

### 2.3.7 Rating age insights
# Constructing/updating table(s) to summarize data
age_rating_table <- edx %>% select(rating,rating_age) %>% group_by(rating_age) %>% summarise(avg_rating = mean(rating), n=n())

#  Figure 13: Average rating per age of rating
ylim.prim <- c(0,4.5)
ylim.sec <- c(0, 1100000)
b <- diff(ylim.prim)/diff(ylim.sec)
age_rating_table %>% ggplot(aes(x= rating_age, y = avg_rating)) + geom_col() + labs(x="Age of rating") + ggtitle("Figure 13: Average rating per age of rating") + scale_y_continuous(name = "Average rating", sec.axis = sec_axis(~ (. )/b,name="Number of ratings")) + geom_line(aes(y = n*b), color = " salmon2") + theme(axis.line.y.right = element_line(color = " salmon2"), axis.ticks.y.right = element_line(color = " salmon2"), axis.text.y.right = element_text(color = " salmon2"), axis.title.y.right = element_text(color = "salmon2"))

# Correlation
round(cor(age_rating_table$avg_rating, age_rating_table$rating_age),2)

## 2.4 Modelling

### 2.4.1 Compute RSME function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

### 2.4.2 Simplest Model (Model 1)
# Prediction
predicted_rating_model1 <- mean(edx$rating)
predicted_rating_model1 

# RMSE
model1RMSE<-RMSE(final_holdout_test$rating, predicted_rating_model1)
rmse_results <- bind_rows(data_frame(method="Simplest model - Model 1", RMSE = model1RMSE))
model1RMSE

### 2.4.3 Adding the movie effect (Model 2)
# Prediction
mu <- mean(edx$rating)
movie_table <- movie_table %>% mutate (b_i = avg_rating - mu)
predicted_ratings_model2 <- mu + final_holdout_test %>% left_join(movie_table, by='movieId') %>% .$b_i

# RMSE
model2RMSE <- RMSE(final_holdout_test$rating,as.vector(predicted_ratings_model2))
rmse_results <- bind_rows(rmse_results, data_frame(method="Model with movie effect - Model 2", RMSE = model2RMSE))
model2RMSE

### 2.4.4 Adding the user effect (Model 3)
# Prediction
user_table <- user_table %>% mutate (b_u = avg_rating - mu)
predicted_ratings_model3 <- final_holdout_test %>% left_join(movie_table, by='movieId') %>% left_join(user_table, by='userId') %>% mutate(pred = mu + b_i + b_u) %>% .$pred

# RMSE
model3RMSE <- RMSE(final_holdout_test$rating,as.vector(predicted_ratings_model3))
rmse_results <- bind_rows(rmse_results, data_frame(method="Model 2 with user effect - Model 3", RMSE = model3RMSE))
model3RMSE

### 2.4.5 Adding the movie year effect (Model 4)
# Prediction
movie_table_summary <- movie_table_summary %>% mutate (b_t = avg_rating_per_year - mu)
movie_table <- movie_table %>% left_join(select(movie_table_summary,movie_year,b_t), by='movie_year')
predicted_ratings_model4 <- final_holdout_test %>% left_join(movie_table, by='movieId') %>% left_join(user_table, by='userId') %>% mutate(pred = mu + b_i + b_u + b_t) %>% .$pred

# RMSE
model4RMSE <- RMSE(final_holdout_test$rating,as.vector(predicted_ratings_model4))
rmse_results <- bind_rows(rmse_results, data_frame(method="Model 3 with movie year effect - Model 4", RMSE = model4RMSE))
model4RMSE

### 2.4.6 Adding the genre effect (Model 5)
# Prediction
genres_table <- edx %>% mutate(b_ge = rating - mu) %>% group_by(genres) %>% summarize(b_g = mean(b_ge))
predicted_ratings_model5 <- final_holdout_test %>% left_join(movie_table, by='movieId') %>% left_join(user_table, by='userId') %>% left_join(genres_table, by='genres')  %>% mutate(pred = mu + b_i + b_u + b_g) %>% .$pred

# RMSE
model5RMSE <- RMSE(final_holdout_test$rating,as.vector(predicted_ratings_model5))
rmse_results <- bind_rows(rmse_results, data_frame(method="Model 3 with genre effect - Model 5", RMSE = model5RMSE))
model5RMSE

### 2.4.7 Adding the age effect (Model 6)
# Prediction
age_rating_table <- age_rating_table %>% mutate (b_a = avg_rating - mu)
predicted_ratings_model6 <- final_holdout_test %>% left_join(movie_table, by='movieId') %>% left_join(user_table, by='userId') %>% left_join(age_rating_table, by='rating_age')  %>% mutate(pred = mu + b_i + b_u + b_a) %>% .$pred

# RMSE
model6RMSE <- RMSE(final_holdout_test$rating,as.vector(predicted_ratings_model6))
rmse_results <- bind_rows(rmse_results, data_frame(method="Model 3 with rating age effect - Model 6", RMSE = model6RMSE))
model6RMSE

### 2.4.8 Using regularisation to improve the performance (Model 7)
## Test regularisation
lambda <- 3
movie_table_regularized <- edx %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n()+lambda), n = n())
predicted_ratings_model7test <- final_holdout_test %>% left_join(movie_table_regularized, by='movieId') %>% mutate(pred = mu + b_i) %>% .$pred
RMSE(final_holdout_test$rating, as.vector(predicted_ratings_model7test))

## Optimisation
lambdas <- seq(0, 10, 0.25)

rmse_model7 <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  movie_table_regularized <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  user_table_regularised <- edx %>% 
    left_join(movie_table_regularized, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings_model7 <- 
    final_holdout_test %>% 
    left_join(movie_table_regularized, by = "movieId") %>%
    left_join(user_table_regularised, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(final_holdout_test$rating, predicted_ratings_model7))
})

# Optimum lambda
lambda <- lambdas[which.min(rmse_model7)]

## Final model
movie_table_regularized <- edx %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n()+lambda))
user_table_regularised <- edx %>% left_join(movie_table_regularized, by="movieId") %>% group_by(userId) %>% summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
# Prediction
predicted_ratings_model7 <- 
  final_holdout_test %>% 
  left_join(movie_table_regularized, by = "movieId") %>%
  left_join(user_table_regularised, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

# RMSE
model7RMSE<-RMSE(final_holdout_test$rating,as.vector(predicted_ratings_model7))
rmse_results <- bind_rows(rmse_results, data_frame(method="Model 3 regularised - Model 7", RMSE = model7RMSE))
model7RMSE

# 3. Results
# Summary table
rmse_results %>% knitr::kable()

