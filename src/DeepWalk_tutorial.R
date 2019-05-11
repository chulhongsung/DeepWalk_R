library(Keras)
library(igraph)
library(igraphdata)
library(tidyverse)
library(reticulate)

data(karate)

plot(karate)

# Random walk generating (100 random walks for each node)

random_walks = c()

for(i in seq_len(34)){
  for(j in 1:100){
    random_walks <- rbind(random_walks, random_walk(karate, start = i, steps = 10) %>% attributes(.) %>% .$names %>% paste0(., collapse = " "))
  }
}

random_walks <- random_walks %>% as.vector()

# Preprocessing 

random_walks <- gsub("Mr Hi", "H", random_walks)
random_walks <- gsub("John A", "A", random_walks)
random_walks <- gsub("Actor", "", random_walks)

tokenizer <- text_tokenizer(num_words = 34, split = " ")

tokenizer %>% fit_text_tokenizer(random_walks)

skipgrams_generator <- function(text, tokenizer, window_size, negative_samples){
  gen <- texts_to_sequences_generator(tokenizer, sample(text))
  
  function() { 
    skip <- generator_next(gen) %>% 
      skipgrams(
        vocabulary_size = tokenizer$num_words,
        window_size = window_size,
        negative_samples = negative_samples
      )
    x <- transpose(skip$couples) %>% map(. %>% unlist %>% as.matrix(ncol = 1))
    y <- skip$labels %>% as.matrix(ncol = 1)
    list(x, y)
  }
  
}

# Skipgram modeling with Keras

embedding_size <- 2  
skip_window <- 3       
num_sampled <- 1       

input_target <- layer_input(shape = c(1, NULL))
input_context <- layer_input(shape = c(1, NULL))

embedding <- layer_embedding(
  input_dim = tokenizer$num_words + 1, 
  output_dim = embedding_size, 
  input_length = 1, 
  name = "embedding"
)

target_vector <- input_target %>% 
  embedding() %>% 
  layer_flatten()

context_vector <- input_context %>%
  embedding() %>%
  layer_flatten()

dot_product <- layer_dot(list(target_vector, context_vector), axes = 1)
output <- layer_dense(dot_product, units = 1, activation = "sigmoid")

model <- keras_model(list(input_target, input_context), output)
model %>% compile(loss = "binary_crossentropy", optimizer = "adam")

summary(model)

model %>%  fit_generator(
  skipgrams_generator(random_walks, tokenizer, skip_window, num_sampled), 
  steps_per_epoch = 340, 
  epochs = 10
)

# Representation matrix 

embedding_matrix <- get_weights(model)[[1]]

words <- data.frame(
  word = names(tokenizer$word_index), 
  id = as.integer(unlist(tokenizer$word_index))
)

words <- words %>%
  arrange(id)

group1 <- c('h', '2', '3', '4', '5', '6', '7', '8', '11', '12', '13', '15', '17', '18', '20', '22')

embedding_matrix <- embedding_matrix[-1,] %>% 
  as.tibble() %>% 
  bind_cols(word = as.tibble(words$word)) %>% 
  rename(word = value) %>% 
  mutate(group = if_else(word %in% group1, 1, 2))


# Visualization with plotly 

library(plotly)


p <- plot_ly(data = embedding_matrix, x = ~V1, y = ~V2,
        color = ~factor(group), colors = c('#4AC6B7', '#FF7070'),
        text = ~word) %>%
  add_markers() %>% 
  add_text(textposition = 'top right') %>% 
  layout(title = 'karate graph',
         yaxis = list(title = "", zeroline = FALSE),
         xaxis = list(title = "", zeroline = FALSE))


