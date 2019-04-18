library(tidyverse)

# Add school type
cnn_fn <- "~/data/cnn_result_names4.csv"
cnn_res <- read_csv(cnn_fn, col_names = TRUE) 

# cnn_res <- round(cnn_res, 4)

cnn_res <- cnn_res %>%
  mutate(pred_class = as.factor(as.character(pred_class)),
         Y = as.factor(as.character(Y))) %>%
  mutate(course_name = paste(a,b,c,d, sep = " "),
         course_name = str_remove(course_name, "NA"),
         course_name = str_remove(course_name, " NA")) %>%
  select(-a,-b,-c,-d,-e,-f)

str(cnn_res)
library(caret)


caret::confusionMatrix(cnn_res$pred_class, cnn_res$Y)
names(cnn_res)


table(cnn_res$Y)

class_check <- cnn_res %>% 
  filter(Y == "5", pred_class == "4" )

str(class_check)
library(stringdist)

c_names <- class_check %>%
                  select(course_name) %>%
                  mutate(course_name = str_remove(course_name, "NA$"),
                         course_name = str_remove(course_name, "[0-9]"))

#                          course_name = str_replace_all(course_name, " ", ""))
#                          course_name = as.character(course_name))
uniquemodels <- c_names$course_name

distancemodels <- stringdistmatrix(uniquemodels,uniquemodels,method = "cosine")

rownames(distancemodels) <- uniquemodels

hc <- hclust(as.dist(distancemodels))

plot(hc)
rect.hclust(hc,k=15)




names(cnn_res)
str(cnn_res)
art <- cnn_res %>% 
  filter(pred_class == "0" & Y == '8') %>%
  select(-pred_class, -arts) %>% 
  group_by(Y) %>%
  summarise_all(mean)

art$Y <- with(art, reorder(Y, social.sciences))

library(ggplot2)
library(reshape2)

art.m <- melt(art)

art.m <- art.m %>% 
  mutate(md = rescale(value))

p <- ggplot(art.m, aes(variable, Y)) + geom_tile(aes(fill = value),
                                                    colour = "white") + scale_fill_gradient(low = "white",
                                                    high = "steelblue")
p + theme(axis.text.x = element_text(angle = 90, hjust = 1))
