library(tidyverse)
library(caret)
library(pryr)

##Transform course table to one feature per course
gr_hist <- "~/data/cadrs/hsCourses.txt"
df_h <- read_delim(gr_hist, delim = "|", col_names = TRUE, na = c("", "NA", "NULL")) 
object_size(df_h)
# how many unique students in this dataset
length(unique(df_h$ResearchID))
# 127,056 students 

#how many unique courses
length(unique(df_h$CourseTitle))

source("/home/joseh/source/cadrs-collaboration/preprocess/course_preprocess.R")

# We need to reshape the dataframe
# Try on whole data (might take some time)
# We expect 127,056 cases and ~7500 columns 
pivot_matrix <- df_h %>%
  select(ResearchID,crs_combo) %>%
  distinct(.) 
  
pivot_matrix <- pivot_matrix %>%
  mutate(present = 1) 

pivot_matrix <- pivot_matrix %>%
  spread(crs_combo,present,fill = 0)
#####
# Create using clean course
# 127,000 x ~6500
#####

pivot_matrix2 <- df_h %>%
  select(ResearchID,crs_copy) %>%
  mutate(crs_copy = str_replace_all(crs_copy,"[[:punct:][:space:]]+","."),
         crs_copy = str_remove(crs_copy, "^\\d+"),
         crs_copy = str_replace_all(crs_copy, "[^[:alnum:]]", ".")) %>% 
  distinct(.)

# t <- pivot_matrix2 %>% 
#   filter(str_detect(crs_copy, "[^[:alnum:]]"))

pivot_matrix2 <- pivot_matrix2 %>%
  mutate(present = 1)

pivot_matrix2 <- pivot_matrix2 %>%
  spread(crs_copy,present,fill = 0)
#####
# attach hs_grad info
#####
source("/home/joseh/source/cadrs-collaboration/preprocess/post_hs_preprocess.R")
hs_grad_courses <- left_join(pivot_matrix, hs_grads, by = c("ResearchID"))

hs_grad_courses <- hs_grad_courses %>%
  filter(dGraduate == 1)
###
hs_grad_courses2 <- left_join(pivot_matrix2, hs_grads, by = c("ResearchID"))

hs_grad_courses2 <- hs_grad_courses2 %>%
  filter(dGraduate == 1)

sum(is.na(hs_grad_courses2))
###
# Add target from NSC
###

hs_grad_courses_1 <- left_join(hs_grad_courses, nsc_enroll %>%
                                                select(ResearchID, v2year4year) %>%
                                                unique(), by = c("ResearchID")) %>%
                      mutate(any_college = ifelse(is.na(v2year4year),0,1),
                             c4_year = case_when(v2year4year == "4" ~ 1,
                                                 v2year4year == "2" ~ 0,
                                                 v2year4year == "L" ~ 0,
                                                 is.na(v2year4year) ~ 0,
                                                 TRUE ~ as.double(.$v2year4year))
                                           
                      )
rm(df_h)
rm(enr)
gc()
table(hs_grad_courses_1$any_college)
table(hs_grad_courses_1$c4_year)
###########
hs_grad_courses_2 <- left_join(hs_grad_courses2, nsc_enroll %>%
                                 select(ResearchID, v2year4year) %>%
                                 unique(), by = c("ResearchID")) %>%
  mutate(any_college = ifelse(is.na(v2year4year),0,1),
         c4_year = case_when(v2year4year == "4" ~ 1,
                             v2year4year == "2" ~ 0,
                             v2year4year == "L" ~ 0,
                             is.na(v2year4year) ~ 0,
                             TRUE ~ as.double(.$v2year4year))
         
  ) %>%
  select(-v2year4year)

table(hs_grad_courses_2$any_college)
table(hs_grad_courses_2$c4_year)
# sum(is.na(hs_grad_courses_2))

qr(hs_grad_courses_2)$rank

###############
library(caret)
library(randomForest)

# mat = lapply(c("LogitBoost", 'xgbTree', 'rf', 'svmRadial'), 
#              function (met) {
#                train(subClasTrain~., method=met, data=smallSetTrain)
#              })

# data.split <- createDataPartition(y=hs_grad_courses_2$any_college,p=0.6,list=FALSE)
# training1<-hs_grad_courses_2[data.split,]
# testing<-hs_grad_courses_2[-data.split,]

qr(M)$rank
# 3950

tapply(hs_grad_courses_2$any_college, hs_grad_courses_2$ExpectedGradYear, mean)
table(hs_grad_courses_2$ExpectedGradYear)
training <- hs_grad_courses_2 %>% filter(ExpectedGradYear <= 2014)
test <- hs_grad_courses_2 %>% filter(ExpectedGradYear == 2015 | ExpectedGradYear == 2016)

prop.table(table(hs_grad_courses_2$any_college))
prop.table(table(training$any_college))
prop.table(table(test$any_college))

rfdata_tr <- training %>% select(-ExpectedGradYear,-c4_year, ResearchID) %>% 
  mutate(any_college = as.character(any_college),
         any_college = as.factor(any_college))
gc()
rm(df_h)
rm(enr)

# (dat <- structure(list(`PCNA-AS1` = c(1, 2, 3), resp = structure(c(2L, 2L, 1L), .Label = c("0", "1"), class = "factor")), .Names = c("PCNAAS1.", "resp"), row.names = c(NA, -3L), class = "data.frame"))
# mod <- randomForest(resp~., data=dat)
str(rfdata_tr$any_college)

fit <- randomForest(any_college ~., data=rfdata_tr, importance = TRUE, mtry = 2)

print(fit)

imp <- fit$importance
###ap calculus ab most important 

# test on the data for earlier times 
rfdata_test <- test %>% select(-ExpectedGradYear,-c4_year, -ResearchID) %>% 
  mutate(any_college = as.character(any_college),
         any_college = as.factor(any_college))

pred <- predict(fit, newdata = rfdata_test)

table(rfdata_tr$any_college)
table(rfdata_test$any_college)

table(pred, rfdata_test$any_college, type='response')

12074/nrow(rfdata_test)

# Only classifying things into one class
table(pred)

misClasificError <- mean(pred != rfdata_test$any_college, na.rm=T)
print(paste('Accuracy',1-misClasificError))
# [1] "Accuracy 0.682879927605905"

accM <- data.frame(cbind(rfdata_test$any_college, pred))
accM <- na.omit(accM)
library(ROSE)
accuracy.meas(accM$V1, accM$pred)

# precision: 0.683
# recall: 1.000
# F: 0.406

roc.curve(accM$V1, accM$pred)
# Area under the curve (AUC): 0.500

#####
# RUN USING CONVENTIONAL TEST/TRAIN SPLIT

data.split <- createDataPartition(y=hs_grad_courses_2$any_college,p=0.6,list=FALSE)
training1 <-hs_grad_courses_2[data.split,]
testing1 <-hs_grad_courses_2[-data.split,]

rfdata_tr2 <- training1 %>% select(-ExpectedGradYear,-c4_year) %>% 
  mutate(any_college = as.character(any_college),
         any_college = as.factor(any_college))
gc()
rm(df_h)
rm(enr)

str(rfdata_tr2$any_college)

fit2 <- randomForest(any_college ~., data=rfdata_tr2, importance = TRUE, mtry = 2)
