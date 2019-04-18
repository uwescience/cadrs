# Potetnial Outcome table for NSC

library(tidyverse)

nsc_fn <- "~/data/cadrs/postSecDems.txt"
nsc <- read_delim(nsc_fn, delim = "|", col_names = TRUE, na = c("", "NA", "NULL")) 

enr_fn <- "~/data/cadrs/enrollments.txt"
enr <- read_delim(enr_fn, delim = "|", col_names = TRUE, na = c("", "NA", "NULL")) 
# Get HS grads
names(enr)

hs_grads = enr %>% 
  filter(dGraduate == 1) %>%
  select(ResearchID, dGraduate,ExpectedGradYear) %>%
  unique()


# Enrollment date span 
library(lubridate)

summary(nsc$EnrollmentBegin)
min(nsc$EnrollmentBegin, na.rm=T)

nsc_enroll <- nsc %>% 
  group_by(ResearchID) %>%
  slice(which.min(EnrollmentBegin))

table(nsc_enroll$v2year4year)
sum(is.na(nsc_enroll$v2year4year))
