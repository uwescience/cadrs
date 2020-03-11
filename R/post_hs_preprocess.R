# Potetnial Outcome table for NSC

library(tidyverse)

nsc_fn <- "/home/joseh/data/cadr_update/postSecDems.txt"
nsc <- fread(nsc_fn, na.strings = c("NA", "NULL"))

enr_fn <- "~/data/cadrs/enrollments.txt"
enr <- read_delim(enr_fn, delim = "|", col_names = TRUE, na = c("", "NA", "NULL")) 
# Get HS grads
names(enr)

hs_grads = enr %>% 
  filter(dGraduate == 1,
         ExpectedGradYear == 2017) %>%
  select(ResearchID, dGraduate,ExpectedGradYear) %>%
  unique()

nsc_sub <- nsc %>%
  filter(ResearchID %in% agg_results$ResearchID)

str(nsc_sub$EnrollmentBegin)

nsc %>%
  filter(ResearchID == 1000511)
# agg results only get student with complete high school records!!!!
# right now there are some that don't have this create a flag 

test2 <- agg_results %>%
  filter(ResearchID == 1000511)
test3 <- results %>%
  filter(ResearchID == 1000511)

tt <- nsc_sub %>%
  filter(v2year4year== 4) %>%
  filter(EnrollmentBegin >= as.Date('2017-01-01') & EnrollmentBegin <= as.Date('2017-12-31')) %>%
  group_by(ResearchID) %>%
  summarise(n = n())
  mutate(EnrollmentBegin = as.Date(EnrollmentBegin)) %>%
  filter(EnrollmentBegin >= as.Date('2017-01-01'))

