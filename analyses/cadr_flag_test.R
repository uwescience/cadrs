# HANDLING ELECTIVES THAT SATISFY CADR REQS WHEN AGGREGATING 
library(tidyverse)
library(data.table)

results_fn <- "/home/joseh/data/svm_cadr_student_predictions_CV.csv"
# results_cadr_fn = "/home/ubuntu/data/db_files/cadrs_results_20082019.csv"

results <- fread(results_fn, na.strings = c("NA", "NULL", ""))
#results_cadrs = fread(results_cadr_fn, na.strings = c("NA", "NULL"))
names(results)
tt <- results %>% 
  filter(dSchoolYear == 2018) %>%
  select(dSchoolYear) %>% table()
table(results$content_area)

# look at science course codes 

sci <- results %>%
  filter(content_area == 'Life and Physical Sciences') %>%
  select(CourseTitle, state_spec_course,p_CADRS, CollegeAcademicDistributionRequirementsFlag) %>%
  unique()

# Create a test dataset, can we capture 1 year of ell eligibility 

stud_jn = results %>%
  filter(ResearchID %in% c(1037790, 1094060))

stud_ell = results %>%
  filter(ResearchID %in% c(1030696, 1127005))



data_test <- bind_rows(
  stud_jn,
  stud_ell
)

test <- data_test %>%
  group_by(ResearchID) %>%
  filter(content_area == 'English Language and Literature' & p_CADRS == 1) %>%
  mutate(english_cadr = case_when(any(sum(CreditsEarned, na.rm=T) < 4) ~ 1, TRUE ~ 0)) %>%
  summarise(english_cadr = max(english_cadr)) %>%
  filter(english_cadr == 1)

ell <- data_test %>%
  filter(ResearchID %in% test$ResearchID) %>%
  group_by(ResearchID) %>%
  filter(content_area == 'English Language and Literature' & str_detect(state_spec_course, 'ELL|ELD')) %>%
  summarise(ell_cred = sum(CreditsEarned, na.rm=T)) %>%
  mutate(ell_eng = case_when(ell_cred >=1 ~ 1, TRUE ~ 0))

  # students who have less than 3 english credits 
    
         
result_test <- data_test %>%
  filter(p_CADRS == 1) %>%
  group_by(ResearchID, content_area) %>%
  summarise(sum_cred = sum(CreditsEarned, na.rm = T)) %>%
  mutate(content_area = str_replace_all(content_area,"[[:punct:]]|\\s+","_")) %>%
  spread(content_area, sum_cred)

final_table <- left_join(result_test, ell, by = "ResearchID")
names(final_table)

# agg test 
agg_test <- final_table %>%
  mutate(eng_cadr = sum(English_Language_and_Literature,ell_eng,
                                   Communications_and_Audio_Visual_Technology, na.rm=T))
####
##
## cohort 17
eng_p <- results %>%
  group_by(ResearchID) %>%
  filter(content_area == 'English Language and Literature' & p_CADRS == 1) %>%
  mutate(english_cadr = case_when(any(sum(CreditsEarned, na.rm=T) < 4) ~ 1, TRUE ~ 0)) %>%
  summarise(english_cadr = max(english_cadr)) %>%
  filter(english_cadr == 1)

ell <- results %>%
  filter(ResearchID %in% eng_p$ResearchID) %>%
  group_by(ResearchID) %>%
  filter(content_area == 'English Language and Literature' & str_detect(state_spec_course, 'ELL|ELD')) %>%
  summarise(ell_cred = sum(CreditsEarned, na.rm=T)) %>%
  mutate(ell_eng = case_when(ell_cred >=1 ~ 1, TRUE ~ 0))

# students who have less than 3 english credits 
result_wide <- results %>%
  filter(p_CADRS == 1) %>%
  group_by(ResearchID, content_area) %>%
  summarise(sum_cred = sum(CreditsEarned, na.rm = T)) %>%
  mutate(content_area = str_replace_all(content_area,"[[:punct:]]|\\s+","_")) %>%
  spread(content_area, sum_cred)

final_table <- left_join(result_wide, ell, by = "ResearchID")
names(final_table)
# agg english
agg_results <- final_table %>%
  mutate(eng_cadr = sum(English_Language_and_Literature,ell_eng,
                        Communications_and_Audio_Visual_Technology, na.rm=T)) %>%
  mutate( cadr = case_when(any(eng_cadr >= 4 & Mathematics >= 3 & Social_Sciences_and_History >= 3 & Life_and_Physical_Sciences >= 2 & Foreign_Language_and_Literature >= 2 & 
                                 Fine_and_Performing_Arts >= 1) ~1, TRUE ~ 0))

## Using Tukwila as a test case 
# attach district membership 
stu_enroll_fn <-"~/data/cadr_update/enrollments.txt"
stu_enroll <- fread(stu_enroll_fn, na.strings = c("NA", "NULL"))

names(stu_enroll)

stu_enroll_2017 <- stu_enroll %>%
  filter(GradeLevelSortOrder == 15, GradReqYear == 2017, dGraduate == 1) %>%
  select(ResearchID, DistrictCode) %>%
  unique()

dist_2017 <- stu_enroll %>%
  filter(GradeLevelSortOrder == 15, GradReqYear == 2017, dGraduate == 1) %>%
  select(DistrictCode, DistrictName) %>%
  unique()

dist_2017 <- dist_2017[c(1:7),]

stu_enroll_2017 %>% 
  select(ResearchID) %>%
  unique() %>% nrow()

stu_enroll_2017 <- left_join(stu_enroll_2017, dist_2017, by = "DistrictCode")

agg_results_dist <- left_join(agg_results, stu_enroll_2017, by = "ResearchID")

agg_results_tuk <- agg_results_dist %>%
  filter(DistrictName == "Tukwila School District")

table(agg_results_tuk$cadr)

agg_results_tuk %>%
  filter(sum(Mathematics, na.rm=T) >= 3) %>%
  nrow()

agg_results_tuk %>%
  filter(sum(eng_cadr, na.rm=T) >= 4) %>%
  nrow()


###
