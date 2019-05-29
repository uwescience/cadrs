# training data clean up
# after running intial cnn it is obvious that the data has some classification errors 
# we have to clean those up to create a real training data set for any modeling 

library(tidyverse)
library(openxlsx)
library(data.table)

gr_hist <- "~/data/cadrs/hsCourses.txt"
df_h <- read_delim(gr_hist, delim = "|", quote = "",col_names = TRUE, na = c("", "NA", "NULL")) 

# state/district course name combination
# if state course exists use that otherwise use the district course code
df_h <- df_h %>%
  mutate(crs_combo = ifelse(is.na(StateCourseName), CourseTitle,StateCourseName))

table(df_h$ReportSchoolYear)

######
# STATE COURSE CATALOGUE FILES YEARLY
# Load xlsx file from ospi
ospi_crs17_fn <- "~/data/cadrs/2016-17StateCourseCodes.xlsx"

ospi_crs17 <- read.xlsx(ospi_crs17_fn, 4, startRow = 2) %>%
  select(State.Course.Code:X6) %>%
  rename(content_area = X6)
#####
ospi_crs16_fn <- "~/data/cadrs/2015-16-StateCourseCodes.xlsx"

ospi_crs16 <- read.xlsx(ospi_crs16_fn, 4, startRow = 2) %>%
  select(State.Course.Code:X6) %>%
  rename(content_area = X6)
####
ospi_crs15_fn <- "~/data/cadrs/2014_15_StateCourseCodes.csv"
ospi_crs15 <- fread(ospi_crs15_fn, skip = 2, header = T, drop = c("V1","V5"))

ospi_crs14_fn <- "~/data/cadrs/2013_14_StateCourseCodes.csv"
ospi_crs14 <- fread(ospi_crs14_fn, skip = 2, header = T, drop = c("V1","V5"))

# Create unique school courses from student file by year and attach course discriptions
names(df_h)

# Look at same state course being classified as both CADR and != CADR
## use for district: DistrictName %in% c("Seattle Public Schools")
st_17 <- df_h %>%
  select(ReportSchoolYear, DistrictName, SchoolCode, CourseID, 
         StateCourseCode, StateCourseName, 
         CollegeAcademicDistributionRequirementsFlag) %>%
  filter(ReportSchoolYear == 2017) %>%
  mutate(StateCourseCode = str_pad(StateCourseCode, 5, pad = "0")) %>% unique()

miss_al_17 <- st_17 %>% 
  group_by(DistrictName, StateCourseName, CollegeAcademicDistributionRequirementsFlag) %>%
  summarise(n = n()) %>%
  mutate(percent = n/sum(n, na.rm=T)) 

total_unique_17 <- miss_al_17 %>%
  select(DistrictName, StateCourseName) %>%
  unique() %>%
  group_by(DistrictName) %>%
  summarise(n = n())

dups_17 <- miss_al_17 %>%
  filter(duplicated(StateCourseName)) %>%
  group_by(DistrictName) %>%
  summarise(n_m = n())

opp_labels_d_17 <- inner_join(total_unique_17, dups_17) %>%
  mutate(percent = n_m/n * 100,
         year = '2016-17')
##### 2016
st_16 <- df_h %>%
  select(ReportSchoolYear, DistrictName, SchoolCode, CourseID, 
         CourseTitle, StateCourseCode, StateCourseName, 
         AdvancedPlacementFlag, CollegeAcademicDistributionRequirementsFlag) %>%
  filter(ReportSchoolYear == 2016) %>%
  mutate(StateCourseCode = str_pad(StateCourseCode, 5, pad = "0")) %>% unique()

miss_al_16 <- st_16 %>% 
  group_by(DistrictName, StateCourseName, CollegeAcademicDistributionRequirementsFlag) %>%
  summarise(n = n()) %>%
  mutate(percent = n/sum(n, na.rm=T)) 

total_unique_16 <- miss_al_16 %>%
  select(DistrictName, StateCourseName) %>%
  unique() %>%
  group_by(DistrictName) %>%
  summarise(n = n())

dups_16 <- miss_al_16 %>%
  filter(duplicated(StateCourseName)) %>%
  group_by(DistrictName) %>%
  summarise(n_m = n())

opp_labels_d_16 <- inner_join(total_unique_16, dups_16) %>%
  mutate(percent = n_m/n * 100,
         year = '2015-16')

## 2015
st_15 <- df_h %>%
  select(ReportSchoolYear, DistrictName, SchoolCode, CourseID, 
         CourseTitle, StateCourseCode, StateCourseName, 
         AdvancedPlacementFlag, CollegeAcademicDistributionRequirementsFlag) %>%
  filter(ReportSchoolYear == 2015) %>%
  mutate(StateCourseCode = str_pad(StateCourseCode, 5, pad = "0")) %>% unique()

miss_al_15 <- st_15 %>% 
  group_by(DistrictName, StateCourseName, CollegeAcademicDistributionRequirementsFlag) %>%
  summarise(n = n()) %>%
  mutate(percent = n/sum(n, na.rm=T)) 

total_unique_15 <- miss_al_15 %>%
  select(DistrictName, StateCourseName) %>%
  unique() %>%
  group_by(DistrictName) %>%
  summarise(n = n())

dups_15 <- miss_al_15 %>%
  filter(duplicated(StateCourseName)) %>%
  group_by(DistrictName) %>%
  summarise(n_m = n())

opp_labels_d_15 <- inner_join(total_unique_15, dups_15) %>%
  mutate(percent = n_m/n * 100,
         year = '2014-15')
## 2014
st_14 <- df_h %>%
  select(ReportSchoolYear, DistrictName, SchoolCode, CourseID, 
         CourseTitle, StateCourseCode, StateCourseName, 
         AdvancedPlacementFlag, CollegeAcademicDistributionRequirementsFlag) %>%
  filter(ReportSchoolYear == 2015) %>%
  mutate(StateCourseCode = str_pad(StateCourseCode, 5, pad = "0")) %>% unique()

miss_al_14 <- st_14 %>% 
  group_by(DistrictName, StateCourseName, CollegeAcademicDistributionRequirementsFlag) %>%
  summarise(n = n()) %>%
  mutate(percent = n/sum(n, na.rm=T)) 

total_unique_14 <- miss_al_14 %>%
  select(DistrictName, StateCourseName) %>%
  unique() %>%
  group_by(DistrictName) %>%
  summarise(n = n())

dups_14 <- miss_al_14 %>%
  filter(duplicated(StateCourseName)) %>%
  group_by(DistrictName) %>%
  summarise(n_m = n())

opp_labels_d_14 <- inner_join(total_unique_14, dups_14) %>%
  mutate(percent = n_m/n * 100,
         year = '2013-14')
## append 
opp_labels_d <- bind_rows(opp_labels_d_17,opp_labels_d_16, opp_labels_d_15, opp_labels_d_14)

write_csv(opp_labels_d, path = "~/data/cadrs/cadrs_dupp_labels.csv")
###########
cadrs_training_17 <- st_17 %>%
  select(StateCourseCode, cadrs = CollegeAcademicDistributionRequirementsFlag) %>%
  unique()

cadrs_17 <- left_join(ospi_crs17, cadrs_training_17, by = c("State.Course.Code" = "StateCourseCode") )

# write_csv(cadrs_training_17, path = "~/data/cadrs/cadrs_training_17_test.csv")
##########
