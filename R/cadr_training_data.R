# training data clean up
# after running intial cnn it is obvious that the data has some classification errors 
# we have to clean those up to create a real training data set for any modeling 

library(tidyverse)
library(openxlsx)
library(data.table)

gr_hist <- "~/data/cadrs/hsCourses.txt"
df_h <- read_delim(gr_hist, delim = "|", quote = "",col_names = TRUE, na = c("", "NA", "NULL")) 

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
## use for district: DistrictName %in% c("CHOOSE DISTRICT")

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

rm(list = ls(pattern = "dups_"))
rm(list = ls(pattern = "opp_"))
rm(list = ls(pattern = "miss_"))
rm(list = ls(pattern = "total_"))
rm(list = ls(pattern = "st_"))

# There are some obvious data errors 
# Create a somewhat cleaned training datset for prototyping baseline model
# Look at the dissagreements for less than 15% disagreeing labels and manually clean those
# Not the best approach but good for now (2016 & 2017)

dist_all <- c(
  "Auburn School District",
  "Federal Way School District",
  "Highline School District",
  "Renton School District",
  "Kent School District",
  "Tukwila School District",
  "Seattle Public Schools"
)

# Districts with lowest known data errors
districts_sub <- c(
  "Auburn School District",
  "Highline School District"
) 

# years 
years_all <- c(as.character(2014:2017))

years_sub <- c("2016", "2017")

unique_labels <- function(years, districts) {
  output <- list()

  for (i in years) {
    setup_df <- df_h %>%
      select(ReportSchoolYear, DistrictName, CourseTitle, 
             StateCourseCode, StateCourseName, 
             cadrs=CollegeAcademicDistributionRequirementsFlag) %>%
      mutate(ReportSchoolYear = as.character(ReportSchoolYear)) %>%
      filter(ReportSchoolYear %in% years,
             DistrictName %in% districts) %>%
      mutate(StateCourseCode = str_pad(StateCourseCode, 5, pad = "0")) %>% 
      unique() %>%
      group_by(ReportSchoolYear, StateCourseCode, StateCourseName, cadrs) %>%
      summarise(n = n()) %>%
      select(-n)
    
    dups <- setup_df %>%
      filter(duplicated(StateCourseName))
    
    dedup_rows <- bind_rows( setup_df %>%
      filter(duplicated(StateCourseName)), 
      setup_df %>% 
      filter(!StateCourseName %in% dups$StateCourseName))
    
    output[[i]] <- dedup_rows
  }
  output_df <- unique(do.call(rbind.data.frame, output))
}

cadrs_unique <- unique_labels(years = years_sub, districts = districts_sub)

# attach course descriptions from ospi
ospi_df <- bind_rows(ospi_crs17 %>% mutate(year = "2017"), 
                     ospi_crs16 %>% mutate(year = "2016"))

cadrs_training <- inner_join(cadrs_unique, ospi_df, by = c("StateCourseCode" = "State.Course.Code", 
                                                           "ReportSchoolYear" = "year")) %>%
  data.frame()


cadrs_training <- cadrs_training %>%
  select(-ReportSchoolYear) %>%
  unique() %>%
  rename(ap_ib=Type..AP.IB., subject = Subject.Area.Code)

write_csv(cadrs_training, path = "~/data/cadrs/cadrs_training.csv")
##########
