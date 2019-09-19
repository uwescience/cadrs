# training data clean up
# after running intial cnn it is obvious that the data has some classification errors 
# we have to clean those up to create a real training data set for any modeling 

library(tidyverse)
library(openxlsx)
library(data.table)
library(here)

source(here("settings.R"))

df_h <- read_delim(gr_hist, delim = "|", quote = "",col_names = TRUE, na = c("", "NA", "NULL")) 


######
# STATE COURSE CATALOGUE FILES YEARLY
# Load xlsx file from ospi
ospi_crs17 <- read.xlsx(ospi_crs17_fn, 4, startRow = 2) %>%
  select(State.Course.Code:X6) %>%
  rename(content_area = X6)
#####


ospi_crs16 <- read.xlsx(ospi_crs16_fn, 4, startRow = 2) %>%
  select(State.Course.Code:X6) %>%
  rename(content_area = X6)
####

# not used in this script, so Jeff commented this out
#ospi_crs15 <- fread(ospi_crs15_fn, skip = 2, header = T, drop = c("V1","V5"))

# not used in this script, so Jeff commented this out
#ospi_crs14 <- fread(ospi_crs14_fn, skip = 2, header = T, drop = c("V1","V5"))

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

# write_csv(opp_labels_d, path = "~/data/cadrs/cadrs_dupp_labels.csv")

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
  "Federal Way School District",
  "Highline School District",
  "Renton School District",
  "Kent School District",
  "Tukwila School District"
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
names(ospi_df)

cadrs_training <- left_join(ospi_df, cadrs_unique, by = c("State.Course.Code" = "StateCourseCode", 
                                                          "year" = "ReportSchoolYear")) %>%
  data.frame()

sum(is.na(cadrs_training$cadrs))/ length(cadrs_training$cadrs)

# 73% missing label
# some courses not offered in these districts
cadrs_training <- cadrs_training %>%
  select(-year, -StateCourseName, -Type..AP.IB.) %>%
  unique() %>%
  rename(subject = Subject.Area.Code)

# write_csv(cadrs_training, path = "~/data/cadrs/cadrs_training.csv")

# try adding more districts 
# ignoring errors 
cadrs_unique_all <- unique_labels(years = years_sub, districts = dist_all)

names(ospi_df)

cadrs_training_all <- left_join(ospi_df, cadrs_unique_all, by = c("State.Course.Code" = "StateCourseCode", 
                                                          "year" = "ReportSchoolYear")) %>%
  data.frame()

sum(is.na(cadrs_training_all$cadrs))/ length(cadrs_training_all$cadrs)

cadrs_training_all <- cadrs_training_all %>%
  select(-year) %>%
  unique()
# save with NAs 
#write_csv(cadrs_training_all, path = "~/data/cadrs/cadrs_training_all.csv")
#Try to clean-up- manual clean up
# use the sub districts without seattle, seattle is not good :( 

########
# using math key-words
math_inc <- c(
  "Trigonometry",
  "Math Analysis",
  "Linear Algebra",
  "Calculus"
)

math_x <- c(
  "Part 1",
  "Part 2",
  "Other",
  "General",
  "Business Math",
  "Particular Topics",
  "Independent Study",
  "Proficiency Development"
)

math_clean <- cadrs_training %>%
  filter(content_area == "Mathematics") %>%
  mutate(cadrs = if_else(is.na(cadrs) & str_detect(Name, paste(math_inc, collapse = '|')), 1 ,cadrs),
         cadrs = if_else(str_detect(Name, paste(math_x, collapse = '|')), 0 ,cadrs)) %>%
  filter(!is.na(cadrs)) %>%
  unique()

# foreign language 
# all but these key_words
lang_x <- c(
  "Field Experience", 
  "Conversation and Culture",
  "Other"
)

lang_inc <- c(
  "Japanese V",
  "Portuguese I",
  "Portuguese II",
  "Portuguese III",
  "Portuguese IV",
  "Latin IV"
)

lan_clean <- cadrs_training %>%
  filter(content_area == "Foreign Language and Literature") %>%
  mutate(cadrs = if_else(is.na(cadrs) & str_detect(Name, paste(lang_x, collapse = '|')), 0 ,1),
         cadrs = if_else(str_detect(Name, paste(lang_inc, collapse = '|')), 1 ,cadrs)) %>%
  filter(!is.na(cadrs)) %>%
  unique()

# sociology

soc_inc <- c(
  "IB",
  "AP"
)

soc_x <- c(
  "Other"
)

soc_clean <- cadrs_training %>%
  filter(content_area == "Social Sciences and History") %>%
  mutate(cadrs = if_else(is.na(cadrs) & str_detect(Name, paste(soc_inc, collapse = '|')), 1 ,cadrs),
         cadrs = if_else(str_detect(Name, paste(soc_x, collapse = '|')), 0 ,cadrs)) %>%
  filter(!is.na(cadrs)) %>%
  unique()

eng_x <- c(
  "Independent Study",
  "Second Language",
  "Development",
  "Debate",
  "Other",
  "Public Speaking",
  "Strategic Reading"
)

eng_clean <- cadrs_training %>%
  filter(content_area == "English Language and Literature") %>%
  mutate(cadrs = if_else(str_detect(Name, paste(eng_x, collapse = '|')), 0 ,cadrs)) %>%
  filter(!is.na(cadrs)) %>%
  unique()

sci_x <- c(
  "Other",
  "Particular Topics",
  "Technological Inquiry",
  "Life and Physical Sciences"
)

sci_clean <- cadrs_training %>%
  filter(content_area == "Life and Physical Sciences") %>%
  mutate(cadrs = if_else(str_detect(Name, paste(sci_x, collapse = '|')), 0 ,cadrs)) %>%
  filter(!is.na(cadrs)) %>%
  unique()

# Create file with additions and clear NAns 
cadrs_training_c <- cadrs_training %>%
  filter(!is.na(cadrs)) %>%
  bind_rows(math_clean,
            lan_clean,
            soc_clean,
            eng_clean,
            sci_clean) %>%
  unique()

table(cadrs_training_c$cadrs)


# Computer and Information Sciences
# IB Computer Science
cadrs_training_c %>%
  filter(Name == "IB Computer Science")

cadrs_training_c <- cadrs_training_c %>%
  mutate(cadrs = if_else(Name == "IB Computer Science", 1, cadrs))


##
#label as non-cadrs
non_cadr_sub <- c(
  "Physical, Health, and Safety Education",
  "Miscellaneous",
  "Military Science",
  "Manufacturing",
  "Health Care Sciences",
  "Hospitality and Tourism",
  "Transportation, Distribution and Logistics",
  "Public, Protective, and Government Service",
  "Nonsubject Specific"
  )

cadrs_training_c <- cadrs_training_c %>%
  mutate( cadrs = if_else(str_detect(content_area, paste(non_cadr_sub, collapse = '|')), 0 ,cadrs))

table(cadrs_training_c$cadrs)

write_csv(cadrs_training_c, path = cadrs_training_path)
