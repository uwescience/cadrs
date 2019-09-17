# Create cohort grade history file
library(tidyverse)
library(openxlsx)
library(data.table)

gr_hist_fn <- "~/data/cadrs/hsCourses.txt"
sch_dim_fn <- "/home/ubuntu/data/Dim.School.txt"
stu_dim_fn <- "~/data/cadrs/Dim_Student.txt"

gr_hist <- fread(gr_hist_fn, quote="", na.strings = c("NA", "NULL"))
sch_dim <- fread(sch_dim_fn, quote="", na.strings = c("NA", "NULL"))
stu_dim <- fread(stu_dim_fn, quote="", na.strings = c("NA", "NULL"))

# get school IDs from RMP region
sch_dim %>%
  select(SchoolType) %>%
  table()

dim_sch_rmp <- sch_dim %>%
  filter(dRoadMapRegionFlag == 1) %>%
  select(SchoolCode, DistrictCode) %>%
  unique()

# get cohort of 2017 HS grads 
hs_grad_id <- stu_dim %>%
  filter(AnyGraduate == 1,
         GradReqYear %in% c(2017)) %>%
  select(ResearchID) 

# look at only the courses of folks who graduated in 2017
crs_hs_grad <- gr_hist %>%
  filter(ResearchID %in% hs_grad_id$ResearchID)

# look only at rmp schools 
crs_hs_grad <- crs_hs_grad %>%
  filter(SchoolCode %in% dim_sch_rmp$SchoolCode & DistrictCode %in% dim_sch_rmp$DistrictCode) %>%
  select(-TermEndDate) %>%
  select(DistrictCode:StateCourseName, dSchoolYear, cadr_ccer=CollegeAcademicDistributionRequirementsFlag) %>%
  mutate(StateCourseCode = as.character(StateCourseCode),
          StateCourseCode = str_pad(StateCourseCode, 5, pad = "0")) %>%
  unique()

rm(gr_hist)
rm(stu_dim)
# attach state course information using stcourse code and year 
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

ospi_crs15 <- fread(ospi_crs15_fn, skip = 2, header = T, drop = c("V1","V5")) %>%
  rename(State.Course.Code = `State Course Code`) %>%
  mutate(State.Course.Code = as.character(State.Course.Code),
         State.Course.Code = str_pad(State.Course.Code, 5, pad = "0"))

ospi_crs14_fn <- "~/data/cadrs/2013_14_StateCourseCodes.csv"

ospi_crs14 <- fread(ospi_crs14_fn, skip = 2, header = T, drop = c("V1","V5")) %>%
  rename(State.Course.Code = `State Course Code`) %>%
  mutate(State.Course.Code = as.character(State.Course.Code),
         State.Course.Code = str_pad(State.Course.Code, 5, pad = "0"))
##
# use years with subject areas
str(ospi_crs)
ospi_crs <- bind_rows(
  ospi_crs17,
  ospi_crs16
  ) %>%
  unique()

test14 <- ospi_crs14 %>%
  filter(!State.Course.Code %in% ospi_crs$State.Course.Code)

test15 <- ospi_crs15 %>%
  filter(!State.Course.Code %in% ospi_crs$State.Course.Code)
# looks like only 23 codes don't exist for previous years use those codes w/out subjects
table(ospi_crs$content_area)

test15$content_area <- c(
  'Mathematics',
  'Life and Physical Sciences',
  'Life and Physical Sciences',
  'Life and Physical Sciences',
  '',
  'Fine and Performing Arts',
  'Fine and Performing Arts',
  'Fine and Performing Arts',
  'Mathematics',
  '',
  '',
  '',
  '',
  '',
  '',
  '',
  '',
  '',
  '',
  '',
  '',
  '',
  '',
  '',
  '',
  ''
)

# append 15 to ospi codes
ospi_crs <- bind_rows(
  ospi_crs,
  test15
) %>%
  unique()
### Clean up ospi cats and append to student file

electives <- c("Business and Marketing",
               "Human Services",
               "Agriculture, Food, and Natural Resources",
               "Hospitality and Tourism", 
               "Health Care Sciences",
               "Transportation, Distribution and Logistics",
               "Manufacturing",
               "Architecture and Construction",
               "Military Science",
               "Public, Protective, and Government Service",
               "Religious Education and Theology",
               "Engineering and Technology",
               "Communications and Audio/Visual Technology")

ospi_crs_ap <- ospi_crs %>%
  mutate(ospi_sub = ifelse(content_area %in% electives, "elective", content_area)) %>%
  select(-`Type.(AP/IB)`, -Subject.Area.Code)

crs_hs_grad <- left_join(crs_hs_grad, 
                              ospi_crs_ap, by=c("StateCourseCode"="State.Course.Code"))

write_csv(crs_hs_grad, "/home/joseh/data/crs_2017_cohort.csv")

# look at edge cases by year 2010 and 2011 seem like online school cases
crs_hs_grad %>%
  select(dSchoolYear) %>%
  table()

crs_hs_grad %>%
  filter(dSchoolYear == 2010)
  #filter(ResearchID == 1070645)

stu_dim_fn <- "~/data/cadrs/cadrs_training.csv"

training <- fread(stu_dim_fn, na.strings = c("NA", "NULL"))
