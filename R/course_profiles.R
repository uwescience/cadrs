# cadrs course explore  

library(tidyverse)
library(openxlsx)
library(data.table)

gr_hist <- "~/data/cadrs/hsCourses.txt"
df_h <- read_delim(gr_hist, delim = "|", quote = "",col_names = TRUE, na = c("", "NA", "NULL")) 

names(df_h)


state_na <- df_h %>% 
  filter(is.na(StateCourseCode))


state_na %>%
  group_by(DistrictName) %>%
  count(sort = T)

dist_all <- c(
  "Auburn School District",
  "Federal Way School District",
  "Highline School District",
  "Renton School District",
  "Kent School District",
  "Tukwila School District"
)

df_h %>% 
  filter(is.na(StateCourseCode),
         DistrictName %in% dist_all,
         ReportSchoolYear >= 2015) %>%
  group_by(ReportSchoolYear) %>%
  count()

rmp_statecode_na <- df_h  %>%
  filter(is.na(StateCourseCode),
         DistrictName %in% dist_all,
         ReportSchoolYear >= 2015)
# look at district counts
rmp_statecode_na %>%
  count(DistrictName, sort = T) 

rmp_statecode_na %>%
  filter(DistrictName == "Auburn School District") %>%
  count(CourseTitle)

rmp_statecode_na %>%
  filter(DistrictName == "Kent School District") %>%
  count(CourseTitle)

rmp_statecode_na %>%
  filter(DistrictName == "Renton School District") %>%
  count(CourseTitle, sort = T)
# Look at the uique courses missing state course indentifiers
uniqueCourse_statecode_na <- rmp_statecode_na %>%
  group_by(CourseTitle) %>%
  count(CourseTitle)
######
# look at the grad classes 
post_df <- "~/data/cadrs/postSecDems.txt"
post <- read_delim(post_df, delim = "|", quote = "",col_names = TRUE, na = c("", "NA", "NULL")) 

# look at students that garduated HS
hs_df <- "~/data/cadrs/Dim_Student.txt"
hs <- read_delim(hs_df, delim = "|", quote = "",col_names = TRUE, na = c("", "NA", "NULL")) 

hs_grad <- hs %>%
  filter(AnyGraduate == 1,
         GradReqYear %in% c(2016, 2017))
hs_grad_id <- hs_grad %>%
  select(ResearchID) %>%
  unique()

# can we look at only the courses of folks who graduated in 2016 and 2017
course_grad <- df_h %>%
  filter(ResearchID %in% hs_grad_id$ResearchID)

test <- course_grad %>%
  filter(ResearchID == 1070645) %>%
  select(-TermEndDate) %>%
  select(ReportSchoolYear:StateCourseName) %>%
  unique()
