# enrollment and 2017 grad cohorts 

# Create cohort grade history file
library(tidyverse)
library(openxlsx)
library(data.table)
library(here)

source("/home/joseh/source/cadrs/settings.R")

stu_enroll_fn <- enrollment_path
sch_dim_fn <- dim_school_path
stu_dim_fn <- dim_student_path
gr_hist_fn <- gr_hist
gr_hist_2017_fn <- course_2017_cohort_path #from SQL
gr_enroll_cohort_fn <- enroll_2017_cohort_path #from SQL

stu_enroll <- fread(stu_enroll_fn, na.strings = c("NA", "NULL"))
sch_dim <- fread(sch_dim_fn, quote="", na.strings = c("NA", "NULL"))
stu_dim <- fread(stu_dim_fn, quote="", na.strings = c("NA", "NULL"))

gr_hist <- fread(gr_hist_fn, quote="", na.strings = c("NA", "NULL", ''))
gr_hist_2017 <- fread(gr_hist_2017_fn, na.strings = c("NA", "NULL", ''))
gr_enroll_cohort <- fread(gr_enroll_cohort_fn, na.strings = c("NA", "NULL", ''))

table(gr_hist_2017$content_area)
table(gr_hist_2017$dTermEndYear)
names(gr_hist)

gr_enroll_cohort %>%
  group_by(DistrictCode) %>%
  summarise(n = n())
# Look at the course file and clean up some of the missing course information
names(gr_hist_2017)

missing_carea <- gr_hist_2017 %>%
  filter(is.na(content_area))

names_NA <- gr_hist_2017 %>%
  filter(is.na(State.Course.Code)) %>%
  select(CourseTitle) %>% unique()


######
re_m <- c(
  "alg", 
  "calc", 
  "math",
  "geom"
  )
re_comp <- c(
  "computer",
  "comp s"
)
re_ss <- c(
  "world hi", 
  "world geo", 
  "socio",
  "hist",
  "govt",
  "econ",
  "polit sci",
  "psych",
  "gov",
  "geo")

re_lang <- c(
  "japa", 
  "span", 
  "frenc", 
  "ara",
  "germ",
  "kor",
  "chin",
  "samoa",
  "sign",
  "latin",
  "ital",
  "somal",
  "viet",
  "russ",
  "ib lan")
re_sci <- c(
  "bio",
  "physics",
  "chem",
  "sci9",
  "science",
  "enviro",
  "physical sci",
  "phys sc"
)
re_eng <- c(
  "english",
  "lamgiage arts",
  "eng l",
  "amer",
  "language a",
  "lit/",
  "lit comp",
  "eng 1",
  "lit com",
  "la 9"
)


comp_cleanup <- names_NA %>%
  mutate(CourseTitle_l = tolower(CourseTitle),
         ospi_label = if_else(str_detect(CourseTitle_l, paste(re_m, collapse = '|')), "Mathematics",''),
         ospi_label = if_else(str_detect(CourseTitle_l, paste(re_comp, collapse = '|')), "Computer and Information Sciences",ospi_label),
         ospi_label = if_else(str_detect(CourseTitle_l, paste(re_ss, collapse = '|')), "Social Sciences and History",ospi_label),
         ospi_label = if_else(str_detect(CourseTitle_l, paste(re_lang, collapse = '|')), "Foreign Language and Literature",ospi_label),
         ospi_label = if_else(str_detect(CourseTitle_l, paste(re_sci, collapse = '|')), "Life and Physical Sciences",ospi_label),
         ospi_label = if_else(str_detect(CourseTitle_l, paste(re_eng, collapse = '|')), "English Language and Literature",ospi_label))

re_art <- c(
  "music",
  "choru",
  "acap",
  "drawin",
  "drama",
  "piano",
  "guita",
  "choir",
  "dance",
  "ap stud",
  "art ",
  "visual",
  "jazz",
  "musc",
  "wind"
)
comp_cleanup <- comp_cleanup %>%
  mutate(
         ospi_label = if_else(str_detect(CourseTitle_l, paste(re_art, collapse = '|')), "Fine and Performing Arts",ospi_label))

left <- comp_cleanup %>%
  filter(ospi_label == '')

###
records_NA <- gr_hist_2017 %>%
  filter(is.na(State.Course.Code)) %>%
  left_join(.,comp_cleanup %>%
              select(CourseTitle, ospi_label), by = "CourseTitle" ) %>%
  select(-content_area) %>% rename(content_area = ospi_label)

gr_hist_2017 <- bind_rows(
  gr_hist_2017 %>%
    filter(!is.na(State.Course.Code)),
  records_NA
)
names(gr_hist_2017)
### add missing course titles to ospi column from school specific file....

gr_hist_2017 <- gr_hist_2017 %>%
  mutate(state_spec_course = StateCourseName)

key_words <- c(
  'Other',
  'Aide',
  'Independent Study',
  'Workplace Experience',
  'Comprehensive'
)

gr_hist_2017 <- gr_hist_2017 %>%
  mutate(
    state_spec_course = if_else(str_detect(state_spec_course, "English as a Second Language"), CourseTitle, state_spec_course),
    state_spec_course = if_else(str_detect(CourseTitle, "SCI9"), CourseTitle,state_spec_course),
    state_spec_course = if_else(str_detect(CourseTitle, "COMMUN ARTS"), CourseTitle,state_spec_course),
    state_spec_course = if_else(str_detect(state_spec_course, paste(key_words, collapse = '|')), "GENERAL",state_spec_course),
    state_spec_course = ifelse(state_spec_course == 'GENERAL', CourseTitle, state_spec_course),
    state_spec_course = ifelse(state_spec_course == '', CourseTitle, state_spec_course),
    state_spec_course = str_remove(state_spec_course, "\\d{5}$")
    )

missing_carea <- gr_hist_2017 %>%
  filter(content_area=='' & dSchoolYear == 2014)
table(missing_carea[,"dSchoolYear"])

results_unique_c <- missing_carea %>% select(state_spec_course) %>% unique()
# Conditionals for 12th grade electives that count towards cadrs (journalism etc...)

write_csv(gr_hist_2017, course_2017_cohort_clean_path)
##