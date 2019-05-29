# training data clean up
# after running intial cnn it is obvious that the data has some classification errors 
# we have to clean those up to create a real training data set for any modeling 

library(tidyverse)
library(openxlsx)

gr_hist <- "~/data/cadrs/hsCourses.txt"
df_h <- read_delim(gr_hist, delim = "|", quote = "",col_names = TRUE, na = c("", "NA", "NULL")) 

# state/district course name combination
# if state course exists use that otherwise use the district course code
df_h <- df_h %>%
  mutate(crs_combo = ifelse(is.na(StateCourseName), CourseTitle,StateCourseName))


table(df_h$ReportSchoolYear)
########
###LOOK AT STATE COURSE FILE COVERAGE
# Load xlsx file from ospi
ospi_crs_fn <- "~/data/cadrs/2016-17StateCourseCodes.xlsx"

ospi_crs17 <- read.xlsx(ospi_crs_fn, 4, startRow = 2) %>%
  select(State.Course.Code:X6) %>%
  rename(content_area = X6)
#####
ospi_crs_fn <- "~/data/cadrs/2015-16-StateCourseCodes.xlsx"

ospi_crs16 <- read.xlsx(ospi_crs_fn, 4, startRow = 2) %>%
  select(State.Course.Code:X6) %>%
  rename(content_area = X6)
####
table(ospi_crs16$content_area)
table(test_c$ospi_sub)
######clean up content areas in ccer to match ospi

test_c <- df_h %>% 
  select(ReportSchoolYear,StateCourseCode,CourseTitle,StateCourseName, ContentAreaName, crs_combo) %>% 
  unique(.)  %>% 
  mutate(state_code = str_pad(StateCourseCode, 5, pad = "0")) %>%
  mutate(ospi_sub = case_when(
    .$ContentAreaName == "Science" ~ "Life and Physical Sciences",
    .$ContentAreaName == "History" ~ "Social Sciences and History",
    .$ContentAreaName == "Civics and Government" ~ "Social Sciences and History",
    .$ContentAreaName == "Economics" ~ "Social Sciences and History",
    .$ContentAreaName == "Theatre" ~ "Fine and Performing Arts",
    .$ContentAreaName == "Music" ~ "Fine and Performing Arts",
    .$ContentAreaName == "Visual Arts" ~ "Fine and Performing Arts",
    .$ContentAreaName == "Dance" ~ "Fine and Performing Arts",
    .$ContentAreaName == "Fine and Performing Arts" ~ "Fine and Performing Arts",
    .$ContentAreaName == "Math" ~ "Mathematics",
    .$ContentAreaName == "Foreign Languages" ~ "Foreign Language and Literature",
    .$ContentAreaName == "English Language Arts" ~ "English Language and Literature",
    .$ContentAreaName == "Physical, Health and Safety Education" ~ "Physical, Health, and Safety Education",
    TRUE ~ as.character(.$ContentAreaName)))

##clean up course titles 
crs_content <- test_c %>% 
  mutate(crs_copy = CourseTitle) %>%
  mutate(crs_copy = str_remove(crs_copy, "\\("),
         crs_copy = str_remove(crs_copy, "\\)"),
         crs_copy = str_remove(crs_copy, "[*]"),
         crs_copy = str_remove(crs_copy, "[\"]"),
         crs_copy = str_replace(crs_copy, "[/]", " "),
         crs_copy = str_replace(crs_copy, "[-]", " "),
         crs_copy = str_trim(crs_copy),
         crs_copy = str_remove(crs_copy, "[0-9]+(th|TH).*(grade|GRADE)"),
         crs_copy = str_remove(crs_copy, "[0-9]"),
         crs_copy = tolower(crs_copy)) 
#######
state_ccer_crs <- left_join(crs_content, ospi_crs16, by = c("state_code"="State.Course.Code"))
# see when they agree 
test_sub <- state_ccer_crs %>%
  filter(ospi_sub == content_area, ospi_sub != "Reading", ospi_sub != "Miscellaneous", content_area != "Miscellaneous",
         ospi_sub != "Non-Instructional time") %>%
  select(ReportSchoolYear,crs_copy, Name, ospi_sub, content_area)

table(test_sub$ReportSchoolYear)
table(state_ccer_crs$ReportSchoolYear)
#######
# CCER data doesn't have content area for recent courses?
# use previous years data to get categories for most recent 
ospi_ccer <- test_sub %>%
  select(crs_copy, Name, ospi_sub, content_area)

table(ospi_ccer$content_area)
table(ospi_ccer$ospi_sub)
##
training_set <- ospi_ccer %>%
  select(crs_copy, ospi_sub) %>% unique()

ospi_cnames <- test_sub %>%
  select(Name, ospi_sub) %>%
  mutate(
    Name = str_remove(Name, "[0-9]+(th|TH).*(grade|GRADE)"),
    Name = str_remove(Name, "\\("),
    Name = str_remove(Name, "\\)"),
    Name = str_remove(Name, "\\d{5}$"),
    Name = str_remove(Name, "[*]"),
    Name = str_replace(Name, "[/]", " "),
    Name = str_replace(Name, "[—]", " "),
    Name = str_trim(Name),
    Name = str_remove(Name, "(9+TH|10+TH|11+TH|12+TH)"),
    Name = tolower(Name)
  ) %>%
  rename(crs_copy=Name)

## ELECTIVES CATS 
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

training_set_a <- bind_rows(training_set, ospi_cnames) %>% 
  unique() %>% # create electives 
  mutate(ospi_sub = ifelse(ospi_sub %in% electives, "elective", ospi_sub)) %>%
  unique()

# course classification contains some bad labels  
relabel_m <- c("math", "calc", "alg", "stat", "ap comp sci", "ap computer", "ib comp")
relabel_ss <- c("econ", "psych", "philo")
relabel_e <- c("acc", "law", "just", "sex")


comp_cleanup <- training_set_a %>%
  filter(ospi_sub == "Computer and Information Sciences") %>%
  mutate(ospi_sub = if_else(str_detect(crs_copy, paste(relabel_m, collapse = '|')), "Mathematics",ospi_sub),
         ospi_sub = if_else(str_detect(crs_copy, paste(relabel_ss, collapse = '|')), "Social Sciences and History",ospi_sub),
         ospi_sub = if_else(str_detect(crs_copy, paste(relabel_e, collapse = '|')), "elective",ospi_sub))
# science
relabel_sc <- c("hist", "geogr", "psych", "philo")
relabel_m <- c('alg', 'calc')
sci_clean <- training_set_a %>%
  filter(ospi_sub == "Life and Physical Sciences") %>%
  mutate(ospi_sub = if_else(str_detect(crs_copy, paste(relabel_sc, collapse = '|')), "Social Sciences and History",ospi_sub),
         ospi_sub = if_else(str_detect(crs_copy, paste(relabel_m, collapse = '|')), "Mathematics",ospi_sub))

# eng 
relabel_e <- c("gov", "hist", "soc")
eng_clean <- training_set_a %>%
  filter(ospi_sub == "English Language and Literature") %>%
  mutate(ospi_sub = if_else(str_detect(crs_copy, paste(relabel_e, collapse = '|')), "Social Sciences and History",ospi_sub))



training_set_b <- training_set_a %>%
  filter(!ospi_sub == "Computer and Information Sciences" & 
         !ospi_sub == "English Language and Literature" &
         !ospi_sub == "Life and Physical Sciences") %>%
  bind_rows(comp_cleanup, sci_clean, eng_clean) %>%
  mutate(ospi_sub = if_else(ospi_sub == "Computer and Information Sciences", "elective", ospi_sub))


# delete bad labels from cnn output...
training_set_b %>%
  filter(str_detect(crs_copy, "american") & str_detect(ospi_sub, "Math")) %>%
  mutate(crs_copy = str_remove(crs_copy, "apex"))


table(training_set_b$ospi_sub)


# save 
write.csv(training_set_b, file = "data/courses_ospi_b.csv")

# manual clean-up AVOID! 
# ## reduce content areas 
# ## "ContentAreaName","crs_copy","cadr_sub"
# 
# # ELECTIVES
# 
# electives <- c("Business and Marketing", 
#                  "Human Services", 
#                  "Agriculture, Food, and Natural Resources",
#                  "Hospitality and Tourism", "Health Care Sciences", 
#                  "Reading", 
#                  "Transportation, Distribution and Logistics",
#                  "Manufacturing", 
#                  "Architecture and Construction", 
#                  "Military Science", 
#                  "Public, Protective, and Government Service", 
#                  "Religious Education and Theology", 
#                  "Engineering and Technology",
#                  "Communications and Audio/Visual Technology")
# # courses to remove
# elective_c <- crs_content %>% select(crs_copy,ContentAreaName) %>%
#   filter(ContentAreaName %in% electives) %>%
#   select(-ContentAreaName) %>%
#   mutate(ContentAreaName = "elective") %>%
#   unique()
# 
# remove_c <- c('visual art',
#               'geo',
#               'econom',
#               'economics',
#               'psychology',
#               'psych',
#               'sign lan',
#               'potter',
#               'sculpture',
#               'jewlery',
#               'environ',
#               'alg',
#               'american') 
# 
# elective_c <- crs_content %>% select(crs_copy,ContentAreaName) %>%
#   filter(ContentAreaName %in% electives) %>%
#   filter(!str_detect(crs_copy, paste(remove_c, collapse = '|'))) 
