# Course title clean-up

library(tidyverse)
library(openxlsx)

gr_hist <- "~/data/cadrs/hsCourses.txt"
df_h <- read_delim(gr_hist, delim = "|", quote = "",col_names = TRUE, na = c("", "NA", "NULL")) 

# state/district course name combination
# if state course exists use that otherwise use the district course code
df_h <- df_h %>%
  mutate(crs_combo = ifelse(is.na(StateCourseName), CourseTitle,StateCourseName))

# df_h %>% select(ResearchID) %>% unique() %>% nrow()
# df_h %>% select(CourseTitle) %>% unique() %>% nrow()
# df_h %>% select(ReportSchoolYear) %>% unique() %>% nrow()

table(df_h$ReportSchoolYear)
########
###LOOK AT STATE COURSE FILE COVERAGE

test_c <- df_h %>% 
  select(StateCourseCode,CourseTitle,StateCourseName, ContentAreaName, crs_combo) %>% 
  mutate(state_code = str_pad(StateCourseCode, 5, pad = "0")) %>%
  unique(.)

# Load xlsx file from ospi
ospi_crs_fn <- "~/data/cadrs/2015-16-StateCourseCodes.xlsx"

ospi_crs <- read.xlsx(ospi_crs_fn, 4, startRow = 2) %>%
  select(State.Course.Code:X6) %>%
  rename(content_area = X6)

table(ospi_crs$content_area)
table(test_c$ContentAreaName)

sci_ospi <- ospi_crs %>% select(Name, content_area) %>%
  filter(content_area == 'Computer and Information Sciences') %>% unique()

sci_ccer <- test_c %>% select(CourseTitle, ContentAreaName) %>%
  filter(ContentAreaName == 'Science') %>% unique()
  
# join to ccer data 
state_ccer_crs <- left_join(test_c, ospi_crs, by = c("state_code"="State.Course.Code"))
names(state_ccer_crs)
###
library(stringr)


sum(is.na(state_ccer_crs$content_area))
sum(is.na(state_ccer_crs$ContentAreaName))

# Do not include unlabeled content area for training 

crs_content <- state_ccer_crs %>% 
  mutate(crs_copy = CourseTitle,
         st_crs_copy = Name) %>%
  mutate(crs_copy = str_remove(crs_copy, "\\("),
    crs_copy = str_remove(crs_copy, "\\)"),
    crs_copy = str_remove(crs_copy, "[*]"),
    crs_copy = str_remove(crs_copy, "[\"]"),
    crs_copy = str_trim(crs_copy))%>%
  mutate(
    st_crs_copy = str_remove(st_crs_copy, "[0-9]+(th|TH).*(grade|GRADE)"),
    st_crs_copy = str_remove(st_crs_copy, "\\("),
    st_crs_copy = str_remove(st_crs_copy, "\\)"),
    st_crs_copy = str_remove(st_crs_copy, "\\d{5}$"),
    st_crs_copy = str_remove(st_crs_copy, "[*]"),
    st_crs_copy = str_trim(st_crs_copy),
    st_crs_copy = str_remove(st_crs_copy, "(9+TH|10+TH|11+TH|12+TH)")
  )

###Clean up some of these courses that are mislabeled 
# Create content area matches {if course is math on both ospi source and ccer then course is math}

table(crs_content$ContentAreaName)

# trouble areas 
# 'elective': 0,
# 'non instructional.time': 6, Get rid of this category 
# 'computer.and.information.sciences': 8, 
# 'engineering.and.technology': 9,
# 'communications.and.audio.visual.technology': 10 # some label mislabels 

m_test <- crs_content %>%
  filter(content_area == 'English Language and Literature') %>%
  select(ContentAreaName,content_area,crs_copy,st_crs_copy)

##Check clustering of "elective" courses can we get fewer than 14 clusters
elective_c <- crs_content %>% select(crs_copy,ContentAreaName, content_area) %>%
  filter(ContentAreaName %in% c("Business and Marketing", "Human Services", "Agriculture, Food, and Natural Resources",
  "Hospitality and Tourism", "Health Care Sciences", "Reading", "Transportation, Distribution and Logistics",
  "Manufacturing", "Architecture and Construction", "Military Science", 
  "Public, Protective, and Government Service", "Religious Education and Theology")) %>%
  mutate(c_name = tolower(crs_copy),
         c_name = str_remove(c_name, "[0-9]+(th|TH).*(grade|GRADE)"),
         c_name = str_remove(c_name, "[/]"),
         c_name = str_remove(c_name, "[&]"),
         c_name = str_remove(c_name, "[-]"),
         c_name = str_remove(c_name, "[0-9]+(th|TH)"),
         c_name = str_replace_all(c_name, " ", "")) %>%
  unique(.)

 
c_name <- elective_c %>% select(c_name) %>% unique()
str(c_name)
rm(crs_content)
gc()
# Levenshtein Distance
library(stringdist)

uniquemodels <- as.character(c_name$c_name)

distancemodels <- stringdistmatrix(uniquemodels,uniquemodels,method = "jw")

rownames(distancemodels) <- uniquemodels

hc <- hclust(as.dist(distancemodels))

plot(hc)

### 
# Clean up content areas to match cadrs

names(crs_content_clean)
table(crs_content_clean$content_comb)
table(ospi_crs$content_area)

nlp_test <- crs_content_clean %>%
  mutate(cadr_sub = case_when(
                              .$content_comb == "Geography" ~ "Life and Physical Sciences",
                              .$content_comb == "Science" ~ "Life and Physical Sciences",
                              
                              .$content_comb == "History" ~ "social sciences",
                              .$content_comb == "Civics and Government" ~ "social sciences",
                              .$content_comb == "Economics" ~ "social sciences",
                              .$content_comb == "Social Sciences and History" ~ "social sciences",
                
                              
                              .$content_comb == "Theatre" ~ "arts",
                              .$content_comb == "Music" ~ "arts",
                              .$content_comb == "Visual Arts" ~ "arts",
                              .$content_comb == "Dance" ~ "arts",
                              .$content_comb == "Fine and Performing Arts" ~ "arts",
                              
                              .$content_comb == "Business and Marketing" ~ "elective",
                              .$content_comb == "Human Services" ~ "elective",
                              .$content_comb == "Agriculture, Food, and Natural Resources" ~ "elective",
                              .$content_comb == "Hospitality and Tourism" ~ "elective",
                              .$content_comb == "Health Care Sciences" ~ "elective",
                              .$content_comb == "Reading" ~ "elective",
                              .$content_comb == "Transportation, Distribution and Logistics" ~ "elective",
                              .$content_comb == "Manufacturing" ~ "elective",
                              .$content_comb == "Architecture and Construction" ~ "elective",
                              .$content_comb == "Military Science" ~ "elective",
                              .$content_comb == "Public, Protective, and Government Service" ~ "elective",
                              .$content_comb == "Religious Education and Theology" ~ "elective",
                              .$content_comb == "More than one core content area (block class)" ~ "elective",
                              
                              # .$content_comb == "Miscellaneous" ~ "other",
                              .$content_comb == "Physical, Health and Safety Education" ~ "PE",
                              .$content_comb == "Physical, Health, and Safety Education" ~ "PE",
                              
                              .$content_comb == "Math" ~ "Mathematics",
                              .$content_comb == "Foreign Languages" ~ "Foreign Language and Literature",
                              
                              .$content_comb == "English Language Arts" ~ "English Language and Literature",
                              TRUE ~ as.character(.$content_comb)))
###

table(nlp_test$cadr_sub)

# create training data course code + course title + cadr_sub 
# append with ospi course title and the subject 
nlp_test <- nlp_test %>% 
  filter(cadr_sub != 'Non-Instructional time' & cadr_sub != 'Nonsubject Specific' & cadr_sub != 'Miscellaneous')

table(nlp_test$cadr_sub)

training_a <- nlp_test %>% select(state_code, crs_copy, crs_combo,cadr_sub, st_crs_copy) %>% na.omit() %>%
              unique()
names(training_a)

# test <-  training_a %>% filter(str_detect(crs_copy, 'FOR LIFE'))
# 
# test <- data.frame(training_a$crs_copy, training_a$crs_combo,training_a$cadr_sub, training_a$st_crs_copy) 

training_b <- nlp_test %>%
                select(state_code, crs_copy = st_crs_copy, cadr_sub)

training1 <- bind_rows(training_a,training_b)


training1 <- training1 %>% na.omit() %>%
  unique(.)

table(training1$cadr_sub)

summary(training)

write.csv(training1, file = "data/courses_cadrs_text_test.csv")
#########


unique_test <- training1 %>% select(-state_code) %>% unique(.) 
####
table(nlp_test$cadr_sub)

#'life.and.physical.sciences': 0, 
#'english.language.and.literature': 1, 
#'mathematics': 2, 
#'elective': 3, 
#'physical.health.and.safety.education': 4, 
#'social.sciences': 5, 
#'arts': 6, 
#'foreign.language.and.literature': 7, 
#'computer.and.information.sciences': 8, 
#'engineering.and.technology': 9, 
#'communications.and.audio.visual.technology': 10}

art_c <- nlp_test %>%
  filter(cadr_sub == "arts") %>%
  filter(str_detect(crs_copy, 'INTRO'))

#get rid of "intro" and "intro to"
#get rid of ledING "0" 

elc <- nlp_test %>%
  filter(content_comb == "Military Science")

table(elc$content_comb)

non_inst <- nlp_test %>%
  filter(ContentAreaName == "Non-Instructional time")

comp_inf <- nlp_test %>%
  filter(ContentAreaName == "Computer and Information Sciences")

courses_cadrs_text_test <- read_csv("data/courses_cadrs_text_test.csv")

View(courses_cadrs_text_test)
