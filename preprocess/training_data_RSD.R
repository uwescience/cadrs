library(tidyverse)
library(openxlsx)
library(data.table)
library(stringi)
library(here)

source("/home/joseh/source/cadrs/settings.R") # When not running using bash R I have to use "source/cadrs/"

ospi_crs17 <- read.xlsx(ospi_crs17_fn, 4, startRow = 2) %>%
  select(State.Course.Code:X6) %>%
  rename(content_area = X6)

ospi_crs17$`State.Course.Code` <- stri_encode(ospi_crs17$`State.Course.Code`, "", "UTF-8")
ospi_crs17$Name <- stri_encode(ospi_crs17$Name, "", "UTF-8")
ospi_crs17$Description <- stri_encode(ospi_crs17$Description, "", "UTF-8")
ospi_crs17$content_area <- stri_encode(ospi_crs17$content_area, "", "UTF-8")

ospi_crs16 <- read.xlsx(ospi_crs16_fn, 4, startRow = 2) %>%
  select(State.Course.Code:X6) %>%
  rename(content_area = X6)

ospi_crs16$`State.Course.Code` <- stri_encode(ospi_crs16$`State.Course.Code`, "", "UTF-8")
ospi_crs16$Name <- stri_encode(ospi_crs16$Name, "", "UTF-8")
ospi_crs16$Description <- stri_encode(ospi_crs16$Description, "", "UTF-8")
ospi_crs16$content_area <- stri_encode(ospi_crs16$content_area, "", "UTF-8")

ospi_crs15 <- fread(ospi_crs15_fn, skip = 2, header = T, drop = c("V1","V5"), encoding='UTF-8') %>%
  rename(State.Course.Code = `State Course Code`) %>%
  mutate(State.Course.Code = as.character(State.Course.Code),
         State.Course.Code = str_pad(State.Course.Code, 5, pad = "0"))

ospi_crs15$`State.Course.Code` <- stri_encode(ospi_crs15$`State.Course.Code`, "", "UTF-8")
ospi_crs15$Name <- stri_encode(ospi_crs15$Name, "", "UTF-8")
ospi_crs15$Description <- stri_encode(ospi_crs15$Description, "", "UTF-8")

ospi_crs15 <- left_join(ospi_crs15, ospi_crs16 %>%
                          select(State.Course.Code, content_area), by = 'State.Course.Code')

missing <- ospi_crs15 %>%
  filter(is.na(content_area))

rsd_crs <- fread(rsd_crs_fn, na.strings = c("NA", "NULL"), encoding='UTF-8') %>%
  mutate(State.Course.Code = as.character(`State Code`),
         State.Course.Code = str_pad(State.Course.Code, 5, pad = "0"),
         cadr = if_else(`CADR Flag` == 'Yes', 1,0),
         State.Course.Code = str_remove(State.Course.Code, "[A-Z]$"))

# Some course CADR classes not correct, fix here using exact string match
not_cadrs <- c(
  'FIN ALGEBRA-A',
  'COMPUTER PRO-I',
  'CREATIVE WRIT',
  'IB BUS&MAN HL-A'
)

rsd_crs <- rsd_crs %>%
  mutate(cadr = if_else(`Course Short` == 'TRIGONOMETRY', 1, cadr),
         cadr = if_else(`Course Short` %in% not_cadrs, 0, cadr))


clean_train <- fread(clean_train_fn, na.strings = c("NA", "NULL"), encoding='UTF-8') %>%
  mutate(State.Course.Code = as.character(State.Course.Code),
         State.Course.Code = str_pad(State.Course.Code, 5, pad = "0"))

# Create combined ospi file with course codes
ospi_crs <- bind_rows(
  ospi_crs17,
  ospi_crs16
) %>%
  unique()

test15 <- ospi_crs15 %>%
  filter(!State.Course.Code %in% ospi_crs$State.Course.Code)
# looks like only 23 codes don't exist for previous years use those codes w/out subjects
table(ospi_crs$content_area)

test15$content_area <- c(
  'Mathematics',
  'Life and Physical Sciences',
  'Life and Physical Sciences',
  'Life and Physical Sciences',
  'elective',
  'Fine and Performing Arts',
  'Foreign Language and Literature',
  'Computer and Information Sciences',
  'elective',
  'elective',
  'elective',
  'elective',
  'elective',
  'elective',
  'elective',
  'elective',
  'elective',
  'elective',
  'elective',
  'elective',
  'elective',
  'elective',
  'elective',
  'elective',
  '',
  ''
)
test15 <- na.omit(test15)


# append 15 to ospi codes
ospi_crs <- bind_rows(
  ospi_crs,
  test15
) %>%
  unique() %>%
  select(-Subject.Area.Code, -`Type.(AP/IB)`)

write_csv(ospi_crs, dim_course_path)

# CLEAN UP RSD FILE 
# Begin with descriptions add course name in description
names(rsd_crs)

# We might not need this step...but will run for a renton specific cohort 
# rsd_crs <- rsd_crs %>%
#   unite_("Description", c("Course Long","Course Description"), sep = " ", remove=F)


# ospi file with rsd cadr flag
# make sure you get rid of Ns on state code
rsd_crs <- left_join(rsd_crs, ospi_crs, by = c('State.Course.Code'))

# need to clean before this step 
rsd_train <- rsd_crs %>%
  select(dist_code=`Course Code`,State.Course.Code, Name = `Course Long`, cadr,dist_description = `Course Description`, Description, content_area)

ospi_train_rsd <- rsd_crs %>%
  select(dist_code=`Course Code`,State.Course.Code, Name, cadr,dist_description = `Course Description`, Description, content_area) # get rid of generic courses
# generic names are bad
key_words <- c(
  'Other',
  'Aide',
  'Independent Study',
  'Workplace Experience',
  'Comprehensive'
)

ospi_train_rsd <- data.table(ospi_train_rsd)

for (i in key_words) {
  ospi_train_rsd <- ospi_train_rsd[!grep( i, ospi_train_rsd$Name),]  
}

###
ospi_rsd_train <- bind_rows(
  rsd_train,
  ospi_train_rsd 
) %>%
  filter(!is.na(Name))


### Find cases that are missing from the rsd + ospi file 
courses_not_covered <- clean_train %>%
  mutate(dist_code = NA,
         dist_description= NA) %>%
  select(dist_code, State.Course.Code, Name, cadr=cadrs, dist_description, Description, content_area) %>%
  filter(!State.Course.Code %in% ospi_rsd_train[, "State.Course.Code"])

table(courses_not_covered[,'content_area'])

courses_not_covered <- data.table(courses_not_covered)

key_words <- c(
  'Other',
  'Aide',
  'Independent Study',
  'Workplace Experience'
)

for (i in key_words) {
  courses_not_covered <- courses_not_covered[!grep( i, courses_not_covered$Name),]  
}

# foreign language 
clean_frl <- courses_not_covered[content_area == 'Foreign Language and Literature']

# append to ospi_rsd_train
glimpse(clean_frl)
glimpse(ospi_rsd_train)

ospi_rsd_train <- bind_rows(
  ospi_rsd_train,
  clean_frl 
)
# have some missing subjects that will need to be filled if I plan to do a subject by subject classifier 
########
# 'Social Sciences and History'
table(courses_not_covered[,content_area])

clean_soc <- courses_not_covered[content_area == 'Social Sciences and History']

# correct cadr flag error
clean_soc[, cadr := ifelse(Name == 'IB Islamic History', 1, cadr)]

ospi_rsd_train <- bind_rows(
  ospi_rsd_train,
  clean_soc 
)

#  Math
table(courses_not_covered[,content_area])

clean_math <- courses_not_covered[content_area == 'Mathematics']

ospi_rsd_train <- bind_rows(
  ospi_rsd_train,
  clean_math
)
##
# Fine and Performing Arts
table(courses_not_covered[,content_area])

clean_fpa <- courses_not_covered[content_area == 'Fine and Performing Arts']

# re-label some cadr rows 
cadr_yes <- c(
  'General Band',
  'Dance Repertory'
)

clean_fpa[, cadr := ifelse(Name %in% cadr_yes, 1, cadr)]

# find 'AP Studio Art—Two-Dimensional' using state course code to work around
# strings not matching in R b/c of encoding issues
clean_fpa[, cadr := ifelse(`State.Course.Code` %in% c('05174'), 1, cadr)]

# Communications and Audio/Visual Technology
clean_comm <- courses_not_covered[content_area == 'Communications and Audio/Visual Technology']

# combine both
ospi_rsd_train <- bind_rows(
  ospi_rsd_train,
  clean_fpa,
  clean_comm
)

#journalism
cadr_yes <- c(
  'JOURNALISM-A',
  'Journalism'
)

ospi_rsd_train <- data.table(ospi_rsd_train)

ospi_rsd_train[, cadr := ifelse(Name %in% cadr_yes, 1, cadr)]

# Life and Physical Sciences 

table(courses_not_covered[,content_area])

clean_sci <- courses_not_covered[content_area == 'Life and Physical Sciences']

ospi_rsd_train <- bind_rows(
  ospi_rsd_train,
  clean_sci
)

# English Language and Literature
clean_eng <- courses_not_covered[content_area == 'English Language and Literature']

ospi_rsd_train <- bind_rows(
  ospi_rsd_train,
  clean_eng
)

# the rest

courses_not_covered <- clean_train %>%
  mutate(dist_code = NA,
         dist_description= NA) %>%
  select(dist_code, State.Course.Code, Name, cadr=cadrs, dist_description, Description, content_area) %>%
  filter(!State.Course.Code %in% ospi_rsd_train[, State.Course.Code])

courses_not_covered <- data.table(courses_not_covered)

key_words <- c(
  'Other',
  'Aide',
  'Independent Study',
  'Workplace Experience'
)

for (i in key_words) {
  courses_not_covered <- courses_not_covered[!grep( i, courses_not_covered$Name),]  
}

ospi_rsd_train <- bind_rows(
  ospi_rsd_train,
  courses_not_covered
)

names(ospi_rsd_train)
table(ospi_rsd_train[,'cadr'])

# NEED TO ADD SCI9 CADR course might want to do this for other weird course names we encounter 
# 
# names(ospi_rsd_train)
# sci1 <- data.frame(dist_code = NA, State.Course.Code = NA, Name ='SCI9 PHYS/ENV-1', cadr=1, dist_description=NA, Description=NA, content_area='Life and Physical Sciences')
# sci2 <- data.frame(dist_code = NA, State.Course.Code = NA, Name ='SCI9 PHYS/ENV-2', cadr=1, dist_description=NA, Description=NA, content_area='Life and Physical Sciences')
sci_cad <- c(
  'SCI9 PHYS/ENV-1',
  'SCI9 PHYS/ENV-2',
  'COE SCIENCE',
  'PHYSICAL SCIENCE',
  'SCIENCE LINKS',
  'PHYS SCIENCE',
  'PHYSICAL SCI',
  'NXTGEN SCIENCE',
  'GENERAL SCI 1M',
  'GENERAL SCI 2M',
  'PHYSICAL SCI 1',
  'Physical Science And Physics',
  'INQ SCI',
  'HONORS BIO',
  'SCIENCE CREDIT',
  'SCIENCE 9',
  'CHEM',
  'Astronomy',
  'Marine Science',
  'Genetics',
  'CHSAstrmy101',
  'ASTRO',
  'MARINE SCI'
) # NEED TO ADD THE OTHERS LIKE ASTRONOMY...
# looping might be more efficient if we find a lot of courses we need to manually add...
datalist = list()
cadr = 1

for (i in sci_cad) {
  dat <- data.frame(dist_code = NA, State.Course.Code = NA, Name =i, cadr=cadr, dist_description=NA, Description=NA, content_area='Life and Physical Science')
  datalist[[i]] <- dat
}
sci_cr = do.call(rbind, datalist)
rownames(sci_cr) <- 1:nrow(sci_cr)

## ELL 
course_NC <- c(
  'ELL LAN ART',
  'ELLIntSkills',
  'ELL Literature/Composition ',
  'BegELLWriting',
  'ADV L/A ELL 1',
  'ELL AM LIT COM ',
  'Beg. Reading & Writing-ELL',
  'ELD LA 12A ML',
  'ELD LA 12B ML',
  'ELD LITERACY',
  'ELD ENG 1 SP',
  'ELD ENG 2 SP',
  'ELL Lit Comp')
 
datalist = list()
cadr = 0

for (i in course_NC) {
   dat <- data.frame(dist_code = NA, State.Course.Code = NA, Name =i, cadr=cadr, dist_description=NA, Description=NA, content_area='English Language and Literature')
   datalist[[i]] <- dat
 }
ell_cr = do.call(rbind, datalist)
rownames(ell_cr) <- 1:nrow(ell_cr)

# english cadr courses 
eng_cad <- c(
  'ENG 10',
  'LANGUAGE ARTS',
  'LIT COMP 9A',
  'LIT COMP 9A',
  'COMMUN ARTS',
  'COMMUNICATIONS',
  'DEBATE',
  'POETRY',
  'COMPETV SPEAK',
  'SPEECH/DEBATE',
  'FILM AS LIT',
  'AMER LIT',
  'LA 10 2 LIT ADV',
  'SPCH/DEBATE')

datalist = list()
cadr = 1

for (i in eng_cad) {
  dat <- data.frame(dist_code = NA, State.Course.Code = NA, Name =i, cadr=cadr, dist_description=NA, Description=NA, content_area='English Language and Literature')
  datalist[[i]] <- dat
}
eng_cr = do.call(rbind, datalist)
rownames(eng_cr) <- 1:nrow(eng_cr)

# Adding art courses not captured

art_cad <- c(
  'GENERAL ART',
  'ART',
  'ApplicArts_1',
  'ART 1 - CTE',
  'Creative Art-Drawing/Painting'
)

datalist = list()
cadr = 1

for (i in art_cad) {
  dat <- data.frame(dist_code = NA, State.Course.Code = NA, Name =i, cadr=cadr, dist_description=NA, Description=NA, content_area='Fine and Performing Arts')
  datalist[[i]] <- dat
}
art_cr = do.call(rbind, datalist)
rownames(art_cr) <- 1:nrow(art_cr)

# social sciences
# courses not captured

soc_cad <- c(
  'Contemporary Global Issues',
  'Contemporary World Problems',
  'Contemporary World Issues',
  'political science',
  'psychology',
  'geography',
  'U.S. History',
  'US Hist'
)

datalist = list()
cadr = 1

for (i in soc_cad) {
  dat <- data.frame(dist_code = NA, State.Course.Code = NA, Name =i, cadr=cadr, dist_description=NA, Description=NA, content_area='Social Sciences and History')
  datalist[[i]] <- dat
}
soc_cr = do.call(rbind, datalist)
rownames(soc_cr) <- 1:nrow(soc_cr)

##
 

ospi_rsd_train <- bind_rows(
  ospi_rsd_train, 
  sci_cr,
  eng_cr,
  ell_cr,
  art_cr,
  soc_cr
)

# Check cadr flag for subset 
cr_name_sub <- c(
  'Literature of a Genre',
  'Literature of a Theme',
  'Civics'
)

ospi_rsd_train <- ospi_rsd_train %>%
  mutate(cadr = if_else(Name %in% cr_name_sub, 1, cadr))


write_csv(ospi_rsd_train, rsd_cadrs_training_path)
