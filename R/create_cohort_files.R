
library(RSQLite)
library(sys)
library(here)

source(here("settings.R"))

exec_sqlite <- function(sql) {
	args <- sqlite_database_path

	sqlite_fullpath <- paste(Sys.getenv("SQLITE_PATH"), Sys.getenv("SQLITE_EXEC"), sep="/")

	print("EXECUTING: ")
	print(sql)

	tmp_path <-here("tmpsql.txt")
	fileConn <- file(tmp_path)
	writeLines(sql, fileConn)
	close(fileConn)

	exec_wait(sqlite_fullpath, args, std_in = tmp_path)

	file.remove(tmp_path)
}

# DO NOTE USE: sqlite cli chokes on fields that start with a double quote char
# that you don't want to be treated as a quoted field value.
import_table_using_sqlite_cli <- function(file_path, table_name, sep=",") {

	sep <- paste(".separator \"", sep, "\"", sep ="")

	s <- paste(
		paste("DROP TABLE IF EXISTS", table_name, ";", sep=" "),
		".mode list",
		sep,
		paste(".import", file_path, table_name, sep=" "),
		sep="\n"
	)

	exec_sqlite(s)
}

# uses a ton of memory b/c we load the entire file into a df, but this is the only way
# to get around quoting issue above
import_table_using_dbi <- function(file_path, table_name, sep=",", quote="\"", file_encoding="UTF-8") {
	print(paste("loading", table_name, sep=" "))

	df <- read.csv(file_path
		,header=TRUE
		,quote=quote
		,sep=sep
		,colClasses=c('character')
		,check.names=FALSE
		,as.is=TRUE
		,na.strings=c('')
		,fileEncoding=file_encoding
	)
	conn <- dbConnect(SQLite(), sqlite_database_path)
	dbExecute(
	  conn,
	  paste("DROP TABLE IF EXISTS", table_name, sep=" ")
	)

	print(paste("writing to database"))
	dbWriteTable(conn, table_name, df)

	dbDisconnect(conn)
	rm(df)
	gc() # force garbage collection
}

import_table_using_dbi(dim_course_path, "DimCourse")
import_table_using_dbi(dim_school_path, "dimSchool", sep="|", quote="", file_encoding="UTF-8-BOM")
import_table_using_dbi(enrollment_path, "enrollment", sep="|", quote="", file_encoding="UTF-8-BOM")
import_table_using_dbi(gr_hist, "courses", sep="|", quote="", file_encoding="UTF-8-BOM")

######

fileConn <- file("SQL/create_course_2017_cohort.sql", "r")
lines <- readLines(fileConn)
close(fileConn)

exec_sqlite(lines)

######

fileConn <- file("SQL/create_enroll_2017_cohort.sql", "r")
lines <- readLines(fileConn)
close(fileConn)

exec_sqlite(lines)
