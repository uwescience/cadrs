.headers on
.mode csv
.output /home/joseh/source/cadrs/output/enroll_2017_cohort.csv

DROP VIEW IF EXISTS enroll_cohort;

-- enroll_2017_cohort.csv
CREATE VIEW enroll_cohort
AS
SELECT *
FROM enrollment enr
JOIN dimSchool sch
    ON enr.SchoolCode = sch.SchoolCode
    AND enr.ReportSchoolYear = sch.AcademicYear
WHERE enr.GradeLevelSortOrder = 15 AND enr.GradReqYear = 2017 AND enr.dGraduate = 1 AND sch.dRoadMapRegionFlag = 1;

select * from enroll_cohort;
