.headers on
.mode csv
.output output/course_2017_cohort.csv

DROP VIEW IF EXISTS cohort_17;

CREATE VIEW cohort_17
AS
SELECT *
FROM courses 
LEFT JOIN ( 
    SELECT `State.Course.Code`, content_area
    FROM DimCourse
) dm ON printf("%05d", StateCourseCode)=dm.`State.Course.Code`
WHERE ResearchID IN (
SELECT DISTINCT ResearchID
FROM enrollment enr
JOIN dimSchool sch
    ON enr.SchoolCode = sch.SchoolCode
    AND enr.ReportSchoolYear = sch.AcademicYear
WHERE enr.GradeLevelSortOrder = 15 AND enr.GradReqYear = 2017 AND enr.dGraduate = 1 AND sch.dRoadMapRegionFlag = 1);

select * from cohort_17;
