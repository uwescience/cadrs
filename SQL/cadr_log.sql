.mode csv
.import /home/joseh/data/student_2017_cohort_results.csv cadrs
.import /home/joseh/data/student_enroll_2017_seniors.csv enroll
.schema cadrs

select cast(count(distinct case when cadr_ccer like '%1%' then CourseID end) as float) / cast(count(distinct CourseID) as float) as prop, sgh.DistrictName
from cadrs sgh
where GradeLevelWhenCourseTaken = '12'
and dSchoolYear = '2017'
group by sgh.DistrictName;

select cast(count(distinct case when svm_CADRS like '%1%' then CourseID end) as float) / cast(count(distinct CourseID) as float) as prop, sgh.DistrictName
from cadrs sgh
where GradeLevelWhenCourseTaken = '12'
and dSchoolYear = '2017'
group by sgh.DistrictName;

.headers on
.mode csv
.import /home/joseh/data/student_2017_cohort_results.csv cadrs
.import /home/joseh/data/student_enroll_2017_seniors.csv enroll
.output cadrs_code.csv
/* Try CADRS DISTRICT */
select a.*, eng.svm_english_cadr, math.svm_math_cadr, sci.svm_sci_cadr, soc.svm_soc_cadr, flang.svm_flang_cadr, art.svm_art_cadr
from enroll a 
left join(
    select distinct ResearchID, 1 as svm_english_cadr
					from cadrs
					where svm_CADRS like '%1%' and ospi_sub in ('English Language and Literature')
					group by ResearchID
					having sum(CreditsEarned) >= 4
) eng on a.ResearchID = eng.ResearchID
left join(
    select distinct ResearchID, 1 as svm_math_cadr
					from cadrs
					where svm_CADRS like '%1%' and ospi_sub in ('Computer and Information Sciences', 'Mathematics')
					group by ResearchID
					having sum(CreditsEarned) >= 3 
) math on a.ResearchID = math.ResearchID
left join(
    select distinct ResearchID, 1 as svm_sci_cadr
					from cadrs
					where svm_CADRS like '%1%' and ospi_sub in ('Life and Physical Sciences')
					group by ResearchID
					having sum(CreditsEarned) >= 2
) sci on a.ResearchID = sci.ResearchID
left join(
    select distinct ResearchID, 1 as svm_soc_cadr
					from cadrs
					where svm_CADRS like '%1%' and ospi_sub in ('Social Sciences and History')
					group by ResearchID
					having sum(CreditsEarned) >= 3
) soc on a.ResearchID=soc.ResearchID
left join(
    select distinct ResearchID, 1 as svm_flang_cadr
					from cadrs
					where svm_CADRS like '%1%' and ospi_sub in ('Foreign Language and Literature')
					group by ResearchID
					having sum(CreditsEarned) >= 2
) flang on a.ResearchID=flang.ResearchID
left join(
    select distinct ResearchID, 1 as svm_art_cadr
					from cadrs
					where svm_CADRS like '%1%' and ospi_sub in ('Fine and Performing Arts')
					group by ResearchID
					having sum(CreditsEarned) >= 1
) art on a.ResearchID = art.ResearchID ;