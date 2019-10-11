
-- load new results file 
.mode csv
.import /home/joseh/data/svm_cadr_student_predictions_CV.csv cadr_result
.tables
.schema cadr_result
-- Create cadr flag by subject (pre-aggregation)
CREATE VIEW ccer_subject_flag
AS
select a.*, eng.b_english_cadr, math.b_math_cadr, sci.b_sci_cadr, soc.b_soc_cadr, flang.b_flang_cadr, art.b_art_cadr
from enroll_cohort a 
left join(
    select distinct ResearchID, 1 as b_english_cadr
					from cadr_result
					where CollegeAcademicDistributionRequirementsFlag like '%1%' and content_area in ('English Language and Literature')
					group by ResearchID
					having sum(CreditsEarned) >= 4
) eng on a.ResearchID = eng.ResearchID
left join(
    select distinct ResearchID, 1 as b_math_cadr
					from cadr_result
					where CollegeAcademicDistributionRequirementsFlag like '%1%' and content_area in ('Computer and Information Sciences', 'Mathematics')
					group by ResearchID
					having sum(CreditsEarned) >= 3 
) math on a.ResearchID = math.ResearchID
left join(
    select distinct ResearchID, 1 as b_sci_cadr
					from cadr_result
					where CollegeAcademicDistributionRequirementsFlag like '%1%' and content_area in ('Life and Physical Sciences')
					group by ResearchID
					having sum(CreditsEarned) >= 2
) sci on a.ResearchID = sci.ResearchID
left join(
    select distinct ResearchID, 1 as b_soc_cadr
					from cadr_result
					where CollegeAcademicDistributionRequirementsFlag like '%1%' and content_area in ('Social Sciences and History')
					group by ResearchID
					having sum(CreditsEarned) >= 3
) soc on a.ResearchID=soc.ResearchID
left join(
    select distinct ResearchID, 1 as b_flang_cadr
					from cadr_result
					where CollegeAcademicDistributionRequirementsFlag like '%1%' and content_area in ('Foreign Language and Literature')
					group by ResearchID
					having sum(CreditsEarned) >= 2
) flang on a.ResearchID=flang.ResearchID
left join(
    select distinct ResearchID, 1 as b_art_cadr
					from cadr_result
					where CollegeAcademicDistributionRequirementsFlag like '%1%' and content_area in ('Fine and Performing Arts')
					group by ResearchID
					having sum(CreditsEarned) >= 1
) art on a.ResearchID = art.ResearchID ;

-- Creare view for svm cadrs
CREATE VIEW svm_cadr_flag
AS
select a.*, eng.svm_english_cadr, math.svm_math_cadr, sci.svm_sci_cadr, soc.svm_soc_cadr, flang.svm_flang_cadr, art.svm_art_cadr
from enroll_cohort a 
left join(
    select distinct ResearchID, 1 as svm_english_cadr
					from cadr_result
					where p_CADRS like '%1%' and content_area in ('English Language and Literature')
					group by ResearchID
					having sum(CreditsEarned) >= 4
) eng on a.ResearchID = eng.ResearchID
left join(
    select distinct ResearchID, 1 as svm_math_cadr
					from cadr_result
					where p_CADRS like '%1%' and content_area in ('Computer and Information Sciences', 'Mathematics')
					group by ResearchID
					having sum(CreditsEarned) >= 3 
) math on a.ResearchID = math.ResearchID
left join(
    select distinct ResearchID, 1 as svm_sci_cadr
					from cadr_result
					where p_CADRS like '%1%' and content_area in ('Life and Physical Sciences')
					group by ResearchID
					having sum(CreditsEarned) >= 2
) sci on a.ResearchID = sci.ResearchID
left join(
    select distinct ResearchID, 1 as svm_soc_cadr
					from cadr_result
					where p_CADRS like '%1%' and content_area in ('Social Sciences and History')
					group by ResearchID
					having sum(CreditsEarned) >= 3
) soc on a.ResearchID=soc.ResearchID
left join(
    select distinct ResearchID, 1 as svm_flang_cadr
					from cadr_result
					where p_CADRS like '%1%' and content_area in ('Foreign Language and Literature')
					group by ResearchID
					having sum(CreditsEarned) >= 2
) flang on a.ResearchID=flang.ResearchID
left join(
    select distinct ResearchID, 1 as svm_art_cadr
					from cadr_result
					where p_CADRS like '%1%' and content_area in ('Fine and Performing Arts')
					group by ResearchID
					having sum(CreditsEarned) >= 1
) art on a.ResearchID = art.ResearchID ;


.headers on
.mode csv
.output cadrs_results_20082019.csv
SELECT a.*, b.svm_english_cadr, b.svm_math_cadr, b.svm_sci_cadr, b.svm_soc_cadr, b.svm_flang_cadr, b.svm_art_cadr
FROM ccer_subject_flag a
LEFT JOIN (
	SELECT ResearchID, svm_english_cadr, svm_math_cadr, svm_sci_cadr, svm_soc_cadr, svm_flang_cadr, svm_art_cadr
	FROM svm_cadr_flag) b ON a.ResearchID = b.ResearchID ;


-- Create aggregated views for ccer flag

select m.DistrictCode, sum(num), sum(total)
from (
	select count(distinct enr.ResearchID) num, DistrictCode
	from enroll_cohort enr

	inner join (
        select DISTINCT ResearchID
	    from cadr_result
	    where CollegeAcademicDistributionRequirementsFlag like '%1%' and content_area in ('English Language and Literature')
	    group by ResearchID
	    having sum(CreditsEarned) >= 4) eng on enr.ResearchID = eng.ResearchID

    inner join (
        select DISTINCT ResearchID
	    from cadr_result
	    where CollegeAcademicDistributionRequirementsFlag like '%1%' and content_area in ('Mathematics', 'Engineering and Technology','Computer and Information Sciences')
	    group by ResearchID
	    having sum(CreditsEarned) >= 3) math on enr.ResearchID = math.ResearchID

    inner join (
        select DISTINCT ResearchID
	    from cadr_result
	    where CollegeAcademicDistributionRequirementsFlag like '%1%' and content_area in ('Life and Physical Sciences')
	    group by ResearchID
	    having sum(CreditsEarned) >= 2) sci on enr.ResearchID = sci.ResearchID
    
    inner join (
        select DISTINCT ResearchID
	    from cadr_result
	    where CollegeAcademicDistributionRequirementsFlag like '%1%' and content_area in ('Social Sciences and History')
	    group by ResearchID
	    having sum(CreditsEarned) >= 3) soc on enr.ResearchID = soc.ResearchID
    
    inner join (
        select DISTINCT ResearchID
	    from cadr_result
	    where CollegeAcademicDistributionRequirementsFlag like '%1%' and content_area in ('Foreign Language and Literature')
	    group by ResearchID
	    having sum(CreditsEarned) >= 2) flang on enr.ResearchID = flang.ResearchID
    
    inner join (
        select DISTINCT ResearchID
	    from cadr_result
	    where CollegeAcademicDistributionRequirementsFlag like '%1%' and content_area in ('Fine and Performing Arts', 'Communications and Audio/Visual Technology')
	    group by ResearchID
	    having sum(CreditsEarned) >= 1) art on enr.ResearchID = art.ResearchID

where ReportSchoolYear = 2017 and dGraduate = 1 and GradeLevelSortOrder = 15
group by DistrictCode ) m 

inner join (	select count(distinct enr.ResearchID) total, DistrictCode
				from enroll_cohort enr
				where ReportSchoolYear = 2017 and dGraduate = 1 and GradeLevelSortOrder = 15
				group by DistrictName
				) n on m.DistrictCode = n.DistrictCode

group by m.DistrictCode ;

-- using the svm 

select m.DistrictCode, sum(num), sum(total)
from (
	select count(distinct enr.ResearchID) num, DistrictCode
	from enroll_cohort enr

	inner join (
        select DISTINCT ResearchID
	    from cadr_result
	    where p_CADRS like '%1%' and content_area in ('English Language and Literature')
	    group by ResearchID
	    having sum(CreditsEarned) >= 4) eng on enr.ResearchID = eng.ResearchID

    inner join (
        select DISTINCT ResearchID
	    from cadr_result
	    where p_CADRS like '%1%' and content_area in ('Mathematics', 'Engineering and Technology','Computer and Information Sciences')
	    group by ResearchID
	    having sum(CreditsEarned) >= 3) math on enr.ResearchID = math.ResearchID

    inner join (
        select DISTINCT ResearchID
	    from cadr_result
	    where p_CADRS like '%1%' and content_area in ('Life and Physical Sciences')
	    group by ResearchID
	    having sum(CreditsEarned) >= 2) sci on enr.ResearchID = sci.ResearchID
    
    inner join (
        select DISTINCT ResearchID
	    from cadr_result
	    where p_CADRS like '%1%' and content_area in ('Social Sciences and History')
	    group by ResearchID
	    having sum(CreditsEarned) >= 3) soc on enr.ResearchID = soc.ResearchID
    
    inner join (
        select DISTINCT ResearchID
	    from cadr_result
	    where p_CADRS like '%1%' and content_area in ('Foreign Language and Literature')
	    group by ResearchID
	    having sum(CreditsEarned) >= 2) flang on enr.ResearchID = flang.ResearchID
    
    inner join (
        select DISTINCT ResearchID
	    from cadr_result
	    where p_CADRS like '%1%' and content_area in ('Fine and Performing Arts', 'Communications and Audio/Visual Technology')
	    group by ResearchID
	    having sum(CreditsEarned) >= 1) art on enr.ResearchID = art.ResearchID

where ReportSchoolYear = 2017 and dGraduate = 1 and GradeLevelSortOrder = 15
group by DistrictCode ) m 

inner join (	select count(distinct enr.ResearchID) total, DistrictCode
				from enroll_cohort enr
				where ReportSchoolYear = 2017 and dGraduate = 1 and GradeLevelSortOrder = 15
				group by DistrictName
				) n on m.DistrictCode = n.DistrictCode

group by m.DistrictCode ;