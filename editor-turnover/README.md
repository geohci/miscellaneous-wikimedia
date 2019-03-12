### Editor Turnover Analysis

The goal of this analysis was to better understand to what degree the edits/editors in a given month were new vs. returning editors from previous months.

It contains the following code:
* monthly_overlap.py: iterate through stub history dump file and record when editor made first edit to that project and how many edits they make in a given month
* compute_overlap.py: process the editor statistics files from monthly_overlap.py to produce aggregate statistics on editor turnover
