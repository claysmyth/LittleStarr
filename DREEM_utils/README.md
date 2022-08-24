Dreem Util README

This directory provides functions for labelling the RC+S data with hypnograms generated via the DREEM headband. 
It requires both DreembandDataToolkit and DREEM-RCS-Synchronization-Tool from Fahim Anjum's Github. It also requires a Project Summary CSV (see rcs-database repo on github.com/claysmyth)

The intended workflow is as follows:
1. Download relevant DREEM sessions (particularly h5 and txt files) via DreembandDataToolkit
2. Manually clean DREEM files by removing duplicate entries. No two H5 and Txt files should share the same morning date (e.g remove spurious recordings)
3. Run create_filepath_csv.py in this directory to create a csv table that matches DREEM and RC+S filepaths, leveraging the Project Summary CSV
4. Run sync_multiple_sessions.m to:
   (A). Create and save (as .mat file) the corrected DREEM data.
   (B). Create and save (as .parquet file) the RC+S intracranial data labelled with the corresponding Sleep Stage.
   (C). Create and save the RC+S eventLog