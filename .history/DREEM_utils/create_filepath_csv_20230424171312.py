import glob
import sys
import pandas as pd
import os
import dateutil.parser as dparser
import pprint

"""This script creates a CSV that contains the relevant filepaths to RCS raw data, DREEM filepaths, and output paths
 in order to run DREEM-RCS-Synchronization-Tool package. Uses project summmary CSV.
 Intended to be called from command line as 'python3 create_filepath_csv.py <outfile_name> RCS02 RCS03...' """

PROJECT_SUMMARY_CSV = "/media/dropbox_hdd/Starr Lab Dropbox/Projects/SleepStimOffset/SleepStimOffset.csv"
DREEM_DATA_BASEPATH = "/media/longterm_hdd/Clay/SleepStimOffset/DREEM/"
DROP_COLUMNS = ["Dropbox_Link", "Data_Server_Hyperlink", 'Unnamed: 0']

if __name__ == "__main__":

    out_file_name = sys.argv[1]
    rcs_participants = sys.argv[2:]
    
    # Read Sleep summary csv, and add column with datetime objects referring to the overnight session end-date
    summary_df = pd.read_csv(PROJECT_SUMMARY_CSV)
    summary_df["DateTime"] = summary_df["TimeEnded"].apply(lambda x: dparser.parse(x, fuzzy=True).date())

    side = ["Left", "Right"]

    for ind, i in enumerate(rcs_participants):
        
        # Access DREEM H5 and txt files
        dreem_part_path = f"{DREEM_DATA_BASEPATH}/{i}"
        dreem_device_path = f"{dreem_part_path}/{os.listdir(dreem_part_path)[0]}"

        participant_dreem_txt_path = f"{dreem_device_path}/SleepData"
        participant_dreem_H5_path = f"{dreem_device_path}/H5"

        h5_file_paths = [f"{participant_dreem_H5_path}/{f}" for f in os.listdir(participant_dreem_H5_path)]
        txt_file_paths = [f"{participant_dreem_txt_path}/{f}" for f in os.listdir(participant_dreem_txt_path)]

        # Create datetime object by parsing H5 file names
        dates = [dparser.parse(k.split("---")[1], fuzzy=True).date() for k in os.listdir(participant_dreem_H5_path)]
        # Order txt files paths to sync with DateTime order. Assumes each datetime object is unique
        txt_file_paths_ordered = []
        for date in dates:
            txt_file_paths_ordered.extend([txt_path for txt_path in txt_file_paths
                                           if date.strftime("%d-%b-%Y") in txt_path])

        # Create pandas dataframe of Dreem file paths and datetime objects
        participant_dreem_df = pd.DataFrame.from_dict({"H5": h5_file_paths, "Txt": txt_file_paths_ordered,
                                                       "DateTime": dates})

        sessions_tmp = summary_df[(summary_df['RCS#'] == i) &
                                  (summary_df["SessionType(s)"] in "Overnight")].copy()

        print(sessions_tmp)
        print(participant_dreem_df)
        # Merge RCS dataframe with dreem dataframe
        tmp_df = participant_dreem_df.merge(sessions_tmp, how='inner', on="DateTime")

        if not ind:
            dreem_filepath_df = tmp_df.drop(DROP_COLUMNS, axis=1)
        else:
            dreem_filepath_df = pd.concat([dreem_filepath_df, tmp_df.drop(DROP_COLUMNS, axis=1)], axis=0)


    dreem_filepath_df.to_csv(out_file_name)
