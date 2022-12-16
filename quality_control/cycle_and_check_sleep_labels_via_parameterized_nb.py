import papermill as pm
import os
import sys

# TODO: Reformat script so that is can be run from command.

# Base path for output notebooks and summaries
OUTPUT_PATH = '/media/longterm_hdd/Clay/Sleep_10day/'
# Data directories
data_dirs = ['/media/longterm_hdd/Clay/Sleep_10day/RCS02L/Overnight',
             '/media/longterm_hdd/Clay/Sleep_10day/RCS02R/Overnight',
             '/media/longterm_hdd/Clay/Sleep_10day/RCS03L/Overnight',
             '/media/longterm_hdd/Clay/Sleep_10day/RCS03R/Overnight',
             '/media/longterm_hdd/Clay/Sleep_10day/RCS16L/Overnight',
             '/media/longterm_hdd/Clay/Sleep_10day/RCS16R/Overnight']
# Names of output notebooks and htmls
OUT_NAMES = ['RCS02L_Sleep_QC.ipynb',
             'RCS02R_Sleep_QC.ipynb',
             'RCS03L_Sleep_QC.ipynb',
             'RCS03R_Sleep_QC.ipynb',
             'RCS16L_Sleep_QC.ipynb',
             'RCS16R_Sleep_QC.ipynb']
FILE_PATH_CSV = '/media/longterm_hdd/Clay/DREEM_data/filepaths_02_03_16_matlab_copy.csv'
# notebook to be executed
input_notebook = '/home/claysmyth/code/LittleStarr/quality_control/viz_psd_sleep_stage.ipynb'
REMOVE_CELL_TAG = 'hide'
REMOVE_CODE_TAG = 'hide_code'


# TODO: Save Metadata (e.g. % disconnects, etc.. ) as table (and also JSON for MongoDB?)

def cycle_and_parameterize(input_notebook, data_dirs, out_names=OUT_NAMES, out_path=OUTPUT_PATH,
                           file_path_csv=FILE_PATH_CSV,
                           remove_cell_tag=REMOVE_CELL_TAG, remove_code_tag=REMOVE_CODE_TAG):
    for i, data_dir in enumerate(data_dirs):
        output_notebook = f'{out_path}{out_names[i]}'
        print(output_notebook)
        pm.execute_notebook(
            input_notebook,
            output_notebook,
            parameters=dict(parent_directories=[data_dir], file_paths_csv=file_path_csv)
        )

        command = (f"jupyter-nbconvert "
                   f"--to html {output_notebook} "
                   f"--TagRemovePreprocessor.enabled=True "
                   f"--TagRemovePreprocessor.remove_cell_tags {remove_cell_tag} "
                   f"--TagRemovePreprocessor.remove_input_tags {remove_code_tag}")
        os.system(command)


def run_with_script_variables():
    cycle_and_parameterize(input_notebook, data_dirs)


if __name__ == '__main__':
    if len(sys.arv) == 1:
        run_with_script_variables()
    else:
        input_notebook = sys.argv[1]
        cycle_and_parameterize(input_notebook, *sys.argv[2:])
