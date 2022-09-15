import papermill as pm
import os

BASE_PATH = ''
OUTPUT_PATH = ''
OUT_NAME = ''


# TODO: Save Metadata (e.g. % disconnects, etc.. ) as table (and also JSON for MongoDB?)

def cycle_and_parameterize(input_notebook, *data_dirs):
    for data_dir in data_dirs:
        output_notebook = f'{OUT_PATH}/{data_dir.split("/")[-1]}/{OUT_NAME}'
        pm.execute_notebook(
            input_notebook,
            output_notebook,
            parameters=dict(parent_directory=data_dir)
        )

        command = (f"jupyter nbconvert "
                   f"--to pdf {output_notebook}"
                   f"--output {output.pdf}")
        # --TagRemovePreprocessor.enabled=True
        # --TagRemovePreprocessor.remove_cell_tags name-of-remove-cell-tag
        # --TagRemovePreprocessor.remove_input_tags name-of-input-to-remove-tag
        os.system(command)


if __name__ == '__main__':
    input_notebook = sys.argv[1]
    cycle_and_parameterize(input_notebook, *sys.argv[2:])
