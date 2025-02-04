%% This script will synchronize multiple nights of DREEM data with corresponding RCS sessions. 
% Output files
%      ../RCS#[L or R]/Overnight/Session#_%mm_%DD_%YY.parquet - has
%               additional column with sleep stage labesl
%      ../RCS#[L or R]/Overnight/Session#_%mm_%DD_%YY_EventLog.csv -
%      includes relevant event information for that session
%      ../DREEM_data/RCS#/[Dreem ID]/CorrectedDreem/[DREEM ID -- Date].mat
%               - Dreem data time-synchronized to RCS session

% Output of create_filepath_csv.py - contains the filepaths of RC+S and
% corrrespongind DREEM data
    %FILE_PATH_CSV_NAME = '/media/longterm_hdd/Clay/DREEM_data/filepaths_07_09_edited.csv';
FILE_PATH_CSV_NAME = '/media/longterm_hdd/Clay/SleepStimOffset/filepaths_02.csv'
% Columns to drop from combinedDataTable prior to writing to parquet file
COLS_TO_DROP = {'TD_samplerate', 'Power_ExternalValuesMask', 'Power_FftSize', 'Power_ValidDataMask'};
% basepath for parquet and eventlog csv
OUT_PATH_BASE = '/media/longterm_hdd/Clay/SleepStimOffset';
% Include Autonomic Data Flag; If False, only Sleep Stage data will be
% added to parquet
autonomic_data = true;

display_text = "Proceed with" + newline + "-file_path_csv: %s" + newline ...
    + "-output_dir: %s" + newline + "-include autonomic data: %s" + ...
    newline + "-Participant(s) Time Zone: America/Los Angeles" + ...
    newline + "[Y/N]";
prompt = sprintf(display_text, FILE_PATH_CSV_NAME, OUT_PATH_BASE, string(autonomic_data));
txt = input(prompt, "s");

if txt ~= "Y" & txt ~= "y"
    error('Chose not to proceed with file paths')
end

% Read output csv created from 'create_filepath_csv.py'
file_paths = readtable(FILE_PATH_CSV_NAME);


if sum(ismember(file_paths.Properties.VariableNames,'SleepStage_Labeled_RCS_Parquet')) == 0
    Sleep_Labeled_RCS_Parquet_outpaths = cell(height(file_paths), 1);
    write_filepaths_csv_copy = true;
end

% Import File_locations and Options necessary for Fahim's dreem sync
% function
[File_locations, Options] = get_sync_package_structs();

% Cycle through each session
for i=1:height(file_paths)
    % Update File_locations fields with appropriate H5, txt, and RCS paths
    File_locations = update_file_paths_from_table(i, file_paths, File_locations);
    % Get synced dreem data and relevant RCS info
    [sleep_DREEM_data, combinedDataTable, eventLog] = sync_package(File_locations, Options);

    % Drop unwanted columns to save disk space
    if ~isempty(COLS_TO_DROP)
        combinedDataTable = removevars(combinedDataTable, COLS_TO_DROP);

        if sum(isnan(combinedDataTable.TD_key1)) == height(combinedDataTable)
            combinedDataTable = removevars(combinedDataTable, {'TD_key1'});
        elseif sum(isnan(combinedDataTable.TD_key0)) == height(combinedDataTable)
            combinedDataTable = removevars(combinedDataTable, {'TD_key0'});
        end
    end

    % Add sleep stage labels to combinedDataTable
    combinedDataTable = add_sleep_stage_labels(combinedDataTable, sleep_DREEM_data, autonomic_data);
    combinedDataTable.localTime.TimeZone = 'America/Los_Angeles';
    
    % Checks if an outpath is non-empty in filepaths. If no, creates one and add
    % to file_paths
    if sum(ismember(file_paths.Properties.VariableNames,'SleepStage_Labeled_RCS_Parquet')) == 0
        % Output assumes a directory structure similar to that in Projects
        % directory in Dropbox.
        [parquet_out_file_path, eventLog_out_path] = get_outpath(OUT_PATH_BASE, file_paths, i);
        Sleep_Labeled_RCS_Parquet_outpaths{i} = parquet_out_file_path;
        write_filepaths_csv_copy = true;
    else
        parquet_out_file_path = file_paths.SleepStage_Labeled_RCS_Path{i};
        eventLog_out_path = [parquet_out_file_path(1:end-8) '_eventLog.csv'];
    end
%%    
    disp('Writing to Parquet...')
    % Save session's sleep-labeled combinedDataTable as parquet to outpath
    if not(isfolder(fileparts(parquet_out_file_path)))
        mkdir(fileparts(parquet_out_file_path))
    end
    
    varNames = combinedDataTable.Properties.VariableNames;
    
    % Initialize the output table
    outTable = combinedDataTable;
    
    % Loop through each column in the table
    for ii = 1:numel(varNames)
        colData = combinedDataTable.(varNames{ii});
        if iscell(colData) && any(cellfun(@isnumeric, colData)) && any(~cellfun(@isnumeric, colData))
            % If the column contains both double and cell array data types,
            % convert the column to a cell array 

            % Should consider keeping column as numerical instead of
            % converting to strings...
            outTable.(varNames{ii}) = cellfun(@num2str_cust, colData, 'UniformOutput', false);
        end
    end
    % Change output table back to combinedDataTable
    % combinedDataTable = outTable;
    parquetwrite(parquet_out_file_path, outTable);
    writetable(eventLog, eventLog_out_path, 'Delimiter',',');
    
    % Close any potential plots
    close all;
end

%%
write_filepaths_csv_copy = true;

if write_filepaths_csv_copy
    table_path = split(FILE_PATH_CSV_NAME, '.');
    file_paths.Properties.VariableNames = file_paths.Properties.VariableDescriptions;
    file_paths.SleepStage_Labeled_RCS_Parquet = Sleep_Labeled_RCS_Parquet_outpaths;
    writetable(file_paths, append(table_path{1}, "_matlab_copy.", table_path{2}), 'Delimiter',',');
end

function out = num2str_cust(input)
    if isequal(class(input), 'double')
        out = num2str(input);
    elseif isequal(class(input), 'cell')
        if isequal(size(input), [1 1])
            out = input{1};
        end
    else
        out = input;
    end
end