function [combinedDataTT] = add_sleep_stage_labels(combinedDataTable, dreem_data)
%ADD_SLEEP_STAGE_LABELS Summary of this function goes here
%   Detailed explanation goes here
combinedDataTT = table2timetable(combinedDataTable, 'RowTimes', 'localTime');

dreem_parsed = dreem_data(:,"/sleep_stage");
dreem_parsed.Properties.VariableNames{"/sleep_stage"} = 'SleepStage';
combinedDataTT = synchronize(combinedDataTT, dreem_parsed, 'first', 'nearest');
end

