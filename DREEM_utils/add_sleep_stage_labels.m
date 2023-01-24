function [combinedDataTT] = add_sleep_stage_labels(combinedDataTable, dreem_data, autonomic_data)
%ADD_SLEEP_STAGE_LABELS Summary of this function goes here
%   Detailed explanation goes here
combinedDataTT = table2timetable(combinedDataTable, 'RowTimes', 'localTime');
if autonomic_data
    cols = ["/pulse_oximeter_infrared/filtered", ... 
        "/pulse_oximeter_infrared/raw", ...
        "/pulse_oximeter_red/filtered", ...
        "/pulse_oximeter_red/raw", ...
        "/positiongram", ...
        "/edf/PulseOxyInfrare", ...
        "/edf/RespirationX", ...
        "/edf/RespirationY", ...
        "/edf/RespirationZ", ...
        "/sleep_stage"];
    dreem_parsed = dreem_data(:,cols);
    dreem_parsed.Properties.VariableNames{"/pulse_oximeter_infrared/filtered"} = 'PO_infrared_filtered';
    dreem_parsed.Properties.VariableNames{"/pulse_oximeter_infrared/raw"} = 'PO_infrared_raw';
    dreem_parsed.Properties.VariableNames{"/pulse_oximeter_red/filtered"} = 'PO_red_filtered';
    dreem_parsed.Properties.VariableNames{"/pulse_oximeter_red/raw"} = 'PO_red_raw';
    dreem_parsed.Properties.VariableNames{"/positiongram"} = 'positiongram';
    dreem_parsed.Properties.VariableNames{"/edf/PulseOxyInfrare"} = 'PulseOxyInfrared';
    dreem_parsed.Properties.VariableNames{"/edf/RespirationX"} = 'RespirationX';
    dreem_parsed.Properties.VariableNames{"/edf/RespirationY"} = 'RespirationY';
    dreem_parsed.Properties.VariableNames{"/edf/RespirationZ"} = 'RespirationZ';
    dreem_parsed.Properties.VariableNames{"/sleep_stage"} = 'SleepStage';
else
    dreem_parsed = dreem_data(:,"/sleep_stage");
    dreem_parsed.Properties.VariableNames{"/sleep_stage"} = 'SleepStage';
end

combinedDataTT = synchronize(combinedDataTT, dreem_parsed, 'first', 'nearest');
end

