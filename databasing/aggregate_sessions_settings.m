% This script takes in a Projects Summary csv, and aggregates the various
% settings - stim, sense, fft&power, adaptive&detector, metadata - into
% individual tables and exports as csvs. If desired, Sessions can be selected by
% sessiontype.


PROJ_SUMMARY_CSV = '/media/dropbox_hdd/Starr Lab Dropbox/Projects/Sleep/Sleep_Summary.csv';
desired_session_types = ['Overnight'];
output_prefix = 'overnight_';
OUT_PATH_BASE = '/media/longterm_hdd/Clay/Sleep_10day_with_autonomic/';


display_text = "Proceed with" + newline + "-proj_summary_csv: %s" + newline ...
    + "-output_dir: %s" + newline + "-session types: %s" + ...
    newline + "-file output prefix: %s" + ...
    newline + "[Y/N]";
prompt = sprintf(display_text, PROJ_SUMMARY_CSV, OUT_PATH_BASE, desired_session_types, output_prefix);
txt = input(prompt, "s");

if txt ~= "Y" & txt ~= "y"
    error('Chose not to proceed with file paths')
end

project_csv = readtable(PROJ_SUMMARY_CSV, 'Delimiter', ',', 'VariableNamingRule','preserve');
project_csv.Device = strcat(project_csv{:,'RCS#'}, cellfun(@(s)s(1),project_csv{:,'Side'}));

rel_inds = ismember(project_csv.('SessionType(s)'), desired_session_types);
relevant_sessions = project_csv(rel_inds, :);

devices = unique(project_csv.Device);

all_data_table = table;
all_data_table.Devices = devices;
filler_array = cell(size(devices));

for i=1:size(devices)
    filler_array{i} = table;
end

% all_data_table.SenseSettings = filler_array;
all_data_table.TDSettings = filler_array;
all_data_table.FftAndPowerSettings = filler_array;
all_data_table.DetectorSettings = filler_array;
all_data_table.AdaptiveSettings = filler_array;
all_data_table.StimSettings = filler_array;
all_data_table.EventLog = filler_array;

%metadata = table;

for i=1:size(relevant_sessions, 1)
    disp(['On row ', int2str(i), ' of ' int2str(size(relevant_sessions, 1))])
    session_identifier = [relevant_sessions.Device{i}(4:end), '_', char(datetime(relevant_sessions.TimeEnded{i}, 'InputFormat', 'MM-dd-yyyy HH:mm:SS', 'Format', 'MM-dd-yy'))];
    session_descriptors = relevant_sessions(i, {'Session#','TimeStarted', 'TimeEnded', 'SessionType(s)', 'Device'});
    session_descriptors.SessionIdentity = session_identifier;
    session_descriptors = renamevars(session_descriptors, {'TimeStarted', 'TimeEnded', 'SessionType(s)'}, {'SessionStartTime', 'SessionEndTime', 'SessionTypes'});

    curr_device = relevant_sessions.Device{i};
    
    raw_data_path = regexprep(char(relevant_sessions.Data_Server_FilePath{i}), "'", '');

    [unifiedDerivedTimes, timeDomainData, timeDomainData_onlyTimeVariables, ...
    timeDomain_timeVariableNames, AccelData, AccelData_onlyTimeVariables, ... 
    Accel_timeVariableNames, PowerData, PowerData_onlyTimeVariables, ...
    Power_timeVariableNames, FFTData, FFTData_onlyTimeVariables, ... 
    FFT_timeVariableNames, AdaptiveData, AdaptiveData_onlyTimeVariables, ...
    Adaptive_timeVariableNames, timeDomainSettings, powerSettings, ...
    fftSettings, eventLogTable, metaData, stimSettingsOut, stimMetaData, ...
    stimLogSettings, DetectorSettings, AdaptiveStimSettings, ...
    AdaptiveEmbeddedRuns_StimSettings] = ProcessRCS(raw_data_path, 2);

    if isempty(all_data_table(strcmp(all_data_table.Devices, curr_device), :).TDSettings{1})
        all_data_table(strcmp(all_data_table.Devices, curr_device), :).TDSettings{1} = denest_and_process_td_settings(timeDomainSettings, metaData, session_descriptors);
        all_data_table(strcmp(all_data_table.Devices, curr_device), :).FftAndPowerSettings{1} = denest_and_process_fft_power_settings(powerSettings, session_descriptors);
        all_data_table(strcmp(all_data_table.Devices, curr_device), :).AdaptiveSettings{1} = denest_and_process_adaptive_settings(AdaptiveStimSettings, session_descriptors);
        all_data_table(strcmp(all_data_table.Devices, curr_device), :).StimSettings{1} = denest_and_process_stim_settings(stimLogSettings, stimMetaData, session_descriptors);
        all_data_table(strcmp(all_data_table.Devices, curr_device), :).DetectorSettings{1} = denest_and_process_detector_settings(DetectorSettings, session_descriptors);
        all_data_table(strcmp(all_data_table.Devices, curr_device), :).EventLog{1} = eventLogTable;
    else
        all_data_table(strcmp(all_data_table.Devices, curr_device), :).TDSettings{1} = ...
            [all_data_table(strcmp(all_data_table.Devices, curr_device),:).TDSettings{1}; denest_and_process_td_settings(timeDomainSettings, metaData, session_descriptors)];

        all_data_table(strcmp(all_data_table.Devices, curr_device), :).FftAndPowerSettings{1} = ...
            [all_data_table(strcmp(all_data_table.Devices, curr_device), :).FftAndPowerSettings{1}; denest_and_process_fft_power_settings(powerSettings, session_descriptors)];

        all_data_table(strcmp(all_data_table.Devices, curr_device), :).DetectorSettings{1} = ...
            [all_data_table(strcmp(all_data_table.Devices, curr_device), :).DetectorSettings{1}; denest_and_process_detector_settings(DetectorSettings, session_descriptors)];

        all_data_table(strcmp(all_data_table.Devices, curr_device), :).AdaptiveSettings{1} = ...
            [all_data_table(strcmp(all_data_table.Devices, curr_device), :).AdaptiveSettings{1}; denest_and_process_adaptive_settings(AdaptiveStimSettings, session_descriptors)];

        all_data_table(strcmp(all_data_table.Devices, curr_device), :).StimSettings{1} = ...
            [all_data_table(strcmp(all_data_table.Devices, curr_device), :).StimSettings{1}; denest_and_process_stim_settings(stimLogSettings, stimMetaData, session_descriptors)];

        all_data_table(strcmp(all_data_table.Devices, curr_device), :).EventLog{1} = ...
            [all_data_table(strcmp(all_data_table.Devices, curr_device), :).EventLog{1}; eventLogTable];
    end

end

%%
for i=1:size(all_data_table,1)
    curr_path = [OUT_PATH_BASE, all_data_table.Devices{i}, '/'];
    writetable(all_data_table(i,:).TDSettings{1}, fullfile(curr_path, [output_prefix, 'TDSettings.csv']))
    writetable(all_data_table(i,:).FftAndPowerSettings{1}, fullfile(curr_path, [output_prefix, 'FftAndPowerSettings.csv']))
    writetable(all_data_table(i,:).DetectorSettings{1}, fullfile(curr_path, [output_prefix, 'DetectorSettings.csv']))
    writetable(all_data_table(i,:).AdaptiveSettings{1}, fullfile(curr_path, [output_prefix, 'AdaptiveSettings.csv']))
    writetable(all_data_table(i,:).StimSettings{1}, fullfile(curr_path, [output_prefix, 'StimSettings.csv']))
    writetable(all_data_table(i,:).EventLog{1}, fullfile(curr_path, [output_prefix, 'EventLogs.csv']))
end

%%
% session_descriptor = table;
% session_descriptor.Device = 'RCS12L';
% td_settings_adjusted = denest_and_process_td_settings(timeDomainSettings, session_descriptor);
% ps = denest_and_process_fft_power_settings(powerSettings, session_descriptor);
% as = denest_and_process_adaptive_settings(AdaptiveStimSettings, session_descriptor);
% stim = denest_and_process_stim_settings(stimLogSettings, stimMetaData, session_descriptor);
% det = denest_and_process_detector_settings(DetectorSettings, session_descriptor);


%%
function [td_settings_adjusted] = denest_and_process_td_settings(td_settings, metaData, session_descriptors)
    ep_mode = zeros(1,4);
    %gains = zeros(1,4);
    hpf = zeros(1,4);
    td_settings_adjusted = td_settings(:,1:end-1);
 
    for i=1:size(td_settings_adjusted, 1)
        ep_mode = [td_settings.TDsettings{i,1}.evokedMode];
        %gains = [td_settings.TDsettings{i,1}.gain];
        hpf = [td_settings.TDsettings{i,1}.hpf];
        
        % Double check that the below is going into the correct row
        td_settings_adjusted.evokedMode{i} = ep_mode;
        %td_settings_adjusted.gain{i} = gains;
        td_settings_adjusted.hpf{i} = hpf;

    end
    
    gains_row = renamevars(struct2table(metaData.ampGains), {'Amp1', 'Amp2', 'Amp3', 'Amp4'}, {'gain_1', 'gain_2', 'gain_3', 'gain_4'});
    gains = repmat(gains_row, size(td_settings_adjusted, 1), 1);

    session_descriptors = repmat(session_descriptors, size(td_settings_adjusted, 1), 1);

    td_settings_adjusted = horzcat(session_descriptors, td_settings_adjusted, gains);

end


function [power_settings_adjusted] = denest_and_process_fft_power_settings(power_settings, session_descriptors)
    power_settings_adjusted = table;
    for i=1:size(power_settings,1)
        fftconfig = struct2table(power_settings(i,:).fftConfig);
        fftconfig = renamevars(fftconfig, {'bandFormationConfig', 'config', 'interval', 'size', 'streamOffsetBins', 'streamSizeBins', 'windowLoad'}, ...
        {'fft_bandFormationConfig', 'fft_config', 'fft_interval', 'fft_size', 'fft_streamOffsetBins', 'fft_numBins', 'fft_windowLoad'});
        fftconfig.fft_binWidth = power_settings(i,:).powerBands.binWidth;

        powerbands = cellfun(@(x) strsplit(x, 'Hz'), power_settings(i,:).powerBands.powerBandsInHz, 'UniformOutput', false);
        powerbins = cellfun(@(x) strsplit(x, 'Hz'), power_settings(i,:).powerBands.powerBinsInHz, 'UniformOutput', false);
        powerband_table = table;
        for j=1:size(powerbands)
            powerband_table.(['Power_Band'  num2str(j)]) = {strjoin(powerbands{j,1}, '')};
        end
        
        for j=1:size(powerbands)
            powerband_table.(['Power_Band'  num2str(j) '_indices']) = {num2str(power_settings(i,:).powerBands.indices_BandStart_BandStop(j,:))};
        end

        for j=1:size(powerbands)
            powerband_table.(['Power_Band'  num2str(j) '_bins']) = {strjoin(powerbins{j,1}, '')};
        end

        row = horzcat(power_settings(i,:), powerband_table, fftconfig);
        row.powerBands = [];
        power_settings_adjusted = vertcat(power_settings_adjusted, row);
    end

    session_descriptors = repmat(session_descriptors, size(power_settings_adjusted,1), 1);

    power_settings_adjusted = horzcat(session_descriptors, power_settings_adjusted);
    power_settings_adjusted.fftConfig = [];
end


function [stim_settings_adjusted] = denest_and_process_stim_settings(stim_settings, stimMetaData, session_descriptors)
    groupA = struct2table(stim_settings.GroupA);
    groupB = struct2table(stim_settings.GroupB);
    groupC = struct2table(stim_settings.GroupC);
    groupD = struct2table(stim_settings.GroupD);

    groupA = renamevars(groupA, {'RateInHz', 'ampInMilliamps', 'pulseWidthInMicroseconds'}, ...
        {'GroupA_RateInHz', 'GroupA_ampInMilliamps', 'GroupA_pulseWidthInMicroseconds'});
    groupB = renamevars(groupB, {'RateInHz', 'ampInMilliamps', 'pulseWidthInMicroseconds'}, ...
        {'GroupB_RateInHz', 'GroupB_ampInMilliamps', 'GroupB_pulseWidthInMicroseconds'});
    groupC = renamevars(groupC, {'RateInHz', 'ampInMilliamps', 'pulseWidthInMicroseconds'}, ...
        {'GroupC_RateInHz', 'GroupC_ampInMilliamps', 'GroupC_pulseWidthInMicroseconds'});
    groupD = renamevars(groupD, {'RateInHz', 'ampInMilliamps', 'pulseWidthInMicroseconds'}, ...
        {'GroupD_RateInHz', 'GroupD_ampInMilliamps', 'GroupD_pulseWidthInMicroseconds'});

    groups = horzcat(groupA, groupB, groupC, groupD);
    
    stim_settings_adjusted = stim_settings;

    for i=1:size(stim_settings,1)
        if all(class(stim_settings.updatedParameters{i,:}) == 'cell')
            stim_settings_adjusted(i,:).updatedParameters = {strjoin(stim_settings.updatedParameters{i,:}, ', ')};
        end
    end
    stim_settings_adjusted.GroupA = [];
    stim_settings_adjusted.GroupB = [];
    stim_settings_adjusted.GroupC = [];
    stim_settings_adjusted.GroupD = [];

    stim_settings_adjusted = horzcat(stim_settings_adjusted, groups);

    stimMeta = table;
    %stimMeta.anodes_prog1 = [stimMetaData.anodes{:,1}];
    %stimMeta.anodes_prog1 = {stimMetaData.anodes{:,1}};
    stimMeta.anodes_prog1 = cellfun(@num2str,{stimMetaData.anodes{:,1}}, 'un',0);
    %stimMeta.cathodes_prog1 = [stimMetaData.cathodes{:,1}];
    %stimMeta.cathodes_prog1 = {stimMetaData.cathodes{:,1}};
    stimMeta.cathodes_prog1 = cellfun(@num2str,{stimMetaData.cathodes{:,1}}, 'un',0);
    stimMeta.validPrograms = {strjoin(strsplit([stimMetaData.validProgramNames{:,1}], '1G'), '1, G')};
    stimMeta = repmat(stimMeta, size(stim_settings_adjusted, 1), 1);

    session_descriptors = repmat(session_descriptors, size(stim_settings_adjusted,1), 1);

    stim_settings_adjusted = horzcat(session_descriptors, stim_settings_adjusted, stimMeta);

end


function [adaptive_settings_adjusted] = denest_and_process_adaptive_settings(adaptiveSettings, session_descriptors)
    adaptive_settings_adjusted = adaptiveSettings;
    adaptive_settings_adjusted.fall = zeros(size(adaptive_settings_adjusted, 1), 4);
    adaptive_settings_adjusted.rise = zeros(size(adaptive_settings_adjusted, 1), 4);
    for i=1:size(adaptiveSettings,1)
        adaptive_settings_adjusted(i,:).fall = [adaptiveSettings.deltas{i,1}.fall];
        adaptive_settings_adjusted(i,:).rise = [adaptiveSettings.deltas{i,1}.rise];
        if all(class(adaptiveSettings.updatedParameters{i,:}) == 'cell')
            adaptive_settings_adjusted(i,:).updatedParameters = {strjoin(adaptiveSettings.updatedParameters{i,:}, ', ')};
        end
    end
    adaptive_settings_adjusted = horzcat(adaptive_settings_adjusted, struct2table(adaptiveSettings.states));
    adaptive_settings_adjusted.states = [];


    session_descriptors = repmat(session_descriptors, size(adaptive_settings_adjusted,1), 1);

    adaptive_settings_adjusted = horzcat(session_descriptors, adaptive_settings_adjusted);
    adaptive_settings_adjusted.deltas = [];

end


function [detector_settings_adjusted] = denest_and_process_detector_settings(detectorSettings, session_descriptors)
    detector_settings_adjusted = detectorSettings;
    for i=1:size(detectorSettings,1)
        if all(class(detectorSettings.updatedParameters{i,:}) == 'cell')
            detector_settings_adjusted(i,:).updatedParameters = {strjoin(detectorSettings.updatedParameters{i,:}, ', ')};
        end
    end

    if size(detectorSettings) > 1
        ld0 = struct2table(detectorSettings.Ld0);
        ld1 = struct2table(detectorSettings.Ld1);
    else
        ld0 = struct2table(detectorSettings.Ld0, 'AsArray', true);
        ld1 = struct2table(detectorSettings.Ld1, 'AsArray', true);
    end

    ld0 = renamevars(ld0, {'biasTerm', 'features', 'fractionalFixedPointValue', 'updateRate', 'blankingDurationUponStateChange', 'onsetDuration', 'holdoffTime', 'terminationDuration', 'detectionInputs_BinaryCode', 'detectionEnable_BinaryCode'}, ...
        {'Ld0_biasTerm', 'Ld0_features', 'Ld0_fractionalFixedPointValue', 'Ld0_updateRate', 'Ld0_blankingDurationUponStateChange', 'Ld0_onsetDuration', 'Ld0_holdoffTime', 'Ld0_terminationDuration', 'Ld0_detectionInputs_BinaryCode', 'Ld0_detectionEnable_BinaryCode'});
    ld0.Ld0_normalizationMultiplyVector = zeros(size(ld0, 1), 4);
    ld0.Ld0_normalizationSubtractVector = zeros(size(ld0, 1), 4);
    ld0.Ld0_weightVector = zeros(size(ld0, 1), 4);
    for i=1:size(ld0, 1)
        ld0(i,:).Ld0_normalizationMultiplyVector = [detectorSettings(i,:).Ld0.features.normalizationMultiplyVector];
        ld0(i,:).Ld0_normalizationSubtractVector = [detectorSettings(i,:).Ld0.features.normalizationSubtractVector];
        ld0(i,:).Ld0_weightVector = [detectorSettings(i,:).Ld0.features.weightVector];
    end


    ld1 = renamevars(ld1, {'biasTerm', 'features', 'fractionalFixedPointValue', 'updateRate', 'blankingDurationUponStateChange', 'onsetDuration', 'holdoffTime', 'terminationDuration', 'detectionInputs_BinaryCode', 'detectionEnable_BinaryCode'}, ...
        {'Ld1_biasTerm', 'Ld1_features', 'Ld1_fractionalFixedPointValue', 'Ld1_updateRate', 'Ld1_blankingDurationUponStateChange', 'Ld1_onsetDuration', 'Ld1_holdoffTime', 'Ld1_terminationDuration', 'Ld1_detectionInputs_BinaryCode', 'Ld1_detectionEnable_BinaryCode'});
    ld1.Ld1_normalizationMultiplyVector = zeros(size(ld1, 1), 4);
    ld1.Ld1_normalizationSubtractVector = zeros(size(ld1, 1), 4);
    ld1.Ld1_weightVector = zeros(size(ld1, 1), 4);
    for i=1:size(ld1, 1)
        ld1(i,:).Ld1_normalizationMultiplyVector = [detectorSettings(i,:).Ld1.features.normalizationMultiplyVector];
        ld1(i,:).Ld1_normalizationSubtractVector = [detectorSettings(i,:).Ld1.features.normalizationSubtractVector];
        ld1(i,:).Ld1_weightVector = [detectorSettings(i,:).Ld1.features.weightVector];
    end

    ld0.Ld0_features = [];
    ld1.Ld1_features = [];

    detector_settings_adjusted = horzcat(detector_settings_adjusted, ld0, ld1);
    detector_settings_adjusted.Ld0 = [];
    detector_settings_adjusted.Ld1 = [];


    session_descriptors = repmat(session_descriptors, size(detector_settings_adjusted,1), 1);

    detector_settings_adjusted = horzcat(session_descriptors, detector_settings_adjusted);
end


