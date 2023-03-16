function [File_locations, Options] = get_sync_package_structs()
%% Set File locations and directories
File_locations=struct;
% Location of Dreem h5 file
File_locations.h5filelocation='';

% Name of Dreem h5 file
File_locations.h5filename='';

% Location of Dreem sleep text file
File_locations.sleepfilelocation='';

% Name of Dreem sleep text file
File_locations.sleeptextfilename='';

% Location of Openmind Analyze RCS data tool
File_locations.rcs_lib_path='/home/claysmyth/code/Analysis-rcs-data/';

% Location of RCS data 
File_locations.rcsfolderPath='';

% Location where Corrected Dreem data is saved
File_locations.savepath='';

% Name of corrected Dreem Data
File_locations.savefilename='';

% Name of corrected h5 data
File_locations.newh5filename=[''];

% Name of corrected h5 data
File_locations.parquetfilename=[''];

% Name of sleep metadata CSV
File_locations.summaryCSVfilename=[''];


%% Select options for synchronization
Options=struct;
% sampling rate of the DREEM data. All data streams will have this sampling
% rate
Options.Fs=250;

% h5 reader option- variable for the timestamps
Options.h5timestampName='eeg_timestamps';
% EDF variables to add and sync
Options.EDFvariablesToAdd={'PulseOxyInfrare';'RespirationX';'RespirationY';'RespirationZ'};

% Display the Original Sleep hypnogram and the hypnogram with timestamps embedded 
Options.DisplaySleepDataWithTimestamps=false;
% Accelerometry in Dreem and RCS go through different filtering process at hardware level.
% There is a +20 dB gain in RCnS data from 0.5 Hz and onwards. This fixes that.
% This is experimental feature DONT USE WITHOUT PROPER OUTPUT INSPECTION
Options.FilterCorrectionAccelerometry=false;
% If Dreem h5 file does not provide timezone, Use RCS timezone for Dreem
% data
Options.UseRCSTimezoneForCorrection=false;
% Display overlap of valid data between Dreem and RCS streams
Options.DisplayOverlaps=false;
% Normalized Accelerometry of Dreem and RCS before calculating
% cross-correlation
Options.NormalizeAccelerometry=true;
% Plot Comparing Dreem and RCS accelerometry before cross-correlation
Options.DisplayAccelerometryBeforeCrosscorr=false;
% Plot Comparing Dreem and RCS accelerometry after cross-correlation
Options.DisplayAccelerometryAfterCrosscorr=false;
% Plot cross-correlation results
Options.DisplayCrosscorr=false;
% Divide data into these many parts, calculate cross-correlations for
% parts and correct Dreem data for each part
Options.CrossCorrParts=2;
% Calculate cross-correlation for full data
Options.CrossCorrFullData=false;
% Validation test for full data cross-corr and cross-corr by parts
Options.CrossCorrValidate=false;
% Display Plot for final hypnogram and accelerometry with comparison to
% original
Options.DisplayFinalHypnogram=false;
% Display Plot for final accelerometry with comparison to
% original
Options.DisplayFinalAccelerometry=false;
% Save corrected Dreem h5 data
Options.SaveH5Result=false;
% Save corrected Dreem mat data
Options.SaveSleepResult=true;
% Save corrected Dreem Parquet data
Options.SaveParquet=false;
% Plot sampling intervals for Corrected dreem data
Options.PlotSamplingIntervals=false;
% Save sync summary CSV
Options.SaveSyncSummaryCSV=true;
