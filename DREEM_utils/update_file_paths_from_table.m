function [File_locations] = update_file_paths_from_table(ind, file_paths, File_locations)

full_txt_path = split(file_paths.Txt{ind},"/");
full_H5_path = split(file_paths.H5{ind}, "/");
full_rcs_path = file_paths.Data_Server_FilePath{ind};
brain_side = file_paths.Side{ind};

% Location of Dreem h5 file
H5_dir = join(full_H5_path(1:end-1), "/");
File_locations.h5filelocation= H5_dir{1};

% Name of Dreem h5 file
File_locations.h5filename= full_H5_path{end};

% Location of Dreem sleep text file
txt_dir = join(full_txt_path(1:end-1), "/");
File_locations.sleepfilelocation= txt_dir{1};

% Name of Dreem sleep text file
File_locations.sleeptextfilename=full_txt_path{end};

% Location of Openmind Analyze RCS data tool
File_locations.rcs_lib_path='/home/claysmyth/code/Analysis-rcs-data/';

% Location of RCS data 
File_locations.rcsfolderPath= full_rcs_path(2:end-1);

% Location where Corrected Dreem data is saved
File_locations.savepath= append('/media/longterm_hdd/Clay/DREEM_data/',file_paths.RCS_{ind},'/', ...
    full_txt_path{end-2},'/CorrectedDreem/');

% Name of corrected Dreem Data
File_locations.savefilename=[File_locations.h5filename(1:end-3) '_' brain_side '.mat'];