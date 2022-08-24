function [parquet_out_path_full, eventlog_out_path_full] = get_outpath(OUT_PATH_BASE, file_paths, ind)
%GET_OUTPATH Returns filepath to save the combinedDataTable as parquet file
%   Assumes dir [OUT_PATH_FULL]/RCS#[L or R]/Overnight/ exists prior to
%   execution
parquet_file_name = append(file_paths.Session_(ind), '_', ...
    datestr(file_paths.DateTime(ind), 'dd-mmm-YYYY'), '.parquet');

eventLog_file_name = append(file_paths.Session_(ind), '_', ...
    datestr(file_paths.DateTime(ind), 'dd-mmm-YYYY'), '_eventLog.csv');

% FIX BELOW.. NOT INSERTING SLASHES
parquet_out_path_full = [OUT_PATH_BASE, '/', append(file_paths.RCS_{ind}, file_paths.Side{ind}(1)), ...
    '/Overnight/', parquet_file_name{1}];

eventlog_out_path_full = [OUT_PATH_BASE, '/', append(file_paths.RCS_{ind}, file_paths.Side{ind}(1)), ...
    '/Overnight/', eventLog_file_name{1}];

end

