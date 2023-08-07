from databasing.database_calls import get_device_as_pl_df, get_gains_from_settings, get_settings_for_pb_calcs
from data_processing.data_transforms import chunk_df_by_timesegment
from rcssim_helpers import add_simulated_ffts
import duckdb
import argparse
import pyarrow as pa

DATABASE_PATH = '/media/shortterm_ssd/Clay/databases/duckdb/rcs-db.duckdb'

"""
This script takes in a device and a duckdb connection, finds the corresponding time series datastreams, and returns a dataframe of time segments with simulated FFTs added as additional columns.
Default settings for the FFT calculations are the corresponding settings used during the RC+S data streaming session.
"""

# TODO: Allow FFT settings to be parameterized... pass in settings as potentially custom dict
def chunk_and_add_ffts(device, con):
    """
    This function takes in a device and a duckdb connection, and returns a dataframe of time segments with simulated FFTs added as additional columns.
    """
        
    df = get_device_as_pl_df(device, con, lazy=True)

    sessions = df.select('SessionIdentity').unique().collect().to_dict(
            as_series=False)['SessionIdentity']
    settings = get_settings_for_pb_calcs(device, con, sessions, 'SessionIdentity')
    gains = get_gains_from_settings(
        sessions, 'SessionIdentity', device, con)


    # TODO: TD_BG need not be gains[0]. Could be gains[1]... I should use the TDSettings table to get correct gain for subcortical channel
    gains_tmp = []
    for channel in ['TD_BG', 'TD_key2', 'TD_key3']:
        if channel == 'TD_BG':
            gains_tmp.append(gains[0])
        if channel == 'TD_key2':
            gains_tmp.append(gains[1])
        if channel == 'TD_key3':
            gains_tmp.append(gains[2])
    gains = gains_tmp

    df_chunked = chunk_df_by_timesegment(df)
    df_chunked = add_simulated_ffts(df_chunked, settings, gains)
    return df_chunked, settings, gains

if __name__ == '__main__':
    '''
    First argument should be the name of the table to create in the database.
    Second+ argument should be the name of the devices to chunk and add FFTs for.
    Flag for replacing existing tables is --replace.

    Example usage:
    python3 feature_engineering/chunk_and_fft_TD_data.py -table_name overnight_simulated_FFTs -devices 02L 02R 03L 03R 07L 07R 09L 09R 16L 16R -schema r02L r02R r03L r03R r07L r07R r09L r09R r16L r16R --replace
    
        This Will add FFTs for all devices listed to the database in the specified schemas (i.e. r02L.overnight_simulated_FFTs), replacing existing tables if they exist.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-table_name', type=str, help='Name of table to create in database')
    parser.add_argument('-devices', type=str, nargs='+', help='Names of devices to chunk and add FFTs for (e.g. 02L 02R 03L)')
    parser.add_argument('-r', '--replace', action='store_true', help='Replace existing table', default=False, required=False)
    parser.add_argument('-database_path', type=str, help='Path to database', default=DATABASE_PATH, required=False)
    parser.add_argument('-schema', type=str, nargs='+', help="Schema to create table in. Either length 1 (all tables go into a single schema), or length [-devices] (table for each device has it's own schema"
                        , required=False)
    args = parser.parse_args()

    con = duckdb.connect(database=args.database_path)

    for i, device in enumerate(args.devices):
        df_chunked, settings, _ = chunk_and_add_ffts(device, con)
        settings_table = pa.Table.from_pydict(settings)

        if len(args.schema) == 1:
            if args.replace:
                # What to do about replacing existing tables?
                con.sql(f'create or replace table {args.schema}.r{device}_{args.table_name} as select * from df_chunked;')
                con.sql(f'create or replace table {args.schema}.r{device}_{args.table_name}_settings as select * from settings_table;')
            else:
                con.sql(f'create table {args.schema}.r{device}_{args.table_name} as select * from df_chunked;')
                con.sql(f'create table {args.schema}.r{device}_{args.table_name}_settings as select * from settings_table;')
        else:
            #for schema in args.schema:
            schema = args.schema[i]
            if args.replace:
                # What to do about replacing existing tables?
                con.sql(f'create or replace table {schema}.{args.table_name} as select * from df_chunked;')
                con.sql(f'create or replace table {schema}.{args.table_name}_settings as select * from settings_table;')
            else:
                con.sql(f'create table {schema}.{args.table_name} as select * from df_chunked;')
                con.sql(f'create table {schema}.{args.table_name}_settings as select * from settings_table;')
    con.close()