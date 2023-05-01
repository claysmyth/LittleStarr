import duckdb
import os
import polars as pl

def create_sorted_view(table_name, sort_column, db_path):
    con = duckdb.connect(dabase=db_path)
    con.sql(f"CREATE VIEW {table_name}_sorted AS SELECT * FROM {table_name} ORDER BY {sort_column}")
    con.close()


def sort_table_by_column(table_name, sort_column, db_path):
    con = duckdb.connect(database=db_path)
    con.sql(f"CREATE TABLE {table_name}_sorted AS SELECT * FROM {table_name} ORDER BY {sort_column}")
    con.sql(f"DROP TABLE {table_name}")
    con.sql(f"ALTER TABLE {table_name}_sorted RENAME TO {table_name}")
    con.close()


def label_and_import_sleep_training_data_parquets(base_path, cols_to_drop, db_path):
    con = duckdb.connect(dabase=db_path)
    devices = os.listdir(base_path)
    for device in devices:
        device_file_path = base_path + device + '/Overnight/'
        device_identifier = device[3:]
        
        ind = 0
        for session_parquet in glob.glob(device_file_path + '/*.parquet'):
            df = pl.read_parquet(session_parquet)
            df = df.drop(cols_to_drop)
            
            end_date = pl.select(df[df.height-1,['localTime']])[0,0].date().strftime('%m-%d-%y')
            session_classifier = device_identifier + '_' + end_date

            session_num = os.path.split(session_parquet)[1].split("_")[0]
            df = df.with_column(pl.lit(session_num).alias('Session#'))
            df = df.with_column(pl.lit(session_classifier).alias('SessionIdentity'))

            if not 'TD_key1' in df.columns:
                df = df.drop(['Power_Band3', 'Power_Band4'])
                df = df.rename({'TD_key0': 'TD_BG'})
            elif not 'TD_key0' in df.columns:
                df = df.drop(['Power_Band1', 'Power_Band2'])
                df = df.rename({'TD_key1': 'TD_BG', 'Power_Band3': 'Power_Band1', 'Power_Band4': 'Power_Band2'})
                print(f"For {session_parquet} recasted TD_key1 as TD_BG and Power Bands 3 and 4 as 1 and 2, respectively.")

            df_arrow = df.to_arrow()
            if ind == 0:
                con.execute(f"CREATE OR REPLACE TABLE overnight.r{device_identifier} AS SELECT * FROM df_arrow")
            else:
                con.execute(f"INSERT INTO overnight.r{device_identifier} SELECT * FROM df_arrow")
            ind += 1
    con.close()