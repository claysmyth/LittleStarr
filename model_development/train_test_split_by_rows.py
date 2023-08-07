import duckdb
import polars as pl
import numpy as np

"""
This script is used to create train and test views of a table in the database. 
Adds the suffix _train or _test to the table name and inserts the train and test splits back into the database.
"""


def train_test_split_pl(df, train_test_split=0.9, random_state=1):
    """
    Split a dataframe into train and test sets.
    """
    np.random.seed(random_state)
    test_inds = np.random.choice(df.height, size=int(df.height*(1-train_test_split)), replace=False)
    train_inds = np.setdiff1d(np.arange(df.height), test_inds)

    df_train = df[train_inds]
    df_test = df[test_inds]

    return (train_inds, df_train), (test_inds, df_test)


def create_train_test_split_views_of_table(in_table_name, con, data_columns, label_column, out_table_prefix, train_test_split=0.9, random_state=1):
    """
    Create views for training and testing data for a device.
    """
    df = con.sql(f'select {", ".join(data_columns)}, {label_column} from {in_table_name};').pl()
    
    (train_inds, df_train), (test_inds, df_test) = train_test_split_pl(df, train_test_split=train_test_split, random_state=random_state)

    con.sql(f'create or replace table {out_table_prefix}_train as select * from df_train;')
    con.sql(f'create or replace table {out_table_prefix}_test as select * from df_test;')

    return df_train.describe(), df_test.describe()

if __name__ == '__main__':
    """
    Cycles through variables in SCHEMAS and creates train and test views for corresponding table (TABLE_NAME)
    """
    SCHEMAS = ['r02L', 'r02R', 'r03L', 'r03R', 'r07L', 'r07R', 'r09L', 'r09R', 'r16L', 'r16R']
    TABLE_NAME = 'overnight_simulated_FFTs'
    columns = ['DerivedTime', 'SessionIdentity', "columns('^TD|^Power|^fft')", 'SleepStage']
    con = duckdb.connect('/media/shortterm_ssd/Clay/databases/duckdb/rcs-db.duckdb')
    for schema in SCHEMAS:
        print('ON: ', schema)
        train_stats, test_stats = create_train_test_split_views_of_table(f"{schema}.{TABLE_NAME}", con, columns, 'SleepStage', f"{schema}.{TABLE_NAME}", train_test_split=0.9, random_state=1)
        print(train_stats, test_stats)



