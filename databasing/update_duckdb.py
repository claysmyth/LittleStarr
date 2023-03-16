import sys
import duckdb
import os
import shutil
import datetime
import subprocess


if __name__ == '__main__':
    """
     This script does the following:
             1) Backup a current DuckDB database file to directory and copies the original .duckdb file
             2) Updates DuckDB
             3) Imports the backup duckDB
     This is necessary because, until a stable release of DuckDB is available, a new database .db file must be made 
     with each version of DuckDB.
     """

    db_path = sys.argv[1]
    backup_dir = sys.argv[2]

    # Connect and export database to backup directory
    con = duckdb.connect(database=db_path)
    print(f"Current Version of DuckDB: {duckdb.__version__}")
    # Save DuckDB version of this database
    with open(os.path.join(backup_dir, 'readme.txt'), 'w') as f:
        f.write(f'DuckDB Version: {duckdb.__version__}')
        f.write(f"\nCreate on: {datetime.datetime.now()}")
    backup_db_path = os.path.join(backup_dir, 'exported_db')
    con.execute(f"EXPORT DATABASE '{backup_db_path}' (FORMAT PARQUET, COMPRESSION ZSTD);")
    con.close()

    # Move .duckdb file to back_up directory, so that database can be recreated in db_path with new version of DuckDB
    backup_db_file_dir = os.path.join(backup_dir, 'original_db_file')
    os.makedirs(backup_db_file_dir)
    backup_db_file = os.path.join(backup_db_file_dir, os.path.basename(db_path))
    shutil.move(db_path, backup_db_file)
    if os.path.exists(db_path+'.wal'):
        shutil.move(db_path+'.wal', backup_db_file+'.wal')

    # Update DuckDB.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "duckdb"])

    # reimport duckdb
    import duckdb

    # Open DuckDB, print version, and import previous database
    # Note: unclear if the below actually works...
    print(f"Updated to DuckDB Version: {duckdb.__version__}")
    con = duckdb.connect(database=db_path)
    con.execute(f"IMPORT DATABASE '{backup_db_path}'")
    con.close()





