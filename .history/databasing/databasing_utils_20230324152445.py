import duckdb

def create_sorted_view(table_name, sort_column, db_path):
    con = duckdb.connect(dabase=db_path)
    con.sql(f"CREATE VIEW {table_name}_sorted AS SELECT * FROM {table_name} ORDER BY {sort_column}")
    con.close()


def sort_table_by_column(table_name, sort_column):
    con = duckdb.connect(database=db_path)
    con.sql(f"CREATE TABLE {table_name}_sorted AS SELECT * FROM {table_name} ORDER BY {sort_column}")
    con.sql(f"DROP TABLE {table_name}")
    con.sql(f"ALTER TABLE {table_name}_sorted RENAME TO {table_name}")
    con.close()