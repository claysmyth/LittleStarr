import duckdb

def create_sorted_view(table_name, sort_column):
    con = duckdb.connect()
    con.execute(f"CREATE VIEW sorted_{table_name} AS SELECT * FROM {table_name} ORDER BY {sort_column}")
    con.close()