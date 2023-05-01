import altair as alt

def create_overlaid_line_plot(df, index_column, columns, width=500, height=300, color_scheme='category10', interactive=True):
    """
    Creates an Altair line plot with the specified columns overlaid on top of each other.
    
    Parameters:
        df (pandas.DataFrame): The data to be plotted.
        columns (list): A list of column names to plot.
        width (int): The width of the plot in pixels. Default is 500.
        height (int): The height of the plot in pixels. Default is 300.
        color_scheme (str): The name of the color scheme to use for the lines. Default is 'category10'.
        interactive (bool): Whether to include interactive elements in the chart, such as zooming and panning.
        
    Returns:
        An Altair chart object.
    """
    # Melt the data so that each row represents one point on the line chart
    melted_df = df[columns].melt(var_name='column', value_name='value', ignore_index=False)
    
    # Create the chart
    chart = alt.Chart(melted_df).mark_line().encode(
        x=alt.X('index', title=None),
        y=alt.Y('value', title=None),
        color=alt.Color('column', scale=alt.Scale(scheme=color_scheme)),
        tooltip=[alt.Tooltip('index:T'), alt.Tooltip('value:Q'), alt.Tooltip('column:N')]
    ).properties(
        width=width,
        height=height
    )
    
    # Add interactivity if desired
    if interactive:
        chart = chart.interactive()
    
    return chart, melted_df
