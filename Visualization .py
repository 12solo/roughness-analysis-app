import plotly.express as px
import plotly.graph_objects as go

def plot_box_distribution(df, y_col, x_col, color_col):
    fig = px.box(df, x=x_col, y=y_col, color=color_col,
                 points="all", title=f"Distribution of {y_col} by {x_col}",
                 template="plotly_white")
    return fig

def plot_ageing_trend(summary_df, y_col, x_col, color_col):
    fig = px.line(summary_df, x=x_col, y=y_col, color=color_col,
                  error_y="ci_95", markers=True,
                  title=f"Evolution of {y_col} over Ageing Time",
                  template="plotly_white")
    return fig