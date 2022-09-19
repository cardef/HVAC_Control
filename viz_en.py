from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import json
import pandas as pd
from pathlib import Path
MAIN_DIR = Path(__file__).parent
config = json.load("config.json")
col_out = config['PARAMETER']['DATA']['col_temp_in'] + config['PARAMETER']['DATA']['col_temp_ext'] 
app = Dash(__name__)


app.layout = html.Div(children = [
    html.Div([
        html.H4('Temperature forecasting'),
        dcc.Graph(id="time-series-chart"),
        html.P("Select zone:"),
        dcc.Dropdown(
            id="ticker",
            options=col_out,
            value="AMZN",
            clearable=False,
        ),
    ]),
])


@app.callback(
    Output("time-series-chart", "figure"), 
    Input("ticker", "value"))
def display_time_series(ticker):
    df = pd.read_csv(MAIN_DIR/'data'/'results'/'temperature'/'results.csv')
    df_= df[ticker]
    # replace with your own data source
    fig = px.line(df, x=df.index, y=ticker)
    return fig


app.run_server(debug=True)