from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import json
import pandas as pd
from pathlib import Path
MAIN_DIR = Path(__file__).parent.parent

with open('config.json') as f:
    config = json.load(f)

with open("cleaned_csv.json") as f:
    cleaned_csv = json.load(f)
len_forecast = config['len_forecast']
time_window = config['time_window']
col_out_temp = cleaned_csv['temp']['col_out']

app = Dash(__name__)

en_res = pd.read_csv(MAIN_DIR/'results'/'outputs'/'energy_prediction.csv', header = [0,1], index_col=[0])
en_res = en_res['hvac'].reset_index().melt(id_vars = ['date'], value_vars = ['true', 'pred'])
en_plot = px.line(en_res, x="date", y="value", color="Type")

en_opt = pd.read_csv(MAIN_DIR/'results'/'opt'/'comparison'/'energy_res_df.csv', header = [0,1], index_col=[0])
en_opt = en_opt['hvac'].reset_index().melt(id_vars = ['date'], value_vars = ['true', 'opt'])
en_opt_plot = px.line(en_opt, x="date", y="value", color="Type")
app.layout = html.Div(children = [
    html.Div([
        html.H4('Energy forecasting'),
        dcc.Graph(id="en-time-series-chart", figure = en_plot),
    ]),
    html.Div([
        html.H4('Temperature forecasting'),
        dcc.Graph(id="temp-time-series-chart"),
        html.P("Select zone:"),
        dcc.Dropdown(
            id="ticker",
            options=col_out_temp,
            value=col_out_temp[0],
            clearable=False,
        ),
    ]),
    
    html.Div([
        html.H4('Energy Optimization'),
        dcc.Graph(id="en-opt-time-series-chart", figure = en_opt_plot),
    ]),
])


@app.callback(
    Output("temp-time-series-chart", "figure"), 
    Input("ticker", "value"))
def display_time_series(ticker):
    temp_res = pd.read_csv(MAIN_DIR/'results'/'outputs'/'temp_prediction.csv', header = [0,1], index_col=[0])
    temp_res_melted= temp_res[ticker].reset_index().melt(id_vars = ['date'], value_vars = ['true', 'pred'])
    # replace with your own data source
    fig = px.line(temp_res_melted, x='date', y='value', color='Type')
    return fig


app.run_server(debug=True, port = 8057)