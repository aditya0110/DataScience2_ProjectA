from __future__ import division
from dash.dependencies import Input, Output
from pandas_datareader import data as web
from loremipsum import get_sentences
from coinmarketcap import Market
from plotly.graph_objs import *
from datetime import datetime, timedelta
import dash
import dash_core_components as dcc
import dash_html_components as html
import sqlite3
import pandas as pd
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
import pickle
import quandl
import datetime as dt
import os
import numpy as np



#Initializing baseline Polinex URL ,timestamps and global variables
base_polo_url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'
start_date = dt.datetime.strptime('2015-01-01', '%Y-%m-%d') # get data from the start of 2015
end_date = dt.datetime.now() # up until today
pediod = 86400 # pull daily data (86,400 seconds per day)
coinmarketcap = Market()
app = dash.Dash()
app.config['suppress_callback_exceptions']=True
altCoinDF = pd.DataFrame()

#Styles
colors = {
    'text':"#003399",
    'left':"#E86361",
    'right':"#3A82B5",
    'background':"#00000"
}

#Dashboard Layout
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='MAAAD Cryptocurrency Dashboard',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
   dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'Bitcoin', 'value': 'BTC'},
	    {'label': 'Bitcoin Cash', 'value': 'BCH'},
            {'label': 'Ethereum', 'value': 'ETH'},
	    {'label': 'Litcoin', 'value': 'LTC'},
            {'label': 'Ripple', 'value': 'XRP'}
        ],
        value='BTC',	
    ),
   dcc.Graph(id='my-table',figure='figure', style={
                
                    #'margin': '15px',
                    #"padding": "20px",
                    "backgroundColor": "rgba (0,0,255)",
		    'width':'auto',
		    'height':'auto',
                }),
   dcc.Graph(id='my-bar-chart',figure='figure', style={
                    'display':'inline-block',
                    #'margin': '15px',
                    #"padding": "20px",
                    "backgroundColor": "rgba (0,0,255)",
		    'width':'auto',
		    'height':'auto',
                }),
   dcc.Graph(id='my-stream',figure='figure', style={
                    'display':'inline-block',
                    #'margin': '15px',
                    #"padding": "20px",
                    "backgroundColor": "rgba (0,0,255)",
		    'width':'auto',
		    'height':'auto',
                }),
  
   # html.Label('Correlation for Year'),
    dcc.RadioItems(id='corr-year', 
        options=[ 
            {'label': '2015', 'value': 2015},
            {'label': '2016', 'value': 2016},
	    {'label': '2017', 'value': 2017},
            {'label': '2018', 'value': 2018}
        ],
        value=2018
    ),
    dcc.Graph(id='corr-mat',figure='figure', style={
                   'display':'inline-block',
                   #'margin': '15px',
                   # "padding": "20px",
                    "backgroundColor": "rgba (0,0,255)",
		    'width':'auto',
		    'height':'auto',
                }),
   

  
   dcc.Graph(id='my-map',figure='figure', style={
                   'display':'inline-block',
                   #'margin': '15px',
                   # "padding": "20px",
                    "backgroundColor": "rgba (0,0,255)",
		    'width':'auto',
		    'height':'auto',
                }),
  dcc.Graph(id='my-ico-table',figure='figure', style={
                    'display':'inline-block',
                    #'margin': '15px',
                   # "padding": "20px",
                    "backgroundColor": "rgba (0,0,255)",
		    'width':'auto',
		    'height':'auto',
                }),
   
 ])
#Fetches coin info selected from the dropdown and plots table
@app.callback(Output('my-table', 'figure'), [Input('my-dropdown', 'value')])
def generate_table(selected_dropdown_value, max_rows=10):
	conn = sqlite3.connect('dashboard.db')
	currValues=[]
	df = pd.read_sql_query("SELECT Coin_Name,USD,AUD,RUB,KRW,INR,CNY,JPY,EUR,CAD FROM Coin where Coin_Name='"+selected_dropdown_value+"'", conn)
	print "dataframe ",df
	df.style.background_gradient(cmap='viridis')
	for i in df:
		print "df['i']", df[i][0]
	return ff.create_table(df)

#Plots bar chart for the sentiment of the selected coin. The sentiment value is calculated by the classifier class and saved to the DB
@app.callback(Output('my-bar-chart', 'figure'), [Input('my-dropdown', 'value')])
def generate_sentiment_bar(value):
	conn = sqlite3.connect('dashboard.db')
	df = pd.read_sql_query("SELECT sentiment FROM Coin where Coin_Name='"+value.strip()+"'", conn)
	positiveValue= df["sentiment"][0]*100
	negativeValue = 100 - positiveValue
	trace1 = go.Bar(
    		y=['Sentiment'],
    		x=[positiveValue],
    		name='Positive',
    		orientation = 'h',
    		marker = dict(
        	color = 'rgba(246, 78, 139, 0.6)',
        	line = dict(
            		color = 'rgba(246, 78, 139, 1.0)',
            		width = 3)
    			)
		)
	trace2 = go.Bar(
    			y=['Sentiment'],
    			x=[negativeValue],
    			name='Negative',
    			orientation = 'h',
    			marker = dict(
       			 color = 'rgba(58, 71, 80, 0.6)',
       			 line = dict(
            		color = 'rgba(58, 71, 80, 1.0)',
            		width = 3)
    			)
		)

	data = [trace1, trace2]
	layout = go.Layout(
    			barmode='stack'
			)

	fig = go.Figure(data=data, layout=layout)
	return fig

#Generates interactive world chloropeth map
@app.callback(Output('my-map', 'figure'), [Input('my-dropdown', 'value')])
def generate_map(selected_dropdown_value):
	conn = sqlite3.connect('dashboard.db')
	currValues=[]
	c = conn.cursor()
	#print "selected_dropdown_value",selected_dropdown_value
	c.execute("SELECT * FROM ICO")
	currValues=c.fetchall()
	df = pd.read_sql_query("SELECT * FROM ICO",conn)
	data = [ dict(
        type = 'choropleth',
        locations = df['country'],
        z = df['Obj_Id'],
        text = df['country'],
        autocolorscale = True,
        reversescale = False,
	showlegend=False,
        marker = dict(
            line = dict (
                color = 'rgb(150,150,150)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = True,
            tickprefix = '',
            title = 'Unique ICO<br>Database ID'),
      	) ]

	layout = dict(
    	title = 'Upcoming ICOs<br>Source:\
            <a href="https://icowatchlist.com/upcoming/">\
            ICO Watchlist</a>',
   	geo = dict(
        showframe = True,
        showcoastlines = True,
	showlegend=False,
        projection = dict(
            type = 'Mercator'
        )
    	)
	)

	fig = dict( data=data, layout=layout )
	return fig


#Renders the table of the ICO's beloning to a certain country when a map region is clicked
@app.callback(Output('my-ico-table', 'figure'), [Input('my-map', 'clickData')])
def map_Selection(clickData):
	points = clickData["points"]
	location_dict = points[0]
	location = location_dict["location"].strip()
	conn = sqlite3.connect('dashboard.db')
	currValues=[]
	c = conn.cursor()
	df = pd.read_sql_query("SELECT ICO_Name,start_time,end_time FROM ICO where country='"+location+"'",conn)
	for i in df:
		print "df['i']", df[i][0]
	
	return ff.create_table(df)



#3D Surface plot not supported on firefox runnning on virtual machine. Retained for future upgrades
def generate_surafacePlot(selected_dropdown_value):
	conn = sqlite3.connect('dashboard.db')
	currValues=[]
	c = conn.cursor()
	c.execute("SELECT * FROM Coin")
	currValues=c.fetchall()
	df = pd.read_sql_query("SELECT ICO_Name,start_time,end_time,sentiment FROM ICO where country='"++"'",conn)
	data = [
    	go.Surface(z=df.as_matrix())
	]
	layout = go.Layout(
    		title='Crypto Prices in Different Currencies',
    		autosize=False,
    		width=500,
    		height=500,
    		margin=dict(
        	l=65,
        	r=50,
        	b=65,
        	t=90
    		)
		)
	fig = go.Figure(data=data, layout=layout)
	return fig

#Download and cache Quandl dataseries
def get_quandl_data(quandl_id):  
    cache_path = '{}.pkl'.format(quandl_id).replace('/','-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df

#Download and cache JSON data, return as a dataframe.
def get_json_data(json_url, cache_path):  
    try:        
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(json_url))
    except (OSError, IOError) as e:
        print('Downloading {}'.format(json_url))
        df = pd.read_json(json_url)
        df.to_pickle(cache_path)
        print('Cached response at {}'.format(json_url, cache_path))
    return df

#Retrieve cryptocurrency data from poloniex
#def get_crypto_data(poloniex_pair):
    #json_url = base_polo_url.format(poloniex_pair, start_date.timestamp(), end_date.timestamp(), pediod)
    #data_df = get_json_data(json_url, poloniex_pair)
    #data_df = data_df.set_index('date')
    #return data_df

#Merge a single column of each dataframe into a new combined dataframe
def merge_dfs_on_column(dataframes, labels, col):
    series_dict = {}
    for index in range(len(dataframes)):
        series_dict[labels[index]] = dataframes[index][col]
        
    return pd.DataFrame(series_dict)

#Download and cache JSON data, return as a dataframe
def get_json_data(json_url, cache_path):
    try:        
        f = open(cache_path, 'rb')
        df = pickle.load(f)   
        print('Loaded {} from cache'.format(json_url))
    except (OSError, IOError) as e:
        print('Downloading {}'.format(json_url))
        df = pd.read_json(json_url)
        df.to_pickle(cache_path)
        print('Cached response at {}'.format(json_url, cache_path))
    return df

#Retrieve cryptocurrency data from poloniex and return dataframe
def get_crypto_data(poloniex_pair):
    json_url = base_polo_url.format(poloniex_pair, totimestamp(start_date), totimestamp(end_date), pediod)
    data_df = get_json_data(json_url, poloniex_pair)
    data_df = data_df.set_index('date')
    return data_df

#Generate a scatter plot of the entire dataframe and return plotly figure
def df_scatter(df, title, seperate_y_axis=False, y_axis_label='', scale='linear', initial_hide=False):
    label_arr = list(df)
    series_arr = list(map(lambda col: df[col], label_arr))
    
    layout = go.Layout(
        title=title,
        legend=dict(orientation="h"),
        xaxis=dict(type='date'),
        yaxis=dict(
            title=y_axis_label,
            showticklabels= not seperate_y_axis,
            type=scale
        )
    )
    
    y_axis_config = dict(
        overlaying='y',
        showticklabels=False,
        type=scale )
    
    visibility = 'visible'
    if initial_hide:
        visibility = 'legendonly'
        
    # Form Trace For Each Series
    trace_arr = []
    for index, series in enumerate(series_arr):
        trace = go.Scatter(
            x=series.index, 
            y=series, 
            name=label_arr[index],
            visible=visibility
        )
        
        # Add seperate axis for the series
        if seperate_y_axis:
            trace['yaxis'] = 'y{}'.format(index + 1)
            layout['yaxis{}'.format(index + 1)] = y_axis_config    
        trace_arr.append(trace)

    fig = go.Figure(data=trace_arr, layout=layout)
    return fig

#Convert date object to timestamps to be passed to Polinex API
def totimestamp(dt, epoch=datetime(1970,1,1)):
    td = dt - epoch
    return (td.microseconds + (td.seconds + td.days * 86400) * 10**6) / 10**6 


#Generate and return scatter plot for all coins to provide consolidated view of prices
#Price may be different if the setup file to update coin values in database was not run for a given day
@app.callback(Output('my-stream', 'figure'), [Input('my-dropdown', 'value')])
def generate_ticker(selected_dropdown_value, max_rows=10):

	btc_usd_price_kraken = get_quandl_data('BCHARTS/KRAKENUSD')
	exchanges = ['COINBASE','BITSTAMP','ITBIT']

	exchange_data = {}

	exchange_data['KRAKEN'] = btc_usd_price_kraken

	for exchange in exchanges:
    		exchange_code = 'BCHARTS/{}USD'.format(exchange)
    		btc_exchange_df = get_quandl_data(exchange_code)
    		exchange_data[exchange] = btc_exchange_df
	
	btc_usd_datasets = merge_dfs_on_column(list(exchange_data.values()), list(exchange_data.keys()), 'Weighted Price')
	btc_usd_datasets.replace(0, np.nan, inplace=True)
	btc_usd_datasets['avg_btc_price_usd'] = btc_usd_datasets.mean(axis=1)
	
	altcoins = ['ETH','LTC','XRP','BCH']

	altcoin_data = {}
	for altcoin in altcoins:
    		coinpair = 'BTC_{}'.format(altcoin)
    		crypto_price_df = get_crypto_data(coinpair)
    		altcoin_data[altcoin] = crypto_price_df
	for altcoin in altcoin_data.keys():
    		altcoin_data[altcoin]['price_usd'] =  altcoin_data[altcoin]['weightedAverage'] * btc_usd_datasets['avg_btc_price_usd']

	combined_df = merge_dfs_on_column(list(altcoin_data.values()), list(altcoin_data.keys()), 'price_usd')
	
	# Add BTC price to the dataframe
	combined_df['BTC'] = btc_usd_datasets['avg_btc_price_usd']
	
	# Check if pearson correlation matrix can be optianed from entire dataframe for specific year
	combined_df_2018 = combined_df[combined_df.index.year == 2018]
	combined_df_2018.pct_change().corr(method='pearson')

	return df_scatter(combined_df, 'Cryptocurrency Prices (USD)', seperate_y_axis=False, y_axis_label='Coin Value (USD)', scale='log')


#Return the pearson correlation heatmap
def correlation_heatmap(df, title, absolute_bounds=True):
    heatmap = go.Heatmap(
        z=df.corr(method='pearson').as_matrix(),
        x=df.columns,
        y=df.columns,
        colorbar=dict(title='Pearson Coefficient'),
    )
    
    layout = go.Layout(title=title)
    
    if absolute_bounds:
        heatmap['zmax'] = 1.0
        heatmap['zmin'] = -1.0
        
    fig = go.Figure(data=[heatmap], layout=layout)
    return fig


#Render the pearson correlation matrix for selected year using radio button
@app.callback(Output('corr-mat', 'figure'),[Input('corr-year', 'value')])
def chart_corr_matrix(value):
	btc_usd_price_kraken = get_quandl_data('BCHARTS/KRAKENUSD')
	exchanges = ['COINBASE','BITSTAMP','ITBIT']
	exchange_data = {}
	exchange_data['KRAKEN'] = btc_usd_price_kraken

	for exchange in exchanges:
    		exchange_code = 'BCHARTS/{}USD'.format(exchange)
    		btc_exchange_df = get_quandl_data(exchange_code)
    		exchange_data[exchange] = btc_exchange_df	
	btc_usd_datasets = merge_dfs_on_column(list(exchange_data.values()), list(exchange_data.keys()), 'Weighted Price')
	btc_usd_datasets.replace(0, np.nan, inplace=True)
	btc_usd_datasets['avg_btc_price_usd'] = btc_usd_datasets.mean(axis=1)
	
	altcoins = ['ETH','LTC','XRP','BCH']

	altcoin_data = {}
	for altcoin in altcoins:
    		coinpair = 'BTC_{}'.format(altcoin)
    		crypto_price_df = get_crypto_data(coinpair)
    		altcoin_data[altcoin] = crypto_price_df
	for altcoin in altcoin_data.keys():
    		altcoin_data[altcoin]['price_usd'] =  altcoin_data[altcoin]['weightedAverage'] * btc_usd_datasets['avg_btc_price_usd']

	combined_df = merge_dfs_on_column(list(altcoin_data.values()), list(altcoin_data.keys()), 'price_usd')
	
	# Add BTC price to the dataframe
	combined_df['BTC'] = btc_usd_datasets['avg_btc_price_usd']
	if(value==2015):
		combined_df_value = combined_df[combined_df.index.year == 2015]
	elif(value==2016):
		combined_df_value = combined_df[combined_df.index.year == 2016]
	elif(value==2017):
		combined_df_value = combined_df[combined_df.index.year == 2017]
	elif(value==2018):
		combined_df_value = combined_df[combined_df.index.year == 2018]

	return correlation_heatmap(combined_df_value.pct_change(), "Cryptocurrency Correlations in "+str(value))

if __name__ == '__main__':
    app.run_server()
