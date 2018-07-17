import base64
import os
import time
import math
import re
import json
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


app = dash.Dash()

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

# ---------------
# function list
# ---------------
# file upload function
def parse_contents(contents, filename):
	content_type, content_string = contents.split(',')
	decoded = base64.b64decode(content_string)

	try:
		df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

	except Exception as e:
		print(e)
		return None

	return df


def build_KDE_series(ages,bw_min,bw_max,smooth_param,max_age_roundup):
	ages = np.array(ages[~np.isnan(ages)]) #remove nans and convert to array
	x_range = np.arange(0,max_age_roundup,1) #sampled every 1 spacing

	smoothing_KDE = [smooth_param for i in ages]
	sum_pdf = x_range-x_range
	for i in range(len(ages)):
		sum_pdf = sum_pdf + norm.pdf(x=x_range,loc=ages[i],scale=smoothing_KDE[i])

	bw_scalars = sum_pdf[[np.where(x_range==int(i)) for i in ages]]
	KDE_bandwidths = [(i-max(bw_scalars))*((bw_max-bw_min)/(min(bw_scalars)-max(bw_scalars)))+bw_min for i in bw_scalars]

	sum_pdf = x_range-x_range
	for i in range(len(ages)):
		sum_pdf = sum_pdf + norm.pdf(x=x_range,loc=ages[i],scale=KDE_bandwidths[i])
		norm_sum_pdf = (sum_pdf) / len(ages)
	return norm_sum_pdf[0]


def calc_stat(x,y,stat):
	x = np.array(x)
	y = np.array(y)

	if stat=='likeness':
		M = abs(x-y)
		s = sum(M)/2
		return s

	if stat=='similarity':
		if np.array_equal(x,y):
			s = 0
		else:
			S = np.sqrt(x*y)
			s = 1-sum(S)
		return s

	if stat=='l2-norm':
		if np.array_equal(x,y):
			s = 0
		else:
			M = np.sqrt(np.square(x-y))
			s = sum(M)
		return s

	if stat=='r2':
		xmean = np.mean(x)
		ymean = np.mean(y)
		xcov = np.zeros(len(x))
		ycov = np.zeros(len(y))

		for i in range(len(x)):
			xcov[i] = x[i] - xmean
		for i in range(len(x)):
			ycov[i] = y[i] - ymean
		numerator = sum(xcov*ycov)

		sumxcov2 = sum(xcov*xcov)
		sumycov2 = sum(ycov*ycov)
		mult2 = sumxcov2*sumycov2
		denominator = np.sqrt(mult2)

		r = numerator/denominator
		r2 = r*r
		s = 1-r2
		return s


# ---------------
# app layout
# ---------------
app.layout = html.Div([

		# site heading div
		html.Div([
			html.H1('Geoclustering Detrital Zircon Samples',
				style={
					'position': 'relative',
					'display': 'inline',
					'font-size': '7rem',
					'color': '#4D637F',
				}),
		], className='row twelve columnns',style={'textAlign':'center','margin':'20px 0px'}),

		# Input data div
		html.Div([
			html.Details([
				html.Summary(children=['Input data'],
							style={'font-size': '2.5rem',
									'color': '#4D637F',
							},
				),
				dcc.Markdown("""
Detrital geochronology datasets can be uploaded from a local drive via the upload function. Data should be uploaded as a .csv file, pre-formatted as follows:

* One column per sample
* First row: Sample name
* Second row: Latitude
* Third row: Longitudes
* Next rows: Ages

An example of this formatted detrital data can be found [here](www.github.com) for reference. Otherwise a test dataset from [Xu et al. (2017)] can be selected below.

[Xu et al. (2017)]: #Xu-2017
				"""),
				html.Button('Load Xu et al. 2017 data', id='data-load1', style={'align':'center','margin': '10px auto'}),
			]),

			dcc.Upload(
				id='upload-data',
				children=html.Div([
					'Drag and drop or click to select a file to upload.'
				]),
				style={
					'width': '100%',
					'height': '60px',
					'lineHeight': '60px',
					'borderWidth': '1px',
					'borderStyle': 'dashed',
					'borderRadius': '5px',
					'textAlign': 'center',
					'margin': '0px auto',
				},
				multiple=False
			), #
			html.Div(id='raw-df-json', style={'display': 'none'}),
			html.Div(id='diss-df-json', style={'display': 'none'}),
		], className='row twelve columnns',style={'margin':'20px 0px'}),

		# Build KDEs dropdown divs
		html.Div([
			html.Details([
				html.Summary(children=['Building KDEs'],
							style={'font-size': '2.5rem',
									'color': '#4D637F',
							},
				),
				dcc.Markdown("""
This step converts the input age data into a continuous analytical function that best represents the underlying age-density population for each sample.

The application builds locally-adaptive KDEs that have a varying bandwidth based on local density of region. The parameters are:

* Minimum and maximum bandwidths: range of bandwidths that kernals will vary between. Set these values equal for a simple KDE plot.
* Smoothing parameter:  represents the size of the local density region over which bandwidths are assigned.

The KDE plot interactively responds for testing the bandwidth parameters. Once satisfied with its appearance, click 'Submit' to save the KDE parameters to be used in the following steps.
							"""),
				html.Details([
					html.Summary(children=['Relevant background theory'],
								style={'color': '#4D637F',
								}),
					dcc.Markdown("""
Commonly the aim with detrital zircon U-Pb dating studies (and many other types of detrital geochronological data) is to understand the distribution of ages that make up a single sample. To better visualise and statistically analysis these age population distributions, analytical functions are fitted to age data.

The main methods to achieve this commonly used within the geochronlogy community are:

(1) Kernal density estimators (**KDEs**) - By fitting stacked kernals (normal distributions in most instances) the underlying density of a continuous function can be approximated. For detrital datasets the bandwidth selected accounts for sampling bias to an extent by providing a smoothed representation of the underlying age-distribution of a sample.

(2) Probablity dentisy plots (**PDPs**) - Detrial dating methods have an inherient uncertainity due to measurement imprecision in recorded age data. PDPs first applied by [Hurford et al. (1984)] to fission track data look to account for this uncertatinity by varying kernal bandwidths according to the analytical uncertainty associated to each recorded age.

Only PDPs explicitly account for the analytical imprecision in the age data. However they don't account for sampling error which can lead to results appearing highly unrepresentative when dealing with low or very high number of samples ([Vermeesch 2012]).
Instead Monte Carlo perturbation tests can be simulated to account for effects of measurement errors. Another recommendation is to set the kernal bandwidth similar to the maximum measurement error on ages of the samples being compared ([Sircombe and Hazleton, 2004]). This avoids bias from different measuring techniques that may be more error prone.

**LA-KDEs** (locally-adaptive), a variation on the KDE is provided is this application that have narrower kernal bandwidths in regions of high density ([Vermeesch 2012]). This is acknowledging that more detail and age peaks can be discerned in regions of higher sample densities whilst lower density regions remain suitably smoothed.

Many researchers has looked for methods to statistically determine 'optimum' KDE bandwidths (e.g. [Silverman 1986], [Botev et al. 2010]) however these typically lead to over-smoothed distributions.
Often for detrtial datasets it is therefore better to visually assess the most suitable bandwidth, allowing the interpretor to essentially bring in priory information on the age density distributions expected.

*Note: Detrital datasets are typically under-sampled and complex, making it hard to evaluate results with any measures of statistical certainty.*

[Hurford et al. (1984)]: #Hurford-1984
[Botev et al. 2010]: #Botev-2010
[Silverman 1986]: #Silverman-1986
[Vermeesch 2012]: #Vermeesch-2012
[Sircombe and Hazleton, 2004]: #Sircombe-hazleton-2004
					"""),
				], style={'margin':'5px 0px'}),
			]),
		], className='row twelve columnns', style={'margin':'20px 0px 10px 0px'}),

		dcc.Dropdown(id='selected-samp'),
		dcc.Graph(id='single-kde-plot'),


		# Build KDEs options divs
		html.Div([
			html.Div([
				html.Div('Min bandwidth (Ma):',style={'textAlign': 'center'}),
			], className='two columns'),
			html.Div([
				dcc.Input(id='bw-min', type='number', value='10', style={'width': 60}),
			], className='one column'),

			html.Div([
				html.Div('Max bandwidth (Ma):',style={'textAlign': 'center'}),
			], className='two columns'),
			html.Div([
				dcc.Input(id='bw-max', type='number', value='20', style={'width': 60}),
			], className='one column'),

			html.Div([
				html.Div('Smoothing parameter (Ma):',style={'textAlign': 'center'}),
			], className='two columns'),
			html.Div([
				dcc.Input(id='smoother', type='number', value='50', style={'width': 80}),
			], className='one column'),

			html.Div([
				html.Button(id='submit-button', n_clicks=0, children='Submit', style={'width': 150,'align':'center'}),
			], className='three columns'),
		], className = 'row', style={'position': 'relative','width':'80%','left':'10%','margin':'20px 0px 0px 0px'}),

		html.Div(
			dcc.Markdown(id='submit-button-text'),className='row twelve columns',
						style={'position': 'relative','textAlign':'center','margin':'40px 0px 20px 0px'}
		), #
		html.Div(id='kdes-df-json', style={'display': 'none'}),


		# Build pairwise dissimilarity matrix
		html.Div([
			html.Details([
				html.Summary(children=['Dissimilarity metrics'],
						style={'font-size': '2.5rem',
								'color': '#4D637F',
						},
			),
			dcc.Markdown("""
To evaluate the similarity in ages between any two samples in a (geologically) meaningful manner, a set of KDE metrics are provided (see relevant background theory for details). Each sample is compared to every other sample in the dataset using the chosen metric to build a matrix of 'pairwise dissimilarities'.
*Note: Sample KDEs are normalised prior to building any dissimilarity matrix to account for varying numbers of age analyses between samples.*
						"""),
			html.Details([
				html.Summary(children=['Relevant background theory'],
							style={'color': '#4D637F',
							}),
				dcc.Markdown("""
Determining a measure for similarity between geochronological samples (that are in turn made up of typically hundreds of age-analyses) is inherently difficult.
Statistical evaluation via hypothesis testing (e.g. calculating 'p-values' or Student's t-tests) would seem the obvious route to go down. However again due to population sample sizes for geochronological datasets often being to small or varied sample to sample it can be difficult to evaluate statistical significance in the data.

An excellent overview of this topic is provided by [Vermeesch (2018)] who evaluates the usefulness of a range of parametric, non-parametric and 'ad-hoc' measures of dissimilarity in detrital geochronology.

In this application KDE-statistics are used to determine dissimilarity, measuring similarity based on the appearances of the calculated KDEs. Whilst these methods may have less statistical significance, it provides a better method for evaluating provenance significance, as we assume that KDEs are a more representative function of the true age population for a given sample than just the individual age-analyses.

The KDE-statistics avaliable in this application are:

(1) **Cross-plot correlation coefficient (R^2)** -  the coefficient indicates similarity from
a cross-plot of two kernel-distributions against each other ([Saylor et al. 2012]).

(2) **Likeness** - a measure of age-distribution similarities, defined as ‘sameness’ percentage between two combined kernel-distributions ([Satkoski et al. 2013]). Similar to L2-norm suggested by [Sircombe and Hazelton (2004)].

(3) **Similarity** - provides a measure of resemblance between two distributions ([Gehrels 2000]).
*Note: all methods do require discretisation of a continuous function.*

[Saylor and Sundell (2016)] review these statistics amongst others concluding that
cross-plot correlations are the most sensitive method to evaluating the presence or
absence of age peaks and their respective changes in magnitude. Empirically speaking, different metric appear to work well in different scenarios. Within the application it is easy to toggle between options using the KDEs plot to qualitatively evaluate the clusters too.

To undergo multivariate analysis (> 2 samples), the degree of dissimilarity between any two samples must be calculated between all pairwise sample combinations, building a dissimilarity matrix that describes the relationship between each and every sample in the set. [Vermeesch (2013)] details the theory of this approach and outlines requirements to be met for building a dissimilarity matrix.
The next section looks at visualisation and interpretation of the dissimilarity matrix.

[Saylor and Sundell (2016)]: #Saylor-sundell-2016
[Satkoski et al. 2013]: #Satkoski-et-al-2013
[Saylor et al. 2012]: #Saylor-et-al-2012
[Sircombe and Hazelton (2004)]: #Sircombe-hazleton-2004
[Vermeesch (2013)]: #Vermeesch-2013
[Gehrels 2000]: #Gehrels-2000
[Vermeesch (2018)]: #Vermeesch-2018
				"""),
				], style={'margin':'5px 0px'}),
			]),
		], className='row twelve columnns',style={'margin':'20px 0px 10px 0px'}),


		# Agglomerative clustering div
		html.Div([
			html.Div([
				html.Details([
					html.Summary(children=['Agglomerative clustering'],
								style={'font-size': '2.5rem',
										'color': '#4D637F',
								},
					),
					dcc.Markdown("""
Clustering of detrital samples is determined based on the similarity of the samples ages distributions. Identifying the natural groupings of detrital samples is customary in provenance studies and plotting the clusters (colours) on a map is a powerful tool for analysing spatial patterns in the data.

The number of clusters (k) in the dataset and the method by which samples are grouped together (linkages) can be updated below.
*Note: The linkage method selected can have a large impact on clusters established (see relevant background theory).* **By dragging over a group of samples on the map, KDEs are visualised to evaluate the data and chosen parameters.**
					"""),
			html.Details([
				html.Summary(children=['Relevant background theory'],
							style={'color': '#4D637F',
							}),
				dcc.Markdown("""
Agglomerative clustering is a well established hierarchical clustering method found across all scientific fields. Agglomerative is 'bottom-up' approach whereby individual data (samples here) are successively merged to larger groups based on distances between the data. Any pre-computed 'distance' metric can be used here in the form of a dissimilarity matrix, where pairwise distances between each and every data-point has been established, hence it is a naturally fitting method by which to evaluate age distribution similarities.

An level (or number)of grouping can be reviewed by this method. Beyond cluster sizes of k=1, how the shortest distance between cluster groups is
calculated is known as the defining ‘linkage’. Descriptions of the linkages are provided [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html).
'Complete' linkage (maximum distances between cluster groups) is likely most suitable for detrital sample clustering as ensures any sample within a cluster is more 'similar' to all samples in that cluster than any other samples.

[Sircombe and Hazelton (2004)] first demonstrated the use for agglomerative clustering of detrital datasets in provenance analysis. The L2-norm between KDEs was used as a dissimilarity metric, similar to 'likeness' provided here.

Other analysis techniques for comparing detrital age distributions include multidimensional scaling (MDS) which put forward by [Vermeesch (2013)] has been widely adopted in detrital zircon provenance studies. MDS reduces the dimensionality of dissimilarity matrix allowing natural groupings in the data to stand out. Agglomerative clustering effectively quantifies these distances and groupings without the need for dimension reduction (which can provide unstable results with non-metric MDS).

Quantified distances between grouped samples are illustrated with dendrograms that graphically represent the levels of similarity within the dataset (*feature coming soon...*)

[Sircombe and Hazelton (2004)]: #Sircombe-hazleton-2004
[Vermeesch (2013)]: #Vermeesch-2013
				"""),
				], style={'margin':'5px 0px'}),
				]),
			], className='twelve columns',style={'position':'relative'}),
		], className='row',style={'margin':'20px 0px'}),


		# map options div
		html.Div([
			html.Div([
				html.Div('Dissimilarity statistic:',style={'position':'relative','textAlign': 'right'}),
			], className='two columns'),
			html.Div([
				dcc.Dropdown(id='diss-matrix-stat-dropdown',
					options=[{'label': i, 'value': i} for i in ['R2','Likeness','Similarity']],
					value='likeness'
				), #
			], className='two columns',style={'position':'relative','width':'300'}),

			html.Div([
				html.Div('Linkage options:',style={'position':'relative','textAlign': 'right'}),
			], className='two columns'),
			html.Div([
				dcc.Dropdown(id='linkage-dropdown',
					options=[{'label': i, 'value': i} for i in ['Single','Complete','Average','Median','Centroid']],
					value='complete'
				),
			], className='three columns',style={'position':'relative','width':'300'}),
		], className = 'row'),


		# main map div row
		html.Div([
			html.Div([
				html.Div([
					dcc.Graph(id='main-map',
						figure=go.Figure(
							data=[go.Scattermapbox(lat=[], lon=[])],
							layout=go.Layout(
								height=400,
								mapbox=dict(
									accesstoken=('pk.eyJ1IjoiYWxpc2hvYmVpcmkiLCJhIjoiY2ozYnM3YTUxMDAxeDMzcGNjbmZyMmplZiJ9.ZjmQ0C2MNs1AzEBC_Syadg'),
									zoom=1,
								),
								hovermode='closest',
								margin=dict(r=0, t=10, l=50, b=10)
							)
						)
					),
				], className='eleven columns',style={'position':'relative','margin-left':'0'}),
				html.Div([
					dcc.Slider(
						id='k-clusters-slider',
						min=2,
						max=8,
						step=None,
						value=4,
						marks={str(year): str(year) for year in range(2,9)},
						vertical=True,
					),
					html.P("No. of clusters (k)", style={'width':'200','text-align':'right', 'transform': 'rotate(-90deg)','float':'left','transform-origin':'top left 0','margin':'-50px 0px 0px 30px'})
				], className='one column',style={'position':'relative','margin-top': '25', 'margin-bottom': '25', 'height': '350'}),
			], className='row'),

			dcc.Graph(id='qc-kde-plot',
					figure=go.Figure(
						data=[],
						layout=go.Layout(
							height=350,
							margin={'l': 60, 'b': 40, 't': 20, 'r': 20},
							hovermode='closest',
							xaxis=dict(title='Age (Ma)'),
							yaxis=dict(title='Density')),
					)
			)
		], className='row twelve columns',style={'position':'relative','margin':'20px 0px'}),


		html.Div([
				html.Details([
					html.Summary(children=['References'],
								style={'font-size': '2.5rem',
										'color': '#4D637F',
										'margin':'10px 0px'
								},
					),
					# reference list #
					html.P(id="Botev-2010"),
					dcc.Markdown('**Botev, Z.I., Grotowski, J.F., Kroese, D.P., 2010.** Kernel density estimation via diffusion. *Annals of Statistics* 38, 2916–2957.'),
					html.P(id="Gehrels-2000"),
					dcc.Markdown('**Gehrels, G.E., 2000.** Introduction to detrital zircon studies of Paleozoic and Triassic \
								strata in western Nevada and northern California. *Special Paper of the Geological \
								Society of America, 347*, pp.1-17.'),
					html.P(id="Hurford-1984"),
					dcc.Markdown('**Hurford, A., Fitch, F., Clarke, A., 1984.** Resolution of the age structure of the detrital zircon \
								populations of two Lower Cretaceous sandstones from the Weald of England \
								by fission track dating. *Geological Magazine* 121, 269–396.'),
					html.P(id="Satkoski-et-al-2013"),
					dcc.Markdown('**Satkoski, A.M., Wilkinson, B.H., Hietpas, J. and Samson, S.D., 2013.** Likeness among \
								detrital zircon populations—An approach to the comparison of age frequency data \
								in time and space. *Geological Society of America Bulletin, 125*(11-12), pp.1783-1799.'),
					html.P(id="Saylor-et-al-2012"),
					dcc.Markdown('**Saylor, J.E., Stockli, D.F., Horton, B.K., Nie, J. and Mora, A., 2012.** Discriminating \
								rapid exhumation from syndepositional volcanism using detrital zircon double \
								dating: Implications for the tectonic history of the Eastern Cordillera, Colombia. \
								*Geological Society of America Bulletin, 124*(5-6), pp.762-779.'),
					html.P(id="Saylor-sundell-2016"),
					dcc.Markdown('**Saylor, J.E. and Sundell, K.E., 2016.** Quantifying comparison of large detrital \
								geochronology data sets. *Geosphere*, pp.GES01237-1.'),
					html.P(id="Silverman-1986"),
					dcc.Markdown('**Silverman, B., 1986.** Density Estimation for Statistics and Data Analysis. Chapman and Hall, London.'),
					html.P(id="Sircombe-hazleton-2004"),
					dcc.Markdown('**Sircombe, K.N. and Hazelton, M.L., 2004.** \
									 Comparison of detrital zircon age distributions by kernel functional estimation. \
									*Sedimentary Geology*, 171(1-4), pp.91-111.'),
					html.P(id="Vermeesch-2012"),
					dcc.Markdown('**Vermeesch, P., 2012.** On the visualisation of detrital age distributions. \
									*Chemical Geology*, 312, pp.190-194.'),
					html.P(id="Vermeesch-2013"),
					dcc.Markdown('**Vermeesch, P., 2013.** Multi-sample comparison of detrital age distributions. \
								*Chemical Geology*, 341, pp.140-146.'),
					html.P(id="Vermeesch-2018"),
					dcc.Markdown('**Vermeesch, P., 2018.** Dissimilarity measures in detrital geochronology \
								*Earth-science Reviews, 178*, pp. 310-321'),
					html.P(id="Xu-2017"),
					dcc.Markdown('**Xu, J., Snedden, J.W., Stockli, D.F., Fulthorpe, C.S. and Galloway, W.E., 2017.** Early \
								Miocene continental-scale sediment supply to the Gulf of Mexico Basin based on \
								detrital zircon analysis. *Geological Society of America Bulletin*, 129(1-2), pp.3-22.'),
				], open = True),
			], className='row twelve columns',style={'position':'relative'})
	],
	# style for the whole site
	style={'align':'center', 'max-width':'1400px', 'margin': '0px auto'}
)



# ---------------
# callbacks
# ---------------
@app.callback(
	Output('raw-df-json', 'children'),
			[Input('upload-data', 'contents'),
			 Input('upload-data', 'filename'),
			 Input('data-load1', 'n_clicks')])
def update_any(contents, filename, data_load_clicks):
	if contents is not None:
		df = parse_contents(contents,filename)
		if df is not None:
			return df.to_json()
	elif data_load_clicks > 0:
		try:
			df = pd.read_csv('data/Xu_et_al_2016_dataset.xlsx')
			return df.to_json()
		except:
			print ('LOCAL UPLOAD FUNCTION DID NOT WORK')
			return [{}]


@app.callback(
	Output('selected-samp', 'options'),
	[Input('raw-df-json', 'children')])
def set_sample_options(raw_df_json):
	df = pd.read_json(raw_df_json)
	df.sort_index(inplace=True)
	sample_names = list(df)
	return [{'label': i, 'value': i} for i in sample_names]


@app.callback(
	Output('selected-samp', 'value'),
	[Input('selected-samp', 'options')])
def set_sample_value(available_options):
	return available_options[0]['value']


@app.callback(Output('single-kde-plot', 'figure'),
			[Input('raw-df-json', 'children'),
			Input('selected-samp','value'),
			Input('bw-max', 'value'),
			Input('bw-min', 'value'),
			Input('smoother', 'value')])
def update_figure(raw_df_json,sample_name,bw_max,bw_min,smooth_val):
	df = pd.read_json(raw_df_json)
	df.sort_index(inplace=True)
	sample_names = list(df)

	max_age = df[sample_name][2:].max()
	max_age_roundup = int(math.ceil(max_age/500.0))*500 # to nearest 500
	x_range = np.arange(0,max_age_roundup,1)

	ages = pd.Series(build_KDE_series(df[sample_name][2:], float(bw_min),float(bw_max),float(smooth_val),max_age_roundup))
	raw_age_list = df[sample_name][2:].dropna().tolist()
	legend_str = 'Ages (n='+str(len(raw_age_list))+')'

	kde_vis = go.Scatter(
				name='KDE',
				x=x_range,
				y=ages,
				mode='lines',
				line = dict(width=3,
					color='rgb(90, 90, 90)'
				)
	)

	ages_vis = go.Scatter(
		name = legend_str,
		x = raw_age_list,
		y = np.zeros(len(raw_age_list)),
		mode = 'markers',
		marker = dict(
			size = 6,
			color = 'rgba(77, 184, 255, 0.2)',
			line = dict(
				width = 1.5,
				color = 'rgb(90, 90, 90)'
			)
		)
	)

	return {'data': [kde_vis, ages_vis],
			'layout': go.Layout(
			height=400,
			margin={'l': 80, 'b': 40, 't': 20, 'r': 20},
			hovermode='closest',
			legend=dict(
				x=0.9,
				y=1),
			xaxis=dict(title='Age (Ma)',
						range=[-20,max_age_roundup]),
			yaxis=dict(title='Density')),
	}


@app.callback(Output('kdes-df-json', 'children'),
			  [Input('raw-df-json', 'children'),
			  Input('submit-button', 'n_clicks')],
			  [State('bw-min', 'value'),
			  State('bw-max', 'value'),
			   State('smoother', 'value')])
def create_new_df(raw_df_json, n_clicks, bw_min, bw_max, smooth_val):
	if int(n_clicks) > 0:
		df = pd.read_json(raw_df_json)#
		df.sort_index(inplace=True)
		max_age = max(df[2:].max())
		max_age_roundup = int(math.ceil(max_age/500.0))*500 # to nearest 500
		KDE_df = pd.DataFrame(data=None, columns=df.columns)

		for col in df:
			KDE_series = pd.Series(build_KDE_series(df[col][2:], float(bw_min),float(bw_max),float(smooth_val),max_age_roundup))
			KDE_df[col] = KDE_series

		return KDE_df.to_json(orient='columns')


@app.callback(Output('submit-button-text', 'children'),
			  [Input('submit-button', 'n_clicks')],
			  [State('bw-min', 'value'),
			  State('bw-max', 'value'),
			   State('smoother', 'value')])
def update_output(n_clicks, bw_min, bw_max, smooth_val):
	if int(n_clicks) > 0:
		if int(bw_min) == int(bw_max):
			return ('The following KDE parameters have been saved: *bandwidth* = **{}** Ma.'.format(bw_min))
		else:
			return ('The following density varying KDE parameters have been saved: \
			*minimum bandwidth* = **{}** Ma, *maximum bandwidth* = **{}** Ma, *smoothing* = **{}** Ma.'.format(bw_min,bw_max,smooth_val))


@app.callback(Output('diss-df-json', 'children'),
			[Input('kdes-df-json', 'children'),
			Input('diss-matrix-stat-dropdown','value')])
def build_diss_matrix(kdes_df_json, stat_value):
	KDE_df = pd.read_json(kdes_df_json)
	KDE_df.sort_index(inplace=True)
	n = len(KDE_df.columns)
	diss = np.zeros((n,n))

	for i in range(n):
		for j in range(n):
			diss[i,j] = calc_stat(KDE_df.iloc[:,i], KDE_df.iloc[:,j],stat_value.lower())

	diss_df = pd.DataFrame(diss)
	return diss_df.to_json()


@app.callback(Output('main-map', 'figure'),
			[Input('raw-df-json', 'children'),
			Input('diss-df-json', 'children'),
			Input('k-clusters-slider','value'),
			Input('linkage-dropdown','value')])
def update_figure(raw_df_json, diss_df_json, k, linkages):
	raw_df = pd.read_json(raw_df_json)
	raw_df.sort_index(inplace=True)
	lats=list(raw_df.iloc[0].values)
	longs=list(raw_df.iloc[1].values)
	sample_names=list(raw_df)

	if diss_df_json is not None:
		diss_df = pd.read_json(diss_df_json)
		try:
			X = linkage(squareform(diss_df.values), linkages.lower())
			m = fcluster(X, k, criterion='maxclust')

			# correctly sort the diss_df results to match order of raw_df input
			df_m = pd.DataFrame(m, index=diss_df.index,columns=['a'])
			df_m.sort_index(inplace=True)
			m = df_m['a'].tolist()

			# assign colours for the labels
			c= ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, k+1)]
			c = [c[i] for i in m]

			return {
				'data': [{
				   'lat': lats,
				   'lon': longs,
				   'text': sample_names,
				   'type': 'scattermapbox',
				   'customdata': c,
				   'mode':'markers', 'marker':dict(size=15,
												   line = dict(width=10,color='black'),
												   color=c,
												   opacity = 0.9)
												   }],
				'layout': go.Layout(
				   height=400,
				   mapbox=dict(
						   accesstoken=('pk.eyJ1IjoiYWxpc2hvYmVpcmkiLCJhIjoiY2ozYnM3YTUxMDAxeDMzcGNjbmZyMmplZiJ9.ZjmQ0C2MNs1AzEBC_Syadg'),
						   center=dict(
							   lat=np.mean(lats),
							   lon=np.mean(longs)),
						   zoom=4,
				   ),
				   hovermode='closest',
				   margin=dict(r=0, t=10, l=50, b=10),
				   dragmode='select'
				)
			}
		except Exception as e:
			print(e)
			print (diss_df.values)


	return {
		'data': [{
		   'lat': lats,
		   'lon': longs,
		   'text': sample_names,
		   'type': 'scattermapbox',
		   'mode':'markers', 'marker':dict(size=15,
										   line = dict(width=10,color='black'),
										   color = 'grey',
										   opacity = 0.5)
										   }],
		'layout': go.Layout(
		   height=400,
		   mapbox=dict(
				   accesstoken=('pk.eyJ1IjoiYWxpc2hvYmVpcmkiLCJhIjoiY2ozYnM3YTUxMDAxeDMzcGNjbmZyMmplZiJ9.ZjmQ0C2MNs1AzEBC_Syadg'),
				   center=dict(
					   lat=np.mean(lats),
					   lon=np.mean(longs)),
				   zoom=4,
		   ),
		   hovermode='closest',
		   margin=dict(r=0, t=10, l=50, b=10),
		   dragmode='select'
		)
	}


@app.callback(Output('qc-kde-plot', 'figure'),
			[Input('main-map', 'selectedData'),
			Input('kdes-df-json', 'children')])
def update_figure(selectedData, kdes_df_json):
	KDE_df = pd.read_json(kdes_df_json)
	KDE_df.sort_index(inplace=True)

	def color_change(inp):
		c = inp.split(',')
		amd = int(c[1][0])+int(9*np.random.rand(1))
		c[1] = str( str(amd) + c[1][1:] )
		amd2 = int(c[2][0])+int(3*np.random.rand(1))
		c[2] = str( str(amd2) + c[2][1:] )
		return ','.join(c)

	c2 = []
	for colors_out in [selectedData['points'][i]['customdata'] for i in range(len(selectedData['points']))]:
		c2.append(color_change(colors_out))

	traces = []
	numerator = 0
	for samp_name in [selectedData['points'][i]['text'] for i in range(len(selectedData['points']))]:
		ys = list(KDE_df[samp_name])
		xs = list(range(len(KDE_df)))
		traces.append(go.Scatter(
			name=samp_name,
			hovertext=samp_name,
			x=xs,
			y=ys,
			mode='lines',
			line = dict(
				width=2,
				color = c2[numerator]
				)
			))
		numerator = numerator + 1

	return {
			'data':traces,
			'layout': go.Layout(
			height=350,
			margin={'l': 80, 'b': 40, 't': 20, 'r': 20},
			hovermode='closest',
			legend=dict(
				x=0.93,
				y=1),
			xaxis=dict(title='Age (Ma)',
						range=[0,max(xs)]),
			yaxis=dict(title='Density')),
	}

#app.css.config.serve_locally = True; app.scripts.config.serve_locally = True

if __name__ == '__main__':
	app.run_server(debug=True, port=3)
