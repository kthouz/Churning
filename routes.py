from flask import Flask, render_template, jsonify, request
import os, json
import module as xyz
import pandas as pd
import numpy as np

global filename, sample, rootname

"""
#load data
sample = pd.read_csv('QWE_Data.csv')

#clean data
sample = xyz.clean_data(sample)
#save a copy and continue
sample.to_csv('cleaned_data.csv',index=False)
sample = pd.read_csv('cleaned_data.csv')
#create inital labels
labels0 = sample.columns

#engineer new variables
maxlong = 67 #this needs to be updated for a new dataset
#data = xyz.engineer_features(data,maxlong)
#summary = xyz.describe_data(data)
#drop outliers: I consider every point located at more than 3 standard deviations of the median as outlier
for col in data.columns: 
	if (data[col]<0).sum()>0: #if the data is symmetric around zero, use mean as central point
		mean = "mean"
	else: #use median is data is not symmetric
		method = "median"
	data = xyz.drop_outlier(data,col,method="mean")
#sample the dataset
pos = data[data.Churn==1].index.tolist()
neg = data[data.Churn==0].index.tolist()
sample = data.loc[xyz.StratSampler(pos,neg,25).sampledIds]
#sample.to_csv("static/js/sample.csv",index=False)
"""
#sample = data.copy()
#labels = sample.columns.tolist()

def scatter_data(sample,variables,jitters=[False,False]):
	"""
	variables (list of str)
	"""
	r = variables[:3]
	#df = sample[r]
	series = []
	jdata = []
	if r[2]!='Default':
		df = sample[r]
		for val in sorted(df[r[2]].unique().tolist()):
			series.append(val)
			jdata.append(map(lambda r:{'x':r[0]+jitters[0]*np.random.normal(0,0.1),
				'y':r[1]+jitters[1]*np.random.normal(0,0.05)},df[df[r[2]]==val].values))
	else:
		series.append(r[2])
		df = sample[r[:2]]
		jdata.append(map(lambda r:{'x':r[0]+jitters[0]*np.random.normal(0,0.1),
			'y':r[1]+jitters[1]*np.random.normal(0,0.1)},df.values))

	data = {'data':jdata,'series':series}
	return data

def box_data(sample,variables,jitters=[False,False]):
	r = variables[:2]
	df = sample[r]
	series = []
	jdata = []
	print type(df[r[0]]), r
	for val in sorted(df[r[0]].unique().tolist()):
		series.append(round(val,2))
		s = df[r[1]][df[r[0]] == val].describe().round(2).T
		jdata.append({'Q1':s['25%'],'Q2':s['50%'],'Q3':s['75%'],
			'whisker_low':s['25%']-1.5*(s['75%']-s['25%']),'whisker_high':s['50%']+1.5*(s['75%']-s['25%'])})
	data = {'data':jdata,'series':series}

	return data

def bar_data(sample,variables,jitters=[False,False]):
	r = variables[:3]
	df = sample.copy()#[[r[0],r[2]]]
	jdata = []
	series = []

	if df[r[0]].unique().shape[0]<20:
		#We proceed generating bar heights
		if r[2] != 'Default':
			df[r[2]] = df[r[2]].round(2)
			df = pd.crosstab(df[r[0]],df[r[2]]).sort_index().reset_index()
			for col in df.columns[1:]:
				series.append(col)
				jdata.append(map(lambda r:{'x':r[0],'y':r[1]},df[[r[0],col]].values))
		else:
			df = df[r[0]].value_counts().sort_index().reset_index()
			series.append(df.columns[-1])
			jdata.append(map(lambda r:{'x':r[0],'y':r[1]},df.values))

	else:
		#We need to bin data into 20 bins by default
		if r[2] != 'Default':
			_,xh = np.histogram(df[r[0]],bins=20)
			for val in sorted(df[r[2]].unique().tolist()):
				yh,_ = np.histogram(df[r[0]][df[r[2]]==val],xh)
				series.append(val)
				jdata.append(map(lambda x,y:{'x':x,'y':y},xh[:-1],yh))
		else:
			yh,xh = np.histogram(df[r[0]],bins=20)
			series.append(df.columns[-1])
			jdata.append(map(lambda x,y:{'x':x,'y':y},xh[:-1],yh))
	data = {'data':jdata,'series':series}
	return data


app = Flask(__name__)


@app.route('/')
def index():

	return render_template('index.html',labels=[''])

@app.route('/dataset')
def dataset():
	global filename, sample, rootname
	filename = request.args.get('variables',0,str).split('\\')[-1]
	rootname = 'clean_'+filename
	#load data
	sample = pd.read_csv(filename)

	#clean data
	#sample = xyz.clean_data(sample)
	#save a copy and continue
	sample.to_csv(rootname,index=False)
	sample = pd.read_csv(rootname)
	alert = str(sample.shape[0])+' lines loaded from '+filename
	return jsonify(filename=filename,alert=alert)

@app.route('/data')
def data():
	global filename, sample, rootname
	variables0 = request.args.get('variables',0,str)
	variables, jitters = variables0.split('_x_')

	jitters = map(lambda j:bool((j=='true')*1),jitters.split('__'))

	r = variables.split('__')
	print "variables",variables, "jitters", jitters
	sample = pd.read_csv(rootname)
	print "sample reloaded"
	if r[-1] == 'ScatterPlot':#prep data for scatter plot
		data = scatter_data(sample,r,jitters)
	elif r[-1] == 'BarPlot':
		data = bar_data(sample,r,jitters)
	elif r[-1] == 'BoxPlot':
		data = box_data(sample,r,jitters)
	elif r[-1] == 'Treemap':
		data = tree_data(sample,r,jitters)

	return jsonify(dataset=data)


@app.route('/outliers')
def outliers():
	global filename, sample, rootname
	variable = request.args.get('variables',0,str)
	sample = pd.read_csv(rootname)
	min1, max1 = sample[variable].min(), sample[variable].max()
	if sample[sample[variable]<0].shape > 0: #if there are data below zero, consider using the mean as center
		sample[variable+'_Out'] = xyz.drop_outliers(sample[variable].copy(),method=2,center='mean')
		sample.to_csv(rootname,index=False)
		sample = pd.read_csv(rootname)
	else: #otherwise use the median. Churn data are usualy asymmetric is there are not about a change
		pass
		sample[variable+'_Out'] = xyz.drop_outliers(sample[variable].copy(),method=2,center='median')
		sample.to_csv(rootname,index=False)
		sample = pd.read_csv(rootname)

	min2=sample[variable+'_Out'].min()
	max2=sample[variable+'_Out'].max()
	labels = ['']+map(lambda x:str(x),sample.columns)
	alert = variable+': oultiers dropped! \n min: '+str(min1)+' -> '+str(min2)+ ' \n max: '+str(max1)+' -> '+str(max2)
	alert +=' \nConsider plottin '+variable+'_Out'
	print labels
	return jsonify(alert=alert,labels=labels)

@app.route('/categories')
def categories():
	global filename, sample, rootname
	variable = request.args.get('variables',0,str)
	sample = pd.read_csv(rootname)
	if 'Change' in map(lambda x:x.strip(),variable.split('-')):
		print "assertion"
		sample[variable+'_Cat'] = xyz.categorize_change(sample[variable].copy())
		sample.to_csv(rootname,index=False)
		sample = pd.read_csv(rootname)
		labels = ['']+map(lambda x:str(x),sample.columns)
		alert = variable+': converted in '+str(sample[variable+'_Cat'].unique().shape[0])+' categories: '
		alert += str(sample[variable+'_Cat'].unique().tolist())+'\n. Replot '+variable+'_Cat'
		print labels
		return jsonify(label=variable,labels=labels,alert=alert)
	else:
		maj_cat = sample[variable].value_counts().index[0]
		sample[variable+'_Cat'] = sample[variable].copy()
		sample[sample[variable+'_Cat'] == maj_cat] == 0
		sample[~(sample[variable+'_Cat'] == maj_cat)] == 1
		sample.to_csv(rootname,index=False)
		sample = pd.read_csv(rootname)
		labels = ['']+map(lambda x:str(x),sample.columns)
		alert = variable+': converted in '+str(sample[variable+'_Cat'].unique().shape[0])+' categories: '
		alert += str(sample[variable+'_Cat'].unique().tolist())+'\n. Replot '+variable+'_Cat'
		print labels
		return jsonify(label=variable,labels=labels,alert=alert)

@app.route('/drop_column')
def drop_column():
	global filename, sample, rootname
	variable = request.args.get('variables',0,str)
	sample = pd.read_csv(rootname)
	sample.drop(variable,axis=1,inplace=True)
	sample.to_csv(rootname,index=False)
	alert = variable+' column dropped'
	labels = ['']+map(lambda x:str(x),sample.columns)
	return jsonify(labels=labels,alert=alert)

if __name__ == '__main__':
	port = int(os.environ.get('PORT',8080))
	app.debug = True
	app.run(host = '0.0.0.0',port=port)