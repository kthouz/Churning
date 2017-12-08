import pandas as pd
import pandas
import numpy as np
import numpy
#import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, chisquare

import tabulate
from itertools import groupby
from random import shuffle
import pickle

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

class StratSampler():
	def __init__(self,pos,neg,posRate):
		"""
		This is a class for stratification sampling
		inputs:
			pos (list of int): list of indices of positive cases in the dataset
			neg (list of int): list of indices of negative cases in the dataset
			posRate (int): new sought positive rate. Note: posRate > initial rate
		outputs:
			sampledIds (list of int): ordered list of sampled indices
		"""
		self.pos = pos
		self.neg = neg
		self.posRate = posRate

		
		nchurners=len(self.pos)
		ncustomers = nchurners+len(self.neg)
		ncneeded = (100/self.posRate - 1)*nchurners
		self.ntotal = ncneeded+nchurners
		print("total number of customers:", ncustomers)
		print("number of actual churners:", nchurners)
		print("total number of non-churners needed to obtain {0}% of churners: {1}".format(self.posRate,ncneeded))
		print("the new sample will be made of {0}. That is {1}% of initial dataset".format(self.ntotal,100*self.ntotal/ncustomers))
		for i in range(100): #randomly reshafle the list of indices
			shuffle(self.pos)
			shuffle(self.neg)
		
		self.posId = self.pos_gen()
		self.negId = self.neg_gen()
		self.sampledIds = self.get_sampled_ids()
	def pos_gen(self): #create a chain (generator) object for positive indices
		for i in self.pos:
			yield i
	def neg_gen(self): #create a chain (generator) object for positive indices
		for i in self.neg:
			yield i
	
	def get_sampled_ids(self):
		"""

		start a loop from 0 to total size of the new sample
		"""
		seed = 123
		#initiate two lists, to save randomly picked positive and negative cases respectively
		positiveIds = []
		negativeIds = []
		i = 0
		print("==> resampling ...  ",)
		while len(positiveIds)+len(negativeIds)<self.ntotal:
			# start a loop from 0 to total size of the new sampe
			# if it catches a number divisable by the sought ratio, update the list of positive cases ids
			# otherwise keep update the list of negative cases ids
			try:
				if i%int(100 / self.posRate) == 0: 
					positiveIds.append(self.posId.next())
				else:
					negativeIds.append(self.negId.next())
			except:
				print("Enter posRate higher than the initial rate")
				break
			i+=1
		print("Done sampling")
		print("positive:", len(positiveIds))
		print("negative:", len(negativeIds))
		print("final size:", len(positiveIds)+len(negativeIds))
		#return sorted list of the two list of ids combined
		return sorted(positiveIds+negativeIds)


def clean_data(data):
	"""
	data (pandas.dataframe): raw dataframe to be cleaned
	"""
	# drop ID column
	#data.drop('ID',axis=1,inplace=True)

	# rename columns
	data.rename(columns = {'Customer Age (in months)':'Longevity - Months',
		'Churn (1 = Yes, 0 = No)':'Churn',
		'CHI Score Month 0':'Happiness Index - Current Month',
		'CHI Score 0-1':'Happiness Index - Change',
		'Support Cases Month 0':'Support Cases - Current Month',
		'Support Cases 0-1':'Support Cases - Change',
		'SP Month 0':'Support Priority - Current Month',
		'SP 0-1':'Support Priority - Change',
		'Logins 0-1':'Logins - Change',
		'Blog Articles 0-1':'Blogs - Change',
		'Views 0-1':'Views - Change',
		' Days Since Last Login 0-1':'Days Since Last Login - Change'},inplace = True)

	return data

def drop_outliers(ps, method=1,center="mean",variable=None):
	"""
	This function removes all outliers. These points are replaced by the mean or median
	method 1: An outlier is defined as a point lying below Q1 - 1.5IQR and above Q3 + 1.5IQR
	Q1: first quartile, Q2: median, Q3: third quartile, IQR: Q3 - Q1.
	method 2: an outlier is any point located 2.5 standard devs, far from the center(mean or median)
	Note: This doesn't work on categorical data
	input: 
		ps (pandas.dataframe or pandas.series): array of numbers to treated
		center (str): default is mean. options: mean, median
	output: pandas.series
	"""
	s = ps.describe().T
	Q1,median,Q3,mean,std = s["25%"],s["50%"],s["75%"],s['mean'], s['std']
	if method == 1:
		IQR = Q3 - Q1
		if IQR == 0:
			print("IQR == 0. ",variable, "needs a closer look")
			return ps
		else:
			ix = ps[(ps < (Q1 - 1.5 * IQR)) | (ps > (Q3 + 1.5 * IQR))].index.tolist()
			return ps
	elif method == 2:
		if center == "mean":
			ix = ps[abs(ps - mean) > 2.5 * std].index.tolist()
			ps.loc[ix] = mean
			return ps
		elif center == "median":
			ix = ps[abs(ps - median) > 2.5 * std].index.tolist()
			ps.loc[ix] = median
			return ps
		else:
			print("unknonw center")
			return ps
	else:
		print("unknonw method")
		return ps

def categorize_change(ps):
	"""
	This function categorizes change in three categories:
	0: no change
	1: increase
	-1: decrease
	"""
	ps[ps == 0] = 0
	ps[ps < 0] = -1
	ps[ps > 1] = 1
	#print("variable transformed in categorical")
	return ps

def engineer_features(data,maxLong):
	pass
	"""
	data['HasSP'] = ~(data.SupportPriority == 0)*1
	data['HasChangeSP'] = ~(data.ChangeSupportPriority == 0)*1
	data['HasNullHI'] = (data.HappinessIndex == 0)*1
	data['PreviousHappinessIndex'] = data.HappinessIndex - data.ChangeHappinessIndex
	data['IsRatedHI'] = ~((data.HappinessIndex == 0) & (data.PreviousHappinessIndex == 0))*1
	data['HasSC'] = ~(data.SupportCases==0)*1
	data['HasChangeSC'] = ~(data.ChangeSupportCases==0)*1
	data['HasChangeNBlogs'] = ~(data.ChangeNumberBlogs == 0)*1
	data['HasChangeNViews'] = ~(data.ChangeNumberViews == 0)*1


	#flip of Longevity with respect to 0 and translated by maxLong
	data['LongevityFlip'] = maxLong - data.Longevity 
	#create remainders of the longevity by 6,12,18,24
	N = [6,12,18,24]
	for n in N:
		data['Longevity'+str(n)] = data.Longevity.apply(lambda x:x%n)
	#Customer Activity
	features = ['ChangeNumberLogins', 'ChangeNumberBlogs','ChangeNumberViews', 'DaysLastLogin']
	data['CustomerActivity1'] = standardizeCustomerActivity(data[features]).sum(axis=1)/4.
	data['CustomerActivity2'] = standardizeCustomerActivity(data[features[:-1]]).sum(axis=1)/4.
	data['IsActive'] = (abs(data.CustomerActivity2 - data.CustomerActivity2.mean())<=data.CustomerActivity2.std())*1
	#happiness as monthly accrued metric
	data['MonthlyHappinessIndex'] = data.HappinessIndex/data.Longevity
	data.MonthlyHappinessIndex.fillna(0,inplace=True)

	return data
	"""

def describe_data(data):
	summary = data.describe()
	summary.loc['uniqueVals'] = data.apply(lambda x:x.unique().shape[0])
	summary.drop(['25%','75%'],axis=0,inplace=True)
	print("size: {0}".format(data.shape))
	print("probability for churning: {0}%".format(100*data.Churn.sum()/data.shape[0]))
	print(tabulate.tabulate(summary.T,tuple(summary.index.tolist())))
	return summary.T

def standardize_vars(df,new = True):
	"""
	df (pandas.dataframe): dataframe to standardize. 
		It has to have these 4 features in right sequence 
			[u'Logins - Change', u'Blogs - Change',u'Views - Change', u'Days Since Last Login - Change']
	new (bool): True if there is a new dataset to standardize, False, otherwise
	"""
	if new:
		df_norm = (df - df.mean())/df.std()
	else:
		dfmean = np.array([ 13.4752322 ,   0.04256966,  30.93343653,   3.11687307])
		dfstd = np.array([   39.14395968,     2.96332588,  1331.21837006,    16.44862321])
		df_norm = (df - dfmean)/df.std
	return df_norm

def test_stat(df,ivar,tvar,equal_var=True,ddof=0):
	"""
	This function calculate statistical test to check if means are the same in groups of ivar
	split by tvar
	inputs:
		df (pandas.dataframe): data frame of interest
		ivar (str): indipendent variable
		tvar (str): target variable
	"""
	ivar_uniques = df[ivar].unique().shape[0]
	tvar_uniques = df[tvar].unique().shape[0]
	if tvar_uniques < 2:
		print ("Only one sample can be generated")
		return None
	if ivar_uniques <= 10: #This the case of a categorical independant variable. We use chisquare
		ss = pd.crosstab(df[ivar],df[tvar])
		ss = (ss.T/ss.sum(axis=1)).T
		s0,s1 = ss[0].values,ss[1].values

		return chisquare(s1,s0,ddof=ddof)

	if ivar_uniques >10: #Consider using ttest
		s0 = df[ivar][df[tvar] == 0]
		s1 = df[ivar][df[tvar] == 1]
		return ttest_ind(s1,s0,equal_var=equal_var)

def explore_continuous_var(ivar,tvar='Churn',df=None,
						   barplotOpts={'bins':10},
						   boxplotOpts = {'ylimit':[0,30]},
						   ttestOpts={'equal_var':False}):
	"""
	This function generates bar-, box- and scatter-plots to compare ivar between two target classes in the tvar.
	It should also be able to output a statistical test between the two samples
	
	ivar (str): independent variable - continuous
	tvar (str): target variable - categorical
	boxplotopts (dict): options of the plots
	"""
	
	fig,ax = plt.subplots(1,3,figsize=(12,2))
	xs = np.linspace(df[ivar].min(),df[ivar].max(),barplotOpts['bins'])
	w = 0.9*(xs[1:] - xs[:-1])[0]
	s0 = df[df[tvar]==0]
	s1 = df[df[tvar]==1]
	h0,_ = np.histogram(s0[ivar],xs)
	h1,_ = np.histogram(s1[ivar],xs)
	
	ax[0].bar(xs[:-1],h0,width=w/2,color='b',edgecolor='none')
	ax[0].bar(xs[:-1]+w/2,h1,width=w/2,color='g',edgecolor='none')

	ax[0].set_xlabel(ivar)
	ax[0].set_ylabel('Count')
	ax[0].legend([0,1],loc='best')
	
	ax[1].boxplot([s0[ivar],s1[ivar]],labels = ['NoChurn','Churn'])
	ax[1].set_ylim(boxplotOpts['ylimit'])
	ax[1].set_ylabel(ivar)
	
	ax[2].scatter(df[ivar],df[tvar]+np.random.normal(loc=0,scale=.05,size=df.shape[0]),
				 s=100,alpha=0.25,edgecolor='none',marker='.')
	ax[2].set_xlabel(ivar)
	ax[2].set_ylabel(tvar)
	#plt.yticks([0,1])

	plt.show()
	
	print(ivar,"t-test results:", ttest_ind(s0[ivar],s1[ivar],equal_var=ttestOpts['equal_var']))

def explore_categorical_var(ivar,tvar='Churn',df=None):
	ctbl = pd.crosstab(df[ivar],df[tvar])
	tot1,tot2 = ctbl.sum(axis=1),ctbl.sum(axis=0)
	ctbl1,ctbl2 = (ctbl.T/tot1).T,(100*ctbl/tot2).round(2)
	
	plt.figure(figsize=(4,2))
	plt.bar(ctbl1.index,ctbl1[0],color='b',width=1)
	plt.bar(ctbl1.index,ctbl1[1],bottom=ctbl1[0],color='g',width=1)
	plt.xticks(ctbl1.index+0.5,ctbl1.index,rotation=90)
	plt.xlabel(ivar)
	plt.ylabel(tvar+' Probability')
	plt.legend([0,1],title='Churn')
	plt.show()
	
	print(ivar, "chi square results:",chisquare(ctbl[0],ctbl[1]))

def evaluateModel(model,xtr,ytr,xva,yva,title,retrain=False):
	print(title)
	print("-------------------------------")
	
	if retrain:
		model.fit(xtr,ytr)
	ycheck = model.predict(xtr)
	ypreds = model.predict(xva)
	
	scores = {
		'roc_score':[roc_auc_score(ytr,ycheck),roc_auc_score(yva,ypreds)],
		'acc_score':[accuracy_score(ytr,ycheck),accuracy_score(yva,ypreds)]
	}
	print(tabulate.tabulate(pd.DataFrame(scores.values(), columns=['train','valid'],
										 index=scores.keys()).T,
						   headers=scores.keys()))
	confusionM = {'train':confusion_matrix(ytr,ycheck), 'valid':confusion_matrix(yva,ypreds)}
	#print "======================================="

	return {'model':model,'scores':scores,'title':title,'confusionM':confusionM}

def modelfit(alg, xtr, ytr, performCV=True, printFeatureImportance=True, cv_folds=5,title=None):
	print(title)
	print("-------------------------------")
	#Fit the algorithm on the data
	alg.fit(xtr, ytr)
		
	#Predict training set:
	dtrain_predictions = alg.predict(xtr)
	dtrain_predprob = alg.predict_proba(xtr)[:,1]
	
	#Perform cross-validation:
	if performCV:
		cv_score = cross_val_score(alg, xtr, ytr, cv=cv_folds, scoring='roc_auc')
	
	#Print(model report:)
	#print("\nModel Report")
	print("Accuracy: %.4g" % accuracy_score(ytr.values, dtrain_predictions))
	print("AUC Score (Train): %f" % roc_auc_score(ytr, dtrain_predprob))
	
	if performCV:
		print("CV Score: Mean = %.4g | Std = %.4g | Min = %.4g | Max = %.4g" % \
		(np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
		
	#Print Feature Importance:
	if printFeatureImportance:
		feat_imp = pd.Series(alg.feature_importances_, xtr.columns.tolist()).sort_values(ascending=False)
		feat_imp.plot(kind='bar', title='Feature Importances')
		plt.ylabel('Feature Importance Score')
		plt.show()
	return {'model':alg,'cv_score':cv_score}