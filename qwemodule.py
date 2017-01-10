import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chisquare
from random import shuffle

from pprint import pprint
from tabulate import tabulate
import pickle

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


## functions
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
            print "IQR == 0. ",variable, "needs a closer look"
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
            print "unknonw center"
            return ps
    else:
        print "unknonw method"
        return ps
    
def clean_data(data):
    """
    data (pandas.dataframe): raw dataframe to be cleaned
    This method will reformat IDs into integers and will rename original variables
    It will also drop outliers and transform some variables to categorical
    
    inputs:
        data (pandas.dataframe)
    return:
        data (pandas.dataframe)
    """
    # ID to integers
    data['ID'] = range(data.shape[0])

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
    # drop outliers
    centers_dict = {'Support Priority - Change': 'skip', #to be transformed into categorical'Views - Change': 'mean', 
                'Longevity - Months': 'median', 
                'Blogs - Change': 'mean', 
                'Support Cases - Change': 'skip',#to be transformed into categorical. Very unbalanced 
                'Logins - Change': 'mean', 
                'Days Since Last Login - Change': 'mean', 
                'Support Priority - Current Month': 'skip', #to be transformed into categorical
                'Happiness Index - Current Month': 'median', 
                'Happiness Index - Change': 'mean', 
                'Churn': 'skip', #target variable
                'ID': 'skip', #identifier
                'Support Cases - Current Month': 'skip',#to be transformed into categorical. Very unbalanced
                'Views - Change': 'mean'}
    for k,v in centers_dict.iteritems():
        if v != 'skip':
            data[k] = drop_outliers(data[k].copy(), method=2, center=v, variable=k)
            
    # support priority variables to categorical
    data['Support Priority - Change'] = ~(data['Support Priority - Change'] == 0)*1
    data['Support Priority - Current Month'] = ~(data['Support Priority - Current Month'] == 0)*1
    data['Support Cases - Change'] = ~(data['Support Cases - Change'] == 0)*1
    data['Support Cases - Current Month'] = ~(data['Support Cases - Current Month'] == 0)*1

    return data


def standardize(ps,center="mean"):
    """
    This method standardize data to their center and a unity variance
    ps (pandas.series): data to be standardize
    center (str): central point {"mean","median"}
    """
    if center == "mean":
        return (ps - ps.mean()) / ps.std()
    if center == "median":
        return (ps - ps.median()[0]) / ps.std()
    else:
        print "unknown central point"
        return ps

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
        print "Only one sample can be generated"
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
        
def biplot(score,coeff,pcax,pcay,labels=None,nm=None):
    """
    This function generate a scatter plot to visualize important features resulting from a principle component analysis
    Given dataset X, the principle components analysis generates loadings (coefficients) L and scores (new features
    in the new basis) C such that X = L*C
    inputs:
        score (matrix): components in the new basis (C)
        coeff (matrix): also called loadings (L)
        pcax (int): one of the principle components to plot
        pcay (int): another principle to plot
    """
    pca1=pcax-1
    pca2=pcay-1
    xs = score[:,pca1]
    ys = score[:,pca2]
    n=score.shape[1]
    if nm == None:
        nm = n
    #construct scales to constrain data between -1 and 1
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())
    
    #scatter data points in the new basis span by pca1 and pca2
    plt.scatter(xs*scalex,ys*scaley, marker='.',edgecolor='none')
    vectors = []
    
    #overlay transforms of original features in the new basis
    for i in range(n):
        #calculate length of vectors in new basis
        vectors.append((labels[i],np.sqrt(coeff[i,pca1]**2 + coeff[i,pca2]**2)))
        #plot arrow vectors
        plt.arrow(0, 0, coeff[i,pca1], coeff[i,pca2],color='g',alpha=0.5) 
        #add labels
        if labels is None:
            plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, "Var"+str(i+1), color='k', ha='center', va='center')
        else:
            plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, labels[i], color='k', ha='center', va='center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(pcax))
    plt.ylabel("PC{}".format(pcay))
    plt.grid()
    plt.show()
    #print "Feature import (PCA)"
    #print "--------------------"
    vectors = sorted(vectors,key=lambda x:x[1],reverse=False)
    
    plt.barh(range(len(vectors)),map(lambda x:x[1],vectors),edgecolor='none')
    plt.yticks(np.arange(len(vectors))+0.4,map(lambda x:x[0],vectors))
    plt.xlabel('Feature importance')
    plt.grid()
    plt.show()
    #pprint(vectors)
    return vectors

def modelfit(alg, xtr, ytr, performCV=True, printFeatureImportance=True, cv_folds=5,title=None):
    """
    This function perform cross validation (CV) of a model over a training dataset.
    It will print results of the cross validation: mean score, standard deviation, min and max score.
    inputs:
        alg: sklearn algorithm to be trained and validated
        xtr: training dataset
        ytr: target classes for the training dataset
        performCV (bool): performs the CV is true
        printFeatureImportance (bool): print the importance of features. It only works for selectedalogorithms.
        cv_folds (int): subset fold of cross validation
        title (str): title to label generated results
    return (dict): dictionary of the trained and validated model and validation score
    """
    print title
    print "-------------------------------"
    #Fit the algorithm on the data
    alg.fit(xtr, ytr)
    #Predict training set:
    dtrain_predictions = alg.predict(xtr)
    dtrain_predprob = alg.predict_proba(xtr)[:,1]
    #Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, xtr, ytr, cv=cv_folds, scoring='roc_auc')

    #Print model report:
    #print "\nModel Report"
    print "Accuracy: %.4g" % accuracy_score(ytr.values, dtrain_predictions)
    print "AUC Score (Train): %f" % roc_auc_score(ytr, dtrain_predprob)
    
    if performCV:
        print "CV Score: Mean = %.4g | Std = %.4g | Min = %.4g | Max = %.4g" % \
        (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, xtr.columns.tolist()).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()
    return {'model':alg,'cv_score':cv_score}

def get_labels(labels_name):
    """
    this returns all labels used in analysis and predictions
    """
    labels = {
        "labels_num":['Blogs - Change', 'Customer Activity - Change', 'Days Since Last Login - Change', 
                      'Happiness Index - Change', 'Happiness Index - Current Month', 'Happiness Index - Monthly', 
                      'Logins - Change', 'Longevity - Modulo 12', 'Longevity - Modulo 18', 'Longevity - Modulo 24', 
                      'Longevity - Months', 'Views - Change'],
        "labels_cat":['Longevity - Modulo 6', 'Support Cases - Change', 'Support Cases - Current Month', 'Support Priority - Change',
                      'Support Priority - Current Month'],
        "target":"Churn",
        "labels_pca":['Happiness Index - Monthly', 'Longevity - Modulo 12', 'Happiness Index - Change', 
                      'Blogs - Change', 'Happiness Index - Current Month', 'Longevity - Modulo 24', 
                      'Customer Activity - Change', 'Logins - Change', 'Longevity - Modulo 18', 
                      'Days Since Last Login - Change']
    }
    return labels[labels_name]

def make_predictions(df):
    """
    This function provides the user a fast way to make predictions over a dataset
    """
    t_labels = get_labels("labels_pca")
    # clean data
    df = clean_data(df)
    # engineer data
    df = engineer_features(df)
    # predict
    with open("model.pkl","r") as mdl:
        model = pickle.load(mdl)
        mdl.close()
    predictions = model.predict(df[t_labels])
    return predictions

## ========================= stratification sampling class ====================== ##   
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
        print "total number of customers:", ncustomers
        print "number of actual churners:", nchurners
        print "total number of non-churners needed to obtain {0}% of churners: {1}".format(self.posRate,ncneeded)
        print "the new sample will be made of {0}. That is {1}% of initial dataset".format(self.ntotal,100*self.ntotal/ncustomers)
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
        print "==> resampling ...  ",
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
                print "Enter posRate higher than the initial rate"
                break
            i+=1
        print "Done sampling"
        print "positive:", len(positiveIds)
        print "negative:", len(negativeIds)
        print "final size:", len(positiveIds)+len(negativeIds)
        #return sorted list of the two list of ids combined
        return sorted(positiveIds+negativeIds)