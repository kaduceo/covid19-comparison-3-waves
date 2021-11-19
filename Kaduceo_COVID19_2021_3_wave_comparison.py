#!/usr/bin/env python
# coding: utf-8

# - **Code for the study named**
# - Evolution of hospitalized patient characteristics through the
# first three COVID-19 waves in Paris area using machine
# learning analysis.
# 
#     - **Authors**
#     - Camille Jung 1 , Jean-Baptiste Excoffier 2* , Mathilde Raphaël-Rousseau 3 , Noémie
# Salaün-Penquer 2 , Matthieu Ortala 2 , Christos Chouaid 4,5
#         - **Institutions**
#         - 1 Clinical Research Center, CHI Créteil, France; firstname.lastname@chicreteil.fr
#         - 2 Kaduceo; firstname.lastname@kaduceo.com
#         - 3 Department of medical information, CHI Créteil, France;
# firstname.lastname@chicreteil.fr
#         - 4 Department of pneumology, CHI Créteil, France

# - **License** : This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# # Packages needed



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import dataframe_image as dfi

import sklearn
from sklearn import model_selection, tree, ensemble

import xgboost as xgb

import shap



from IPython.core.display import display, HTML
display(HTML("<style>.container{width:80%!important;}</style>"))





from IPython.display import display_html
from itertools import chain, cycle

def display_side_by_side( *args,titles=cycle(['']) ) :
    "https://stackoverflow.com/a/44923103"
    
    html_str=''
    for df, title in zip( args, chain(titles,cycle(['</br>'])) ) :
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2>{title}</h2>'
        html_str+=df.to_html().replace('table','table style="display:inline"')
        html_str+='</td></th>'
        
    display_html(html_str,raw=True)


# # Data
# - Due to privacy reasons, data cannot be made available. Hence, this code does not work *per se*. Nevertheless, this code is made public so as to allow any person to check the validity and soundness of methods and implementations used, especially concerning machine learning model building and validation.




### Data loading (a dataset per wave)

wave_files_names = ["Data/Wave_1.csv", "Data/Wave_2.csv", "Data/Wave_3.csv"]

list_wave_df = []

for name in wave_files_names :
    list_wave_df.append(pd.read_csv(name, sep=";"))





target_variable_name = 'Severe Case'

explanatory_variable_names = ['Age', 
                         'Gender', 
                         'Cancer', 
                         'Diabetes',
                         'Embolism', 
                         'Overweight or Obesity', 
                         'Cardiovascular',
                         'Cirrhosis',
                         'Sickle Cell', 
                         'IBD', 
                         'Mental Retardation', 
                         'Cognitive Impairment', 
                         'Pregnancy',
                         'Trisomy',
                         'Heart Failure', 
                         'Dementia',
                         'Psychiatric Disorders', 
                         'Pulmonary problems',
                         'Organ Transplant',
                         'Stroke Sequelae']


# # Machine Learning Model construction
# - Based on **XGBoost** (https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
# - A **nested cross-validation** is used, using inner and outer cross-validation splitting (https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)




scoring_rule_ = "roc_auc" #Scoring rule for inner cross-validation optimization of hyperparameters.


def fct_XGB_gridsearch( X_, y_ ) :
    type_CV = model_selection.RepeatedStratifiedKFold( n_splits=5, n_repeats=2 )

    
    sqrt_colsample_bytree = np.sqrt( X_.shape[1] ) / X_.shape[1]
    balanced_scale_pos_weight = (y_.squeeze() == 0).sum() / (y_.squeeze() == 1).sum()
    
    grille_params = { "objective":["binary:logistic"], 
                  "eval_metric":["error"], 
                  "max_depth":[2, 3, 4, 5, 6], 
                  "n_estimators":[100, 200, 400], 
                  "learning_rate":[0.05, 0.1, 0.2], 
                  "min_child_weight":[5, 10, 20],
                  "subsample":[0.7, 1],
                  "colsample_bytree":[0.5, 0.8, 1], 
                  "scale_pos_weight":[1, balanced_scale_pos_weight], 
                  "use_label_encoder":[False]}

    gridsearch_repeatedCV = model_selection.GridSearchCV( estimator=xgb.XGBClassifier(), param_grid=grille_params, cv=type_CV, scoring=scoring_rule_, refit=False )
    
    gridsearch_repeatedCV.fit( X_, y_ )
    
    return xgb.XGBClassifier( **gridsearch_repeatedCV.best_params_ )


function_model = fct_XGB_gridsearch  





### Outer Cross validation

Kfold_5 = model_selection.StratifiedKFold( n_splits=5, shuffle=True )





dict_df_waves = dict()
dict_X = dict()
dict_y_pred = dict()
dict_shap_values = dict()


for i, df_wave_x in enumerate(list_wave_df) :
    j = i + 1
    print("Wave", j, "\n")
    
    X = df_wave_x[ explanatory_variable_names ].astype(float).values
    y = df_wave_x[target_variable_name].astype(float).values.flatten()
    
    y_pred = np.zeros( shape=(y.shape[0], 2) ) #Target variable is binary.
    X_shap_values = np.zeros( shape=X.shape ) #As target variable is binary, we only look at the influences for the positive class.
    

    splits = Kfold_5.split( X, y )

    for index_train, index_test in splits : 
        X_train = X[index_train, :] ; y_train = y[index_train]
        X_test = X[index_test, :] ; y_test = y[index_test]

        modele_temp = function_model( X_train, y_train )
        modele_temp.fit( X_train, y_train )

        y_pred[index_test, :] = modele_temp.predict_proba( X_test )


        #### Inlfuences are computed using only the training set background.
        TreeSHAP_model = shap.TreeExplainer( modele_temp, data=X_train, feature_perturbation="interventional", model_output="probability" )

        #### We only retain influences for positive class.
        temp_shap_ = TreeSHAP_model.shap_values( X=X_test, approximate=False, check_additivity=True )
        X_shap_values[index_test, :] = temp_shap_
    
    
    
    dict_df_waves[j] = df_wave_x
    dict_X[j] = pd.DataFrame(X, columns=explanatory_variable_names)
    dict_y_pred[j] = pd.DataFrame(y_pred)
    dict_shap_values[j] = pd.DataFrame(X_shap_values, columns=explanatory_variable_names)
    
    print("\n")
    
    
print("Finished.")


# # Model performances




for i in range(1, 4) :
    print("Wave", i)
    
    y = dict_df_waves[i][target_variable_name].values
    y_pred = dict_y_pred[i].values
    
    
    prop_1 = y.sum() / y.shape[0] * 100
    
    accuracy = (y_pred.argmax(axis=1) == y).sum() / y.shape[0] * 100

    fpr, tpr, _ = sklearn.metrics.roc_curve(y, y_pred[:, 1])
    auc_roc_score = sklearn.metrics.auc(fpr, tpr) * 100

    df_confusion_matrix = sklearn.metrics.confusion_matrix( y, y_pred.argmax(axis=1), labels=[0, 1] )
    df_confusion_matrix = pd.DataFrame( df_confusion_matrix, index=[0, 1], columns=[0, 1] )
    df_confusion_matrix.index.name = "Classe"
    df_confusion_matrix.columns.name = "Pred"

    precision_, recall_, f1score_, n_, = sklearn.metrics.precision_recall_fscore_support( y, y_pred.argmax(axis=1), beta=1.0, labels=[0, 1] )
    df_stats_ = pd.DataFrame( {"Sensitivity":recall_, "Precision":precision_, "F1-Score":f1score_}, index=[0, 1] ) * 100
    
    
    df_base = pd.DataFrame( {"Prop 1":[prop_1], "Precision":[accuracy], "ROC score":[auc_roc_score]} )
    display_side_by_side( df_base.round(2), df_confusion_matrix, df_stats_.round(2) )
    print("\n")


# # Influence analysis




### We put all influences in percents

for i in range(1, 4) : 
    dict_shap_values[i] = dict_shap_values[i] * 100





### Colors and markers for each wave for graph

list_colors = ["deepskyblue", "dodgerblue", "mediumblue"]
list_markers = ["o", "P", "D"]


# ### Global views




### Global importance ranking graph

df_abs_mean = pd.concat( [ (dict_shap_values[i].abs().mean()).rename("Wave " + str(i)) for i in range(1, 4) ], axis=1 )

#### Normalization for each wave for a better comparison between waves
df_abs_mean = df_abs_mean / df_abs_mean.sum()


plt.figure( figsize=(14, 11) )

df_abs_mean.plot( kind="barh", color=list_colors, width=0.8, ax=plt.gca() )

plt.grid(axis="both", linewidth=0.5, linestyle="--")
plt.gca().invert_yaxis()
plt.tick_params(axis='both', labelsize=14)
plt.legend(loc="best", prop={'size': 16})



plt.xlabel("Normalized Mean Absolute Influence", fontsize=15)
plt.show()





### Distribution graph

def fct_shap_plot( i ) :
    color_bar_ = True
    if i != 3 :
        color_bar_ = False

    shap.summary_plot(shap_values=dict_shap_values[i].values, 
                      features=dict_X[i].values, max_display=None,
                      feature_names=explanatory_variable_names, 
                       show=False, plot_size=None,
                      plot_type="dot", 
                      color_bar_label='Feature Value',
                      sort=False, 
                      cmap=plt.cm.RdYlBu_r, 
                      color_bar=color_bar_ )

    plt.tick_params(axis='both', labelsize=14)

    if i != 1 :
        plt.yticks( ticks=plt.yticks()[0], labels=["" for x in explanatory_variable_names] )

    plt.plot( [0, 0], [-6, 100], "-", color="dimgrey", linewidth=1.5 )

    if i == 2 :
        plt.xlabel("Distribution of influences with respect to feature value", fontsize=16)
    else :
        plt.xlabel("")
    
    plt.title("Wave " + str(i), fontsize=18)
    plt.grid(axis="y", linewidth=0.3, linestyle="-") 

    
    
    




plt.figure( figsize=(16, 12) )

for i in range(1, 4) :
    plt.subplot(1, 3, i)
    fct_shap_plot(i)

plt.tight_layout()
plt.show()


# ### Univariate views




### Choose an explanatory variable name
col = "Age" 

xlim_global = pd.concat( [ dict_X[i] for i in range(1, 4) ] )[col]
xlim_global = ( xlim_global.min()*0.95, xlim_global.max()*1.05 )

ylim_global = pd.concat( [ dict_shap_values[i] for i in range(1, 4) ] )[col]
ylim_global = ( ylim_global.min()*0.95, ylim_global.max()*1.05 )





fig, ax = plt.subplots( 1, 3, figsize=(25, 9) )


for i in range(1, 4) :
    ax[i-1].scatter( dict_X[i][col], dict_shap_values[i][col], color=list_colors[i-1], marker=list_markers[i-1] )

    ax[i-1].set_xlabel(col + " value", fontsize=20)
    ax[i-1].set_ylabel("Influence (%)", fontsize=20)
    ax[i-1].set_title("Wave " + str(i), fontsize=22)

    ax[i-1].grid(axis="both", linewidth=0.5, linestyle="-")
    
    ax[i-1].set_xlim(xlim_global)
    ax[i-1].set_ylim(ylim_global)
    
    
    xlim_ = ax[i-1].get_xlim()
    ax[i-1].plot( xlim_, [0, 0], linewidth=1.4, color="grey" )
    ax[i-1].set_xlim( xlim_ )
    
    
    ax[i-1].tick_params(axis='both', labelsize=17)


plt.tight_layout()
plt.show()


# ### Bivariate views




### Choose two explanatory variable names
col_x = "Age"
col_y = "Diabetes"


df_global = pd.concat( [ dict_shap_values[i] for i in range(1, 4) ] )[ [col_x, col_y] ]

xlim_global = ( df_global[col_x].min()*0.95, df_global[col_x].max()*1.05 )
ylim_global = ( df_global[col_y].min()*0.95, df_global[col_y].max()*1.05 )





fig, ax = plt.subplots( 1, 3, figsize=(22, 8) )


for i in range(1, 4) :
    min_col_x = dict_X[i][col_x].min() ; max_col_x = dict_X[i][col_x].max()
    list_color_temp = [ plt.cm.get_cmap('RdYlBu_r')( (x - min_col_x) / (max_col_x - min_col_x) ) for x in dict_X[i][col_x] ]

    min_col_y = dict_X[i][col_y].min() ; max_col_y = dict_X[i][col_y].max()
    list_markersize_temp = [ 20 + 60*( (y - min_col_y) / (max_col_y - min_col_y) ) for y in dict_X[i][col_y] ]

    
    ax[i-1].scatter( dict_shap_values[i][col_x], dict_shap_values[i][col_y], color=list_color_temp, s=list_markersize_temp )

    ax[i-1].set_xlabel(col_x + " influence (%)", fontsize=16)
    ax[i-1].set_ylabel(col_y + " influence (%)", fontsize=16)

    ax[i-1].set_title("Wave " + str(i), fontsize=18)

    ax[i-1].grid(axis="both", linewidth=0.5, linestyle="-")

    ax[i-1].set_xlim(xlim_global)
    ax[i-1].set_ylim(ylim_global)
    
    
    xlim_ = ax[i-1].get_xlim()
    ax[i-1].plot( xlim_, [0, 0], linewidth=1.4, color="grey" )
    ax[i-1].set_xlim( xlim_ )

    ylim_ = ax[i-1].get_ylim()
    ax[i-1].plot( [0, 0], ylim_, linewidth=1.4, color="grey" )
    ax[i-1].set_ylim( ylim_ )
    
    
    ax[i-1].tick_params(axis='both', labelsize=12)
    
    
plt.tight_layout()
plt.show()













