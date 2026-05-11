import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv('bengaluru_house_prices.csv')
df
df.groupby('area_type')['area_type'].agg('count')

df_o=df.drop(['society','balcony','availability','area_type'],axis='columns')		
df_o
df_o.isna().sum()
df_o['bath']=df_o['bath'].fillna(1)
df_o.dropna(inplace=True)
df_o.isna().sum()
df_o['size'].unique()
df_o['BHK']=df_o['size'].apply(lambda x:int(x.split(' ')[0]))
df_o
df_o['BHK'].unique()
df_o[df_o.BHK>10]
df_o['total_sqft'].unique()
def to_float(x):
    try:
        float(x)
        return True
    except:
        return False
 
df_o.isna().sum()
df_o[~df_o['total_sqft'].apply(to_float)].head(20)

df_o.shape
def convert_sqft_to_num(x):
 tokens = x.split('-')
 if len(tokens) == 2:
   return (float(tokens[0]) + float(tokens[1]))//2
 try:
   return float(x)
 except:
   return None
df1=df_o.copy()
df1['total_sqft']=df_o['total_sqft'].apply(convert_sqft_to_num)
df1
df1.isna().sum()
df1.dropna(inplace=True)
df1.isna().sum()
df2=df1.copy()
df2
df2['PPSQ']=df1['price']*100000/df1['total_sqft']
df2
len(df2.location.unique())
df2.location=df2.location.apply(lambda x: x.strip())
location_stats=df2.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats.head(50)
len(location_stats[location_stats<10])

 
location_stats_less_than=location_stats[location_stats<10]
location_stats_less_than
df2.location=df2.location.apply(lambda x: 'other' if x in location_stats_less_than else x) 
df2.location
df2.location.unique()
len(df2.location.unique())
df2
df2[df2.total_sqft/df2.BHK<300].head()
df2.shape
df3=df2[~(df2.total_sqft/df2.BHK<300)]
df3.shape
df3.PPSQ.describe()
def remove_pps_outlier(df):
 df_out=pd.DataFrame()
 for key,subdf in df.groupby('location'):
     m=np.mean(subdf.PPSQ)
     st=np.std(subdf.PPSQ)
     reduced_df=subdf[(subdf.PPSQ>(m-st)) & (subdf.PPSQ<=(m+st))]
     df_out=pd.concat([df_out,reduced_df],ignore_index=True)
 return df_out
df4=remove_pps_outlier(df3)
df4.shape

df4=remove_pps_outlier(df3)
df4.shape

df4.head(50)
def plotter_scat(df,location):
 bhk2=df[(df.location==location) & (df.BHK==2)]
 bhk3=df[(df.location==location) & (df.BHK==3)]
 plt.scatter(bhk2.total_sqft,bhk2.PPSQ,color='orange')
 plt.scatter(bhk3.total_sqft,bhk3.PPSQ,color='red')
 
plotter_scat(df4,"Hebbal")
df4.BHK.shape[0]
def remove_bhk_outliers (df) :
 exclude_indices = np.array([])
 for location, location_df in df.groupby('location'):
  bhk_stats = {}
  for bhk, bhk_df in location_df.groupby ('BHK'):
      bhk_stats [bhk]  = {
     'mean': np.mean(bhk_df.PPSQ),
     'std': np.std(bhk_df.PPSQ),
     'count': bhk_df.shape[0]
      }
  for bhk, bhk_df in location_df.groupby('BHK'):
      stats = bhk_stats.get(bhk-1)
      if stats and stats['count']>=5:
          exclude_indices= np.append(exclude_indices, bhk_df[bhk_df.PPSQ<(stats['mean'])]. index.values)
 return df.drop(exclude_indices,axis='index')
df5=remove_bhk_outliers(df4)
df5.shape
plotter_scat(df5,"Hebbal")
import matplotlib.pyplot as plt
plt.hist(df5.PPSQ)

df5.bath.unique()
df5[df5.bath>10]
df6=df5[df5.bath<df5.BHK+2]
df6
df7=df6.drop(['size','PPSQ','bath'],axis='columns')
df7
dummies=pd.get_dummies(df7.location)
dummies
df8=pd.concat([df7,dummies.drop('other',axis='columns')],axis='columns')
df8
df9=df8.drop('location',axis='columns')
df9
x=df9.drop('price',axis='columns')
y=df9.price
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10)
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(x_train,y_train)
lr_clf.score(x_test,y_test)
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), x, y, cv=cv)
x.isna().sum()
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(x,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                
                'fit_intercept': [True, False],
                'positive': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(x,y)
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(x.columns==location)[0][0]

    X = np.zeros(len(x.columns))
    X[0] = sqft
    X[1] = bath
    X[2] = bhk
    if loc_index >= 0:
        X[loc_index] = 1

    return lr_clf.predict([X])[0]
predict_price('1st Phase JP Nagar',1000,2,2)
lr_clf.score(x_test,y_test)
predict_price('Indira Nagar',1000,3,3)
