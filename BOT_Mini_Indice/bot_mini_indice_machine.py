import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set(palette='deep', style='white')
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import My_Regression_Class

#Functions
def autolabel_without_pct(rects,ax): #autolabel
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy = (rect.get_x() + rect.get_width()/2, height),
                    xytext= (0,3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, rotation=90)

# Importando o arquivo
ohlc = pd.read_csv('dataframe_completo.csv', index_col='time')
ohlc.head(5)

# Informação dos dados
ohlc.info()

# Shift na coluna
df = ohlc.copy() 
df['close_win'] = df['close_win'].shift(-1)

# Removendo valores NA
df = df.dropna()

# Definindo o percentual de treino, teste e  validação
indice_total = len(df)
indice_treino = np.round(indice_total*.70)
indice_teste = np.round(indice_total*.20 + indice_treino)
inndice_validacao = np.round(indice_total*.10 + indice_teste) 

# Definindo X e y
df.columns
X = df.drop('close_win', axis=1)
y = df['close_win']

#Applying Statmodel (p_value <=0.05)
import statsmodels.api as sm

Xc = sm.add_constant(X)
model = sm.OLS(y, Xc)
model_v1 = model.fit()
model_v1.summary()

# Feature selection with extra trees classifier
from sklearn.ensemble import ExtraTreesClassifier
modelo = ExtraTreesClassifier()
modelo.fit(X, y)

print(X.columns)
print(modelo.feature_importances_) 

mean_feature = np.mean(modelo.feature_importances_)
label = X.columns
y_mean = [mean_feature] * len(label) 
ind = np.arange(0,len(label))
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_title('The Feature Importances Plot')
rect = ax.bar(label,np.round(modelo.feature_importances_,3), label='Data')
ax.grid(b=True, which='major', linestyle='--')
ax.set_xlabel('Model Features')
ax.set_ylabel('Level of Importance')
ax.set_xticks(ind)
ax.tick_params(axis='x', labelrotation=90)
autolabel_without_pct(rect,ax)
ax.plot(label,y_mean, color='red', label='Mean', linestyle='--')
ax.legend()
plt.plot()

#Selecting the most relevant features
features_importance = dict(zip(X.columns,modelo.feature_importances_))
features_importance = pd.DataFrame(features_importance, index=[0])
features_importance_names = [features_importance.columns[i] for i in np.arange(0,len(features_importance.columns)) if features_importance.iloc[0,i] > mean_feature]
X = X[features_importance_names]

# Definindo treino e teste
X_train = X[:indice_treino+1]
X_test =  X[indice_treino:indice_teste+1]

y_train = y[:indice_treino+1]
y_test = y[indice_treino:indice_teste+1]


