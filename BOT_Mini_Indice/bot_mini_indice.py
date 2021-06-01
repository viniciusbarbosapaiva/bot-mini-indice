import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import MetaTrader5 as mt
from datetime import datetime
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
import pytz
import calendar
from My_Regression_Class import MyRegressionClass
import matplotlib.pyplot as plt
sns.set(style='white', palette='deep')

#Funções para iniciar o MetaTrader 5 
def initialize_metatrader():
    print('Versão:', mt.__version__)
    print('Autor:', mt.__author__)
    
    if not mt.initialize():
        print('Initialize() falhou ao iniciar, código do error= ',mt.last_error())
        mt.shutdown()
    else:
        print('Initialize() funcionou.')
        print('Terminal Informação: ', mt.terminal_info())
        print('MetaTrader versão: ', mt.version())

def login_verication():
    initialize_metatrader()
    user= 50390953
    senha = 'oY3a3X'
    print('Usuário é: {}'.format(user))
    authorized = mt.login(user, password=senha)
    
    if authorized:
        print('Authorized ok')
        initialize_metatrader()
        print(mt.account_info)
        print('Mostrando informação da conta:')
        account_info_dict = mt.account_info()._asdict() 
        for prop in account_info_dict:
            print('   {} = {}'.format(prop,account_info_dict[prop]))
        
    else:
        print('Inicialização falhou ao conectar na conta {}, código do error= {}'.format(user, mt.last_error()))
        mt.shutdown()

# Função para coletar preços
def coletando_preco(ativo, data_0, timeframe, data_1):
    'data_0 e data_1 são em pd.date_range'
    
    select = mt.symbol_select(ativo,True)
    work_day = []
    weekend_day = []
    if not select:
        print('O ativo {} não existe. Verificar se o código está correto. Coódigo do error = {}'.format(ativo, mt.last_error()))
    else:
        print('O ativo {} existe.'.format(ativo))
        for i in np.arange(0,len(data_0)):
    
            if calendar.day_name[data_0[i].weekday()] =='Saturday' or calendar.day_name[data_0[i].weekday()] =='Sunday':
                weekend_day.append(i)
    
            else:
                work_day.append(i)
    data_0 = data_0[work_day]
    data_1 = data_1[work_day]
    
    precos = pd.DataFrame()
    for i in np.arange(0,len(data_0)):
        rates_unico = mt.copy_rates_range(ativo,timeframe,data_0[i].to_pydatetime(),data_1[i].to_pydatetime())
        df_unico = pd.DataFrame(rates_unico)
        precos= pd.concat([precos,df_unico]) 
        
    precos['time'] = pd.to_datetime(precos['time'], unit='s') 
    print('Os preços foram carregados com sucesso')
    print('Total de {} registros'.format(len(precos)))
    print('A coluna Time foi convertida pata Datetime.')
    return precos   

# Engenharia de atributos
def engenharia_de_atributo(ohlc,ma_period,me_period, bollinger_period,base):
    df = ohlc.copy() #Copiando o dataset
    df = df.set_index('time', drop=True) #Configurando a coluna time para index
    df['mahigh'] = df['high'].rolling(ma_period).mean() #Calculando a média móvel da máxima
    df['malow'] = df['low'].rolling(ma_period).mean() #Calculando a média móvel da mínima
    df['meclose'] = df['close'].ewm(span=me_period, adjust=False, min_periods=me_period).mean() #Calculando a média móvel exponencial do fechamento
 
    #Cálculodas Bandas de Bollinger
    typical_price = ((df['high']+df['low']+df['close'])/3).apply(lambda x: base * np.round(x/base)) 
    df['bbupper'] = typical_price.rolling(bollinger_period).mean()+2*typical_price.rolling(bollinger_period).std()
    df['bblow'] = typical_price.rolling(bollinger_period).mean()-2*typical_price.rolling(bollinger_period).std()
    
    #Retirando os valores N.A
    #df = df.dropna()
        
    #Arredondando os preços
    for i in df[['mahigh','malow','meclose', 'bbupper','bblow']]:
        df[i] = df[i].apply(lambda x: base * np.round(x/base)) 
    
    return df   

# Plotagem de barra
def autolabel_without_pct(rects,ax): #autolabel
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy = (rect.get_x() + rect.get_width()/2, height),
                    xytext= (0,3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, rotation=90)
    
# Abrindo o terminal do MetaTrader 5
login_verication()

# Coletando os preços de Mini Índice
utc_timezone = pytz.timezone('Etc/UTC')
win = 'WIN$'
wdo = 'WDO$'
dias_inicio = pd.date_range(start=datetime(2021,5,1,9,2,00, tzinfo=utc_timezone), end=datetime(2021,5,28, 9,2,00, tzinfo=utc_timezone))
dias_fim = pd.date_range(start=datetime(2021,5,1,17,00,00, tzinfo=utc_timezone), end=datetime(2021,5,28, 17,00,00, tzinfo=utc_timezone))
timeframe= mt.TIMEFRAME_M1

ohlc_win = coletando_preco(win,dias_inicio,timeframe,dias_fim)
ohlc_wdo = coletando_preco(wdo,dias_inicio,timeframe,dias_fim)

# Informação sobre o dataset gerado
ohlc_win.columns
ohlc_win.info()
estatistica_win = ohlc_win.describe()

ohlc_wdo.columns
ohlc_wdo.info()
estatistica_wdo = ohlc_wdo.describe()

# Engenharia de atributos
period_ma = 20
period_me = 9
period_bollinger = 20
base_win = 5
base_wdo = 0.5

df_win = engenharia_de_atributo(ohlc_win,period_ma,period_me,period_bollinger,base_win)
df_wdo = engenharia_de_atributo(ohlc_wdo,period_ma,period_me,period_bollinger,base_wdo)

# Plotando o gráfico
fig = make_subplots(rows=2,cols=2, row_heights=[0.8,0.2])
fig.add_trace(go.Scatter(x=df_win.index,
                         y=df_win['close'],
                         line=dict(color='black'),
                         mode='lines',
                         name='Preço de fechamento do Mini Índice'), row=1, col=1)
fig.add_trace(go.Scatter(x=df_wdo.index,
                         y=df_wdo['close'],
                         line=dict(color='black'),
                         mode='lines',
                         name='Preço de fechamento do Mini Dólar'), row=1, col=2)
fig.add_trace(go.Bar(x=df_win.index,
                     y=df_win['real_volume'],
                     name='Volume Real',
                     marker=dict(line=dict(color='black'))), row=2, col=1)
fig.add_trace(go.Bar(x=df_wdo.index,
                     y=df_wdo['real_volume'],
                     name='Volume Real',
                     marker=dict(line=dict(color='black'))), row=2, col=2)
fig.update_layout(title='Cotação dos dias {} - {}'.format(date_from.date(),date_to.date()))
plot(fig)

# Alterando os nomes das colunas
colunas = df_win.columns
colunas_win = [i+'_win' for i in colunas]
colunas_wdo = [i+'_wdo' for i in colunas]
df_win.columns = colunas_win
df_wdo.columns = colunas_wdo

# Juntando as colunas
df = pd.concat([df_win,df_wdo], axis=1)
df.to_csv('dataframe_completo.csv')

# Shift na coluna
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
X_train = X[:int(indice_treino)+1]
X_test =  X[int(indice_treino):int(indice_teste)+1]

y_train = y[:int(indice_treino)+1]
y_test = y[int(indice_treino):int(indice_teste)+1]

print(len(X_train), len(X_test), len(y_train), len(y_test))

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_train = pd.DataFrame(sc_x.fit_transform(X_train), columns=X.columns.values)
X_test = pd.DataFrame(sc_x.transform(X_test), columns=X.columns.values)
y_test = pd.DataFrame(sc_y.fit_transform(np.array(y_test.iloc[:]).reshape(len(y_test),1)), columns=[y_test.name])
y_train = pd.DataFrame(sc_y.transform(np.array(y_train.iloc[:]).reshape(len(y_train),1)), columns=[y_train.name])

#### Model Building ####
### Comparing Models
## Multiple Linear Regression Regression
from sklearn.linear_model import LinearRegression
k = X_test.shape[1]
n = len(X_test)
lr_regressor = LinearRegression(fit_intercept=True)
lr_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = lr_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

results = pd.DataFrame([['Multiple Linear Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

## Ridge Regression
from sklearn.linear_model import Ridge
rd_regressor = Ridge(alpha=50)
rd_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = rd_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['Ridge Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

## Lasso Regression
from sklearn.linear_model import Lasso
la_regressor = Lasso(alpha=500)
la_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = la_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['Lasso Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

## Polynomial Regressor
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X_train)
lr_poly_regressor = LinearRegression(fit_intercept=True)
lr_poly_regressor.fit(X_poly, y_train)

# Predicting Test Set
y_pred = lr_poly_regressor.predict(poly_reg.fit_transform(X_test))
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['Polynomial Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

## Suport Vector Regression 
'Necessary Standard Scaler '
from sklearn.svm import SVR
svr_regressor = SVR(kernel = 'rbf')
svr_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = svr_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['Support Vector RBF', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

## Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = dt_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['Decision Tree Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

## Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=300, random_state=0)
rf_regressor.fit(X_train,y_train)

# Predicting Test Set
y_pred = rf_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['Random Forest Regression', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

## Ada Boosting
from sklearn.ensemble import AdaBoostRegressor
ad_regressor = AdaBoostRegressor()
ad_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = ad_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['AdaBoost Regressor', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

##Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
gb_regressor = GradientBoostingRegressor()
gb_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = gb_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['GradientBoosting Regressor', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

##Xg Boosting
from xgboost import XGBRegressor
xgb_regressor = XGBRegressor()
xgb_regressor.fit(X_train, y_train)

# Predicting Test Set
y_pred = xgb_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['XGB Regressor', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)

##Ensemble Voting regressor
from sklearn.ensemble import VotingRegressor
voting_regressor = VotingRegressor(estimators= [('lr', lr_regressor),
                                                ('rd', rd_regressor),
                                                ('la', la_regressor),
                                                ('lr_poly', lr_poly_regressor),
                                                ('svr', svr_regressor),
                                                ('dt', dt_regressor),
                                                ('rf', rf_regressor),
                                                ('ad', ad_regressor),
                                                ('gr', gb_regressor),
                                                ('xg', xgb_regressor)])

for clf in (lr_regressor,lr_poly_regressor,svr_regressor,dt_regressor,
            rf_regressor, ad_regressor,gb_regressor, xgb_regressor, voting_regressor):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, metrics.r2_score(y_test, y_pred))

# Predicting Test Set
y_pred = voting_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
MAPE = np.array(np.mean( np.abs( (y_test-np.array(y_pred).reshape(len(y_pred),1)) / y_test  )  )*100)[0]

model_results = pd.DataFrame([['Ensemble Voting', mae, mse, rmse, r2,adj_r2,MAPE]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score', 'MAPE'])

results = results.append(model_results, ignore_index = True)  

#The Best Classifier
print('The best regressor is:')
print('{}'.format(results.sort_values(by='Adj. R2 Score',ascending=False).head(5)))

#Applying K-fold validation
from sklearn.model_selection import cross_val_score
def display_scores (scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard:', scores.std())

lin_scores = cross_val_score(estimator= lr_regressor, X=X, y=y, 
                             scoring= 'neg_mean_squared_error',cv=10) # Era X_train e y_train. Passei para X e y.
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

#Validation of model. Analyzing the loss function
from yellowbrick.regressor import ResidualsPlot
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(1,1,1)
residual = ResidualsPlot(lr_regressor, ax=ax1)
residual.fit(X_train, y_train.values.flatten())  
residual.score(X_test, y_test.values.flatten())  
residual.show() 





