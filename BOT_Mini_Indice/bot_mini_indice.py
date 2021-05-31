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
    select = mt.symbol_select(ativo,True)
    if not select:
        print('O ativo {} não existe. Verificar se o código está correto. Coódigo do error = {}'.format(ativo, mt.last_error()))
    else:
        print('O ativo {} existe.'.format(ativo))
        #date_datetime = datetime.strptime(data, '%d/%m/%y %H:%M:%S')
        rates = mt.copy_rates_range(ativo,timeframe,data_0,data_1)
        print('Os preços foram carregados com sucesso')
        precos = pd.DataFrame(rates)
        precos['time'] = pd.to_datetime(precos['time'], unit='s')
        print('Os preços foram convertidos com sucesso.')
        print('Total de {} registros'.format(len(precos)))
    return precos   

# Engenharia de atributos
def engenharia_de_atributo(ohlc,ma_period,me_period, bollinger_period,base):
    df = ohlc.copy() #Copiando o dataset
    df = df.set_index('time', drop=True) #Configurando a coluna time para index
    df['ma20high'] = df['high'].rolling(ma_period).mean() #Calculando a média móvel da máxima
    df['ma20low'] = df['low'].rolling(ma_period).mean() #Calculando a média móvel da mínima
    df['me9close'] = df['close'].ewm(span=me_period, adjust=False, min_periods=me_period).mean() #Calculando a média móvel exponencial do fechamento
 
    #Cálculodas Bandas de Bollinger
    typical_price = ((df['high']+df['low']+df['close'])/3).apply(lambda x: base * np.round(x/base)) 
    df['bbupper'] = typical_price.rolling(bollinger_period).mean()+2*typical_price.rolling(bollinger_period).std()
    df['bblow'] = typical_price.rolling(bollinger_period).mean()-2*typical_price.rolling(bollinger_period).std()
    
    #Retirando os valores N.A
    df = df.dropna()
        
    #Arredondando os preços
    for i in df[['ma20high','ma20low','me9close', 'bbupper','bblow']]:
        df[i] = df[i].apply(lambda x: base * np.round(x/base)) 
    
    return df    
    
# Abrindo o terminal do MetaTrader 5
login_verication()

# Coletando os preços de Mini Índice
utc_timezone = pytz.timezone('Etc/UTC')
win = 'WIN$'
wdo = 'WDO$'
dias_inicio = pd.date_range(start=datetime(2021,5,1,9,2,00, tzinfo=utc_timezone), end=datetime(2021,5,28, 9,2,00, tzinfo=utc_timezone)).tolist()
dias_fim = pd.date_range(start=datetime(2021,5,1,17,00,00, tzinfo=utc_timezone), end=datetime(2021,5,28, 17,00,00, tzinfo=utc_timezone)).tolist()

print(date_from, date_to)
timeframe= mt.TIMEFRAME_M1
ohlc_win = coletando_preco(win,date_from,timeframe,date_to)
ohlc_wdo = coletando_preco(wdo,date_from,timeframe,date_to)

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






