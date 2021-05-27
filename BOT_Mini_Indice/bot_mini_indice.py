import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import MetaTrader5 as mt
from datetime import datetime
import plotly.graph_objects as go
from plotly.offline import plot


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
def coletando_preco(ativo, data, timeframe, count):
    select = mt.symbol_select(ativo,True)
    if not select:
        print('O ativo {} não existe. Verificar se o código está correto. Coódigo do error = {}'.format(ativo, mt.last_error()))
    else:
        print('O ativo {} existe.'.format(ativo))
        #date_datetime = datetime.strptime(data, '%d/%m/%y %H:%M:%S')
        print(data)
        rates = mt.copy_rates_from(ativo,timeframe,data,count)
        print('Os preços foram carregados com sucesso')
        precos = pd.DataFrame(rates)
        precos['time'] = pd.to_datetime(precos['time'][0], unit='s')
        print('Os preços foram convertidos com sucesso')
    return precos
    
# Abrindo o terminal do MetaTrader 5
login_verication()

# Coletando os preços de Mini Índice
ativo = 'WIN$'
date = datetime(2021,5,27)
timeframe= mt.TIMEFRAME_M1
barras = 500
ohlc = coletando_preco(ativo,date,timeframe,barras)

# Informação sobre o dataset gerado
ohlc.columns
ohlc.info()
estatistica = ohlc.describe()

# Engenharia de atributos
df = ohlc.copy()
df['date'] = df['time'].apply(lambda x: x.date())

# Plotando o gráfico
fig = go.Figure(data=[go.Candlestick(x=df['date'],
                                        open=df['open'],
                                        high= df['high'],
                                        low=df['low'],
                                        close=df['close'])])
plot(fig)





