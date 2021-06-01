import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import MetaTrader5 as mt
from datetime import datetime
import pytz
import calendar
import streamlit as st
import matplotlib.pyplot as plt
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

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

# Abrindo o terminal do MetaTrader 5
login_verication()

# Definindo o timezone
utc_timezone = pytz.timezone('Etc/UTC')

# Definindo o código dos índices (Provisório)
indices = ('WIN$','WDO$')

# Data de início e fim
dias_inicio = pd.date_range(start=datetime(2021,5,1,9,2,00, tzinfo=utc_timezone), end=datetime(2021,5,28, 9,2,00, tzinfo=utc_timezone))
dias_fim = pd.date_range(start=datetime(2021,5,1,17,00,00, tzinfo=utc_timezone), end=datetime(2021,5,28, 17,00,00, tzinfo=utc_timezone))
timeframe= mt.TIMEFRAME_M1

# Define o título do Dashboard
st.title("App para compra e venda Mini Índice ou Mini Dólar")
st.title("Dashboard em Tempo Rea")

ohlc_win = coletando_preco(win,dias_inicio,timeframe,dias_fim)
ohlc_wdo = coletando_preco(wdo,dias_inicio,timeframe,dias_fim)







