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

# Define o título do Dashboard
st.title("APP para compra e venda Mini Índice ou Mini Dólar")
st.header("by Vinícius B. Paiva ([viniciusbarbosapaiva](https://www.linkedin.com/in/vinicius-barbosa-paiva/))")

# Definindo o código dos índices (Provisório)
st.sidebar.markdown('# Propriedades do APP')
indices = ('WIN$','WDO$')

# Definindo qual Índice usaremos por vez
indice_selecionado = st.sidebar.selectbox('Qual será o ativo?', indices)

# Defindo o timeframe
timeframes = {'1min': mt.TIMEFRAME_M1,
              '2min': mt.TIMEFRAME_M2,
              '5min': mt.TIMEFRAME_M5,
              '10min': mt.TIMEFRAME_M10,
              '15min': mt.TIMEFRAME_M15,
              '30min': mt.TIMEFRAME_M30,
              '1h': mt.TIMEFRAME_H1,
              '4h':mt.TIMEFRAME_H4}
timeframe_selecionado = st.sidebar.select_slider('Escolha o timeframe onde o BOT irá realizar as operações:',
                                         list(timeframes.keys()))

# Definindo a hora início e fim que será coletado os dados
import datetime
hora_inicio = st.sidebar.time_input('Horário inicial para coleta dos dados',
                                    datetime.time(9, 00))
hora_fim = st.sidebar.time_input('Horário final para coleta dos dados',
                                    datetime.time(17, 00))

# Definindo as datas de início e fim
data_inicio = st.sidebar.date_input('Data inicial para coleta dos dados',
                               datetime.date(2021, 5, 1))
data_fim = st.sidebar.date_input('Data final para coleta dos dados',
                               datetime.date(2021, 5, 28))

# Data de início e fim
dias_inicio = pd.date_range(start=datetime.datetime.combine(data_inicio,hora_inicio), end=datetime.datetime.combine(data_fim,hora_inicio),tz=utc_timezone)
dias_fim = pd.date_range(start=datetime.datetime.combine(data_inicio,hora_fim), end=datetime.datetime.combine(data_fim,hora_fim),tz=utc_timezone)

# Extraindo os dados
st.cache()
mensagem = st.text('Carregando os dados...')
dados = coletando_preco(indice_selecionado,dias_inicio,
                        timeframes[timeframe_selecionado],dias_fim)
mensagem.text('Carregando os dados...Concluído!')

# Sub Título
st.subheader('Visualização dos Dados Brutos')
st.write(dados.head())
st.write(dados.tail())

# Função para o plot dos dados brutos
def plot_dados_brutos():
    st.subheader('Preço de Abertura e Fechamento do {}'.format(indice_selecionado))
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=dados['time'],
                             open=dados['open'],
                             high=dados['high'],
                             low=dados['low'],
                             close=dados['close'],
                             name='Preço de Abertura do {}'.format(indice_selecionado)))
    
    fig.layout.update(xaxis_rangeslider_visible=True) #title_text='Preço de Abertura e Fechamento do {}'.format(indice_selecionado)
    st.plotly_chart(fig)

# Executa a função
plot_dados_brutos()

# Texto Previsão
st.subheader('Previsão de Machine Learning')

# Preparando os dados para as previsões
df_treino = dados[['time', 'close']]
df_treino = df_treino.rename(columns={'time':'ds', 'close':'y'})

# Criando o modelo
model = Prophet()

# Condicional para treinamento do modelo
treinar = (True,False)
mensagem_treinamento = st.text('Modelo ainda não está treinado!')
treinar_modelo = st.sidebar.selectbox("Treinar o modelo?",treinar)

# Treinando o modelo
if mensagem_treinamento == True:
    model.fit(df_treino)
    mensagem_treinamento.text('Modelo treinado!')

# Definindo o horizonte de previsão
dia_futuro = data_fim + + datetime.timedelta(days=1)
dia_futuro_inicio = datetime.datetime.combine(dia_futuro,hora_inicio)
dia_futuro_fim = datetime.datetime.combine(dia_futuro,hora_fim)
teste = pd.date_range(start = dia_futuro_inicio, end= dia_futuro_fim, freq=list(timeframes.keys())[list(timeframes.values()).index(timeframes[timeframe_selecionado])])

list(timeframes.keys())[list(timeframes.values()).index(timeframes[timeframe_selecionado])]
mt.TIMEFRAME_H1
mt.TIMEFRAME_H4
datetime.datetime.combine(data_inicio,hora_inicio)
pd.to_datetime(mt.TIMEFRAME_H1, unit='s').time()
