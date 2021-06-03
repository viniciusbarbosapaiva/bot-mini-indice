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
import time
from PIL import Image
from sklearn.metrics import r2_score

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
    if ativo == 'WIN$':
        precos['close'] = precos['close'].apply(lambda x: 5 * np.round(x/5))
    else:
        precos['close'] = precos['close'].apply(lambda x: 0.5 * np.round(x/0.5))
    print('Os preços foram carregados com sucesso')
    print('Total de {} registros'.format(len(precos)))
    print('A coluna Time foi convertida pata Datetime.')
    return precos   

# Função para o Expert Advisor
def expert():
    codigo = ['WIN', 'WDO']
    st.selectbox('Qual o código?',codigo)
    st.button('Voltar para o Painel Principal', key='back_again')

# Abrindo o terminal do MetaTrader 5
login_verication()

# Definindo o timezone
utc_timezone = pytz.timezone('Etc/UTC')

# Configurando layout página
st.set_page_config(layout="wide")

# Define o título do Dashboard
image = Image.open(r'C:\Users\eng2\Desktop\bot-mini-indice\BOT_Mini_Indice\logo\LOGO 01-03.png')
image = image.resize((200, 200), Image.ANTIALIAS)
st.markdown('---')
c1,c2,c3 = st.beta_columns((1,1,1))
c1.image(image)
c2.title("APP para compra e venda Mini Índice ou Mini Dólar")
c2.subheader("Autor: Vinícius B. Paiva ([LinkedIn](https://www.linkedin.com/in/vinicius-barbosa-paiva/)) ([GitHub](https://github.com/viniciusbarbosapaiva))")

# Definindo o código dos índices (Provisório)
st.sidebar.markdown('# Propriedades do APP')
indices = ('WDO$','WIN$')

# Definindo qual Índice usaremos por vez
indice_selecionado = st.sidebar.selectbox('Qual será o ativo?', indices)

# Defindo o timeframe
timeframes = {'1min': mt.TIMEFRAME_M1,
              '2min': mt.TIMEFRAME_M2,
              '5min': mt.TIMEFRAME_M5,
              '10min': mt.TIMEFRAME_M10,
              '15min': mt.TIMEFRAME_M15,
              '30min': mt.TIMEFRAME_M30,
              '1h': mt.TIMEFRAME_H1}
timeframe_selecionado = st.sidebar.selectbox('Escolha o timeframe onde o BOT irá realizar as operações:',
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
                               datetime.date(2021, 6, 1))

# Data de início e fim
dias_inicio = pd.date_range(start=datetime.datetime.combine(data_inicio,hora_inicio), end=datetime.datetime.combine(data_fim,hora_inicio),tz=utc_timezone)
dias_fim = pd.date_range(start=datetime.datetime.combine(data_inicio,hora_fim), end=datetime.datetime.combine(data_fim,hora_fim),tz=utc_timezone)

# Extraindo os dados
st.cache()
dados = coletando_preco(indice_selecionado,dias_inicio,
                            timeframes[timeframe_selecionado],dias_fim)

# Função para o plot dos dados brutos
def plot_dados_brutos():
    #st.subheader('Preço de Abertura e Fechamento do {}'.format(indice_selecionado))
    fig = go.Figure(layout=go.Layout(height=600, width=1800))
    fig.add_trace(go.Candlestick(x=dados['time'],
                             open=dados['open'],
                             high=dados['high'],
                             low=dados['low'],
                             close=dados['close'],
                             name='Preço de Abertura do {}'.format(indice_selecionado)))
    
    fig.layout.update(xaxis_rangeslider_visible=True,
                      title_text='Preço de Abertura e Fechamento do {}'.format(indice_selecionado)) #title_text='Preço de Abertura e Fechamento do {}'.format(indice_selecionado)
    st.plotly_chart(fig)

# Tabela dos Dados
col4,col5,col6 = st.beta_columns((1,1,1))
col4.subheader('Visualização dos Primeiros Cinco Dias do Mês {}.'.format(data_inicio.month))
col4.write(dados[['time','close']].head())
col5.subheader('Parâmetros do Expert Advisor (Robô)')
expert_advisor = col5.button('Click Aqui para Configurar o EA')
st.markdown('---')
if expert_advisor:
    expert()
    st.markdown('---')
else:
    # Executa a função plot 
    plot_dados_brutos()
    st.markdown('---')
col6.subheader('Visualização dos Últimos Cinco Dias do Mês {}.'.format(data_fim.month))
col6.write(dados[['time','close']].tail()) 
    
# Texto Previsão
st.subheader('Previsão de Machine Learning')
mensagem_treinamento = st.text('Modelo ainda não está treinado!')

# Preparando os dados para as previsões
df_treino = dados[['time', 'close']]
df_treino = df_treino.rename(columns={'time':'ds', 'close':'y'})

# Criando o modelo
model = Prophet(changepoint_prior_scale=0.15,yearly_seasonality=True,daily_seasonality=True)

# Condicional para treinamento do modelo
treinar = (False,True)
treinar_modelo = st.sidebar.selectbox("Treinar o modelo?",treinar)

# Treinando o modelo
if treinar_modelo == True:
# Prepara as datas futuras para as previsões
    dia_futuro = data_fim + datetime.timedelta(days=1)
    dia_futuro_inicio = datetime.datetime.combine(dia_futuro,hora_inicio)
    dia_futuro_fim = datetime.datetime.combine(dia_futuro,hora_fim)
    dia_futuro_range = pd.date_range(start = dia_futuro_inicio, end= dia_futuro_fim, freq=list(timeframes.keys())[list(timeframes.values()).index(timeframes[timeframe_selecionado])])
    dia_futuro_range.names = ['ds']
    dia_futuro_range = dia_futuro_range.to_frame()
    dia_futuro_range.index = [i for i in np.arange(0,len(dia_futuro_range))]

    futuro = pd.concat([df_treino['ds'].to_frame(),dia_futuro_range]).reset_index(drop=True)
    fit_model = model.fit(df_treino)    
    mensagem_treinamento.text('Modelo treinado!')
    
# Faz as previsões
    forecast = model.predict(futuro)
    if indice_selecionado == 'WIN$':
        forecast['yhat'] = forecast['yhat'].apply(lambda x: 5 * np.round(x/5))
        forecast['yhat'] = forecast['yhat'].apply(lambda x:np.float64(x))
    else:
        forecast['yhat'] = forecast['yhat'].apply(lambda x: 0.5 * np.round(x/0.5))
        forecast['yhat'] = forecast['yhat'].apply(lambda x:np.float64(x))
        
# Sub título
    st.markdown('---')
    st.subheader('Dados Previstos')
# Dados Previstos
    new_forecast = forecast.set_index('ds', drop=True)
    new_df_treino = df_treino.set_index('ds', drop=True)
    comparacao_real_previsao = pd.concat([new_df_treino.loc[data_inicio:data_fim,'y'],
                                          new_forecast.loc[data_inicio:data_fim,'yhat']], axis=1)
    comparacao_real_previsao = comparacao_real_previsao.rename(columns={'y':'Valor Real',
                                                                'yhat': 'Valor Previsto'})
    comparacao_real_previsao['Error'] = np.abs(comparacao_real_previsao['Valor Real']-comparacao_real_previsao['Valor Previsto'])
    c1,c2,c3 = st.beta_columns((0.5,1,0.5))
    c1.text("")
    st.markdown('---')
    c2.write(comparacao_real_previsao.tail(10))
    c3.text("")
# plotando os dados previstos
    metric_df = forecast.set_index('ds')[['yhat']].join(df_treino.set_index('ds').y).reset_index()
    metric_df.dropna(inplace=True)
    R2_score = r2_score(metric_df.y, metric_df.yhat)
    fig = go.Figure(layout=go.Layout(height=600, width=1800))
    fig.add_trace(go.Scatter(x = df_treino['ds'],
                             y = df_treino['y'],
                             mode='markers',
                             marker=dict(color='red'),
                             name='Preço Atual'))
    fig.add_trace(go.Scatter(x = forecast['ds'],
                             y = forecast['yhat'],
                             mode='markers',
                             marker=dict(color='blue'),
                             name='Preço Previsto'))
    fig.update_layout(title='O Modelo Obteve uma Acurácia de {}% para a Previsão do Preço no Timeframe de {} para o {}'.format(np.round(R2_score*100,2),
                      list(timeframes.keys())[list(timeframes.values()).index(timeframes[timeframe_selecionado])],
                      indice_selecionado),xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    st.markdown('---')
        #st.subheader('Previsão de {} para o {}'.format(list(timeframes.keys())[list(timeframes.values()).index(timeframes[timeframe_selecionado])],
         #    indice_selecionado))
        #grafico2 = plot_plotly(model,forecast, xlabel='Data', ylabel='Previsão')            
        #grafico2.update_traces(marker=dict(size=5,
         #                                  line=dict(width=2,
          #                                 color='DarkSlateGrey')),
           #                                selector=dict(mode='markers'))
    
        #st.plotly_chart(grafico2)
    
#datetime.datetime.now().time() - dia_futuro_range[i].to_pydatetime().time()

