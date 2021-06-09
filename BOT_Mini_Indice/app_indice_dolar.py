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

def futuro_dia(fim_data):
    dia_acrescentar = 1
    futuro = fim_data + datetime.timedelta(days=dia_acrescentar)
    while calendar.day_name[futuro.weekday()] =='Saturday' or calendar.day_name[futuro.weekday()] =='Sunday':
        dia_acrescentar+=1
        futuro = data_fim + datetime.timedelta(days=dia_acrescentar)
    else:
        return futuro 

# Função para o Expert Advisor
def expert(indice_escolhido):
    codigo = indice_escolhido.split('$')[0]
    meses_win = ['G','J','M','Q','V','Z']
    meses_wdo = ['F', 'G', 'H','J','K','M','N,","Q','U','V','X','Z']
    c1,c2,c3 = st.beta_columns((1,1,1))
    c1.subheader('Código Vigente')
    if codigo == 'WIN':
        mes_codigo = c1.selectbox('Selecione o código refente ao vencimento', meses_win)
    else:
        mes_codigo = c1.selectbox('Selecione o código refente ao vencimento', meses_wdo)
    ano_contrato = datetime.datetime.now().strftime("%Y")[2:]
    c1.text(codigo+mes_codigo+ano_contrato)
    c2.subheader('Número de Contratos')
    c2.slider('Selecione Quantidade',1,100)
    c3.subheader('Stop Loss')
    c3.number_input('Pontos para Stop Loss', value=int(100))
    c4,c5,c6= st.beta_columns((1,1,1))
    c4.subheader('Take Profit')
    c4.number_input('Pontos para Take Profit', value=int(100))
    c5.subheader('Martingale')
    martingale_list = ['Não Utilizar', 'Sim. Moderado','Sim. Agressivo']
    c5.selectbox('Qual tipo de Martingale?',martingale_list)
    st.button('Voltar para o Painel Principal', key='back_again')
    
# Abrindo o terminal do MetaTrader 5
login_verication()

# Definindo o timezone
utc_timezone = pytz.timezone('Etc/UTC')

# Configurando layout página
st.set_page_config(layout="wide")

# Define o título do Dashboard
image = Image.open(r'LOGO 01-03.png')
image = image.resize((200, 200), Image.ANTIALIAS)
st.markdown('---')
c1,c2,c3 = st.beta_columns((1,1,1))
c1.image(image)
c2.title("APP para compra e venda Mini Índice ou Mini Dólar")
c2.subheader("Autor: Vinícius B. Paiva ([LinkedIn](https://www.linkedin.com/in/vinicius-barbosa-paiva/)) ([GitHub](https://github.com/viniciusbarbosapaiva))")

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
                               datetime.date(2021, 1, 1))
data_fim = st.sidebar.date_input('Data final para coleta dos dados',
                               datetime.date(2021, 6, 7))

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
symbol_info = mt.symbol_info(indice_selecionado)  
account_info_string = mt.account_info()  
col5.text('Código: {}'.format(symbol_info.description))
col5.text('Path: {}'.format(symbol_info.path))
col5.text('Moeda: {}'.format(account_info_string.currency))
col5.text('Valor Conta: R${}'.format(account_info_string.balance ))
col5.text('Capital: R${}'.format(account_info_string.equity))   
col5.text('Margem: R${}'.format(account_info_string.margin))  
col5.text('Lucro: R${}'.format(account_info_string.profit ))  
st.markdown('---')
if expert_advisor:
    expert(indice_selecionado)
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
treinar = ('Não','Sim')
treinar_modelo = st.sidebar.selectbox("Treinar o modelo?",treinar)

# Opção de treinar o modelo
if treinar_modelo == 'Sim':
# Prepara as datas futuras para as previsões
    dia_futuro = futuro_dia(data_fim)
    dia_futuro_inicio = datetime.datetime.combine(dia_futuro,hora_inicio)
    dia_futuro_fim = datetime.datetime.combine(dia_futuro,hora_fim)
    dia_futuro_range = pd.date_range(start = dia_futuro_inicio, end= dia_futuro_fim, freq=list(timeframes.keys())[list(timeframes.values()).index(timeframes[timeframe_selecionado])])
    dia_futuro_range.names = ['ds']
    dia_futuro_range = dia_futuro_range.to_frame()
    dia_futuro_range.index = [i for i in np.arange(0,len(dia_futuro_range))]
    futuro = pd.concat([df_treino['ds'].to_frame(),dia_futuro_range]).reset_index(drop=True)
    
# Treinando o modelo
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
    comparacao_real_previsao[['Valor Real', 'Valor Previsto']] = comparacao_real_previsao[['Valor Real', 'Valor Previsto']].apply(lambda x: np.round(x,2))
    comparacao_real_previsao['Error'] = np.abs(comparacao_real_previsao['Valor Real']-comparacao_real_previsao['Valor Previsto'])
    c1,c2,c3 = st.beta_columns((0.5,1,0.5))
    c1.text("")
    st.markdown('---')
    c2.write(comparacao_real_previsao.tail(10))
    c3.text("")
    analise_futuro = comparacao_real_previsao.to_csv('comparacao_real_previsto.csv')
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
    st.subheader('Previsão de {} para o {}'.format(list(timeframes.keys())[list(timeframes.values()).index(timeframes[timeframe_selecionado])],
                 indice_selecionado))
    grafico2 = plot_plotly(model,forecast, xlabel='Data', ylabel='Previsão')            
    grafico2.update_traces(marker=dict(size=5,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))
    
    # Analysing the first rule
    index_first = [i for i in np.arange(0,len(analise_futuro)) if analise_futuro['ds'][i].time() == datetime.time(9,00) or analise_futuro['ds'][i].time() == datetime.time(17,00)]
    first = analise_futuro.iloc[index_first,]
    
    st.plotly_chart(grafico2)
    
# Definindo função para enviar ordem
def position_open(symbol, order_type,volume,price,sl,tp,comment):
    if order_type != mt.ORDER_TYPE_BUY and order_type != mt.ORDER_TYPE_SELL:
        print(mt.TRADE_RETCODE_INVALID)
        print('Ordem Inválida')
        return(False)
    else:
        deviation = 50
        request = {
                "action": mt.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                    "type": order_type,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "deviation": deviation,
                    "magic": 234000,
                    "comment": "python script open",
                    "type_time": mt.ORDER_TIME_GTC,
                    "type_filling": mt.ORDER_FILLING_RETURN,
                    }   
        result= mt.order_send(request)
        return result
        

s = symbol_info.description
indice = s[s.find('(')+1:s.find(')')]
select = mt.symbol_select(indice,True)
point = mt.symbol_info(indice).point
price_buy = mt.symbol_info_tick(indice).ask
SL = 100
TP = 100
VOLUME = 1      
bsl = price_buy - SL * point
btp = price_buy + TP * point
deviation = 50
request = {
        "action": mt.TRADE_ACTION_DEAL,
        "symbol": indice,
        "volume": VOLUME,
        "type": mt.ORDER_TYPE_BUY,
        "price": price_buy,
        "sl": bsl,
        "tp": btp,
        "deviation": deviation,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt.ORDER_TIME_GTC,
        "type_filling": mt.ORDER_FILLING_RETURN,
        }   
result= mt.order_send(request)

position_open(indice,mt.ORDER_TYPE_BUY,VOLUME,price_buy,bsl,btp,'EA Test')

def buy(volume, symbol=None, price=0.0, sl=0.0, tp=0.0,comment=""):
    
    # Checando volume
    if volume <= 0:
        print(mt.TRADE_RETCODE_INVALID_VOLUME)
        return(False)
    if price==0:
        price=symbol_info.ask
    else:
        return position_open(symbol,mt.ORDER_TYPE_BUY,volume,price,sl,
                             tp,comment)
        
def sell(volume, symbol=None, price=0.0, sl=0.0, tp=0.0,comment=""):
    # Checando volume
    if volume <= 0:
        print(mt.TRADE_RETCODE_INVALID_VOLUME)
        return(False)
    if price==0:
        price=symbol_info.bid
    else:
        return position_open(symbol,mt.ORDER_TYPE_SELL,volume,price,sl,
                             tp,comment)

def open_orders(symbol):
    position = mt.positions_get(symbol)
    if position == None or position==0:
        print('Não há nenhuma posição aberta para o {}'.format(symbol))
        return 0
    elif len(position)>0:
        print('Há no total {} posição aberta no {}'.format(len(position),symbol))
        return len(position)
    
def order_entry(direcao, lot,stop_loss,take_profit):
    s = symbol_info.description
    indice = s[s.find('(')+1:s.find(')')]
    select = mt.symbol_select(indice,True)
    point = mt.symbol_info(indice).point
    
    if not select:
        print('O ativo {} não existe. Verificar se o código está correto. Coódigo do error = {}'.format(ativo, mt.last_error()))
    
    if direcao == 0 and open_orders(indice) < 1 :
        price_buy = mt.symbol_info_tick(indice).ask
        if SL > 0:
            bsl = price_buy - SL * point
        else: bsl=0
        
        if TP > 0:
            btp = price_buy + TP * point
        else: btp=0
       
        position_open(indice,mt.ORDER_TYPE_BUY,VOLUME,price_buy,bsl,
                             btp,'EA Teste')
        buy(VOLUME,indice,price_buy,bsl,btp,'EA Teste')
        
  
order_entry(0,VOLUME,SL,TP)    

# exibimos dados sobre o pacote MetaTrader5
print("MetaTrader5 package author: ", mt.__author__)
print("MetaTrader5 package version: ", mt.__version__)
 
# estabelecemos a conexão ao MetaTrader 5
if not mt.initialize():
    print("initialize() failed, error code =",mt.last_error())
    quit()


symbol = 'ITUB4'
symbol_info = mt.symbol_info(symbol)
if symbol_info is None:
    print(symbol, "not found, can not call order_check()")
    mt.shutdown()
    quit()
 
# se o símbolo não estiver disponível no MarketWatch, adicionamo-lo
if not symbol_info.visible:
    print(symbol, "is not visible, trying to switch on")
    if not mt.symbol_select(symbol,True):
        print("symbol_select({}}) failed, exit",symbol)
        mt.shutdown()
        quit()

lot = 100
point = mt.symbol_info(symbol).point
price = mt.symbol_info_tick(symbol).ask
deviation = 1
request = {
    "action": mt.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": lot,
    "type": mt.ORDER_TYPE_BUY,
    "price": price,
    "sl": price - 100 * point,
    "tp": price + 100 * point,
    "deviation": deviation,
    "magic": 123456,
    "comment": "python script open",
    "type_time": mt.ORDER_TIME_DAY,
    "type_filling": mt.ORDER_FILLING_RETURN,
}
result = mt.order_send(request)

# verificamos o resultado da execução
print("1. order_send(): by {} {} lots at {} with deviation={} points".format(symbol,lot,price,deviation));


#datetime.datetime.now().time() - dia_futuro_range[i].to_pydatetime().time()
#type(dia_futuro_range['ds'][len(dia_futuro_range)-1].to_pydatetime().time())
#dia_futuro_range[dia_futuro_range['ds']=='11:30:00']
#rates = mt.copy_rates_from(indice_selecionado,timeframes[timeframe_selecionado],dia_futuro_range['ds'][index_01].to_pydatetime(),1)

#index_01 = 0
#index_02 = 1

#condicional = True
#while condicional:
   # tick = mt.symbol_info_tick(indice_selecionado)
    #date = datetime.datetime.fromtimestamp(tick.time,tz=utc_timezone)
    #if date.time() > dia_futuro_range['ds'][index_01].to_pydatetime().time() and date.time() < dia_futuro_range['ds'][index_02].to_pydatetime().time():
    #    print('Valeu', dia_futuro_range['ds'][index_01],date.time() ,dia_futuro_range['ds'][index_02])
    #    rates = mt.copy_rates_from(indice_selecionado,timeframes[timeframe_selecionado],dia_futuro_range['ds'][index_01].to_pydatetime(),1)
    #    print(datetime.datetime.fromtimestamp(rates['time'],tz=utc_timezone ),rates['close'])
    #    break
        
    #else:    
    #    print('Não valeu')
    #    condicional = False
    #index_01 += 1
    #index_02 += 1
    #condicional = True
    














  
    
