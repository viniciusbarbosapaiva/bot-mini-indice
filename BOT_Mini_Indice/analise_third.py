import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
from datetime import datetime
import datetime
import matplotlib.pyplot as plt

# Functions
def autolabel(rects,ax): #autolabel
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy = (rect.get_x() + rect.get_width()/2, height),
                    xytext= (0,3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=15)

# Importing dataset
df = pd.read_csv('comparacao_real_previsto.csv')
df['ds'] = pd.to_datetime(df['ds'])

# Droping error column
df = df.drop('Error', axis=1)

# info Dataset
df.info()

# verifying number of elements
df['date'] = df['ds'].apply(lambda x: x.date())
df = df.set_index('ds', drop=True)
df_group = df.groupby('date').count()


# deleting the 17/02/21 line
index_drop = df.loc['2021-02-17',].index
df = df.drop(index_drop)

# Creating correlation among Valor real and Valor Previsto columns
d = 0
d_1 = 1
length = len(df)
real_diferenca = np.array([])
previsto_diferenca = np.array([])
df = df.reset_index(drop=True)

while d_1 != length:
    diferenca_real = df.iloc[d,0] - df.iloc[d_1,0]
    diferenca_previsto = df.iloc[d,1] - df.iloc[d_1,1]
    
    real_diferenca = np.append(real_diferenca,diferenca_real)
    previsto_diferenca = np.append(previsto_diferenca,diferenca_previsto)
    
    d+=1
    d_1+=1
print(len(real_diferenca),len(previsto_diferenca))

real_diferenca = pd.DataFrame(real_diferenca,columns=['Real Diferenca'])
previsto_diferenca = pd.DataFrame(previsto_diferenca, columns=['Previsto Diferenca'])

# The first rule performance
first_rule = pd.concat([real_diferenca,previsto_diferenca], axis=1)
first_rule.columns

success_first = np.array([])
for real,previsto in zip(first_rule['Real Diferenca'],first_rule['Previsto Diferenca']):
    
    if real > 0 and previsto > 0:
        success_first = np.append(success_first,True)
    elif real < 0 and previsto < 0:
        success_first = np.append(success_first,True)
    else:
        success_first = np.append(success_first,False)

# Creating gain column
first_rule['gain'] = success_first

# Analysing performance
gain = (len(first_rule[first_rule['gain'] == True]['gain'])/len(first_rule))*100
gain = np.round(gain,2)
loss = (len(first_rule[first_rule['gain'] == False]['gain'])/len(first_rule))*100
loss = np.round(loss,2)
print('A terceira estratégia obteve {:.2f}% de gain'.format(gain))
print('A terceira estratégia obteve {:.2f}% de loss'.format(loss))

# Plotting the second rule performance
width = 0.35
labels = ['Gain','Loss']
ind = np.arange(len(labels))
values = [gain, loss]
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.set_title('Quantidade de Operações Vencedoras e Perdedoras \n Realizada pela Terceira Estratégia de negociação', fontsize=15)
ax.set_xlabel('Tipo de Retorno', fontsize=15)
ax.set_ylabel('Quantidade de Operações Ganhas/Perdas', fontsize=15)
ax.set_xticklabels(['Gain', 'Loss'], fontsize=15)
#ax.set_yticklabels(np.arange(0,501,100), fontsize=15)
rects1= ax.bar('Gain', values[0], width, edgecolor='black')
rects2=ax.bar('Loss', values[1], width, edgecolor='black')
ax.set_xticks(ind)
autolabel(rects1,ax)
autolabel(rects2,ax)
plt.tight_layout() 
plt.savefig('Terceira_estrategia_15m.png')

#################################################################################

'''
 # Definindo parâmetros para o robô
    robo = ('Não','Sim')
    ativar_robo = st.sidebar.selectbox("Ativar o Robô?",robo)
    st.cache()
    if ativar_robo == 'Sim':
        st.subheader('Parâmetros para o EA (Robô)')
        codigo = indice_selecionado.split('$')[0]
        c_1,c_2,c_3 = st.beta_columns((1,1,1))
        c_1.subheader('Código Vigente')
        if codigo == 'WIN':
            s = symbol_info.description
            indice = s[s.find('(')+1:s.find(')')]
            c_1.text(indice)
            c_2.subheader('Número de Contratos')
            volume = c_2.slider('Selecione Quantidade',1,100)
            c_3.subheader('Stop Loss')
            stop_loss = c_3.number_input('Pontos para Stop Loss', value=int(1000))
            c_4,c_5,c_6= st.beta_columns((1,1,1))
            c_4.subheader('Take Profit')
            profit = c_4.number_input('Pontos para Take Profit', value=int(1000))
            c_5.subheader('Martingale')
            martingale_list = ['Não Utilizar', 'Sim. Moderado','Sim. Agressivo']
            c_5.selectbox('Qual tipo de Martingale?',martingale_list)
            enviar_ordem = c_6.selectbox('Enviar Ordem?',robo)
            
        
            new_forecast_file = pd.read_csv('new_forecast_database.csv')
            new_forecast_file['ds'] = pd.to_datetime(new_forecast_file['ds'])
            new_forecast_file = new_forecast_file.set_index('ds', drop=True)               
            SL = stop_loss
            TP = profit
            VOLUME = volume  
            last_day = new_forecast_file.index[len(new_forecast_file)-1].date().strftime("%Y-%m-%d")
            today = new_forecast_file.loc[last_day, 'yhat']

            while working_hour() and enviar_ordem=='Sim':
                rates = mt.copy_rates_from(indice_selecionado,timeframes[timeframe_selecionado],
                                           dia_futuro_range['ds'][0].to_pydatetime(),2)
                pregao_hoje = pd.DataFrame(rates)
                pregao_hoje['time'] = pd.to_datetime(pregao_hoje['time'], unit='s')
                print('Tempo anterior: ',pregao_hoje['time'][len(pregao_hoje)-2],'Preço anterior: ', pregao_hoje['close'][len(pregao_hoje)-2])
                print('Tempo agora: ',pregao_hoje['time'][len(pregao_hoje)-1],'Preço previsto: ',today.loc[pregao_hoje['time'][1]])
                time.sleep(1)
    
                if today.loc[pregao_hoje['time'][1]] < pregao_hoje['close'][len(pregao_hoje)-2]:
                    print('Sell signal')
                    time.sleep(1)
                    order_entry(1,VOLUME,SL,TP)
                    position = mt.positions_get(symbol=indice)
                    position_type = position[0][5]
                    if today.loc[pregao_hoje['time'][1]] < pregao_hoje['close'][len(pregao_hoje)-2] and position_type == 1 and open_orders(indice) >=1 :
                        print('Tendência de baixa continua')
                    else:
                        close_all(1)
   
                if today.loc[pregao_hoje['time'][1]] > pregao_hoje['close'][len(pregao_hoje)-2]:
                    print('Buy signal')
                    time.sleep(1)
                    order_entry(0,VOLUME,SL,TP) 
                    position = mt.positions_get(symbol=indice)
                    position_type = position[0][5]
                    if today.loc[pregao_hoje['time'][1]] > pregao_hoje['close'][len(pregao_hoje)-2] and position_type == 0 and open_orders(indice) >=1 :
                        print('Tendência de alta continua')
                        time.sleep(1)
                    else:
                        close_all(0)
                        time.sleep(1)
                                
         
        if codigo == 'WDO':
            s = symbol_info.description
            indice = s[s.find('(')+1:s.find(')')]
            c_1.text(indice)
            c_2.subheader('Número de Contratos')
            volume = c_2.slider('Selecione Quantidade',1,100)
            c_3.subheader('Stop Loss')
            stop_loss = c_3.number_input('Pontos para Stop Loss', value=int(50000))
            c_4,c_5,c_6= st.beta_columns((1,1,1))
            c_4.subheader('Take Profit')
            profit = c_4.number_input('Pontos para Take Profit', value=int(50000))
            c_5.subheader('Martingale')
            martingale_list = ['Não Utilizar', 'Sim. Moderado','Sim. Agressivo']
            c_5.selectbox('Qual tipo de Martingale?',martingale_list)
            enviar_ordem = c_6.button('Click Aqui para Enviar Ordem')
            c_6.text('Robô não está ativado')
            
          
            new_forecast_file = pd.read_csv('new_forecast_database.csv')
            new_forecast_file['ds'] = pd.to_datetime(new_forecast_file['ds'])
            new_forecast_file = new_forecast_file.set_index('ds', drop=True)               
            SL = stop_loss
            TP = profit
            VOLUME = volume  
            last_day = new_forecast.index[len(new_forecast)-1].date().strftime("%Y-%m-%d")
            today = new_forecast.loc[last_day, 'yhat']
            while working_hour() and enviar_ordem=='Sim':
                rates = mt.copy_rates_from(indice_selecionado,timeframes[timeframe_selecionado],
                                           dia_futuro_range['ds'][0].to_pydatetime(),2)
                pregao_hoje = pd.DataFrame(rates)
                pregao_hoje['time'] = pd.to_datetime(pregao_hoje['time'], unit='s')
                print('Tempo anterior: ',pregao_hoje['time'][len(pregao_hoje)-2],'Preço anterior: ', pregao_hoje['close'][len(pregao_hoje)-2])
                print('Tempo agora: ',pregao_hoje['time'][len(pregao_hoje)-1],'Preço previsto: ',today.loc[pregao_hoje['time'][1]])
                time.sleep(1)
                
                if today.loc[pregao_hoje['time'][1]] < pregao_hoje['close'][len(pregao_hoje)-2]:
                    print('Sell signal')
                    time.sleep(1)
                    order_entry(1,VOLUME,SL,TP)
                    position = mt.positions_get(symbol=indice)
                    position_type = position[0][5]
                    if today.loc[pregao_hoje['time'][1]] < pregao_hoje['close'][len(pregao_hoje)-2] and position_type == 1 and open_orders(indice) >=1 :
                        print('Tendência de baixa continua')
                    else:
                        close_all(1)
   
                if today.loc[pregao_hoje['time'][1]] > pregao_hoje['close'][len(pregao_hoje)-2]:
                    print('Buy signal')
                    time.sleep(1)
                    order_entry(0,VOLUME,SL,TP) 
                    position = mt.positions_get(symbol=indice)
                    position_type = position[0][5]
                    if today.loc[pregao_hoje['time'][1]] > pregao_hoje['close'][len(pregao_hoje)-2] and position_type == 0 and open_orders(indice) >=1 :
                        print('Tendência de alta continua')
                        time.sleep(1)
                    else:
                        close_all(0)
                        time.sleep(1)              

'''








