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













