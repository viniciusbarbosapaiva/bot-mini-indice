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

# Importando o arquivo
df = pd.read_csv('dataframe_completo.csv')
df.head(5)

# Informação dos dados
df.info()

# Verificando Null values
null_values = (df.isnull().sum()/len(df))*100
null_values = pd.DataFrame(null_values, columns=['% of Null Values']) 
df[df['open_win'].isnull()]
np.where?
