import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.io as pio

# Carica il dataset
df = pd.read_csv('/home/davide/Scrivania/weatherAUS.csv')
# Filtra i dati di Sydney e seleziona solo le colonne necessarie
syd = df[df['Location'] == 'Sydney']

# Seleziona i dati fino al 2012 per l'addestramento
train_data = syd[syd['Date'] <= '2012-12-31']

# Seleziona i dati dal 2013 al 2017 per le previsioni
future_data = syd[(syd['Date'] >= '2013-01-01') & (syd['Date'] <= '2017-12-31')]

# Seleziona solo le colonne necessarie
train_features = ['Date', 'Temp9am', 'Humidity9am', 'Rainfall']
future_features = ['Date', 'Temp9am', 'Humidity9am', 'Rainfall']

# Rimuovi le righe con valori mancanti nei dati di addestramento
train_data = train_data[train_features].dropna()

# Rinomina le colonne
train_data.columns = ['ds', 'Temp9am', 'Humidity', 'Rainfall']

# Converti la colonna 'Date' in formato datetime
train_data['ds'] = pd.to_datetime(train_data['ds'])

# Crea variabili aggiuntive come mese e anno
train_data['month'] = train_data['ds'].dt.month
train_data['year'] = train_data['ds'].dt.year

# Dividi i dati di addestramento in set di addestramento e test
train, test = train_test_split(train_data, test_size=0.2, shuffle=False)

# Seleziona le variabili di input (features) per l'addestramento
features = ['month', 'year', 'Temp9am', 'Humidity', 'Rainfall']

# Addestra il modello Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42) 
model.fit(train[features], train['Temp9am'])

# Prepara i dati futuri per le previsioni
future_data = future_data[future_features]
future_data = future_data.dropna()

# Rinomina le colonne nel dataset future
future_data.columns = ['ds', 'Temp9am', 'Humidity', 'Rainfall']

# Converti la colonna 'Date' in formato datetime
future_data['ds'] = pd.to_datetime(future_data['ds'])

# Crea variabili aggiuntive come mese e anno
future_data['month'] = future_data['ds'].dt.month
future_data['year'] = future_data['ds'].dt.year

# Effettua previsioni
forecast = model.predict(future_data[features])

# Calcola l'errore quadratico medio tra le previsioni e i dati effettivi
mse = mean_squared_error(future_data['Temp9am'], forecast)
print(f'Errore quadratico medio (MSE): {mse:.2f}')

# Visualizza i risultati
plt.figure(figsize=(10, 6))
plt.plot(future_data['ds'], future_data['Temp9am'], label='Dati effettivi', color='blue')
plt.plot(future_data['ds'], forecast, label='Previsioni Random Forest', color='red')
plt.legend()
plt.title('Confronto tra dati effettivi e previsioni con MSE')
plt.xlabel('Data')
plt.ylabel('Temperatura (y)')
plt.text(future_data['ds'].iloc[0], future_data['Temp9am'].min(), f'MSE: {mse:.2f}', verticalalignment='bottom')
plt.show()
# Visualizza i risultati con Plotly
fig = px.line(future_data, x='ds', y='Temp9am', title='Confronto tra dati effettivi e previsioni (Temperatura 9am)')
fig.add_scatter(x=future_data['ds'], y=forecast, mode='lines', name='Previsioni Random Forest', line=dict(color='red'))
fig.show()