from prophet import Prophet
import pandas as pd
from matplotlib import pyplot as plt 
import pickle
from prophet.plot import add_changepoints_to_plot
from sklearn.metrics import mean_squared_error

# Carica il dataset
df = pd.read_csv('/home/davide/Scrivania/weatherAUS.csv')

# Filtra i dati solo per la località di Sydney
syd = df[df['Location'] == 'Sydney']

# Converte la colonna 'Date' in formato datetime
syd['Date'] = pd.to_datetime(syd['Date'])

# Seleziona solo le colonne 'Date' e 'Temp9am'
data = syd[['Date', 'Temp9am']]

# Rimuovi le righe con valori mancanti
data.dropna(inplace=True)

# Rinomina le colonne
data.columns = ['ds', 'y']

# Crea e addestra il modello Prophet
m = Prophet()
m.fit(data)

# Crea un DataFrame future per le previsioni
future = m.make_future_dataframe(periods=1095)

# Effettua le previsioni
forecast = m.predict(future)

# Visualizza i risultati
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Calcola l'errore quadratico medio tra i dati effettivi e le previsioni
mse = mean_squared_error(data['y'], forecast['yhat'][:len(data)])
print(f'Errore quadratico medio (MSE): {mse:.2f}')

# Plot dei risultati
fig = m.plot(forecast)
plt.show()

fig = m.plot_components(forecast)

