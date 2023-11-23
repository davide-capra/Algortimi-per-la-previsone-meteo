from prophet import Prophet
import pandas as pd
from matplotlib import pyplot as plt

# Carica il dataset
df = pd.read_csv('/home/davide/Scrivania/weatherAUS.csv')

# Converti la colonna "Date" in oggetti datetime
df['Date'] = pd.to_datetime(df['Date'])

# Filtra i dati degli ultimi 3 anni
df = df[df['Date'].dt.year > df['Date'].dt.year.max() - 3]

# Seleziona i dati relativi a Sydney
sydney_data = df[df['Location'] == 'Sydney']

# Seleziona solo le colonne "Date" e "Sunshine" e rinomina le colonne
data = sydney_data[['Date', 'Sunshine']]
data.columns = ['ds', 'y']

# Crea il modello Prophet
model = Prophet()
model.fit(data)

# Crea un DataFrame "future" con date nel periodo da '2015-01-01' a '2020-12-31'
future = pd.DataFrame(pd.date_range(start='2015-01-01', end='2020-12-31'), columns=['ds'])

# Effettua previsioni
forecast = model.predict(future)

# Crea grafici
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Pannello 1: Dataset originale
axs[0].plot(data['ds'], data['y'], label='Dataset originale')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Sunshine')
axs[0].set_title('Dataset originale')

# Pannello 2: Previsione
axs[1].plot(forecast['ds'], forecast['yhat'], label='Previsione')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Sunshine')
axs[1].set_title('Previsione')

# Mostra i grafici
plt.tight_layout()
plt.show()