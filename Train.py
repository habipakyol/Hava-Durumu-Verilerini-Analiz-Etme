import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data = pd.read_csv("DailyDelhiClimateTrain.csv")
print(data.head())
print("--------------------------------")

#bu verinin tanımlayıcı istatistiklerine bakalım
print(data.describe())
print("--------------------------------")

#bu verinin bilgilerine bakalım
print(data.info())
print("--------------------------------")

#Bu veri kümesindeki tarih sütunu bir datetime veri tipine sahip değildi
print(data["date"].head())

#Şehirdeki sıcaklık bilgilerine bakalım
figure = px.line(data, x="date",
                 y="meantemp",
                 title='Yıllar içinde Delhi de ortalama sıcaklık')
figure.show()

#Nem oranına bakalım
figure = px.line(data, x="date",
                 y="humidity",
                 title='Yıllar içinde Delhi de nem oranı')
figure.show()

# sıcaklık ve nem arasındaki ilişkiye bir göz atalım
sns.scatterplot(x="meantemp", y="humidity", size = "meantemp", data=data)
plt.show()

data["date"] = pd.to_datetime(data["date"], format = '%Y-%m-%d')
data['year'] = data['date'].dt.year
data["month"] = data["date"].dt.month
print(data.head())

plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.title("Temperature Change in Delhi Over the Years")
sns.lineplot(data = data, x='month', y='meantemp', hue='year')
plt.show()

forecast_data = data.rename(columns = {"date": "ds",
                                       "meantemp": "y"})
print(forecast_data)
#bu kodda şunu yaptık: veri setimizdeki "date" sütununu "ds" olarak, "meantemp" sütununu "y" olarak yeniden adlandırdık


from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
model = Prophet()
model.fit(forecast_data)
forecasts = model.make_future_dataframe(periods=365)
predictions = model.predict(forecasts)

plot_plotly(model, predictions, xlabel='Date', ylabel='Temperature')

#bu kodda şunu yaptık: Prophet kütüphanesini kullanarak bir model oluşturduk ve bu modeli eğittik. Daha sonra gelecekteki 365 gün için tahminlerde bulunduk ve bu tahminleri çizdirdik.
