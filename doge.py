import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Получение данных о ценах Dogecoin за последнюю неделю
def get_dogecoin_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Используем данные за последний месяц для лучшего анализа
    url = f'https://api.coingecko.com/api/v3/coins/dogecoin/market_chart/range?vs_currency=usd&from={int(start_date.timestamp())}&to={int(end_date.timestamp())}'
    response = requests.get(url)
    data = response.json()
    return data['prices']

# Построение графика цен
def plot_prices(prices, ax):
    dates = [datetime.fromtimestamp(price[0] / 1000) for price in prices]
    values = [price[1] for price in prices]
    ax.plot(dates, values)
    ax.set_title('Dogecoin Prices Over the Last Month')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')

# Анализ данных и прогноз с использованием линейной регрессии
def analyze_data(prices):
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['day'] = (df['date'] - df['date'].min()).dt.days
    X = df[['day']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    df['predicted_price'] = model.predict(df[['day']])
    
    return df

# Построение графика реальных и предсказанных цен
def plot_prediction(df, ax):
    ax.plot(df['date'], df['price'], label='Actual Price')
    ax.plot(df['date'], df['predicted_price'], label='Predicted Price')
    ax.set_title('Dogecoin Actual vs Predicted Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()

# Создание графического интерфейса
def create_gui():
    root = tk.Tk()
    root.title("Dogecoin Analysis")
    
    # Создание двух графиков
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    prices = get_dogecoin_data()
    plot_prices(prices, ax1)
    df = analyze_data(prices)
    plot_prediction(df, ax2)
    
    # Отображение графиков в Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    root.mainloop()

if __name__ == '__main__':
    create_gui()
