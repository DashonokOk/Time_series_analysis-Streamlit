import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from darts import TimeSeries
from darts.models import ExponentialSmoothing

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')

# Заголовок приложения
st.title('Анализ и прогнозирование временного ряда')

st.image('logo.png', use_container_width=True)

# Описание приложения
st.markdown("""
Это приложение позволяет загрузить файл с временным рядом данных, провести анализ и прогнозирование, а также отобразить результаты.
""")

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите файл с временным рядом (CSV или Excel)", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    # Чтение файла
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    # Отображение первых 10 строк данных
    st.write("Первые 10 строк данных:")
    st.write(data.head(10))

    # Преобразование столбца 'Order Date' в формат datetime
    data['Order Date'] = pd.to_datetime(data['Order Date'])

    # Установка 'Order Date' в качестве индекса
    data.set_index('Order Date', inplace=True)

    # Ресемплирование данных по неделям и суммирование продаж
    weekly_sales = data['Sales'].resample('W').sum()

    # Визуализация данных по неделям
    st.write("Еженедельные продажи:")
    st.write("График показывает суммарные продажи по неделям. Это позволяет увидеть динамику продаж на недельной основе.")
    st.line_chart(weekly_sales)

    # Декомпозиция временного ряда
    decomposition = seasonal_decompose(weekly_sales, model='additive')
    st.write("Декомпозиция временного ряда:")
    st.write("График декомпозиции показывает тренд, сезонность и остатки временного ряда. Тренд показывает общую динамику продаж, сезонность - повторяющиеся паттерны, а остатки - случайные колебания.")
    st.pyplot(decomposition.plot())

    # Построение модели ARIMA
    model_arima = ARIMA(weekly_sales, order=(5, 1, 0))
    model_fit_arima = model_arima.fit()
    st.write("Прогноз модели ARIMA:")
    st.write("График показывает реальные еженедельные продажи и прогноз модели ARIMA. Модель ARIMA используется для прогнозирования временных рядов на основе авторегрессии и скользящего среднего.")
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_sales, label='Еженедельные продажи')
    plt.plot(model_fit_arima.fittedvalues, color='red', label='Прогноз ARIMA')
    plt.title('Прогноз модели ARIMA')
    plt.xlabel('Дата')
    plt.ylabel('Продажи')
    plt.legend()
    st.pyplot(plt.gcf())

    # Построение модели Prophet
    df_prophet = weekly_sales.reset_index().rename(columns={'Order Date': 'ds', 'Sales': 'y'})
    model_prophet = Prophet()
    model_prophet.fit(df_prophet)
    future = model_prophet.make_future_dataframe(periods=52, freq='W')
    forecast = model_prophet.predict(future)
    st.write("Прогноз модели Prophet:")
    st.write("График показывает реальные еженедельные продажи и прогноз модели Prophet. Модель Prophet используется для прогнозирования временных рядов с учетом сезонности и трендов.")
    fig = model_prophet.plot(forecast)
    st.pyplot(fig)

    # Построение модели Random Forest
    X = np.arange(len(weekly_sales)).reshape(-1, 1)
    y = weekly_sales.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Среднеквадратичная ошибка (MSE) модели Random Forest: {mse}')
    st.write("Прогноз модели Random Forest:")
    st.write("График показывает реальные значения и прогноз модели Random Forest. Модель Random Forest используется для регрессионного анализа и прогнозирования временных рядов.")
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Реальные значения')
    plt.plot(y_pred, label='Прогноз Random Forest', color='red')
    plt.title('Прогноз модели Random Forest')
    plt.xlabel('Индекс')
    plt.ylabel('Продажи')
    plt.legend()
    st.pyplot(plt.gcf())

    # Построение модели Darts
    series = TimeSeries.from_dataframe(weekly_sales.reset_index(), 'Order Date', 'Sales')
    model_darts = ExponentialSmoothing()
    model_darts.fit(series)
    forecast_darts = model_darts.predict(52)
    st.write("Прогноз модели Darts:")
    st.write("График показывает реальные еженедельные продажи и прогноз модели Darts. Модель Darts используется для прогнозирования временных рядов с использованием метода экспоненциального сглаживания.")
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_sales.index, weekly_sales.values, label='Еженедельные продажи')
    plt.plot(forecast_darts.time_index, forecast_darts.values(), color='green', label='Прогноз Darts')
    plt.title('Прогноз модели Darts')
    plt.xlabel('Дата')
    plt.ylabel('Продажи')
    plt.legend()
    st.pyplot(plt.gcf())
