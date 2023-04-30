import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import seaborn as sns
from matplotlib import pyplot as plt
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64

st.title('Делаем магию')

"""
### Step 1: Что там по данным
"""
# Читаем данные и выводим как таблицу
data = pd.read_csv('final_data_forecast.csv')
data['date_date'] = pd.to_datetime(data['date']).dt.date
st.dataframe(data)
list_of_product = data['Склад'].unique()

"""
#### Step 1.1: Построим графики
"""
option_col = st.selectbox(
    'Давайте выберем колонку для построения графика',
    ('Количество Начальный остаток', 'Стоимость Начальный остаток',
     'Количество приход', 'Стоимость приход', 'Количество расход',
     'Стоимость расход', 'Себест', 'Цена продажи', 'Сумма продажи',
     'Количество Конечный остаток', 'Стоимость Конечный остаток',
     'Profit', 'profitPR'))

options_product = st.multiselect('По каким продуктам строим график', list_of_product, list_of_product[0])

start_time = st.slider(
    "Начало периода",
    min_value=data['date_date'].min(),
    value=data['date_date'].min(),
    max_value=data['date_date'].max(),
    step=timedelta(days=30),
    format="MM/DD/YY")

end_time = st.slider(
    "Конец периода",
    min_value=start_time,
    value=data['date_date'].max(),
    max_value=data['date_date'].max(),
    step=timedelta(days=30),
    format="MM/DD/YY")

fig, ax = plt.subplots(figsize=(16,10))
sns.barplot(x='date_date', y=option_col,
             data=data[
                 (start_time <= data['date_date']) &
                 (data['date_date'] <= end_time) &
                 (data['Склад'].isin(options_product))
                 ],
            hue='Склад', ax=ax, dodge=True)
plt.xticks(rotation=45)
st.write(fig)

"""
#### Step 1.2: А теперь выберем кокретный продукт
"""
product = st.selectbox('Давайте выберем продукт для прогноза', list_of_product)

##########################################################################################

"""
### Step 2: А что будем прогнозировать?

Имейте в виду, что прогнозы становятся менее точными с увеличением горизонта прогнозирования.
"""
option = st.selectbox(
    'Давайте выберем колонку для прогноза',
    ('Количество Начальный остаток', 'Стоимость Начальный остаток',
     'Количество приход', 'Стоимость приход', 'Количество расход',
     'Стоимость расход', 'Себест', 'Цена продажи', 'Сумма продажи',
     'Количество Конечный остаток', 'Стоимость Конечный остаток',
     'Profit', 'profitPR'))


periods_input = st.number_input('А теперь определимся: на какой период будем прогнозировать?',
min_value = 1, max_value = 365)
data_for_forecast = data[data['Склад'] == product][['date', option]].rename(columns={'date': 'ds',
                                                                                     option: 'y'})
max_date = data_for_forecast['ds'].max()
m = Prophet()
m.fit(data_for_forecast)

##########################################################################################

"""
### Step 3: Ну посмотрим что получилось

На изображении ниже показаны будущие прогнозируемые значения. «yhat» — это прогнозируемое значение, а верхний и нижний пределы — это (по умолчанию) 80-процентные доверительные интервалы.
"""
future = m.make_future_dataframe(periods=periods_input, freq = 'M')
forecast = m.predict(future)
fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
fcst_filtered =  fcst[fcst['ds'] > max_date]    
st.dataframe(fcst_filtered)
"""
На следующем изображении показаны фактические (черные точки) и прогнозируемые (синяя линия) значения с течением времени.
"""
fig1 = m.plot(forecast)
st.write(fig1)

##########################################################################################

"""
### Step 4: Можем скачать что сделали

Ссылка ниже позволяет загрузить вновь созданный прогноз на ваш компьютер для дальнейшего анализа и использования.
"""
csv_exp = fcst_filtered.to_csv(index=False)
# When no file name is given, pandas returns the CSV as a string, nice.
b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
href = f'<a href="data:file/csv;base64,{b64}">Скачать CSV-файл</a> (щелкните правой кнопкой мыши и сохраните как ** &lt;прогноз_имя&gt;.csv**)'
st.markdown(href, unsafe_allow_html=True)
