import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64

st.title('Делаем магию')

"""
### Step 1: Что там по данным
"""
#df = st.file_uploader('Import the time series csv file here. Columns must be labeled ds and y. The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.', type='csv')

#st.info()

data = pd.read_csv('final_data_forecast.csv')
st.dataframe(data.head())

list_of_product = data['Склад'].unique()

product = st.selectbox('Давайте выберем продукт для прогноза', list_of_product)

#st.text('Можем строить прогнозы по продуктам:\n')
#st.text('\n'.join(list_of_product))


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

#st.write('Выбор сделан:', option)

periods_input = st.number_input('А теперь определимся: на какой период будем прогнозировать?',
min_value = 1, max_value = 365)

#st.write('Выбор сделан:', periods_input)


data_for_forecast = data[data['Склад'] == product][['date', option]].rename(columns={'date': 'ds',
                                                                                     option: 'y'})
max_date = data_for_forecast['ds'].max()

m = Prophet()
m.fit(data_for_forecast)

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


"""
### Step 4: Можем скачать что сделали

Ссылка ниже позволяет загрузить вновь созданный прогноз на ваш компьютер для дальнейшего анализа и использования.
"""
csv_exp = fcst_filtered.to_csv(index=False)
# When no file name is given, pandas returns the CSV as a string, nice.
b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
href = f'<a href="data:file/csv;base64,{b64}">Скачать CSV-файл</a> (щелкните правой кнопкой мыши и сохраните как ** &lt;прогноз_имя&gt;.csv**)'
st.markdown(href, unsafe_allow_html=True)
