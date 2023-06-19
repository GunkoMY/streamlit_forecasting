import streamlit as st
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
#pip install streamlit-authenticator
#pip install streamlit-option-menu
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
import io
# buffer to use for excel writer
buffer = io.BytesIO()

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

import yaml
from yaml.loader import SafeLoader

st.set_page_config(page_title="Аналитическая система сопровождения складских запасов",)

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# hashed_passwords = stauth.Hasher(['abc', 'def']).generate()
# print(hashed_passwords)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')
# print(name, authentication_status, username)

data = pd.read_csv('final_data_forecast.csv')
data['date_date'] = pd.to_datetime(data['date']).dt.date
data['Склад'] = data['Склад'].apply(lambda x: x.split('.')[1])
list_of_product = data['Склад'].unique()

def do_analysis():
    """
    ### Сравнительный анализ исторических данных
    """
    option_col = st.selectbox(
        'Параметр для построения графика',
        ('Количество Начальный остаток', 'Стоимость Начальный остаток',
         'Количество приход', 'Стоимость приход', 'Количество расход',
         'Стоимость расход', 'Себест', 'Цена продажи', 'Сумма продажи',
         'Количество Конечный остаток', 'Стоимость Конечный остаток',
         'Profit', 'profitPR'))

    options_product = st.multiselect('Продукты для построения графика', list_of_product, list_of_product[0])

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

    """
    #### Таблица по товарам на исторических данных
    """
    st.dataframe(data[(start_time <= data['date_date']) & (data['date_date'] <= end_time) & (data['Склад'].isin(options_product))].drop('date_date', axis=1).T, use_container_width=True,)


    """
    #### Гистограмма по выбранному параметру и товарам на исторических данных
    """

    fig, ax = plt.subplots(figsize=(12,8))
    sns.barplot(x='date_date', y=option_col,
                 data=data[
                     (start_time <= data['date_date']) &
                     (data['date_date'] <= end_time) &
                     (data['Склад'].isin(options_product))
                     ],
                hue='Склад', ax=ax, dodge=True)
    plt.xticks(rotation=45)
    st.write(fig)

def do_forecast():
    global fcst
    """
    #### Выбор конкретного продукта, параметра и периода для прогнозирования
    Имейте в виду, что прогнозы становятся менее точными с увеличением горизонта прогнозирования.
    **Прогноз строится по месяцам!**
    """
    product = st.selectbox('Необходимо продукт для прогноза', list_of_product)

    option = st.selectbox(
        'Необходимо выбрать параметр для прогноза',
        ('Количество Начальный остаток', 'Стоимость Начальный остаток',
         'Количество приход', 'Стоимость приход', 'Количество расход',
         'Стоимость расход', 'Себест', 'Цена продажи', 'Сумма продажи',
         'Количество Конечный остаток', 'Стоимость Конечный остаток',
         'Profit', 'profitPR'))


    periods_input = st.number_input('Необходимо выбрать период для поргноза',
                                    min_value = 5, max_value = 24)

    ##########################################################################################

    """
    ### Прогноз по выбранному продукту

    На изображении ниже показаны будущие прогнозируемые значения. «прогноз» — это прогнозируемое значение по выбранному параметру и продукту,
    а верхний и нижний пределы — это 80-процентные доверительные интервалы.
    """
    data_for_forecast = data[data['Склад'] == product][['date', option]].rename(columns={'date': 'ds',
                                                                                         option: 'y'})
    train = data_for_forecast.iloc[:20]
    test = data_for_forecast.iloc[20:]
    max_date = train['ds'].max()

    m = Prophet(weekly_seasonality=True)
    m.fit(train)

    future = m.make_future_dataframe(periods=periods_input, freq = 'M')
    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    fcst_filtered =  fcst[fcst['ds'] > max_date]    
    st.dataframe(fcst_filtered.rename(columns={'yhat': 'прогноз',
                                               'yhat_lower': 'нижний предел',
                                               'yhat_upper': 'верхний предел'}).set_index('ds'))

    """
    На следующем изображении показаны исторические значения, использумые для обучения (черные точки),
    тестовые значения, на которыых проверялась работа алгоритма (зеленые точки) и прогнозируемые значения (синяя линия с красными точками) с течением времени.
    """
    fig1 = m.plot(forecast)
    plt.scatter(pd.to_datetime(test['ds']).dt.to_pydatetime(), test['y'], color='g', label='test')
    plt.scatter(forecast['ds'].iloc[-periods_input:].dt.to_pydatetime(), forecast['yhat'].iloc[-periods_input:], color='r', label='predict')
    plt.text(x=forecast['ds'].iloc[0], y=0.8 * m.history['y'].min(), fontsize=15, s='История')
    plt.text(x=pd.to_datetime(test['ds']).iloc[0], y=0.8 * m.history['y'].min(), fontsize=15, s='Тест', color='g')
    plt.text(x=forecast['ds'].iloc[23], y=0.8 * m.history['y'].min(), fontsize=15, s='Прогноз', color='r')
    plt.axvline(x=pd.to_datetime(test['ds']).iloc[0], color='g')
    plt.axvline(x=forecast['ds'].iloc[23], color='r')
    plt.legend(loc='upper left')
    plt.ylabel(option)
    plt.xlabel('Дата')
    plt.title(product)
    st.write(fig1)


    f"""
    ##### Средняя абсолютная ошибка
    на истории: {round(mean_absolute_error(train['y'], forecast['yhat'][:20]))},
    на тесте: {round(mean_absolute_error(test['y'], forecast['yhat'][20:24]))}
    """

    """
    ### Выгрузка обработанных данных с прогнозом

    Ссылка ниже позволяет загрузить вновь созданный прогноз на ваш компьютер для дальнейшего анализа и использования.
    """
    # download button 2 to download dataframe as xlsx
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Write each dataframe to a different worksheet.
        fcst.to_excel(writer, sheet_name='Sheet1', index=False)
        # Close the Pandas Excel writer and output the Excel file to the buffer
        writer.close()

        download2 = st.download_button(
            label="Download data as Excel",
            data=buffer,
            file_name='forecast.xlsx',
            mime='application/vnd.ms-excel'
        )

def do_download():
    """
    ### Выгрузка обработанных данных с прогнозом

    Ссылка ниже позволяет загрузить вновь созданный прогноз на ваш компьютер для дальнейшего анализа и использования.
    """
    csv_exp = fcst.to_csv(index=False)
    #st.dataframe(fcst)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Скачать CSV-файл</a> (щелкните правой кнопкой мыши и сохраните как ** &lt;прогноз_имя&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)

def do_logout():
    authenticator.logout('Logout', 'main', key='unique_key')

menu_dict = {
    "Анализ" : {"fn": do_analysis},
    "Прогноз" : {"fn": do_forecast},
    #"Выгрузка" : {"fn": do_download},
    "Выход" : {"fn": do_logout},
}

if st.session_state["authentication_status"]:
    #authenticator.logout('Logout', 'main', key='unique_key')
    #st.write(f'Welcome *{st.session_state["name"]}*')
    st.title('Аналитическая система сопровождения складских запасов')

    #with st.sidebar:
    selected = option_menu(None,
                           options=[
                               "Анализ",
                               "Прогноз",
                               #"Выгрузка",
                               "Выход"
                               ],
                           icons=[
                               "pie-chart-fill",
                               "graph-up-arrow",
                               #"download",
                               "door-open"],
                           default_index=0,
                           orientation="horizontal")

    if selected in menu_dict.keys():
        menu_dict[selected]["fn"]()

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
