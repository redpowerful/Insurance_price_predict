import streamlit as st
import pandas as pd
import pickle
import numpy as np

car_list = ('Renault Duster', 'Lada 4x4', 'Lada Granta', 'Datsun on-DO',
            'KIA Rio', 'Hyundai Solaris', 'Renault Logan', 'Toyota Camry',
            'VW Polo', 'Hyundai Creta', 'Lada Vesta', 'Lada X-Ray',
            'Mazda CX-5', 'KIA Sportage', 'Skoda Rapid', 'VW Tiguan',
            'Toyota RAV 4', 'Renault Kaptur', 'Chevrolet NIVA', 'Lada Largus',
            'Skoda Octavia A7', 'Nissan Qashqai', 'Renault Sandero',
            'Lada Kalina', 'Nissan X-Trail')

car_exp = ('2-5', '0-1', '6-10', '26-50', '11-25', '51+')

dtp_types = ('Без ДТП', 'Легкое ДТП', 'Угон, ДТП без востановления авто',
             'Крупное ДТП')

st.title("Расчет стоимости страхования")

age = st.sidebar.slider('Сколько вам лет?', 18, 100)
car = st.sidebar.selectbox('Выберите марку автомобиля', car_list)
insurance = st.sidebar.selectbox("Выберите тип страховки", ("Полное страхование", "Частичное"))
exp_car = st.sidebar.selectbox('Какой у вас стаж вождения?', car_exp)
type_of_dtp = st.sidebar.selectbox('Бывали ли вы в дтп?', dtp_types)
car_price = st.sidebar.number_input(label='Стоимость автомобиля', min_value=0, max_value=10000000, step=100000)


def get_data(age, car, insurance, exp_car, type_of_dtp, car_price) -> pd.DataFrame:
    """
       Gets params and return dataframe by them

       :param age: how old are client
       :param car: name of car
       :param insurance: type of insurance
       :param exp_car: how long has been driving car
       :param type_of_dtp: type of traffic accident
       :param car_price: price of car
       :return: data frame
       """
    df = pd.DataFrame(data={
        'Возраст': age,
        'Марка автомобиля': car,
        'Тип страхования': insurance,
        'Стаж': exp_car,
        'ДТП': type_of_dtp,
        'Медианное стоимость авто': car_price
    }, index=[0])
    return df


df = get_data(age, car, insurance, exp_car, type_of_dtp, car_price)  # get df
load_cat_clf = pickle.load(open('insurance_cat_clf.pkl', 'rb'))  # load model

prediction = load_cat_clf.predict(df)  # get predict


def get_price(cost):
       if cost < 0:  # fix some issue
              cost = 20000 * 1.15
       if type_of_dtp == 'Крупное ДТП':
              if cost < 250000:
                     cost = 250000 * 1.15
              else:
                     cost = cost * 1.15
       elif type_of_dtp == 'Легкое ДТП':
              if cost < 55000:
                     cost = 60000 * 1.15
              else:
                     cost = cost * 1.15
       elif type_of_dtp == 'Угон, ДТП без востановления авто':
              if cost < 450000:
                     cost = 450000 * 1.15
              else:
                     cost = cost * 1.15
       return cost


# Out df and predict
st.markdown(
    f"""
       Мы можем предсказать стоимость выплаты страхования основанной на искуственном интеллекте
       
       Выберите в левом окне нужные вам параметры
       
       У нас действует акция за без аварийность
       
       Стоимость вашего тарифа: {int(get_price(prediction[0]))}
       
       """
)