import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import Pool
from sklearn.preprocessing import LabelEncoder
import pickle
# Preprocessing data:
price_car_df = pd.read_csv('sub.csv')
data = pd.read_excel('df.xlsx')

# PANDAS settings


# Create intervals of experience
def create_interval(x):
    """
    Classifies continuous values to
    most popular experience values

    :param: x
    :return: class_of_exp
    """
    if x <= 1:
        val = '0-1'
    elif (x > 1) & (x <= 5):
        val = '2-5'
    elif (x > 5) & (x <= 10):
        val = '6-10'
    elif (x > 10) & (x <= 25):
        val = '11-25'
    elif (x > 25) & (x <= 50):
        val = '26-50'
    elif (x > 50):
        val = '51+'
    return val


def preprocessing(data, data2) -> pd.DataFrame:
    """
    Preprocessing pipeline
    :param data:
    :return: data
    """
    data2.columns = ['Марка автомобиля', 'Средняя стоимость авто', 'Медианное стоимость авто']
    data2 = data2.set_index('Марка автомобиля').sort_index()
    data['Стаж'] = data['Стаж вождения'].apply(lambda x: create_interval(x))
    data['ДТП'] = data[['Легкое ДТП', 'Крупное ДТП', 'Угон, ДТП без востановления авто']].idxmax(axis=1)
    data.loc[(data['Легкое ДТП'] == 0) & (data['Крупное ДТП'] == 0) & (data['Угон, ДТП без востановления авто'] == 0), 'ДТП'] = 'Без ДТП'
    data = data.join(data2, on='Марка автомобиля')
    return data[['Возраст', 'Марка автомобиля', 'Тип страхования', 'Стаж', 'ДТП', 'Медианное стоимость авто', 'Выплата, руб.']]

data = preprocessing(data, price_car_df)
y = data['Выплата, руб.']
X = data.drop(columns='Выплата, руб.')

#catboost
cat_features = ['ДТП', 'Марка автомобиля', 'Стаж', 'Тип страхования']
cat = CatBoostRegressor(depth = 8, iterations = 150, learning_rate = 0.1, l2_leaf_reg = 3)
pool=Pool(X, y, cat_features=cat_features, feature_names=list(X.columns))
cat.fit(pool,verbose=False)


#cat.save_model("model")

pickle.dump(cat, open('insurance_cat_clf.pkl', 'wb'))

print('finish!')