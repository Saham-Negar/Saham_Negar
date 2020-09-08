import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
# from tkinter import *
# import tkinter
# from tkinter import filedialog, Text , Label , messagebox
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
# import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, RNN, SimpleRNN

st.write("""
# اپ سهام نگر 

## ...بزودی

""")

st.sidebar.header('انتخاب فایل اکسل سهام مورد نظر')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    loop_test_num = 0

    loop2345 = []

    loop = []
    num_test_1 = 10
    error_persent = 0.025


    # df = pd.read_csv(file)
    

    df2 = df['Adj Close']

    df2 = df2[-1:]

    df2 = int(df2)

    df_adj_close = df2

    close = df['Close']

    close = close[-1:]

    df_close = close

    val = df['Volume']

    val = val[-1:]

    df_val = val

    df3 = df2

    df3_2 = df2

    df2 = str(df2)

    df2 = df2[:-1]

    df2 = int(df2)

    df2 = df2 * 0.05

    df2 = str(df2)

    q = False
    a = df2
    m = 0
    i = 0
    while q == False:
        try:
            h = a[m:]
            v = False
            while v == False:
                try:
                    i = i + 1
                    k = h[:i]
                    if '.' in k:
                        k = k[:-1]

                        ttxt = k

                        st.write(k)
                        q = True
                        v = True

                except:
                    continue

        except:
            continue

    df2 = k + '0'
    df3 = df3 - int(df2)
    df4 = df3_2 + int(df2)


    for j in range(num_test_1):
        df = pd.read_csv(uploaded_file)

        data = df.filter(['Adj Close'])

        dataset = data.values

        shaaa = df.shape[0]

#         print()
        st.write(shaaa)
#         print()
        st.write(j)

        # for io in range(0, 105):
        #     if shaaa == io:
        #         epock = 5000
        #         batch = 1

        for io in range(120, 200):
            if shaaa == io:
                epock = 1650
                batch = 32

        for io in range(200, 300):
            if shaaa == io:
                epock = 10
                batch = 8

        for io in range(300, 400):
            if shaaa == io:
                epock = 1450
                batch = 36

        for io in range(400, 500):
            if shaaa == io:
                epock = 1350
                batch = 38

        for io in range(500, 600):
            if shaaa == io:
                epock = 1250
                batch = 40

        for io in range(600, 700):
            if shaaa == io:
                epock = 1150
                batch = 42

        for io in range(700, 800):
            if shaaa == io:
                epock = 1050
                batch = 44

        for io in range(800, 900):
            if shaaa == io:
                epock = 950
                batch = 46

        for io in range(900, 1000):
            if shaaa == io:
                epock = 850
                batch = 48

        for io in range(1000, 1100):
            if shaaa == io:
                epock = 750
                batch = 50

        for io in range(1100, 1200):
            if shaaa == io:
                epock = 650
                batch = 52

        for io in range(1200, 1300):
            if shaaa == io:
                epock = 550
                batch = 54

        for io in range(1300, 1400):
            if shaaa == io:
                epock = 450
                batch = 56

        for io in range(1400, 1500):
            if shaaa == io:
                epock = 350
                batch = 58

        for io in range(1500, 1600):
            if shaaa == io:
                epock = 250
                batch = 60

        for io in range(1600, 1700):
            if shaaa == io:
                epock = 150
                batch = 62

        for io in range(1700, 1800):
            if shaaa == io:
                #  45 * 38 = 1710
                epock = 50
                batch = 64

        for io in range(1800, 1900):
            if shaaa == io:
                epock = 70
                batch = 66

        for io in range(1900, 2000):
            if shaaa == io:
                epock = 90
                batch = 68

        for io in range(2000, 2100):
            if shaaa == io:
                epock = 110
                batch = 70

        for io in range(2100, 2200):
            if shaaa == io:
                epock = 130
                batch = 72

        for io in range(2200, 2300):
            if shaaa == io:
                epock = 150
                batch = 74

        for io in range(2300, 2400):
            if shaaa == io:
                epock = 170
                batch = 76

        for io in range(2400, 2500):
            if shaaa == io:
                epock = 190
                batch = 78

        for io in range(2500, 2600):
            if shaaa == io:
                epock = 85
                batch = 98

        # print(data[-1:])

        training_data_len = math.ceil(len(dataset) * .8)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        train_data = scaled_data[0:training_data_len, :]

        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()

        model.add(LSTM(units=60, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=60, return_sequences=True))
        model.add(LSTM(units=60, return_sequences=True))
        model.add(LSTM(units=60, return_sequences=True))
        model.add(LSTM(units=60))
        model.add(Dense(units=1))

        

        model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')
        model.fit(x_train, y_train, batch_size=batch, epochs=epock)


       
        new_df = df.filter(['Adj Close'])
        last_60_days = new_df[-60:].values
        last_60_days_scaled = scaler.transform(last_60_days)
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price234 = scaler.inverse_transform(pred_price)

       
        st.write(pred_price234)



        # y_test = dataset[training_data_len:, :]

        # rmse = np.sqrt(np.mean(((pred_price234 - y_test) ** 2)))
        #
        # print()
        #
        # print(str(rmse))

        # name234 = name + '2'
        #
        # name222 = 'C:/Users/New/Documents/TseClient 2.0/' + name234 + '.csv'
        # # سمگا.csv
        # file2 = name222

        # with open(file, 'a') as f:
        #
        #     f.write('\n')
        #     # line_1 = '20200311,13120.00,13120.00,13120.00,'
        #     # pred_price00 = str(pred_price234)
        #     # pred_price23456 = pred_price00[2:-2]
        #     # f.write(line_1 + str(pred_price23456))

        ttext = str(pred_price234)
        text = ttext[2:-2]

        # len() for text ^^^^^

        q = False
        a = text
        m = 0
        i = 0
        while q == False:
            try:
                h = a[m:]
                v = False
                while v == False:
                    try:
                        i = i + 1
                        k = h[:i]
                        if '.' in k:
                            k = k[:-1]

                            ttxt = k

                            st.write(k)
                            q = True
                            v = True

                    except:
                        continue

            except:
                continue

        text = int(ttxt)




        ttxt = int(text)
        st.write(str(ttxt))
        # ttxt = int(ttxt) + 500
        loop.append(ttxt)


    vv = 0
    for i in range(num_test_1):
        vv += loop[i]


    # vv = loop[0] + loop[1] + loop[2] + loop[3] + loop[4]
    vv = vv / num_test_1
    vv = vv

    loop2345.append(vv)
    loop_test_num += vv





    

    for i in range(num_test_1):
#         print()
        st.write(str(i), '-pred : ')
        st.write(str(loop[i]))

#     print()
    st.write('Avj : ')
    st.write(vv)
#     print()

    # df = m_qoute
    # df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')

    loop = []
    for j in range(num_test_1):
        df = pd.read_csv(uploaded_file)

#         df = pd.read_csv(file)

        data = df.filter(['Adj Close'])

        dataset = data.values

        shaaa = df.shape[0]

#         print()
        st.write(shaaa)
#         print()
        st.write(j)

        training_data_len = math.ceil(len(dataset) * .8)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        train_data = scaled_data[0:training_data_len, :]

        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()

        model.add(LSTM(60, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(60, return_sequences=True))
        model.add(LSTM(60, return_sequences=True))
        model.add(LSTM(60, return_sequences=True))
        # model.add(Dense(25))
        model.add(LSTM(60))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')
        model.fit(x_train, y_train, batch_size=batch, epochs=epock)

        

        new_df = df.filter(['Adj Close'])
        last_60_days = new_df[-60:].values
        last_60_days_scaled = scaler.transform(last_60_days)
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price234 = scaler.inverse_transform(pred_price)

        # loop.append(pred_price234)

        st.write(pred_price234)

        # y_test = dataset[training_data_len:, :]

        # rmse = np.sqrt(np.mean(((pred_price234 - y_test) ** 2)))
        #
        # print()
        #
        # print(str(rmse))

        # name234 = name + '2'
        #
        # name222 = 'C:/Users/New/Documents/TseClient 2.0/' + name234 + '.csv'
        # # سمگا.csv
        # file2 = name222

        # with open(file, 'a') as f:
        #
        #     f.write('\n')
        #     # line_1 = '20200311,13120.00,13120.00,13120.00,'
        #     # pred_price00 = str(pred_price234)
        #     # pred_price23456 = pred_price00[2:-2]
        #     # f.write(line_1 + str(pred_price23456))

        ttext = str(pred_price234)
        text = ttext[2:-2]

        # len() for text ^^^^^

        q = False
        a = text
        m = 0
        i = 0
        while q == False:
            try:
                h = a[m:]
                v = False
                while v == False:
                    try:
                        i = i + 1
                        k = h[:i]
                        if '.' in k:
                            k = k[:-1]

                            ttxt = k

                            st.write(k)
                            q = True
                            v = True

                    except:
                        continue

            except:
                continue

        text = int(ttxt)

  

        ttxt = int(text)
        st.write(str(ttxt))
        # ttxt = int(ttxt) + 500
        loop.append(ttxt)

    # df = pd.read_csv(file)
    #
    # df2 = df['Adj Close']
    #
    # df2 = df2[-1:]
    #
    # df2 = int(df2)
    #
    # df_adj_close = df2
    #
    # close = df['Close']
    #
    # close = close[-1:]
    #
    # df_close = close
    #
    # val = df['Volume']
    #
    # val = val[-1:]
    #
    # df_val = val
    #
    # df3 = df2
    #
    # df3_2 = df2
    #
    # df2 = str(df2)
    #
    # df2 = df2[:-1]
    #
    # df2 = int(df2)

    # data2 = [k for k in data_saham.split('\n') if len(k)>3]
    # data3 = {k.split(':')[0].strip():k.split(':')[1].strip() for k in data2}
    # b = data3[name]
    # qww = 'http:' + b[4:]
    # print(qww)
    #
    # os.system('taskkill /f /im firefox.exe')
    # options = webdriver.FirefoxOptions()
    # options.add_argument('-headless')
    # driver = webdriver.Firefox(firefox_options=options)
    #
    # # qqw = 'http://www.tsetmc.com/Loader.aspx?ParTree=151311&i=70391097626818082'
    # # qqw = 'http://www.tsetmc.com/Loader.aspx?ParTree=151311&i=46741025610365786'
    #
    # # /html/body/div[4]/form/div[3]/div[1]
    #
    # driver.get(qww)
    #
    # time.sleep(1)
    #
    # g = False
    #
    # while g == False:
    #     try:
    #         search = driver.find_element_by_xpath('/html/body/div[4]/form/div[3]/div[1]')
    #         k = search.text
    #         k = k[39:]
    #         print(k)
    #
    #         if 'بازار پايه نارنجي فرابورس' in k :
    #             print('0.02')
    #             num_saham = 0.02
    #
    #         elif 'بازار پايه قرمز فرابورس' in k :
    #             print('0.01')
    #             num_saham = 0.01
    #
    #         elif 'بازار پايه زرد فرابورس' in k :
    #             print('0.03')
    #             num_saham = 0.03
    #
    #         elif 'بازار اول فرابورس' in k :
    #             print('0.05')
    #             num_saham = 0.05
    #
    #         elif 'بازار دوم فرابورس' in k :
    #             print('0.05')
    #             num_saham = 0.05
    #
    #         driver.quit()
    #         os.system('taskkill /f /im firefox.exe')
    #         g = True
    #
    #     except:
    #         time.sleep(1)
    #         continue

    # df2 = df2 * 0.05

    # df2 = str(df2)

    # df2 = df2 + '0'

    # df3 = df3 - int(df2)


    # q = False
    # a = df2
    # m = 0
    # i = 0
    # while q == False:
    #     try:
    #         h = a[m:]
    #         v = False
    #         while v == False:
    #             try:
    #                 i = i + 1
    #                 k = h[:i]
    #                 if '.' in k:
    #                     k = k[:-1]
    #
    #                     ttxt = k
    #
    #                     print(k)
    #                     q = True
    #                     v = True
    #
    #             except:
    #                 continue
    #
    #     except:
    #         continue
    #
    # df2 = k + '0'
    # df3 = df3 - int(df2)
    # df4 = df3_2 + int(df2)


    vv = 0
    for i in range(num_test_1):
        vv += loop[i]

    # vv = loop[0] + loop[1] + loop[2] + loop[3] + loop[4]

    vv = vv / num_test_1
    vv = vv
    loop2345.append(vv)

    loop_test_num += vv

    loop_test_num = loop_test_num / 2

    last_loop = loop2345[0] + loop2345[1]
    last_loop = last_loop / 2

#     print()
    st.write('last_befor_lopp_test :')
    st.write(loop_test_num)
#     print()
    st.write('last_befor_lopp :')
    st.write(last_loop)
#     print()

    dff = False

    while dff == False:

        if loop_test_num >= df3:
            # loop_test_num += 500
            st.write(loop_test_num)
            dff = True


        else:
            loop_test_num += 1

    st.write(loop_test_num)

    dff2 = False

    while dff2 == False:

        if loop_test_num >= df4:
            loop_test_num -= 1


        else:
            # loop_test_num -= 500
            st.write(loop_test_num)
            dff2 = True

    st.write(loop_test_num)

    
    for i in range(num_test_1):
#         print()
        st.write(str(i), '-pred : ')
        st.write(str(loop[i]))

#     print()
    st.write('Avj : ')
    st.write(vv)
#     print()
#     print()
    st.write('Avj_loop_test_num : ')
    st.write(loop_test_num)
#     print()
#     print()
    # print('Avj_last_loop : ')
    # print(last_loop)
    # print()

    test_i = int(df_val)

    st.write(test_i)

    loop_test_num_range = int(loop_test_num) * error_persent

    loop_test_num_range = int(loop_test_num_range)

    range_1 = loop_test_num_range + int(loop_test_num)

    dff = False

    while dff == False:

        if range_1 >= df3:
            st.write(range_1)
            dff = True


        else:
            range_1 += 1

    st.write(range_1)

    dff2 = False

    while dff2 == False:

        if range_1 >= df4:
            range_1 -= 1


        else:
            st.write(range_1)
            dff2 = True

    range_2 = loop_test_num_range - int(loop_test_num)

    dff = False

    while dff == False:

        if range_2 >= df3:
            st.write(range_2)
            dff = True


        else:
            range_2 += 1

    st.write(range_2)

    dff2 = False

    while dff2 == False:

        if range_2 >= df4:
            range_2 -= 1


        else:
            st.write(range_2)
            dff2 = True

    st.write('the highest range in pred is : ')
    st.write(range_1)
#     print()
    st.write('the lowes range in pred is : ')
    st.write(range_2)
#     print()



    # if test_i >= 10000000:
    #     print('More than 10,000,000 volume !')
    #     i2 = str(test_i)
    #     range_val = i2[:4]
    #     range_val = int(range_val)
    #
    #
    # elif test_i < 10000000:
    #     print('lower than 10,000,000 volume')
    #     i2 = str(test_i)
    #     range_val = i2[:3]
    #     range_val = int(range_val)


    if int(loop_test_num) > int(df_adj_close):
        st.write('the price will get higher  ^')
#         print()

    elif int(loop_test_num) < int(df_adj_close):
        st.write('the price will get Lower   v')
#         print()


    elif int(loop_test_num) == int(df_close):
        st.write('the price will be in range !')
#         print()
    
    
    
    
    
    
    
    
    
else:
#     def user_input_features():
     st.write('هیچ فایلی پیدا نشد')
      
      
      
# st.sidebar.markdown("""
# [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
# """)

# Collects user input features into dataframe
# uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
# if uploaded_file is not None:
#     input_df = pd.read_csv(uploaded_file)
# else:
#     def user_input_features():
# #         island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
# #         sex = st.sidebar.selectbox('Sex',('male','female'))
# #         bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
# #         bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
# #         flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
# #         body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
# #         data = {'island': island,
# #                 'bill_length_mm': bill_length_mm,
# #                 'bill_depth_mm': bill_depth_mm,
# #                 'flipper_length_mm': flipper_length_mm,
# #                 'body_mass_g': body_mass_g,
# #                 'sex': sex}
# #         features = pd.DataFrame(data, index=[0])
# #         return features
#     input_df = user_input_features()

# # Combines user input features with entire penguins dataset
# # This will be useful for the encoding phase
# penguins_raw = pd.read_csv('penguins_cleaned.csv')
# # penguins = penguins_raw.drop(columns=['species'])
# # df = pd.concat([input_df,penguins],axis=0)
# df = pd.readcsv)

# # Encoding of ordinal features
# # https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
# # encode = ['sex','island']
# # for col in encode:
# #     dummy = pd.get_dummies(df[col], prefix=col)
# #     df = pd.concat([df,dummy], axis=1)
# #     del df[col]
# # df = df[:1] # Selects only the first row (the user input data)

# # Displays the user input features
# st.subheader('User Input features')

# if uploaded_file is not None:
#     st.write(df)
# else:
#     st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
#     st.write(df)

# # Reads in saved classification model
# load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# # Apply model to make predictions
# # prediction = load_clf.predict(df)
# # prediction_proba = load_clf.predict_proba(df)


# # st.subheader('Prediction')
# # penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
# # st.write(penguins_species[prediction])

# st.subheader('Prediction Probability')
# # st.write(prediction_proba)
