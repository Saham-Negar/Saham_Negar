import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, RNN, SimpleRNN
import os

st.write("""
# اپ سهام نگر 

""")

# st.sidebar.header('انتخاب فایل اکسل سهام مورد نظر')
# uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
uploaded_file = pd.read_csv('سمگا.csv')
if uploaded_file is not None:
    df = pd.read_csv('سمگا.csv')
    st.write(df)
    st.write('در حال ذخیره سازی داده')
    with open('test.txt', 'w') as f:
        f.write('test is done!')
        st.write('that is finished!')
    
    
#     loop_test_num = 0

#     loop2345 = []

#     loop = []
#     num_test_1 = 1
#     error_persent = 0.025
# #     df = pd.read_csv(uploaded_file)
#     df = pd.read_csv('سمگا.csv')
#     df2 = df['Adj Close']

#     df2 = df2[-1:]

#     df2 = int(df2)

#     df_adj_close = df2

#     close = df['Close']

#     close = close[-1:]

#     df_close = close

#     val = df['Volume']

#     val = val[-1:]

#     df_val = val

#     df3 = df2

#     df3_2 = df2

#     df2 = str(df2)

#     df2 = df2[:-1]

#     df2 = int(df2)

#     df2 = df2 * 0.05

#     df2 = str(df2)

#     q = False
#     a = df2
#     m = 0
#     i = 0
#     while q == False:
#         try:
#             h = a[m:]
#             v = False
#             while v == False:
#                 try:
#                     i = i + 1
#                     k = h[:i]
#                     if '.' in k:
#                         k = k[:-1]

#                         ttxt = k

#                         st.write(k)
#                         q = True
#                         v = True

#                 except:
#                     continue

#         except:
#             continue

#     df2 = k + '0'
#     df3 = df3 - int(df2)
#     df4 = df3_2 + int(df2)
    
#     my_bar = st.progress(0)


#     for j in range(num_test_1):

#         df = pd.read_csv('سمگا.csv')

#         data = df.filter(['Adj Close'])

#         dataset = data.values

#         shaaa = df.shape[0]

# #         st.write()
# #         st.write(shaaa)
# #         st.write()
# #         st.write(j)

#         # for io in range(0, 105):
#         #     if shaaa == io:
#         #         epock = 5000
#         #         batch = 1

#         for io in range(120, 200):
#             if shaaa == io:
#                 epock = 1650
#                 batch = 32

#         for io in range(200, 300):
#             if shaaa == io:
#                 epock = 10
#                 batch = 8

#         for io in range(300, 400):
#             if shaaa == io:
#                 epock = 1450
#                 batch = 36

#         for io in range(400, 500):
#             if shaaa == io:
#                 epock = 1350
#                 batch = 38

#         for io in range(500, 600):
#             if shaaa == io:
#                 epock = 1250
#                 batch = 40

#         for io in range(600, 700):
#             if shaaa == io:
#                 epock = 1150
#                 batch = 42

#         for io in range(700, 800):
#             if shaaa == io:
#                 epock = 1050
#                 batch = 44

#         for io in range(800, 900):
#             if shaaa == io:
#                 epock = 950
#                 batch = 46

#         for io in range(900, 1000):
#             if shaaa == io:
#                 epock = 850
#                 batch = 48

#         for io in range(1000, 1100):
#             if shaaa == io:
#                 epock = 750
#                 batch = 50

#         for io in range(1100, 1200):
#             if shaaa == io:
#                 epock = 650
#                 batch = 52

#         for io in range(1200, 1300):
#             if shaaa == io:
#                 epock = 550
#                 batch = 54

#         for io in range(1300, 1400):
#             if shaaa == io:
#                 epock = 450
#                 batch = 56

#         for io in range(1400, 1500):
#             if shaaa == io:
#                 epock = 350
#                 batch = 58

#         for io in range(1500, 1600):
#             if shaaa == io:
#                 epock = 250
#                 batch = 60

#         for io in range(1600, 1700):
#             if shaaa == io:
#                 epock = 150
#                 batch = 62

#         for io in range(1700, 1800):
#             if shaaa == io:
#                 #  45 * 38 = 1710
#                 epock = 50
#                 batch = 64

#         for io in range(1800, 1900):
#             if shaaa == io:
#                 epock = 70
#                 batch = 66

#         for io in range(1900, 2000):
#             if shaaa == io:
#                 epock = 90
#                 batch = 68

#         for io in range(2000, 2100):
#             if shaaa == io:
#                 epock = 110
#                 batch = 70

#         for io in range(2100, 2200):
#             if shaaa == io:
#                 epock = 130
#                 batch = 72

#         for io in range(2200, 2300):
#             if shaaa == io:
#                 epock = 150
#                 batch = 74

#         for io in range(2300, 2400):
#             if shaaa == io:
#                 epock = 170
#                 batch = 76

#         for io in range(2400, 2500):
#             if shaaa == io:
#                 epock = 190
#                 batch = 78

#         for io in range(2500, 2600):
#             if shaaa == io:
#                 epock = 85
#                 batch = 98

#         # print(data[-1:])
        
#         prognum = int(j) * 10
#         my_bar.progress(prognum)

#         training_data_len = math.ceil(len(dataset) * .8)

#         scaler = MinMaxScaler(feature_range=(0, 1))
#         scaled_data = scaler.fit_transform(dataset)

#         train_data = scaled_data[0:training_data_len, :]

#         x_train = []
#         y_train = []

#         for i in range(60, len(train_data)):
#             x_train.append(train_data[i - 60:i, 0])
#             y_train.append(train_data[i, 0])

#         x_train, y_train = np.array(x_train), np.array(y_train)

#         x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#         model = Sequential()

#         model.add(LSTM(units=60, return_sequences=True, input_shape=(x_train.shape[1], 1)))
#         model.add(LSTM(units=60, return_sequences=True))
#         model.add(LSTM(units=60, return_sequences=True))
#         model.add(LSTM(units=60, return_sequences=True))
#         model.add(LSTM(units=60))
#         model.add(Dense(units=1))

#         model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')
#         model.fit(x_train, y_train, batch_size=batch, epochs=epock)


       
#         new_df = df.filter(['Adj Close'])
#         last_60_days = new_df[-60:].values
#         last_60_days_scaled = scaler.transform(last_60_days)
#         X_test = []
#         X_test.append(last_60_days_scaled)
#         X_test = np.array(X_test)
#         X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#         pred_price = model.predict(X_test)
#         pred_price234 = scaler.inverse_transform(pred_price)



# #         st.write(pred_price234)


        
#         ttext = str(pred_price234)
#         text = ttext[2:-2]


#         q = False
#         a = text
#         m = 0
#         i = 0
#         while q == False:
#             try:
#                 h = a[m:]
#                 v = False
#                 while v == False:
#                     try:
#                         i = i + 1
#                         k = h[:i]
#                         if '.' in k:
#                             k = k[:-1]

#                             ttxt = k

#                             st.write(k)
#                             q = True
#                             v = True

#                     except:
#                         continue

#             except:
#                 continue

#         text = int(ttxt)

       

#         ttxt = int(text)
# #         st.write(str(ttxt))
#         # ttxt = int(ttxt) + 500
#         loop.append(ttxt)


#     vv = 0
#     for i in range(num_test_1):
#         vv += loop[i]



#     vv = vv / num_test_1
#     vv = vv

#     loop2345.append(vv)
#     loop_test_num += vv






#     for i in range(num_test_1):
#         st.write()
#         st.write(str(i), '-pred : ')
#         st.write(str(loop[i]))

# #     st.write()
# #     st.write('Avj : ')
# #     st.write(vv)
# #     st.write()
    
    
    
# #     part 2!

#     loop = []
#     my_bar.progress(0)
#     for j in range(num_test_1):

#         df = pd.read_csv('سمگا.csv')

#         data = df.filter(['Adj Close'])

#         dataset = data.values

#         shaaa = df.shape[0]

# #         st.write()
# #         print(shaaa)
# #         print()
# #         print(j)

#         training_data_len = math.ceil(len(dataset) * .8)

#         scaler = MinMaxScaler(feature_range=(0, 1))
#         scaled_data = scaler.fit_transform(dataset)

#         train_data = scaled_data[0:training_data_len, :]

#         x_train = []
#         y_train = []

#         for i in range(60, len(train_data)):
#             x_train.append(train_data[i - 60:i, 0])
#             y_train.append(train_data[i, 0])

#         x_train, y_train = np.array(x_train), np.array(y_train)

#         x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
#         prognum = int(j) * 10
#         my_bar.progress(prognum)

#         model = Sequential()

#         model.add(LSTM(60, return_sequences=True, input_shape=(x_train.shape[1], 1)))
#         model.add(LSTM(60, return_sequences=True))
#         model.add(LSTM(60, return_sequences=True))
#         model.add(LSTM(60, return_sequences=True))
#         # model.add(Dense(25))
#         model.add(LSTM(60))
#         model.add(Dense(1))

#         model.compile(optimizer='adam', loss='mean_squared_error', metrics='accuracy')
#         model.fit(x_train, y_train, batch_size=batch, epochs=epock)

       

#         new_df = df.filter(['Adj Close'])
#         last_60_days = new_df[-60:].values
#         last_60_days_scaled = scaler.transform(last_60_days)
#         X_test = []
#         X_test.append(last_60_days_scaled)
#         X_test = np.array(X_test)
#         X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#         pred_price = model.predict(X_test)
#         pred_price234 = scaler.inverse_transform(pred_price)

        

# #         print(pred_price234)

       

#         ttext = str(pred_price234)
#         text = ttext[2:-2]

#         # len() for text ^^^^^

#         q = False
#         a = text
#         m = 0
#         i = 0
#         while q == False:
#             try:
#                 h = a[m:]
#                 v = False
#                 while v == False:
#                     try:
#                         i = i + 1
#                         k = h[:i]
#                         if '.' in k:
#                             k = k[:-1]

#                             ttxt = k

# #                             print(k)
#                             q = True
#                             v = True

#                     except:
#                         continue

#             except:
#                 continue

#         text = int(ttxt)

        

#         ttxt = int(text)
# #         print(str(ttxt))
#         # ttxt = int(ttxt) + 500
#         loop.append(ttxt)

   


#     vv = 0
#     for i in range(num_test_1):
#         vv += loop[i]



#     vv = vv / num_test_1
#     vv = vv
#     loop2345.append(vv)

#     loop_test_num += vv

#     loop_test_num = loop_test_num / 2

#     last_loop = loop2345[0] + loop2345[1]
#     last_loop = last_loop / 2

# #     print()
# #     print('last_befor_lopp_test :')
# #     print(loop_test_num)
# #     print()
# #     print('last_befor_lopp :')
# #     print(last_loop)
# #     print()

#     dff = False

#     while dff == False:

#         if loop_test_num >= df3:
#             # loop_test_num += 500
# #             print(loop_test_num)
#             dff = True


#         else:
#             loop_test_num += 1

# #     print(loop_test_num)

#     dff2 = False

#     while dff2 == False:

#         if loop_test_num >= df4:
#             loop_test_num -= 1


#         else:
#             # loop_test_num -= 500
# #             print(loop_test_num)
#             dff2 = True

# #     print(loop_test_num)

   
#     for i in range(num_test_1):
#         st.write()
#         st.write(str(i), '-pred : ')
#         st.write(str(loop[i]))

# #     print()
# #     print('Avj : ')
# #     print(vv)
# #     print()
# #     print()
#     st.write('Avj_loop_test_num : ')
#     st.write(loop_test_num)
# #     print()
# #     print()


#     test_i = int(df_val)

# #     print(test_i)

#     loop_test_num_range = int(loop_test_num) * error_persent

#     loop_test_num_range = int(loop_test_num_range)

#     range_1 = loop_test_num_range + int(loop_test_num)

#     dff = False

#     while dff == False:

#         if range_1 >= df3:
# #             print(range_1)
#             dff = True


#         else:
#             range_1 += 1

# #     print(range_1)

#     dff2 = False

#     while dff2 == False:

#         if range_1 >= df4:
#             range_1 -= 1


#         else:
# #             print(range_1)
#             dff2 = True

#     range_2 = loop_test_num_range - int(loop_test_num)

#     dff = False

#     while dff == False:

#         if range_2 >= df3:
# #             print(range_2)
#             dff = True


#         else:
#             range_2 += 1

# #     print(range_2)

#     dff2 = False

#     while dff2 == False:

#         if range_2 >= df4:
#             range_2 -= 1


#         else:
# #             print(range_2)
#             dff2 = True

#     st.write('the highest range in pred is : ')
#     st.write(range_1)
#     st.write()
#     st.write('the lowes range in pred is : ')
#     st.write(range_2)
#     st.write()



   


#     if int(loop_test_num) > int(df_adj_close):
#         st.write('the price will get higher  ^')
#         st.write()

#     elif int(loop_test_num) < int(df_adj_close):
#         st.write('the price will get Lower   v')
#         st.write()


#     elif int(loop_test_num) == int(df_close):
#         st.write('the price will be in range !')
#         st.write()
    
    
    
    
    
    
    
    
    
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
