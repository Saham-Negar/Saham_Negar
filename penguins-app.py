import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
from PIL import Image
from datetime import datetime, timedelta

image_1 = Image.open('aa.ico')
st.beta_set_page_config(page_title='سهام نگر', page_icon=image_1)





# body


st.write("""
# اپ سهام نگر 
مسیری به سوی اینده
""")







# side bar


# account Log in
st.sidebar.header(''' حساب کاربری''')
username_enter = st.sidebar.text_input('نام حساب : ')
password_enter = st.sidebar.text_input('رمز عبور حساب : ')
button_finish_1 = st.sidebar.button('ورود')

if button_finish_1:
    try: 
        with open(username_enter + '.txt', 'r') as f:
            data = f.read()
            data2 = [k for k in data.split('\n') if len(k)>3]
            data3 = {k.split(':')[0].strip():k.split(':')[1].strip() for k in data2}
            if username_enter in data3:
                if password_enter == data3[username_enter]:
                    account_maximum_date = data3['date']
                    account_maximum_date = account_maximum_date.replace('-', '')
                    account_maximum_date = int(account_maximum_date)
                    date_now = datetime.today()
                    date_now = str(date_now)[:10]
                    date_now = date_now.replace('-', '')
                    date_now = int(date_now)
                    if date_now < account_maximum_date:
                        image = Image.open('C:/Users/New/Desktop/12.png')
                        st.image(image=image, caption='سهام شبریز 12 روز اینده', use_column_width=True)
                        image = Image.open('C:/Users/New/Desktop/15.png')
                        st.image(image=image, caption='سهام بترانس 15 روز اینده', use_column_width=True)
                    else:
                        st.write('!تاریخ معتبر نمی باشد')
                    
                else:
                    st.write('!رمز عبور وارد شده صحیح نمی باشد')
            
    except:
        st.write(' !هیچ حساب کاربری یافت نشد')









# account making 
st.sidebar.header('\n')
st.sidebar.header('\n')
st.sidebar.header('\n')
st.sidebar.header('''ساخت حساب کاربری''')
username = st.sidebar.text_input('نام کاربری : ')
password = st.sidebar.text_input('رمز عبور : ')
email = st.sidebar.text_input('ایمیل : ')
phone = st.sidebar.text_input('شماره همراه : ')
account_days_30 = st.sidebar.checkbox('یک ماهه')
account_days_90 = st.sidebar.checkbox('سه ماهه')
account_days_183 = st.sidebar.checkbox('شش ماهه')
account_days_366 = st.sidebar.checkbox('دوازده ماهه')
button_finish_2 = st.sidebar.button('ساخت حساب')


if button_finish_2: 
    try:
        with open(username + '.txt', 'r') as f:
            st.write('با این مشخصات اکانت دیگر وجود دارد')
    except:
        with open(username + '.txt', 'w') as f:
            if account_days_30:
                date = datetime.today() + timedelta(30)
            elif account_days_90:
                date = datetime.today() + timedelta(90)
            elif account_days_183:
                date = datetime.today() + timedelta(183)
            elif account_days_366:
                date = datetime.today() + timedelta(366)
            date = str(date)[:10]
            f.write(username + ' : ' + password)
            f.write('\n')
            f.write('date' + ' : ' + date)       
            st.write('!با موفقیت انجام گردید')
