import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

# body

st.write("""
# اپ سهام نگر 

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
                    st.write('done!')
                else:
                    st.write('رمز عبور وارد شده صحیح نمی باشد')
            
    except:
        st.write(' !هیچ حساب کاربری یافت نشد')





# account making 
st.sidebar.header('''ساخت حساب کاربری''')
username = st.sidebar.text_input('نام کاربری : ')
password = st.sidebar.text_input('رمز عبور : ')
email = st.sidebar.text_input('ایمیل : ')
phone = st.sidebar.text_input('شماره همراه : ')
button_finish_2 = st.sidebar.button('ساخت اکانت')


if button_finish_2: 
    with open(str(username) + '.txt', 'w') as f:
        f.write( username + ' : ' + password)

    st.write('از همکاری شما متشکریم')
