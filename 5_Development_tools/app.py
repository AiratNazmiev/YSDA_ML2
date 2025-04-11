import streamlit as st

st.title("This is a title")
st.header("This is a header")
st.subheader("This is a subheader")
st.text("This is a text")
st.markdown("# This is a markdown header 1")
st.markdown("## This is a markdown header 2")
st.markdown("### This is a markdown header 3")
st.markdown("This is a markdown: *bold* **italic** `inline code` ~strikethrough~")
st.markdown("""This is a code block with syntax highlighting
```python
print("Hello world!")
```
""")
st.html(
    "image from url example with html: "
    "<img src='https://www.wallpaperflare.com/static/450/825/286/kitten-cute-animals-grass-5k-wallpaper.jpg' width=400px>",
)


st.write("Text with write")
st.write(range(10))

st.success("Success")
st.info("Information")
st.warning("Warning")
st.error("Error")
exp = ZeroDivisionError("Trying to divide by Zero")
st.exception(exp)



from urllib import request
request.urlretrieve(
    "http://craphound.com/images/1006884_2adf8fc7.jpg",
    "image_example.jpg",
)

from PIL import Image
img = Image.open("image_example.jpg")

st.image(img, width=1000)


# чекбокс
if st.checkbox("Show/Hide"):
    st.text("Showing the widget")
else:
    st.warning("Not showing what is inside")
    
# выбор опции кружочками
status = st.radio("Select Gender: ", ('Male', 'Female'))
if (status == 'Male'):
    st.success("Male")
else:
    st.success("Female")
    
# выбор опции выпадающим меню
hobby = st.selectbox(
    "Hobbies: ",
    ['Dancing', 'Reading', 'Sports'],
)
st.write("Your hobby is: ", hobby)

# выбор нескольких опций
hobbies = st.multiselect(
    "Hobbies: ",
    ['Dancing', 'Reading', 'Sports'],
)
st.write("You selected", len(hobbies), 'hobbies')

# кнопка без функционала
st.button("Click me for no reason")

# кнопка, показывающая текст, когда нажата
if(st.button("Click me")):
    st.text("You did it, you clicked me!!!")
    
# текстовый input: label - название, value - что написано по дефолту
name = st.text_input(label="Enter Your name", value="Type Here ...")
if(st.button('Submit')):
    result = name.title()
    st.success(result)

# слайдер
level = st.slider("Select the level", 1, 5)
st.text('Selected: {}'.format(level))


# инициализируем переменные
st.session_state.key1 = 'value1'     # Attribute API
st.session_state['key2'] = 'value2'  # Dictionary like API

# посмотреть что в st.session_state
st.write(st.session_state)

# magic
st.session_state

# ошибка если неправильный ключ
#st.write(st.session_state['missing_key'])

# key - позволяет указать в какое поле session_state записать объект
st.text_input("Please input something", key="my input")
st.session_state


import streamlit as st
import pandas as pd

@st.cache_data  # кэширование
def load_data(url):
    df = pd.read_csv(url)  # скачивание датасета
    return df

df = load_data("https://github.com/plotly/datasets/raw/master/uber-rides-data1.csv")
st.dataframe(df)

st.button("Rerun")


import streamlit as st
from transformers import pipeline

@st.cache_resource  # кэширование
def load_model():
    return pipeline("sentiment-analysis")  # скачивание модели

model = load_model()

query = st.text_input("Your query", value="I love Streamlit! 🎈")
if query:
    result = model(query)[0]  # классифицируем
    st.write(query)
    st.write(result)