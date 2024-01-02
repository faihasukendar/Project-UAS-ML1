import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori

st.set_page_config(page_title="Market Basket Analysis", page_icon="🥖", layout="wide", initial_sidebar_state="expanded", menu_items=None)

st.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">', unsafe_allow_html=True)

with open("style.css") as f:
  st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown(
"""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #fcba03;">
  <a href="/" target="_self" id="main-btn">Market Basket Analysis</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav"">
      <li class="nav-item">
        <a id="notebook" class="nav-link active" href="https://www.kaggle.com/code/danielsimamora/market-basket-analysis" target="_blank">📄Notebook</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html = True)

st.markdown("""<p id="title-1z2x">Market Basket Analysis</p>""", unsafe_allow_html=True)
st.markdown("""<p id="caption-1z2x">Analisis Asosiasi Item Dengan Menggunakan Apriori</p>""", unsafe_allow_html=True)
st.markdown("""<p id="caption-1z2x">Nama : Faiha Atsaa S | NIM : 211351052</p>""", unsafe_allow_html=True)

# Processing the CSV as Pandas DataFrame
df = pd.read_csv("BreadBasket_DMS.csv")
df['Date'] = pd.to_datetime(df['Date'])

df["month"] = df['Date'].dt.month
df["day"] = df['Date'].dt.weekday

df["month"].replace([i for i in range(1, 12 + 1)], ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], inplace = True)
df["day"].replace([i for i in range(6 + 1)], ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], inplace = True)

# Filter the data based on User Inputs
def get_data(month = '', day = ''):
    data = df.copy()
    filtered = data.loc[
        (data["month"].str.contains(month.title())) &
        (data["day"].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else "No result!"


def user_input_features():
    item = st.selectbox("Item", ['Alfajores', 'Hot chocolate', 'Cookies', 'Muffin', 'Coffee', 'Pastry', 'Medialuna', 'Tea', 'none','Tartine', 'Basket', 'Mineral water', 'Farm house', 'Fudge','Juice', "Ella's kitchen pouches", 'Victorian sponge', 'Frittata','Hearty & seasonal', 'Soup', 'Pick and mix bowls', 'Smoothies','Cake', 'Mighty protein', 'Chicken sand', 'Coke','My-5 fruit shoot', 'Focaccia', 'Sandwich', 'Alfajores', 'Eggs','Brownie', 'Dulce de leche', 'Honey', 'The bart', 'Granola','Fairy doors', 'Empanadas', 'Keeping it local', 'Art tray','Bowl nic pitt', 'Bread pudding', 'Adjustment', 'Truffles','Chimichurri oil', 'Bacon', 'Spread', 'Kids biscuit', 'Siblings','Caramel bites', 'Jammie dodgers', 'Tiffin', 'Olum & polenta','Polenta', 'The nomad', 'Hack the stack', 'Bakewell','Lemon and coconut', 'Toast', 'Scone', 'Crepes', 'Vegan mincepie','Bare popcorn', 'Muesli', 'Crisps', 'Pintxos', 'Gingerbread syrup','Panatone', 'Brioche and salami', 'Afternoon with the baker','Salad', 'Chicken stew', 'Spanish brunch','Raspberry shortbread sandwich', 'Extra salami or feta','Duck egg', 'Baguette', "Valentine's card", 'Tshirt','Vegan feast', 'Postcard', 'Nomad bag', 'Chocolates','Coffee granules', 'Drinking chocolate spoons', 'Christmas common','Argentina night', 'Half slice monster', 'Gift voucher','Cherry me dried fruit', 'Mortimer', 'Raw bars', 'Tacos/fajita'])
    month = st.select_slider("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    day = st.select_slider('Day', ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], value="Sat")

    return month, day, item

month, day, item = user_input_features()

data = get_data(month, day)

st.text("")
st.text("")

# ==========================================================================================================================================================================

def encode(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1

if type(data) != type("No result!"):
  item_count = data.groupby(['Transaction','Item'])['Item'].count().reset_index(name = "Count")
  item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
  item_count_pivot = item_count_pivot.applymap(encode)

  support = 0.01 # atau 1%
  frequent_items = apriori(item_count_pivot, min_support = support, use_colnames = True)

  metric = "lift"
  min_threshold = 1

  rules = association_rules(frequent_items, metric = metric, min_threshold = min_threshold)[["antecedents", "consequents", "support", "confidence", "lift"]]
  rules.sort_values('confidence', ascending = False, inplace = True)

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)
    
def return_item_df(item_antecedents):
    data = rules[["antecedents", "consequents"]].copy()
    
    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)
    
    return list(data.loc[data["antecedents"] == item_antecedents].iloc[0,:])


# ==========================================================================================================================================================================

st.text("")
st.text("")

if type(data) != type("No result!"):
  st.markdown("""<p id="recommendation-1z2x">Recommendation:</p>""", unsafe_allow_html=True)
  st.success(f"Customer who buys **{item}**, also buys **{return_item_df(item)[1]}**!")