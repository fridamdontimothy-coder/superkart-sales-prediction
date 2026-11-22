import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='SuperKart Sales Predictor',layout='wide')
st.title('SuperKart | Product–Store Sales Prediction')
st.caption('Predicts product–store sales revenue, not a validated next-quarter time-series forecast.')
@st.cache_resource
def load():
 return joblib.load(ROOT/'superkart_model.joblib'),json.loads((ROOT/'superkart_metadata.json').read_text())
model,meta=load()
features=meta['features']
tab1,tab2=st.tabs(['Single prediction','Batch predictions'])
with tab1:
 with st.form('single'):
  a,b=st.columns(2)
  with a:
   weight=st.number_input('Product weight',min_value=0.,value=12.66)
   sugar=st.selectbox('Sugar content',['Low Sugar','Regular','No Sugar'])
   area=st.number_input('Allocated area ratio',min_value=0.,max_value=1.,value=.027,format='%.3f')
   mrp=st.number_input('Product MRP',min_value=0.,value=117.08)
   size=st.selectbox('Store size',['Small','Medium','High'])
  with b:
   tier=st.selectbox('City tier',['Tier 1','Tier 2','Tier 3'])
   store=st.selectbox('Store type',['Supermarket Type1','Supermarket Type2','Supermarket Type3','Departmental Store','Food Mart'])
   prefix=st.selectbox('Product ID prefix',['FD','DR','NC'])
   age=st.number_input('Store age in years (2025 reference)',min_value=0,value=16)
   cat=st.selectbox('Product type category',['Perishables','Non Perishables'])
  submit=st.form_submit_button('Predict single record')
 if submit:
  row=[weight,sugar,area,mrp,size,tier,store,prefix,age,cat]
  x=pd.DataFrame([row],columns=features)
  st.metric('Predicted sales revenue',f'{model.predict(x)[0]:,.2f}')
  st.dataframe(x,use_container_width=True)
with tab2:
 up=st.file_uploader('Upload CSV containing the 10 required input columns',type='csv')
 if up is not None:
  data=pd.read_csv(up)
  missing=[c for c in features if c not in data]
  if missing:st.error('Missing columns: '+', '.join(missing))
  else:
   result=data.copy()
   try:
    result['Predicted_Sales_Revenue']=model.predict(data[features])
    st.success(f'Predicted {len(result)} rows')
    st.dataframe(result,use_container_width=True)
    st.download_button('Download predictions CSV',result.to_csv(index=False).encode(),'SuperKart_predictions.csv','text/csv')
   except Exception as exc:st.error(f'Prediction failed: {exc}')
