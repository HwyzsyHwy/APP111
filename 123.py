import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image

# 允许加载高分辨率图片
Image.MAX_IMAGE_PIXELS = None

# 加载已经训练好的模型（请替换为您实际的模型文件路径）
Biochar = joblib.load('XGB-biochar.pkl')
Gas = joblib.load('XGB-gas.pkl')
Oil = joblib.load('XGB-oil.pkl')

# 模型字典
models = {
    'XGB-biochar': Biochar,
    'XGB-gas': Gas,
    'XGB-oil': Oil
}

# 标题
st.title("Biomass Pyrolysis Product Application Forecast")

# 描述
st.write("""
This application predicts the yield of biomass pyrolysis products based on input features.
The corresponding product prediction model is selected and the feature values are entered to obtain the prediction.
""")

# 侧边栏：选择使用的模型
selected_models = st.sidebar.multiselect("Select models to use for prediction", list(models.keys()), default=list(models.keys()))

# 侧边栏：输入特征值
st.sidebar.header("Input Features")

# 修改数值输入控件，使其可以接受至少两位小数
Moisture = st.sidebar.number_input("M", min_value=0.0, max_value=25.0, value=5.0, step=0.01)
Ash = st.sidebar.number_input("Ash", min_value=0.0, max_value=35.0, value=5.0, step=0.01)
Volatile_Matter = st.sidebar.number_input("VM", min_value=0.0, max_value=100.0, value=80.0, step=0.01)
Fixed_Carbon = st.sidebar.number_input("FC", min_value=0.0, max_value=35.0, value=10.0, step=0.01)
C = st.sidebar.number_input("C", min_value=0.0, max_value=70.0, value=50.0, step=0.01)
H = st.sidebar.number_input("H", min_value=0.0, max_value=30.0, value=5.0, step=0.01)
O = st.sidebar.number_input("O", min_value=20.0, max_value=80.0, value=50.0, step=0.01)
N = st.sidebar.number_input("N", min_value=0.0, max_value=20.0, value=1.0, step=0.01)
Particle_Size = st.sidebar.number_input("PS", min_value=0.0, max_value=200.0, value=1.0, step=0.01)
Final_Temperature = st.sidebar.number_input("FT", min_value=200.0, max_value=1000.0, value=500.0, step=0.01)
Heating_rate = st.sidebar.number_input("HR", min_value=0.0, max_value=700.0, value=10.0, step=0.01)
Final_Residue = st.sidebar.number_input("FR", min_value=0.0, max_value=10000.0, value=100.0, step=0.01)

# 将输入数据转换为 DataFrame，供模型预测使用
input_data = pd.DataFrame({
    'M': [Moisture],
    'Ash': [Ash],
    'VM': [Volatile_Matter],
    'FC': [Fixed_Carbon],
    'C': [C],
    'H': [H],
    'O': [O],
    'N': [N],
    'PS': [Particle_Size],
    'FT': [Final_Temperature],
    'HR': [Heating_rate],
    'FR': [Final_Residue]
})

# 添加预测按钮
if st.sidebar.button("Predict"):
    # 为每个选定的模型展示预测值
    for model_name in selected_models:
        model = models[model_name]
        # 进行预测
        prediction = model.predict(input_data)[0]
        
        # 显示每个模型的预测结果
        st.write(f"## Model: {model_name}")
        st.write(f"**Predicted Yield**: {prediction:.2f}")

# 显示 PNG 图片
st.subheader("1. workflows")
image1 = Image.open("Basic_Information.png")
st.image(image1, caption="Information of the surveyed medical experts", use_column_width=True)

st.subheader("2. workflows")
image2 = Image.open("accuracy.png")
st.image(image2, caption="Evaluation of the website-based tool by the medical experts", use_column_width=True)
