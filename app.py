import streamlit as st
import joblib
import pandas as pd

# Load model and preprocessing tools
model = joblib.load('xgb_model.pkl')
selector = joblib.load('selector.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Feature names
selected_features = [
    'نسبة السيولة', 'الرافعة المالية', 'صافي الربح', 'معدل دوران الاصول',
    'نسبة الديون الى حقوق الملكية', 'الديون الى اجمالي الخصوم'
]

# Page config
st.set_page_config(page_title="🔮 تنبؤ التصنيف الائتماني", page_icon="💳", layout="centered")

# CSS Styling
st.markdown("""
    <style>
        body {
            background-color: #f2f6fc;
        }
        .main {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
        }
        h1 {
            font-family: 'Segoe UI', sans-serif;
            font-size: 40px;
            text-align: center;
            color: #222;
            margin-bottom: 10px;
        }
        p {
            font-size: 18px;
            color: #555;
            text-align: center;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: #18E1D9;
            color: white;
            padding: 10px 25px;
            font-size: 18px;
            font-weight: bold;
            border-radius: 12px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #0c7c7c;
        }
        .css-1offfwp {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Main app container
with st.container():
    st.markdown("<h1>🔮 تنبؤ التصنيف الائتماني</h1>", unsafe_allow_html=True)
    st.markdown("<p>ادخل البيانات يدويًا أو ارفع ملف للحصول على التصنيف المالي للشركة.</p>", unsafe_allow_html=True)

    # Upload file
    uploaded_file = st.file_uploader("📁 ارفع ملف البيانات", type=["csv", "xlsx"])

    # Manual input
    st.markdown("## 📝 إدخال البيانات يدويًا")
    col1, col2 = st.columns(2)

    with col1:
        نسبة_السيولة = st.number_input("💧 نسبة السيولة", value=0.0, step=0.01)
        الرافعة_المالية = st.number_input("📊 الرافعة المالية", value=0.0, step=0.01)
        معدل_دوران_الاصول = st.number_input("♻️ معدل دوران الأصول", value=0.0, step=0.01)

    with col2:
        صافي_الربح = st.number_input("💰 صافي الربح (%)", value=0.0, step=0.01)
        نسبة_الديون_الى_حقوق_الملكية = st.number_input("⚖️ نسبة الديون إلى حقوق الملكية", value=0.0, step=0.01)
        الديون_الى_اجمالي_الخصوم = st.number_input("📉 الديون إلى إجمالي الخصوم", value=0.0, step=0.01)

    # From file
    if uploaded_file is not None:
        if uploaded_file.name.endswith("csv"):
            input_data_file = pd.read_csv(uploaded_file)
        else:
            input_data_file = pd.read_excel(uploaded_file)

        if all(feature in input_data_file.columns for feature in selected_features):
            input_data_file = input_data_file[selected_features]
            input_data_file.fillna(input_data_file.median(), inplace=True)
            input_data_file['صافي الربح'] = input_data_file['صافي الربح'].astype(str).str.rstrip('%').astype(float) / 100

            input_data_scaled_file = scaler.transform(input_data_file)
            input_data_selected_file = selector.transform(input_data_scaled_file)

            if st.button("🔍 تنبؤ من الملف"):
                prediction = model.predict(input_data_selected_file)
                predicted_class = label_encoder.inverse_transform(prediction)
                st.success("تم التنبؤ بالتصنيفات بنجاح!")
                st.dataframe(pd.DataFrame(predicted_class, columns=["💡 التصنيف المتوقع"]))
        else:
            st.error("⚠️ بعض الأعمدة ناقصة في الملف!")

    # Manual Prediction
    if st.button("🚀 تنبؤ من البيانات المدخلة"):
        try:
            manual_input_df = pd.DataFrame([{
                'نسبة السيولة': نسبة_السيولة,
                'الرافعة المالية': الرافعة_المالية,
                'صافي الربح': صافي_الربح / 100,
                'معدل دوران الاصول': معدل_دوران_الاصول,
                'نسبة الديون الى حقوق الملكية': نسبة_الديون_الى_حقوق_الملكية,
                'الديون الى اجمالي الخصوم': الديون_الى_اجمالي_الخصوم
            }])
            manual_input_scaled = scaler.transform(manual_input_df)
            manual_input_selected = selector.transform(manual_input_scaled)
            prediction = model.predict(manual_input_selected)
            predicted_class = label_encoder.inverse_transform(prediction)
            st.balloons()
            st.success(f"🎯 التصنيف الائتماني المتوقع هو: **{predicted_class[0]}**")
        except Exception as e:
            st.error(f"حدث خطأ أثناء التنبؤ: {e}")
