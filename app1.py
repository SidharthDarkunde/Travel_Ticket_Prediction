import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

# 1. Page Configuration
st.set_page_config(
    page_title="Global Travel Insights Engine",
    page_icon="💎",
    layout="wide"
)

# 2. Precision CSS to match the image exactly
def apply_custom_design():
    st.markdown(f"""
    <style>
    /* Background Image with Dark Overlay */
    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), 
                    url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }}

    /* Header Styling */
    .main-title {{
        font-size: 38px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 0px;
        color: white;
    }}
    .sub-title {{
        font-size: 18px;
        text-align: center;
        margin-bottom: 30px;
        opacity: 0.9;
    }}

    /* Glassmorphism Input Cards */
    [data-testid="stVerticalBlock"] > div:has(div.stNumberInput, div.stSelectbox, div.stSlider) {{
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin-bottom: 20px;
    }}

    /* Input Field Styling */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stTextInput input {{
        background-color: rgba(0, 0, 0, 0.3) !important;
        color: white !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }}

    /* Luxury Gold Button */
    div.stButton > button:first-child {{
        background: linear-gradient(180deg, #8a6d3b 0%, #634f2a 100%);
        border: 1px solid #d4af37;
        color: white;
        border-radius: 30px;
        padding: 10px 40px;
        font-weight: bold;
        display: block;
        margin: 0 auto;
        transition: 0.3s;
    }}
    div.stButton > button:first-child:hover {{
        box-shadow: 0px 0px 15px #d4af37;
        transform: translateY(-2px);
    }}

    /* Result Cards */
    .result-card {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}
    
    .confidence-box {{
        background: rgba(0, 0, 0, 0.4);
        border: 2px solid #007bff;
        border-radius: 15px;
        text-align: center;
        padding: 15px;
    }}

    /* Label Colors */
    label {{ color: white !important; font-weight: 500 !important; }}
    </style>
    """, unsafe_allow_html=True)

apply_custom_design()

# 3. Header
st.markdown('<p class="sub-title">Global Travel Insights Engine</p>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">💎 Customer Conversion Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predicting premium product adoption with AI precision.</p>', unsafe_allow_html=True)

# 4. Load Pipeline
@st.cache_resource
def load_pipeline():
    # Make sure your joblib files are in the same directory
    preprocessor = joblib.load("preprocessor.joblib")
    model = joblib.load("model.joblib")
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

pipeline = load_pipeline()

# 5. Input Layout (3 Columns with Glass Cards)
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("### 👤 Profile")
    Age = st.number_input("Age", 18, 100, 30)
    MaritalStatus = st.selectbox("Marital Status", ["Unmarried", "Married", "Divorced"])
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    MonthlyIncome = st.number_input("Monthly Income ($)", 1000, 500000, 30000)

with col2:
    st.markdown("### ✈️ Travel Habits")
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    NumberOfTrips = st.slider("Number Of Trips", 0, 20, 2)
    Passport = st.radio("Has Passport?", [0, 1], format_func=lambda x: "YES" if x==1 else "NO", horizontal=True)
    OwnCar = st.radio("Owns Car?", [0, 1], format_func=lambda x: "YES" if x==1 else "NO", horizontal=True)

with col3:
    st.markdown("### 🎯 Pitch Details")
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    PitchDuration = st.number_input("Pitch Duration (min)", 0, 200, 30)
    SatisfactionScore = st.select_slider("Satisfaction Score", options=[1, 2, 3, 4, 5], value=3)
    TotalVisiting = st.slider("Total Visiting", 0, 50, 5)

st.markdown("<br>", unsafe_allow_html=True)

# 6. Prediction Logic
if st.button("EXECUTE PREDICTION"):
    # Create DF for model
    input_df = pd.DataFrame([{
        "Age": Age, "TypeofContact": TypeofContact, "CityTier": CityTier,
        "DurationOfPitch": PitchDuration, "Occupation": Occupation,
        "Gender": "Male", "NumberOfFollowups": 3, # Dummy values for missing fields
        "ProductPitched": ProductPitched, "PreferredPropertyStar": 3,
        "MaritalStatus": MaritalStatus, "NumberOfTrips": NumberOfTrips,
        "Passport": Passport, "PitchSatisfactionScore": SatisfactionScore,
        "OwnCar": OwnCar, "Designation": Designation,
        "MonthlyIncome": MonthlyIncome, "TotalVisiting": TotalVisiting
    }])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0].max()

    st.markdown("---")
    
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        if prediction == 1:
            st.success(f"### ✅ PROSPECTIVE BUYER \n This customer is likely to accept the offer.")
        else:
            st.error(f"### ❌ LOW CONVERSION \n This customer is unlikely to accept the offer.")

    with res_col2:
        st.markdown(f"""
            <div class="confidence-box">
                <p style="margin:0; font-size:14px;">Confidence Score</p>
                <h2 style="margin:0; color:#007bff;">{round(probability * 100, 2)}%</h2>
            </div>
        """, unsafe_allow_html=True)

        