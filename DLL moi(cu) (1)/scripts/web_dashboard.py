import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="HR Analytics Dashboard", layout="wide")

st.title("HR Analytics Data Mining Dashboard")

# Helper function to read if path exists

def try_read(path):
    if os.path.exists(path):
        try:
            if path.endswith('.csv'):
                return pd.read_csv(path)
            if path.endswith('.parquet'):
                return pd.read_parquet(path)
        except Exception as e:
            st.warning(f"Không đọc được {path}: {e}")
    return None

# Load raw and processed data
raw_path = 'data/raw/HR_Analytics.csv'
processed_path = 'data/processed/HR_Processed.parquet'

raw_df = try_read(raw_path)
proc_df = try_read(processed_path)

# 1. Overview + EDA tables
st.header("1. Data Overview & Summary")

col1, col2 = st.columns(2)

with col1:
    if raw_df is not None:
        st.subheader('Raw Data Sample')
        st.dataframe(raw_df.head(10))
        st.write(f"{raw_df.shape[0]} rows x {raw_df.shape[1]} columns")
    else:
        st.info('Không tìm thấy dữ liệu raw ở data/raw/HR_Analytics.csv')

with col2:
    if proc_df is not None:
        st.subheader('Processed Data Sample')
        st.dataframe(proc_df.head(10))
        st.write(f"{proc_df.shape[0]} rows x {proc_df.shape[1]} columns")
    else:
        st.info('Không tìm thấy dữ liệu processed ở data/processed/HR_Processed.parquet')

# 2. Attrition chart based on raw data if available
if raw_df is not None:
    if 'Attrition' in raw_df.columns:
        st.subheader('Attrition Rate (raw)')
        attr_counts = raw_df['Attrition'].value_counts().reset_index()
        attr_counts.columns = ['Attrition', 'Count']
        fig = px.pie(attr_counts, names='Attrition', values='Count', title='Tỷ lệ Attrition')
        st.plotly_chart(fig, use_container_width=True)

    if 'Department' in raw_df.columns and 'Attrition' in raw_df.columns:
        st.subheader('Attrition by Department')
        # convert Attrition numeric if not
        if raw_df['Attrition'].dtype != 'O':
            this_df = raw_df.copy()
            this_df['Attrition'] = this_df['Attrition'].map({1:'Yes', 0:'No'})
        else:
            this_df = raw_df
        dept = this_df.groupby(['Department', 'Attrition']).size().reset_index(name='Count')
        fig2 = px.bar(dept, x='Department', y='Count', color='Attrition', barmode='group', title='Attrition theo Department')
        st.plotly_chart(fig2, use_container_width=True)

    if 'SalarySlab' in raw_df.columns and 'Attrition' in raw_df.columns:
        st.subheader('Attrition by Salary Slab')
        slab = raw_df.groupby(['SalarySlab', 'Attrition']).size().reset_index(name='Count')
        fig3 = px.bar(slab, x='SalarySlab', y='Count', color='Attrition', barmode='group', title='Attrition theo SalarySlab')
        st.plotly_chart(fig3, use_container_width=True)

# 3. Load and display data from notebook output tables
st.header('2. Reports / Tables from Notebook')
report_dirs = ['outputs/reports', 'outputs/tables']

for report_dir in report_dirs:
    if os.path.exists(report_dir):
        files = sorted([f for f in os.listdir(report_dir) if f.lower().endswith('.csv')])
        if files:
            st.subheader(report_dir)
            for f in files:
                full = os.path.join(report_dir, f)
                df_report = try_read(full)
                if df_report is not None:
                    st.markdown(f"### {f}")
                    st.write(df_report.head(200))
                else:
                    st.warning(f'Không đọc được {full}')
    else:
        st.info(f'Không tìm thấy thư mục {report_dir}')

# 4. Add chart from notebooks if exists
st.header('3. Auto Charts from Existing Figures')
figure_dir = 'outputs/figures'
if os.path.exists(figure_dir):
    images = [f for f in os.listdir(figure_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if images:
        for img in images:
            st.subheader(img)
            st.image(os.path.join(figure_dir, img), use_column_width=True)
    else:
        st.info('Không tìm thấy hình ảnh biểu đồ trong outputs/figures')
else:
    st.info('Không tìm thấy thư mục outputs/figures')

# 5. Policy and next steps
st.header('4. Policy Recommendations')
st.markdown("""
- Theo dõi tỷ lệ `Attrition` theo từng Department và SalarySlab.
- Thực hiện phân cụm nhân sự với kết quả từ notebooks.
- Cập nhật model khi có dữ liệu mới và so sánh PR-AUC.
""")
