import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# 页面配置
st.set_page_config(
    page_title="车辆故障数据质检平台",
    page_icon="",
    layout="wide"
)

# 标题
st.title("车辆故障数据质检平台")
st.markdown("上传 `vehicle_fleet.csv` 与 `fault_events.csv`，自动完成数据质量检测与分析")

# ========== 文件上传 ==========
col1, col2 = st.columns(2)

with col1:
    fleet_file = st.file_uploader("上传 vehicle_fleet.csv", type=['csv'])
with col2:
    faults_file = st.file_uploader("上传 fault_events.csv", type=['csv'])

# 只有两张表都上传了才继续
if fleet_file is not None and faults_file is not None:

    # 读取数据
    fleet = pd.read_csv(fleet_file)
    faults = pd.read_csv(faults_file)

    # ========== 数据质量评分 ==========
    st.header("数据质量总览")

    # 计算指标
    fleet_total = len(fleet)
    faults_total = len(faults)
    fleet_missing = fleet.isnull().sum().sum()
    faults_missing = faults.isnull().sum().sum()
    fleet_complete = (1 - fleet_missing / (len(fleet) * len(fleet.columns))) * 100
    faults_complete = (1 - faults_missing / (len(faults) * len(faults.columns))) * 100

    # 异常值统计
    fleet_anomaly = 0
    if 'current_mileage' in fleet.columns:
        fleet_anomaly += (fleet['current_mileage'] < 0).sum()
    if 'purchase_year' in fleet.columns:
        fleet_anomaly += ((fleet['purchase_year'] < 1990) | (fleet['purchase_year'] > 2026)).sum()

    faults_anomaly = 0
    if 'mileage_at_fault' in faults.columns:
        faults_anomaly += (faults['mileage_at_fault'] < 0).sum()
    if 'temperature_celsius' in faults.columns:
        faults_anomaly += ((faults['temperature_celsius'] < -50) | (faults['temperature_celsius'] > 150)).sum()

    # 展示指标卡片
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("车辆档案数", f"{fleet_total}")
    c2.metric("故障事件数", f"{faults_total}")
    c3.metric("档案完整率", f"{fleet_complete:.1f}%")
    c4.metric("事件完整率", f"{faults_complete:.1f}%")
    c5.metric("异常值总数", f"{fleet_anomaly + faults_anomaly}")

    # 综合评分
    overall_score = (fleet_complete + faults_complete) / 2
    st.progress(overall_score / 100)
    st.markdown(f"<h4 style='text-align: center;'>综合数据质量评分: {overall_score:.1f}/100</h4>",
                unsafe_allow_html=True)

    # ========== 各字段缺失率 ==========
    st.header("各字段缺失率")

    tab1, tab2 = st.tabs(["vehicle_fleet", "fault_events"])

    with tab1:
        fleet_missing_rate = (fleet.isnull().sum() / len(fleet) * 100).sort_values(ascending=False)
        fleet_missing_rate = fleet_missing_rate[fleet_missing_rate > 0]
        if len(fleet_missing_rate) > 0:
            st.bar_chart(fleet_missing_rate)
        else:
            st.success("vehicle_fleet 无缺失值")

    with tab2:
        faults_missing_rate = (faults.isnull().sum() / len(faults) * 100).sort_values(ascending=False)
        faults_missing_rate = faults_missing_rate[faults_missing_rate > 0]
        if len(faults_missing_rate) > 0:
            st.bar_chart(faults_missing_rate)
        else:
            st.success("fault_events 无缺失值")

    # ========== 关键字段分布 ==========
    st.header("关键字段分布")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("故障时里程分布")
        if 'mileage_at_fault' in faults.columns:
            fig, ax = plt.subplots()
            ax.hist(faults['mileage_at_fault'], bins=40, color='steelblue', edgecolor='white')
            ax.axvline(x=5000, color='red', linestyle='--', label='磨合期边界')
            ax.set_xlabel('里程 (km)')
            ax.set_ylabel('故障数')
            ax.legend()
            st.pyplot(fig)

    with col_b:
        st.subheader("故障严重程度分布")
        if 'severity' in faults.columns:
            severity_dist = faults['severity'].value_counts().sort_index()
            st.bar_chart(severity_dist)

    # ========== 数据预览 ==========
    st.header("数据预览")

    tab3, tab4 = st.tabs(["vehicle_fleet", "fault_events"])
    with tab3:
        st.dataframe(fleet.head(20), use_container_width=True)
    with tab4:
        st.dataframe(faults.head(20), use_container_width=True)

    # ========== 数据清洗 + 下载 ==========
    st.header("一键数据清洗")

    if st.button("执行清洗并下载"):
        with st.spinner("正在清洗数据..."):
            # 清洗 fleet
            fleet_clean = fleet.copy()
            if 'current_mileage' in fleet_clean.columns:
                fleet_clean['current_mileage'] = fleet_clean['current_mileage'].fillna(
                    fleet_clean['current_mileage'].median())
            if 'manufacturer' in fleet_clean.columns:
                fleet_clean['manufacturer'] = fleet_clean['manufacturer'].fillna('未知')
            fleet_clean = fleet_clean.drop_duplicates()

            # 清洗 faults
            faults_clean = faults.copy()
            if 'mileage_at_fault' in faults_clean.columns:
                faults_clean['mileage_at_fault'] = faults_clean['mileage_at_fault'].fillna(
                    faults_clean['mileage_at_fault'].median())
            if 'fault_type' in faults_clean.columns:
                faults_clean['fault_type'] = faults_clean['fault_type'].fillna('未知')
            faults_clean = faults_clean.drop_duplicates()

        # 提供下载
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="下载清洗后 vehicle_fleet.csv",
                data=fleet_clean.to_csv(index=False).encode('utf-8'),
                file_name='vehicle_fleet_cleaned.csv',
                mime='text/csv'
            )
        with col_d2:
            st.download_button(
                label="下载清洗后 fault_events.csv",
                data=faults_clean.to_csv(index=False).encode('utf-8'),
                file_name='fault_events_cleaned.csv',
                mime='text/csv'
            )

        st.success("清洗完成！已去除异常值与重复记录")

else:
    st.info("请上传两张 CSV 文件以开始分析")