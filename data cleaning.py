import pandas as pd
import numpy as np
import sqlite3
import json
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# ========== 读取四张表 ==========
fleet = pd.read_csv('vehicle_fleet.csv')
faults = pd.read_csv('fault_events.csv')
ner = pd.read_csv('ner_annotations.csv')
kg = pd.read_csv('knowledge_graph_triples.csv')

print("=== 四张表概览 ===")
for name, df in [("vehicle_fleet", fleet), ("fault_events", faults),
                  ("ner_annotations", ner), ("knowledge_graph_triples", kg)]:
    print(f"{name}: {len(df)} 行 × {len(df.columns)} 列")
    print(f"  字段: {df.columns.tolist()}")


print("\n=== 清洗 vehicle_fleet ===")

# 缺失值
fleet['current_mileage'] = fleet['current_mileage'].fillna(fleet['current_mileage'].median())
fleet['annual_mileage'] = fleet['annual_mileage'].fillna(fleet['annual_mileage'].median())
fleet['manufacturer'] = fleet['manufacturer'].fillna('未知')
fleet['vehicle_type'] = fleet['vehicle_type'].fillna('未知')
fleet['fuel_type'] = fleet['fuel_type'].fillna('未知')

# 异常值
fleet = fleet[fleet['current_mileage'] >= 0]
fleet = fleet[fleet['annual_mileage'] <= 500000]
fleet = fleet[(fleet['purchase_year'] >= 1990) & (fleet['purchase_year'] <= 2026)]

# 日期转换
fleet['purchase_date'] = pd.to_datetime(fleet['purchase_date'], errors='coerce')
fleet['last_maintenance_date'] = pd.to_datetime(fleet['last_maintenance_date'], errors='coerce')
fleet['insurance_expiry'] = pd.to_datetime(fleet['insurance_expiry'], errors='coerce')

# 新增特征
fleet['vehicle_age'] = 2026 - fleet['purchase_year']
fleet['mileage_group'] = pd.cut(
    fleet['current_mileage'],
    bins=[0, 10000, 30000, 50000, 100000, 200000, 9999999],
    labels=['0-1万', '1-3万', '3-5万', '5-10万', '10-20万', '20万+']
)

# 去重
fleet = fleet.drop_duplicates(subset=['vehicle_id'], keep='first')
print(f"清洗后: {len(fleet)} 条")


print("\n=== 清洗 fault_events ===")

# 缺失值处理
faults['mileage_at_fault'] = faults['mileage_at_fault'].fillna(faults['mileage_at_fault'].median())
faults['temperature_celsius'] = faults['temperature_celsius'].fillna(faults['temperature_celsius'].median())
faults['pressure_kpa'] = faults['pressure_kpa'].fillna(faults['pressure_kpa'].median())
faults['engine_rpm'] = faults['engine_rpm'].fillna(faults['engine_rpm'].median())
faults['severity'] = faults['severity'].fillna(faults['severity'].mode()[0])
faults['fault_description_raw'] = faults['fault_description_raw'].fillna('无描述')

# 日期转换
faults['occurrence_date'] = pd.to_datetime(faults['occurrence_date'], errors='coerce')
faults['fault_year'] = faults['occurrence_date'].dt.year
faults['fault_month'] = faults['occurrence_date'].dt.month
faults['fault_quarter'] = faults['occurrence_date'].dt.quarter

# 异常值
faults = faults[faults['mileage_at_fault'] >= 0]
faults = faults[faults['temperature_celsius'] > -50]  # 温度不可能低于-50
faults = faults[faults['temperature_celsius'] < 150]   # 温度不可能高于150
faults = faults[faults['pressure_kpa'] > 0]
faults = faults[faults['engine_rpm'] >= 0]

# 严重等级映射
faults['severity_num'] = faults['severity']
faults['severity_num'] = faults['severity'].fillna(2)

# 季节
season_map = {12: '冬', 1: '冬', 2: '冬',
              3: '春', 4: '春', 5: '春',
              6: '夏', 7: '夏', 8: '夏',
              9: '秋', 10: '秋', 11: '秋'}
faults['season'] = faults['fault_month'].map(season_map)

print(f"清洗后: {len(faults)} 条")


print("\n=== 清洗 ner_annotations ===")

# entities 字段可能是 JSON 字符串，尝试解析
def safe_parse_entities(x):
    try:
        if pd.isna(x):
            return []
        return json.loads(x)
    except:
        return []

ner['entities_list'] = ner['entities'].apply(safe_parse_entities)
ner['n_entities'] = ner['n_entities'].fillna(ner['entities_list'].apply(len))
ner['fault_code'] = ner['fault_code'].fillna('未知')

# 去重
ner = ner.drop_duplicates(subset=['id'], keep='first')
print(f"清洗后: {len(ner)} 条")


print("\n=== 清洗 knowledge_graph_triples ===")

kg = kg.dropna(subset=['subject', 'predicate', 'object'])
kg['confidence'] = kg['confidence'].fillna(kg['confidence'].median())
kg = kg[kg['confidence'] >= 0.5]  # 过滤低置信度三元组
kg = kg.drop_duplicates()

print(f"清洗后: {len(kg)} 条")


conn = sqlite3.connect('vehicle_analysis.db')

fleet.to_sql('fleet', conn, if_exists='replace', index=False)
faults.to_sql('faults', conn, if_exists='replace', index=False)
ner_for_sql = ner.drop(columns=['entities_list'])
ner_for_sql.to_sql('ner', conn, if_exists='replace', index=False)
kg.to_sql('kg', conn, if_exists='replace', index=False)

print("\n=== 数据库创建完成 ===")


q1 = """
SELECT 
    f.manufacturer,
    COUNT(DISTINCT f.vehicle_id) as total_vehicles,
    COUNT(ft.event_id) as fault_count,
    ROUND(COUNT(ft.event_id) * 1.0 / COUNT(DISTINCT f.vehicle_id), 2) as fault_rate,
    ROUND(AVG(ft.severity_num), 2) as avg_severity,
    ROUND(AVG(ft.mileage_at_fault), 0) as avg_fault_mileage
FROM fleet f
LEFT JOIN faults ft ON f.model_id = ft.model_id
GROUP BY f.manufacturer
ORDER BY fault_rate DESC
"""
df1 = pd.read_sql(q1, conn)
print("【各品牌故障率对比】")
print(df1)


q2 = """
SELECT 
    f.mileage_group,
    ft.fault_system,
    COUNT(ft.event_id) as fault_count,
    ROUND(AVG(ft.temperature_celsius), 1) as avg_temp,
    ROUND(AVG(ft.pressure_kpa), 1) as avg_pressure
FROM fleet f
JOIN faults ft ON f.model_id = ft.model_id
WHERE f.mileage_group IS NOT NULL AND ft.fault_system IS NOT NULL
GROUP BY f.mileage_group, ft.fault_system
ORDER BY f.mileage_group, fault_count DESC
"""
df2 = pd.read_sql(q2, conn)
print("\n【里程段 × 故障系统 交叉分析】")
print(df2.head(20))


q3 = """
SELECT 
    ft.season,
    ft.fault_system,
    COUNT(ft.event_id) as fault_count,
    ROUND(AVG(ft.seasonal_risk_score), 3) as avg_risk_score
FROM fleet f
JOIN faults ft ON f.model_id = ft.model_id
WHERE ft.season IS NOT NULL AND ft.fault_system IS NOT NULL
GROUP BY ft.season, ft.fault_system
ORDER BY ft.season, fault_count DESC
"""
df3 = pd.read_sql(q3, conn)
print("\n【季节性故障分析】")
print(df3)


q4 = """
SELECT 
    CASE 
        WHEN ft.temperature_celsius > 110 THEN '高温异常'
        WHEN ft.temperature_celsius < 20 THEN '低温异常'
        ELSE '温度正常'
    END as temp_status,
    CASE 
        WHEN ft.pressure_kpa > 300 THEN '高压异常'
        WHEN ft.pressure_kpa < 50 THEN '低压异常'
        ELSE '压力正常'
    END as pressure_status,
    COUNT(ft.event_id) as count,
    ROUND(AVG(ft.severity_num), 2) as avg_severity,
    ROUND(AVG(ft.system_degradation_index), 3) as avg_degradation
FROM fleet f
JOIN faults ft ON f.model_id = ft.model_id
GROUP BY temp_status, pressure_status
ORDER BY count DESC
"""
df4 = pd.read_sql(q4, conn)
print("\n【传感器异常 × 故障严重程度】")
print(df4)


q5 = """
SELECT 
    subject,
    predicate,
    object,
    COUNT(*) as occurrence,
    ROUND(AVG(confidence), 3) as avg_confidence
FROM kg
WHERE predicate LIKE '%引起%' OR predicate LIKE '%导致%' OR predicate LIKE '%原因%'
GROUP BY subject, predicate, object
ORDER BY occurrence DESC
LIMIT 15
"""
df5 = pd.read_sql(q5, conn)
print("\n【知识图谱 - 故障因果关系 TOP 15】")
print(df5)


q6 = """
SELECT 
    fault_code,
    AVG(n_entities) as avg_entities,
    COUNT(*) as record_count
FROM ner
GROUP BY fault_code
ORDER BY avg_entities DESC
"""
df6 = pd.read_sql(q6, conn)
print("\n【NER - 各故障码实体数量】")
print(df6)



print("\n" + "=" * 50)
print("车辆故障数据质量报告")
print("=" * 50)

report = {
    "vehicle_fleet 总记录数": len(fleet),
    "fault_events 总记录数": len(faults),
    "ner_annotations 总记录数": len(ner),
    "knowledge_graph_triples 总记录数": len(kg),

    "车辆档案缺失率": f"{fleet.isnull().sum().sum() / (len(fleet) * len(fleet.columns)) * 100:.2f}%",
    "故障事件缺失率": f"{faults.isnull().sum().sum() / (len(faults) * len(faults.columns)) * 100:.2f}%",

    "有故障记录的车辆数": faults['vehicle_id'].nunique(),
    "无故障记录的车辆数": len(fleet) - faults['vehicle_id'].nunique(),
    "车辆故障覆盖率": f"{faults['vehicle_id'].nunique() / len(fleet) * 100:.1f}%",

    "故障码种类数": faults['fault_code'].nunique(),
    "故障系统种类数": faults['fault_system'].nunique(),
    "知识图谱关系种类数": kg['predicate'].nunique(),

    "平均故障里程": f"{faults['mileage_at_fault'].mean():.0f} km",
    "平均故障温度": f"{faults['temperature_celsius'].mean():.1f} °C",
    "平均系统退化指数": f"{faults['system_degradation_index'].mean():.3f}",

    "数据完整率": f"{(1 - faults.isnull().sum().sum() / (len(faults) * len(faults.columns))) * 100:.2f}%"
}

for k, v in report.items():
    print(f"{k}: {v}")

# 保存报告
with open('data_quality_report.txt', 'w', encoding='utf-8') as f:
    for k, v in report.items():
        f.write(f"{k}: {v}\n")
print("\n报告已保存到data_quality_report.txt")




# 图1：各品牌故障率
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 品牌故障率
df1_plot = df1.head(8)
axes[0,0].barh(df1_plot['manufacturer'], df1_plot['fault_rate'], color='coral')
axes[0,0].set_xlabel('故障率')
axes[0,0].set_title('各品牌故障率对比')

# 月度故障趋势
monthly = faults.groupby('fault_month').size()
axes[0,1].plot(monthly.index, monthly.values, marker='o', color='steelblue')
axes[0,1].set_xlabel('月份')
axes[0,1].set_ylabel('故障数')
axes[0,1].set_title('月度故障趋势')
axes[0,1].set_xticks(range(1, 13))

# 里程段故障分布
mileage_dist = faults['mileage_at_fault'].hist(bins=30, ax=axes[1,0], color='green', alpha=0.7)
axes[1,0].set_xlabel('故障时里程 (km)')
axes[1,0].set_ylabel('故障数')
axes[1,0].set_title('故障里程分布')

# 严重程度分布
severity_dist = faults['severity_num'].value_counts().sort_index()
axes[1,1].bar(severity_dist.index.astype(str), severity_dist.values, color='orange')
axes[1,1].set_xlabel('严重等级')
axes[1,1].set_ylabel('故障数')
axes[1,1].set_title('故障严重程度分布')

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=150)
plt.show()
print("可视化已保存到eda_overview.png")

#早期故障分析（双图）
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：全量分布，标注磨合期
axes[0].hist(faults['mileage_at_fault'], bins=40, color='steelblue', alpha=0.7, edgecolor='white')
axes[0].axvline(x=5000, color='red', linestyle='--', label='磨合期边界(5000km)')
axes[0].set_xlabel('故障时里程 (km)')
axes[0].set_ylabel('故障数')
axes[0].set_title('故障里程分布（全量）')
axes[0].legend()

# 右图：0~2万公里放大看
early_faults = faults[faults['mileage_at_fault'] <= 20000]
axes[1].hist(early_faults['mileage_at_fault'], bins=30, color='coral', alpha=0.8, edgecolor='white')
axes[1].set_xlabel('故障时里程 (km)')
axes[1].set_ylabel('故障数')
axes[1].set_title('早期故障分布（0~2万公里放大）')

plt.tight_layout()
plt.savefig('mileage_analysis_early_fault.png', dpi=150)
plt.show()
print("早期故障分析图已保存到mileage_analysis_early_fault.png")

#浴盆曲线，按里程段统计故障密度
faults['mileage_bin'] = pd.cut(faults['mileage_at_fault'], bins=20)
bin_stats = faults.groupby('mileage_bin',observed=True).size().reset_index(name='fault_count')
bin_stats['midpoint'] = bin_stats['mileage_bin'].apply(lambda x: x.mid)

plt.figure(figsize=(10, 5))
plt.plot(bin_stats['midpoint'], bin_stats['fault_count'], marker='o', color='darkred')
plt.xlabel('里程 (km)')
plt.ylabel('故障数')
plt.title('故障率随里程变化趋势（浴盆曲线特征）')
plt.grid(True, alpha=0.3)
plt.savefig('bathtub_curve.png', dpi=150)
plt.show()
print("浴盆曲线已保存到 bathtub_curve.png")


# 保存所有分析结果
df1.to_csv('analysis_brand_fault_rate.csv', index=False, encoding='utf-8-sig')
df2.to_csv('analysis_mileage_system.csv', index=False, encoding='utf-8-sig')
df3.to_csv('analysis_seasonal.csv', index=False, encoding='utf-8-sig')
df4.to_csv('analysis_sensor_severity.csv', index=False, encoding='utf-8-sig')
df5.to_csv('analysis_knowledge_graph.csv', index=False, encoding='utf-8-sig')
df6.to_csv('analysis_ner_entities.csv', index=False, encoding='utf-8-sig')

print("所有分析结果已保存到 CSV 文件")


# 1. 看 mileage_at_fault 为 0 的有多少
zero_mileage = faults[faults['mileage_at_fault'] == 0]
print(f"里程为0的故障记录: {len(zero_mileage)} 条")

# 2. 看 0~5000 公里的故障都是什么类型
low_mileage = faults[faults['mileage_at_fault'] <= 5000]
print(f"\n0~5000公里故障数: {len(low_mileage)}")
print("\n低里程故障的故障系统分布:")
print(low_mileage['fault_system'].value_counts())

print("\n低里程故障的严重程度分布:")
print(low_mileage['severity_num'].value_counts())

# 3. 看 0~1000 公里的具体数值分布
very_low = faults[faults['mileage_at_fault'] <= 1000]
print(f"\n0~1000公里故障数: {len(very_low)}")
print("具体里程值统计:")
print(very_low['mileage_at_fault'].describe())
