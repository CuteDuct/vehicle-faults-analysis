import pandas as pd

fleet = pd.read_csv('vehicle_fleet.csv')
faults = pd.read_csv('fault_events.csv')
ner = pd.read_csv('ner_annotations.csv')
kg = pd.read_csv('knowledge_graph_triples.csv')

# 检查 vehicle_id 的格式
print("fleet 表 vehicle_id 示例:", fleet['vehicle_id'].head(3).tolist())
print("fleet 表 vehicle_id 类型:", fleet['vehicle_id'].dtype)

print("faults 表 vehicle_id 示例:", faults['vehicle_id'].head(3).tolist())
print("faults 表 vehicle_id 类型:", faults['vehicle_id'].dtype)

# 看有多少能匹配上
matched = set(fleet['vehicle_id']).intersection(set(faults['vehicle_id']))
print(f"fleet 表车辆数: {fleet['vehicle_id'].nunique()}")
print(f"faults 表车辆数: {faults['vehicle_id'].nunique()}")
print(f"能匹配上的车辆数: {len(matched)}")

# 如果匹配数为0，看是不是格式问题
if len(matched) == 0:
    print("⚠️ 警告：vehicle_id 完全匹配不上！")
    print("fleet 示例:", fleet['vehicle_id'].iloc[0])
    print("faults 示例:", faults['vehicle_id'].iloc[0])



print("fleet 表 model_id 示例:", fleet['model_id'].head(3).tolist())
print("fleet 表 model_id 类型:", fleet['model_id'].dtype)

print("faults 表 model_id 示例:", faults['model_id'].head(3).tolist())
print("faults 表 model_id 类型:", faults['model_id'].dtype)

# 看 model_id 能匹配多少
matched_model = set(fleet['model_id']).intersection(set(faults['model_id']))
print(f"fleet 表 model_id 种类数: {fleet['model_id'].nunique()}")
print(f"faults 表 model_id 种类数: {faults['model_id'].nunique()}")
print(f"能匹配上的 model_id 数: {len(matched_model)}")

# 如果 model_id 也对不上，检查 vehicle_name
if len(matched_model) == 0:
    print("\n⚠️ model_id 也匹配不上！")
    print("fleet 表 vehicle_name 示例:", fleet['vehicle_name'].head(3).tolist())
    print("faults 表 vehicle_id 示例:", faults['vehicle_id'].head(3).tolist())


