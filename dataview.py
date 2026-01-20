import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Bitstream Vera Sans']
plt.rcParams['axes.unicode_minus'] = False


# 1. 指定日志文件路径
# 确保路径是正确的，包括 'data' 目录
log_file_path = os.path.join("data", "mamba_infer.log") # 使用 os.path.join 构建路径

# 2. 定义一个宽松的模式来匹配包含层数和时间的行
# 这个模式用来初步筛选可能包含计时信息的行
log_line_pattern = re.compile(r'\[Layer \d+\] .*?: \d+\.\d+ ms')

# 3. 解析日志文件
parsed_data = []
try:
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 先检查行是否符合基本的日志格式和包含时间信息
            if '- INFO -' in line and log_line_pattern.search(line):
                 try:
                    # 按 '- INFO -' 分割，获取消息部分
                    parts = line.split(' - INFO - ')
                    if len(parts) < 2:
                        continue # 如果分割不成功，跳过

                    message = parts[1].strip() # 获取消息部分，去除前后空白

                    # 从消息开头提取层数
                    layer_match = re.match(r'\[Layer (\d+)\]', message)
                    if not layer_match:
                        continue # 如果层数格式不正确，跳过

                    layer = int(layer_match.group(1))

                    # 提取层数后面的字符串，包含组件名称和时间
                    component_time_str = message[layer_match.end():].strip()

                    # 找到最后一个冒号，以此分割组件名称和时间值
                    last_colon_index = component_time_str.rfind(':')
                    if last_colon_index == -1:
                        continue # 如果没有冒号，跳过

                    # 分割出原始组件名称和时间字符串
                    component_raw = component_time_str[:last_colon_index].strip()
                    time_str_with_ms = component_time_str[last_colon_index + 1:].strip()

                    # 从时间字符串中提取浮点数值
                    time_match = re.match(r'(\d+\.\d+)', time_str_with_ms)
                    if not time_match:
                        continue # 如果时间格式不正确，跳过

                    time_ms = float(time_match.group(1))

                    # --- 清理原始组件名称并映射到规范名称 ---
                    # 根据日志中可能出现的原始名称，映射到我们用于图表的规范名称
                    component_mapping = {
                        'RMSNorm time': 'RMSNorm',
                        'in_proj inference time': 'in_proj',
                        'Convolution sequence transformation time': 'Convolution',
                        'SSM parameters time': 'SSM parameters',
                        'selective_state_update': 'SSM scan', # 这个名称后面没有 ' time'
                        'out_proj time': 'out_proj',
                        # 如果以后出现新的计时项，可以在这里添加映射
                    }
                    # 使用映射，如果原始名称不在映射中，则使用原始名称（作为备用）
                    component = component_mapping.get(component_raw, component_raw)

                    # 将提取的数据添加到列表中
                    parsed_data.append({'layer': layer, 'component': component, 'time_ms': time_ms})

                 except Exception as e:
                      # 捕获解析单行时的错误，并打印警告，不中断整个文件读取
                      print(f"Warning: Could not fully parse line: {line.strip()}. Error: {e}")
                      continue # 跳过当前行，继续下一行


except FileNotFoundError:
    print(f"Error: Log file not found at '{log_file_path}'")
    exit() # 文件找不到，退出程序
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit() # 其他读取错误，退出程序


# 检查是否成功解析到任何数据
if not parsed_data:
    print(f"No timing data found in the log file '{log_file_path}' that matches the expected format.")
    print("Please ensure the log file contains lines like '[Layer X] Component Name: Y.YYY ms' or '[Layer X] Component Name time: Y.YYY ms'.")
    exit()

# 4. 将解析的数据加载到 pandas DataFrame
df = pd.DataFrame(parsed_data)

# 5. 数据分析

# 5a. 计算每个组件在所有层上的总耗时
# 按规范的组件名称分组并求和，然后按时间降序排序
aggregated_time = df.groupby('component')['time_ms'].sum().sort_values(ascending=False)

print("--- Aggregated Time per Component (Sum across all layers) (ms) ---")
print(aggregated_time)
print("-" * 40)

# 5b. 准备各层时间细分数据（用于堆叠柱状图）
# 使用规范的组件名称作为列
pivot_df = df.pivot_table(index='layer', columns='component', values='time_ms', fill_value=0)

# 定义绘制时组件的显示顺序（对应于堆叠的顺序）
# 根据Mamba Block的结构和日志顺序定义一个逻辑顺序
component_order = ['RMSNorm', 'in_proj', 'Convolution', 'SSM parameters', 'SSM scan/update', 'out_proj']

# 过滤出在实际数据中存在的组件，并按照定义的顺序排列
# 这确保即使日志中缺少某个组件，代码也不会出错
ordered_columns = [c for c in component_order if c in pivot_df.columns]
pivot_df = pivot_df[ordered_columns]


# 6. 数据可视化

# 6a. 绘制各层时间细分堆叠柱状图
plt.figure(figsize=(14, 7))
# 直接在当前 figure 上绘制堆叠柱状图，使用排序好的列
ax = pivot_df.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='viridis')

plt.title('Layer-wise Time Breakdown (Stacked Bar Chart)')
plt.xlabel('Layer')
plt.ylabel('Time (ms)')
plt.xticks(rotation=0) # 层数通常不多，不旋转 x 轴标签
plt.legend(title='Component', bbox_to_anchor=(1.05, 1), loc='upper left') # 将图例放在图外
plt.grid(axis='y', linestyle='--') # 添加水平网格线
plt.tight_layout(rect=[0, 0, 0.85, 1]) # 调整布局以给图例留出空间
plt.show()


# 6b. 绘制各组件总耗时柱状图
plt.figure(figsize=(10, 6))
# 使用 hue 参数以符合 FutureWarning，并关闭图例（x 轴标签已足够说明）
sns.barplot(x=aggregated_time.index, y=aggregated_time.values, hue=aggregated_time.index, palette='viridis', legend=False)

plt.title('Aggregated Component Time (Sum across all layers)')
plt.xlabel('Component')
plt.ylabel('Time (ms)')
plt.xticks(rotation=45, ha='right') # 旋转 x 轴标签以防重叠
plt.grid(axis='y', linestyle='--') # 添加水平网格线
plt.tight_layout() # 自动调整布局
# 您可以在这里添加 plt.savefig(...) 来保存图表
plt.savefig('images/saggregated_component_time.png', dpi=300)
