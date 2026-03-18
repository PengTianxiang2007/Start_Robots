import pandas as pd

# 直接读取第一个数据块（包含了前面的一千多帧，肯定覆盖了 episode 0 和 1 的交界）
parquet_file = "data/chunk-000/file-000.parquet"
df = pd.read_parquet(parquet_file)

# 直接筛选出第二个 episode (episode_index == 1) 的第一行数据
ep1_start = df[df['episode_index'] == 1].iloc[0]

print("=== 第二个 Episode 的起始状态 ===")
print(f"Frame Index: {ep1_start['frame_index']}")
print(f"Timestamp:   {ep1_start['timestamp']}")

# 结论判断
if ep1_start['frame_index'] == 0:
    print("\n结论: 是相对于当前 Episode 的 (重置为0了)")
else:
    print("\n结论: 是相对于全局的 (继承了上一个 episode 的数值累加)")