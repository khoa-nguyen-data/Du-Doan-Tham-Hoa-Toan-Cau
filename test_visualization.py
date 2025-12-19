"""Test script để kiểm tra visualization modules"""

# Test import
print("Testing imports...")
from visualization.utils.data_loader import load_data, aggregate_by_disaster_type
from visualization.components.disaster_distribution import visualize_disaster_distribution

# Test load data
print("\n1. Loading data...")
df = load_data()
print(f"   ✅ Data shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# Test aggregate
print("\n2. Testing aggregate_by_disaster_type...")
agg = aggregate_by_disaster_type(df)
print(f"   ✅ Aggregated shape: {agg.shape}")
print(f"   Columns: {agg.columns.tolist()}")
print(f"   Top 3:\n{agg.head(3)}")

# Test visualization
print("\n3. Testing visualize_disaster_distribution...")
fig1, fig2 = visualize_disaster_distribution()
print(f"   ✅ Sunburst traces: {len(fig1.data)}")
print(f"   ✅ Bar chart traces: {len(fig2.data)}")

print("\n✅ ALL TESTS PASSED!")
