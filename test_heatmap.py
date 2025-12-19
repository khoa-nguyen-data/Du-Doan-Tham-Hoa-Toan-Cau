"""Test heatmap component"""

from visualization.components.monthly_heatmap import visualize_heatmap

print("Testing heatmap visualization...")
try:
    fig = visualize_heatmap()
    print(f"✅ Heatmap created successfully!")
    print(f"   Shape: {fig.data[0].z.shape}")
    print(f"   X labels: {fig.data[0].x}")
    print(f"   Y labels: {list(fig.data[0].y)}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
