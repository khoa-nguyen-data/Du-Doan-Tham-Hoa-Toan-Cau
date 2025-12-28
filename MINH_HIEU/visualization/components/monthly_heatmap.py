"""
Monthly Heatmap Component
Heatmap phân bố thảm họa theo tháng và loại
"""

import plotly.graph_objects as go
import pandas as pd

# Import utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_data
from config.styles import HEATMAP_CONFIG, DEFAULT_SIZE
from config.translations import UI_TEXT


def visualize_heatmap() -> go.Figure:
    """
    Tạo heatmap phân bố thảm họa theo tháng và loại
    
    Returns:
        Figure object chứa heatmap
    """
    # Load data
    df = load_data()
    
    # Tạo pivot table
    heatmap_data = df.groupby(['month', 'loai_tham_hoa']).size().reset_index(name='count')
    pivot_data = heatmap_data.pivot(index='loai_tham_hoa', columns='month', values='count')
    pivot_data = pivot_data.fillna(0)
    
    # Tên tháng tiếng Việt
    month_labels = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 
                    'T7', 'T8', 'T9', 'T10', 'T11', 'T12']
    
    # Tạo heatmap
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=pivot_data.values,
        x=month_labels,
        y=pivot_data.index,
        **HEATMAP_CONFIG,
        text=pivot_data.values,
        texttemplate='%{text:.0f}',
        textfont=dict(size=10),
        hovertemplate='<b>%{y}</b><br>' +
                      'Tháng: %{x}<br>' +
                      'Số sự kiện: %{z:.0f}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': f"{UI_TEXT['heatmap_title']}<br><sub>{UI_TEXT['heatmap_subtitle']}</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'weight': 'bold'}
        },
        xaxis_title=UI_TEXT['month'],
        yaxis_title=UI_TEXT['disaster_type'],
        width=DEFAULT_SIZE['width'],
        height=DEFAULT_SIZE['height'],
        template='plotly_white',
        font={'family': 'Arial', 'size': 12},
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        xaxis=dict(
            side='bottom',
            tickmode='array',
            tickvals=list(range(12)),
            ticktext=month_labels,
            tickangle=0
        ),
        yaxis=dict(
            tickfont=dict(size=11)
        ),
        margin={'l': 150, 'r': 100, 't': 100, 'b': 80}
    )
    
    return fig


if __name__ == "__main__":
    """Test standalone execution"""
    print("🔄 Đang tạo heatmap thảm họa theo tháng...")
    fig = visualize_heatmap()
    print("✅ Hoàn thành!")
    print(f"📊 Kích thước heatmap: {fig.data[0].z.shape}")
    
    # Show figure
    fig.show(config={'displayModeBar': False})
