"""
Disaster Distribution Component
Phân bố loại thảm họa: Sunburst chart và Bar chart
"""

import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple

# Import utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_data, aggregate_by_disaster_type
from utils.figure_factory import create_base_figure, format_number
from config.styles import COLOR_PALETTE, SUNBURST_CONFIG, DEFAULT_SIZE
from config.translations import UI_TEXT


def visualize_disaster_distribution() -> Tuple[go.Figure, go.Figure]:
    """
    Tạo biểu đồ phân bố loại thảm họa
    
    Returns:
        Tuple chứa (sunburst_figure, bar_figure)
    """
    # Load data
    df = load_data()
    agg_df = aggregate_by_disaster_type(df)
    
    # 1. SUNBURST CHART
    fig1 = px.sunburst(
        df,
        path=['loai_tham_hoa', 'country'],
        values='severity_index',
        color='loai_tham_hoa',
        color_discrete_map=COLOR_PALETTE,
        title=UI_TEXT['distribution_sunburst']
    )
    
    fig1.update_traces(**SUNBURST_CONFIG)
    
    fig1.update_layout(
        width=DEFAULT_SIZE['width'],
        height=DEFAULT_SIZE['height'],
        title={
            'text': UI_TEXT['distribution_sunburst'],
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'weight': 'bold'}
        },
        font={'family': 'Arial', 'size': 12},
        paper_bgcolor='#FFFFFF',
        showlegend=False
    )
    
    # 2. BAR CHART
    fig2 = go.Figure()
    
    fig2.add_trace(go.Bar(
        x=agg_df['loai_tham_hoa'],
        y=agg_df['so_su_kien'],
        marker=dict(
            color=[COLOR_PALETTE[x] for x in agg_df['loai_tham_hoa']],
            line=dict(color='white', width=2)
        ),
        text=agg_df['so_su_kien'],
        textposition='outside',
        texttemplate='%{text:,.0f}',
        hovertemplate='<b>%{x}</b><br>' +
                      'Số sự kiện: %{y:,.0f}<br>' +
                      '<extra></extra>'
    ))
    
    fig2.update_layout(
        title={
            'text': UI_TEXT['distribution_bar'],
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'weight': 'bold'}
        },
        xaxis_title=UI_TEXT['disaster_type'],
        yaxis_title=UI_TEXT['total_events'],
        width=DEFAULT_SIZE['width'],
        height=DEFAULT_SIZE['height'],
        template='plotly_white',
        font={'family': 'Arial', 'size': 12},
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#F8F9FA',
        showlegend=False,
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            gridcolor='#E5E5E5',
            gridwidth=0.5
        ),
        margin={'l': 80, 'r': 30, 't': 100, 'b': 120}
    )
    
    return fig1, fig2


if __name__ == "__main__":
    """Test standalone execution"""
    print("🔄 Đang tạo biểu đồ phân bố loại thảm họa...")
    fig1, fig2 = visualize_disaster_distribution()
    print("✅ Hoàn thành!")
    print(f"📊 Sunburst chart: {len(fig1.data)} traces")
    print(f"📊 Bar chart: {len(fig2.data)} traces")
    
    # Show figures
    fig1.show(config={'displayModeBar': False})
    fig2.show(config={'displayModeBar': False})
