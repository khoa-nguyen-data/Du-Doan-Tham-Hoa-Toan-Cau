"""
World Map Component
Bản đồ thảm họa toàn cầu với scatter points
"""

import plotly.express as px
import plotly.graph_objects as go

# Import utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_data
from config.styles import COLOR_PALETTE, MAP_CONFIG, DEFAULT_SIZE
from config.translations import UI_TEXT


def visualize_world_map() -> go.Figure:
    """
    Tạo bản đồ thảm họa toàn cầu
    
    Returns:
        Figure object chứa bản đồ scatter
    """
    # Load data
    df = load_data()
    
    # Tạo bản đồ scatter
    fig = px.scatter_geo(
        df,
        lat='latitude',
        lon='longitude',
        color='loai_tham_hoa',
        size='severity_index',
        hover_name='country',
        hover_data={
            'loai_tham_hoa': True,
            'severity_index': ':.2f',
            'casualties': ':,.0f',
            'economic_loss_usd': ':,.0f',
            'latitude': False,
            'longitude': False
        },
        color_discrete_map=COLOR_PALETTE,
        size_max=30,
        title=UI_TEXT['map_title']
    )
    
    # Cập nhật layout bản đồ
    fig.update_layout(
        title={
            'text': f"{UI_TEXT['map_title']}<br><sub>{UI_TEXT['map_subtitle']}</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'weight': 'bold'}
        },
        width=DEFAULT_SIZE['width'],
        height=DEFAULT_SIZE['height'],
        template='plotly_white',
        font={'family': 'Arial', 'size': 12},
        paper_bgcolor='#FFFFFF',
        **MAP_CONFIG
    )
    
    # Tối ưu hover template
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>' +
                      'Loại: %{customdata[0]}<br>' +
                      'Mức độ: %{customdata[1]:.1f}<br>' +
                      'Thương vong: %{customdata[2]:,.0f}<br>' +
                      'Thiệt hại: $%{customdata[3]:,.0f}<br>' +
                      '<extra></extra>'
    )
    
    # Legend position
    fig.update_layout(
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=0.02,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#CCCCCC',
            borderwidth=1,
            font={'size': 10}
        )
    )
    
    return fig


if __name__ == "__main__":
    """Test standalone execution"""
    print("🔄 Đang tạo bản đồ thảm họa toàn cầu...")
    fig = visualize_world_map()
    print("✅ Hoàn thành!")
    print(f"📊 Số điểm trên bản đồ: {len(fig.data[0].lat)}")
    
    # Show figure
    fig.show(config={'displayModeBar': False})
