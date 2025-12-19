"""
Top Countries Component
Top 10 quốc gia bị ảnh hưởng: treemap và donut chart
"""

import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple

# Import utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_data, aggregate_by_country
from utils.figure_factory import format_number
from config.styles import COLOR_PALETTE, DEFAULT_SIZE
from config.translations import UI_TEXT


def visualize_top_countries() -> Tuple[go.Figure, go.Figure]:
    """
    Tạo biểu đồ top 10 quốc gia bị ảnh hưởng
    
    Returns:
        Tuple chứa (treemap_figure, donut_figure)
    """
    # Load data
    df = load_data()
    top_countries = aggregate_by_country(df, top_n=10)
    
    # 1. TREEMAP
    fig1 = px.treemap(
        top_countries,
        path=['country'],
        values='so_su_kien',
        color='so_su_kien',
        color_continuous_scale='Reds',
        title=UI_TEXT['top_countries_treemap']
    )
    
    fig1.update_traces(
        textposition='middle center',
        texttemplate='<b>%{label}</b><br>%{value:,.0f} sự kiện',
        marker=dict(line=dict(width=2, color='white')),
        hovertemplate='<b>%{label}</b><br>' +
                      'Số sự kiện: %{value:,.0f}<br>' +
                      'Tỷ lệ: %{percentParent}<br>' +
                      '<extra></extra>'
    )
    
    fig1.update_layout(
        title={
            'text': UI_TEXT['top_countries_treemap'],
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'weight': 'bold'}
        },
        width=DEFAULT_SIZE['width'],
        height=DEFAULT_SIZE['height'],
        template='plotly_white',
        font={'family': 'Arial', 'size': 12},
        paper_bgcolor='#FFFFFF',
        coloraxis_colorbar=dict(
            title='Số sự kiện',
            thickness=15,
            len=0.7
        )
    )
    
    # 2. DONUT CHART
    # Tạo màu gradient cho donut
    colors_donut = px.colors.sequential.Reds_r[:10]
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Pie(
        labels=top_countries['country'],
        values=top_countries['so_su_kien'],
        hole=0.4,
        marker=dict(
            colors=colors_donut,
            line=dict(color='white', width=2)
        ),
        textposition='auto',
        texttemplate='%{label}<br>%{percent}',
        hovertemplate='<b>%{label}</b><br>' +
                      'Số sự kiện: %{value:,.0f}<br>' +
                      'Tỷ lệ: %{percent}<br>' +
                      '<extra></extra>'
    ))
    
    # Thêm text ở giữa
    total_events = top_countries['so_su_kien'].sum()
    
    fig2.update_layout(
        title={
            'text': UI_TEXT['top_countries_donut'],
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'weight': 'bold'}
        },
        width=DEFAULT_SIZE['width'],
        height=DEFAULT_SIZE['height'],
        template='plotly_white',
        font={'family': 'Arial', 'size': 11},
        paper_bgcolor='#FFFFFF',
        annotations=[dict(
            text=f'<b>Tổng</b><br>{format_number(total_events, "short")}<br>sự kiện',
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False,
            font_family='Arial'
        )],
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.02,
            font={'size': 10}
        )
    )
    
    return fig1, fig2


if __name__ == "__main__":
    """Test standalone execution"""
    print("🔄 Đang tạo biểu đồ top 10 quốc gia...")
    fig1, fig2 = visualize_top_countries()
    print("✅ Hoàn thành!")
    print(f"📊 Treemap: {len(fig1.data[0].labels)} quốc gia")
    print(f"📊 Donut chart: {len(fig2.data[0].labels)} quốc gia")
    
    # Show figures
    fig1.show(config={'displayModeBar': False})
    fig2.show(config={'displayModeBar': False})
