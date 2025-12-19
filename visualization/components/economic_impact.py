"""
Economic Impact Component
Thiệt hại kinh tế và thương vong: dual bar charts
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_data, aggregate_by_country
from utils.figure_factory import format_number
from config.styles import DEFAULT_SIZE
from config.translations import UI_TEXT


def visualize_economic_casualties() -> go.Figure:
    """
    Tạo biểu đồ thiệt hại kinh tế và thương vong
    
    Returns:
        Figure object với 2 subplots (economic loss, casualties)
    """
    # Load data
    df = load_data()
    top_countries = aggregate_by_country(df, top_n=10)
    
    # Tạo subplot với 1 hàng, 2 cột
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            UI_TEXT['economic_loss_chart'],
            UI_TEXT['casualties_chart']
        ),
        horizontal_spacing=0.12
    )
    
    # 1. THIỆT HẠI KINH TẾ
    fig.add_trace(
        go.Bar(
            x=top_countries['tong_thiet_hai'],
            y=top_countries['country'],
            orientation='h',
            marker=dict(
                color='#E74C3C',
                line=dict(color='white', width=1.5)
            ),
            text=[format_number(x, 'currency') for x in top_countries['tong_thiet_hai']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>' +
                          'Thiệt hại: $%{x:,.0f}<br>' +
                          '<extra></extra>',
            name='Thiệt hại kinh tế'
        ),
        row=1, col=1
    )
    
    # 2. THƯƠNG VONG
    fig.add_trace(
        go.Bar(
            x=top_countries['tong_thuong_vong'],
            y=top_countries['country'],
            orientation='h',
            marker=dict(
                color='#3498DB',
                line=dict(color='white', width=1.5)
            ),
            text=[format_number(x, 'short') for x in top_countries['tong_thuong_vong']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>' +
                          'Thương vong: %{x:,.0f} người<br>' +
                          '<extra></extra>',
            name='Thương vong'
        ),
        row=1, col=2
    )
    
    # Cập nhật layout
    fig.update_layout(
        title={
            'text': UI_TEXT['economic_title'],
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'weight': 'bold'}
        },
        width=1200,
        height=600,
        template='plotly_white',
        font={'family': 'Arial', 'size': 11},
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#F8F9FA',
        showlegend=False,
        margin={'l': 150, 'r': 100, 't': 100, 'b': 60}
    )
    
    # Cập nhật axes
    fig.update_xaxes(
        title_text='Thiệt hại (USD)',
        gridcolor='#E5E5E5',
        gridwidth=0.5,
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text='Số người',
        gridcolor='#E5E5E5',
        gridwidth=0.5,
        row=1, col=2
    )
    
    fig.update_yaxes(
        title_text='',
        tickfont=dict(size=10),
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text='',
        tickfont=dict(size=10),
        row=1, col=2
    )
    
    return fig


if __name__ == "__main__":
    """Test standalone execution"""
    print("🔄 Đang tạo biểu đồ thiệt hại kinh tế và thương vong...")
    fig = visualize_economic_casualties()
    print("✅ Hoàn thành!")
    print(f"📊 Số quốc gia: {len(fig.data[0].y)}")
    
    # Show figure
    fig.show(config={'displayModeBar': False})
