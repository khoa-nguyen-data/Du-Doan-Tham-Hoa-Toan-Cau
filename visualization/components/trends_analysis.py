"""
Trends Analysis Component
Xu hướng thảm họa theo thời gian: yearly và monthly trends
"""

import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple

# Import utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_data
from config.styles import COLOR_PALETTE, DEFAULT_SIZE
from config.translations import UI_TEXT


def visualize_trends() -> Tuple[go.Figure, go.Figure]:
    """
    Tạo biểu đồ xu hướng thảm họa theo thời gian
    
    Returns:
        Tuple chứa (yearly_figure, monthly_figure)
    """
    # Load data
    df = load_data()
    
    # 1. XU HƯỚNG HÀNG NĂM
    yearly_data = df.groupby(['year', 'loai_tham_hoa']).size().reset_index(name='count')
    
    fig1 = px.line(
        yearly_data,
        x='year',
        y='count',
        color='loai_tham_hoa',
        markers=True,
        color_discrete_map=COLOR_PALETTE,
        title=UI_TEXT['trends_yearly']
    )
    
    fig1.update_traces(
        line=dict(width=3),
        marker=dict(size=8, line=dict(width=1, color='white')),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Năm: %{x}<br>' +
                      'Số sự kiện: %{y:,.0f}<br>' +
                      '<extra></extra>'
    )
    
    fig1.update_layout(
        title={
            'text': UI_TEXT['trends_yearly'],
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'weight': 'bold'}
        },
        xaxis_title=UI_TEXT['year'],
        yaxis_title=UI_TEXT['total_events'],
        width=DEFAULT_SIZE['width'],
        height=DEFAULT_SIZE['height'],
        template='plotly_white',
        font={'family': 'Arial', 'size': 12},
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#F8F9FA',
        xaxis=dict(
            gridcolor='#E5E5E5',
            gridwidth=0.5,
            dtick=1
        ),
        yaxis=dict(
            gridcolor='#E5E5E5',
            gridwidth=0.5
        ),
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.98,
            xanchor='right',
            x=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#CCCCCC',
            borderwidth=1,
            font={'size': 10}
        ),
        hovermode='x unified'
    )
    
    # 2. XU HƯỚNG THEO THÁNG (TRUNG BÌNH)
    monthly_data = df.groupby(['month', 'loai_tham_hoa']).size().reset_index(name='count')
    
    # Dịch tên tháng
    month_names = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 
                   'T7', 'T8', 'T9', 'T10', 'T11', 'T12']
    monthly_data['month_label'] = monthly_data['month'].apply(lambda x: month_names[x-1])
    
    fig2 = px.line(
        monthly_data,
        x='month_label',
        y='count',
        color='loai_tham_hoa',
        markers=True,
        color_discrete_map=COLOR_PALETTE,
        title=UI_TEXT['trends_monthly']
    )
    
    fig2.update_traces(
        line=dict(width=3),
        marker=dict(size=8, line=dict(width=1, color='white')),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Tháng: %{x}<br>' +
                      'Trung bình: %{y:.1f} sự kiện<br>' +
                      '<extra></extra>'
    )
    
    fig2.update_layout(
        title={
            'text': UI_TEXT['trends_monthly'],
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'weight': 'bold'}
        },
        xaxis_title=UI_TEXT['month'],
        yaxis_title='Số sự kiện trung bình',
        width=DEFAULT_SIZE['width'],
        height=DEFAULT_SIZE['height'],
        template='plotly_white',
        font={'family': 'Arial', 'size': 12},
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#F8F9FA',
        xaxis=dict(
            gridcolor='#E5E5E5',
            gridwidth=0.5,
            tickmode='array',
            tickvals=month_names,
            ticktext=month_names
        ),
        yaxis=dict(
            gridcolor='#E5E5E5',
            gridwidth=0.5
        ),
        legend=dict(
            orientation='v',
            yanchor='top',
            y=0.98,
            xanchor='right',
            x=0.98,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#CCCCCC',
            borderwidth=1,
            font={'size': 10}
        ),
        hovermode='x unified'
    )
    
    return fig1, fig2


if __name__ == "__main__":
    """Test standalone execution"""
    print("🔄 Đang tạo biểu đồ xu hướng thảm họa...")
    fig1, fig2 = visualize_trends()
    print("✅ Hoàn thành!")
    print(f"📊 Yearly trends: {len(fig1.data)} loại thảm họa")
    print(f"📊 Monthly trends: {len(fig2.data)} loại thảm họa")
    
    # Show figures
    fig1.show(config={'displayModeBar': False})
    fig2.show(config={'displayModeBar': False})
