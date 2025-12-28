"""
Response Efficiency Component
Hiệu quả ứng phó: scatter plot và radar chart
"""

import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple
import pandas as pd

# Import utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_data, aggregate_by_disaster_type
from config.styles import COLOR_PALETTE, DEFAULT_SIZE
from config.translations import UI_TEXT


def visualize_response_efficiency() -> Tuple[go.Figure, go.Figure]:
    """
    Tạo biểu đồ hiệu quả ứng phó và hồi phục
    
    Returns:
        Tuple chứa (scatter_figure, radar_figure)
    """
    # Load data
    df = load_data()
    
    # 1. SCATTER PLOT: Response time vs Recovery time
    fig1 = px.scatter(
        df,
        x='response_time_hours',
        y='recovery_days',
        color='loai_tham_hoa',
        size='severity_index',
        hover_name='country',
        hover_data={
            'response_time_hours': ':.1f',
            'recovery_days': ':.1f',
            'severity_index': ':.2f',
            'loai_tham_hoa': True
        },
        color_discrete_map=COLOR_PALETTE,
        size_max=20,
        title=UI_TEXT['response_scatter']
    )
    
    fig1.update_traces(
        marker=dict(
            line=dict(width=1, color='white'),
            opacity=0.7
        ),
        hovertemplate='<b>%{hovertext}</b><br>' +
                      'Loại: %{customdata[3]}<br>' +
                      'Ứng phó: %{x:.1f} giờ<br>' +
                      'Phục hồi: %{y:.1f} ngày<br>' +
                      'Mức độ: %{customdata[2]:.1f}<br>' +
                      '<extra></extra>'
    )
    
    fig1.update_layout(
        title={
            'text': UI_TEXT['response_scatter'],
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'weight': 'bold'}
        },
        xaxis_title='Thời gian ứng phó (giờ)',
        yaxis_title='Thời gian phục hồi (ngày)',
        width=DEFAULT_SIZE['width'],
        height=DEFAULT_SIZE['height'],
        template='plotly_white',
        font={'family': 'Arial', 'size': 12},
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#F8F9FA',
        xaxis=dict(
            gridcolor='#E5E5E5',
            gridwidth=0.5
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
        )
    )
    
    # 2. RADAR CHART: So sánh hiệu suất theo loại thảm họa
    agg_df = aggregate_by_disaster_type(df)
    
    # Chuẩn hóa các giá trị về scale 0-100 để dễ so sánh
    agg_df['response_normalized'] = 100 - (agg_df['thoi_gian_ung_pho_tb'] / agg_df['thoi_gian_ung_pho_tb'].max() * 100)
    agg_df['recovery_normalized'] = 100 - (agg_df['thoi_gian_phuc_hoi_tb'] / agg_df['thoi_gian_phuc_hoi_tb'].max() * 100)
    agg_df['severity_normalized'] = agg_df['muc_do_nghiem_trong_tb'] / agg_df['muc_do_nghiem_trong_tb'].max() * 100
    
    fig2 = go.Figure()
    
    # Chọn top 6 loại thảm họa phổ biến nhất để radar chart không quá tải
    top_disasters = agg_df.nlargest(6, 'so_su_kien')
    
    for idx, row in top_disasters.iterrows():
        fig2.add_trace(go.Scatterpolar(
            r=[
                row['response_normalized'],
                row['recovery_normalized'],
                100 - row['severity_normalized'],  # Đảo ngược để cao = tốt
                (row['so_su_kien'] / agg_df['so_su_kien'].max() * 100)
            ],
            theta=['Tốc độ ứng phó', 'Tốc độ phục hồi', 'Mức độ an toàn', 'Tần suất'],
            fill='toself',
            name=row['loai_tham_hoa'],
            marker=dict(
                color=COLOR_PALETTE.get(row['loai_tham_hoa'], '#999999')
            ),
            opacity=0.6,
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          '%{theta}: %{r:.1f}<br>' +
                          '<extra></extra>'
        ))
    
    fig2.update_layout(
        title={
            'text': UI_TEXT['response_radar'],
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'weight': 'bold'}
        },
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix='',
                gridcolor='#E5E5E5'
            ),
            angularaxis=dict(
                gridcolor='#E5E5E5'
            ),
            bgcolor='#F8F9FA'
        ),
        width=DEFAULT_SIZE['width'],
        height=DEFAULT_SIZE['height'],
        template='plotly_white',
        font={'family': 'Arial', 'size': 12},
        paper_bgcolor='#FFFFFF',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5,
            font={'size': 10}
        )
    )
    
    return fig1, fig2


if __name__ == "__main__":
    """Test standalone execution"""
    print("🔄 Đang tạo biểu đồ hiệu quả ứng phó...")
    fig1, fig2 = visualize_response_efficiency()
    print("✅ Hoàn thành!")
    print(f"📊 Scatter plot: {sum([len(trace.x) for trace in fig1.data])} điểm")
    print(f"📊 Radar chart: {len(fig2.data)} loại thảm họa")
    
    # Show figures
    fig1.show(config={'displayModeBar': False})
    fig2.show(config={'displayModeBar': False})
