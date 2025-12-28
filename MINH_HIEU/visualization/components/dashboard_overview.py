"""
Dashboard Overview Component
Dashboard tổng quan với 6 panels chính
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Import utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_data, aggregate_by_disaster_type, aggregate_by_country
from utils.figure_factory import format_number
from config.styles import COLOR_PALETTE, DASHBOARD_SIZE
from config.translations import UI_TEXT


def visualize_dashboard() -> go.Figure:
    """
    Tạo dashboard tổng quan với 6 panels
    
    Returns:
        Figure object chứa dashboard
    """
    # Load data
    df = load_data()
    agg_disaster = aggregate_by_disaster_type(df)
    agg_country = aggregate_by_country(df, top_n=10)
    
    # Tạo subplot layout: 3 rows x 2 cols
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '📊 Phân bố theo loại',
            '🌍 Top 10 quốc gia',
            '📈 Xu hướng hàng năm',
            '💰 Thiệt hại kinh tế',
            '🔥 Mức độ nghiêm trọng',
            '⚡ Thời gian ứng phó'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'scatter'}, {'type': 'bar'}],
            [{'type': 'bar'}, {'type': 'scatter'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # 1. PHÂN BỐ THEO LOẠI (Top 5)
    top5_disaster = agg_disaster.head(5)
    fig.add_trace(
        go.Bar(
            x=top5_disaster['loai_tham_hoa'],
            y=top5_disaster['so_su_kien'],
            marker=dict(color=[COLOR_PALETTE[x] for x in top5_disaster['loai_tham_hoa']]),
            text=top5_disaster['so_su_kien'],
            textposition='outside',
            texttemplate='%{text:,.0f}',
            showlegend=False,
            hovertemplate='%{x}<br>%{y:,.0f} sự kiện<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. TOP 10 QUỐC GIA
    fig.add_trace(
        go.Bar(
            x=agg_country['so_su_kien'][:5],  # Top 5 cho gọn
            y=agg_country['country'][:5],
            orientation='h',
            marker=dict(color='#E74C3C'),
            text=agg_country['so_su_kien'][:5],
            textposition='outside',
            texttemplate='%{text:,.0f}',
            showlegend=False,
            hovertemplate='%{y}<br>%{x:,.0f} sự kiện<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. XU HƯỚNG HÀNG NĂM
    yearly = df.groupby('year').size().reset_index(name='count')
    fig.add_trace(
        go.Scatter(
            x=yearly['year'],
            y=yearly['count'],
            mode='lines+markers',
            line=dict(color='#3498DB', width=3),
            marker=dict(size=8, color='#3498DB'),
            showlegend=False,
            hovertemplate='%{x}<br>%{y:,.0f} sự kiện<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. THIỆT HẠI KINH TẾ (Top 5 quốc gia)
    fig.add_trace(
        go.Bar(
            x=agg_country['tong_thiet_hai'][:5] / 1e9,  # Đơn vị: tỷ USD
            y=agg_country['country'][:5],
            orientation='h',
            marker=dict(color='#F39C12'),
            text=[format_number(x, 'currency') for x in agg_country['tong_thiet_hai'][:5]],
            textposition='outside',
            showlegend=False,
            hovertemplate='%{y}<br>$%{x:.2f}B<extra></extra>'
        ),
        row=2, col=2
    )
    
    # 5. MỨC ĐỘ NGHIÊM TRỌNG (trung bình theo loại)
    fig.add_trace(
        go.Bar(
            x=agg_disaster['loai_tham_hoa'][:5],
            y=agg_disaster['muc_do_nghiem_trong_tb'][:5],
            marker=dict(color='#E67E22'),
            text=agg_disaster['muc_do_nghiem_trong_tb'][:5].round(2),
            textposition='outside',
            texttemplate='%{text:.2f}',
            showlegend=False,
            hovertemplate='%{x}<br>Mức độ: %{y:.2f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 6. THỜI GIAN ỨNG PHÓ vs PHỤ HỒI
    fig.add_trace(
        go.Scatter(
            x=agg_disaster['thoi_gian_ung_pho_tb'],
            y=agg_disaster['thoi_gian_phuc_hoi_tb'],
            mode='markers+text',
            marker=dict(
                size=agg_disaster['so_su_kien'] / 10,
                color=[COLOR_PALETTE[x] for x in agg_disaster['loai_tham_hoa']],
                line=dict(width=1, color='white')
            ),
            text=[x[:6] + '...' if len(x) > 6 else x for x in agg_disaster['loai_tham_hoa']],
            textposition='top center',
            textfont=dict(size=8),
            showlegend=False,
            hovertemplate='%{text}<br>Ứng phó: %{x:.1f} giờ<br>Phục hồi: %{y:.1f} ngày<extra></extra>'
        ),
        row=3, col=2
    )
    
    # Cập nhật layout chung
    fig.update_layout(
        title={
            'text': f"{UI_TEXT['dashboard_title']}<br><sub>{UI_TEXT['dashboard_subtitle']}</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'weight': 'bold'}
        },
        width=DASHBOARD_SIZE['width'],
        height=DASHBOARD_SIZE['height'],
        template='plotly_white',
        font={'family': 'Arial', 'size': 10},
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#F8F9FA',
        showlegend=False,
        margin={'l': 100, 'r': 50, 't': 120, 'b': 60}
    )
    
    # Cập nhật axes
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9), row=1, col=1)
    fig.update_xaxes(title_text='Số sự kiện', row=1, col=2)
    fig.update_xaxes(title_text='Năm', dtick=1, row=2, col=1)
    fig.update_xaxes(title_text='Tỷ USD', row=2, col=2)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9), row=3, col=1)
    fig.update_xaxes(title_text='Thời gian ứng phó (giờ)', row=3, col=2)
    
    fig.update_yaxes(title_text='Số sự kiện', row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=9), row=1, col=2)
    fig.update_yaxes(title_text='Số sự kiện', row=2, col=1)
    fig.update_yaxes(tickfont=dict(size=9), row=2, col=2)
    fig.update_yaxes(title_text='Mức độ TB', row=3, col=1)
    fig.update_yaxes(title_text='Thời gian phục hồi (ngày)', row=3, col=2)
    
    # Grid cho tất cả subplots
    fig.update_xaxes(gridcolor='#E5E5E5', gridwidth=0.5)
    fig.update_yaxes(gridcolor='#E5E5E5', gridwidth=0.5)
    
    return fig


if __name__ == "__main__":
    """Test standalone execution"""
    print("🔄 Đang tạo dashboard tổng quan...")
    fig = visualize_dashboard()
    print("✅ Hoàn thành!")
    print(f"📊 Số panels: {len(fig.data)}")
    
    # Show figure
    fig.show(config={'displayModeBar': False})
