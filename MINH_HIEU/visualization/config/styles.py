"""
Styles Configuration - Cấu hình giao diện và màu sắc
Tối ưu cho máy yếu với cấu hình rendering nhẹ
"""

# Cấu hình màu sắc cho các loại thảm họa
COLOR_PALETTE = {
    'Động đất': '#FF6B6B',
    'Lũ lụt': '#4ECDC4',
    'Bão': '#95E1D3',
    'Cháy rừng': '#FF8B94',
    'Hạn hán': '#FFA07A',
    'Lốc xoáy': '#B4A7D6',
    'Sạt lở đất': '#8B7355',
    'Nắng nóng cực đoan': '#FFD93D',
    'Bão biển': '#6C5B7B',
    'Núi lửa': '#C06C84'
}

# Cấu hình chung cho tất cả biểu đồ (tối ưu hiệu năng)
CHART_CONFIG = {
    'displayModeBar': False,  # Ẩn thanh công cụ để nhẹ hơn
    'staticPlot': False,      # Cho phép tương tác nhưng nhẹ
    'responsive': True,        # Responsive design
    'scrollZoom': False,       # Tắt zoom bằng chuột để tránh lag
}

# Template layout mặc định (tối ưu cho máy yếu)
DEFAULT_LAYOUT = {
    'template': 'plotly_white',
    'font': {
        'family': 'Arial, sans-serif',
        'size': 12,
        'color': '#2C3E50'
    },
    'paper_bgcolor': '#FFFFFF',
    'plot_bgcolor': '#F8F9FA',
    'margin': {'l': 60, 'r': 30, 't': 80, 'b': 60},
    'hoverlabel': {
        'bgcolor': 'white',
        'font_size': 11,
        'font_family': 'Arial'
    },
    'hovermode': 'closest',
    'showlegend': True,
    'legend': {
        'orientation': 'h',
        'yanchor': 'bottom',
        'y': -0.2,
        'xanchor': 'center',
        'x': 0.5,
        'font': {'size': 10}
    }
}

# Cấu hình cho các loại biểu đồ cụ thể
SUNBURST_CONFIG = {
    'marker': {
        'line': {'width': 2, 'color': 'white'}
    },
    'branchvalues': 'total',
    'textinfo': 'label+percent parent',
}

MAP_CONFIG = {
    'geo': {
        'showframe': False,
        'showcoastlines': True,
        'projection_type': 'natural earth',
        'bgcolor': 'rgba(0,0,0,0)',
        'coastlinecolor': '#95A5A6',
        'coastlinewidth': 0.5,
        'landcolor': '#ECF0F1',
        'showland': True,
        'showcountries': True,
        'countrycolor': '#BDC3C7',
        'countrywidth': 0.3,
    }
}

HEATMAP_CONFIG = {
    'colorscale': 'Reds',
    'colorbar': {
        'title': {
            'text': 'Số sự kiện',
            'side': 'right'
        },
        'thickness': 15,
        'len': 0.7
    },
    'xgap': 1,
    'ygap': 1
}

# Kích thước biểu đồ mặc định (tối ưu cho notebook)
DEFAULT_SIZE = {
    'width': 1000,
    'height': 600
}

DASHBOARD_SIZE = {
    'width': 1200,
    'height': 900
}

SMALL_CHART_SIZE = {
    'width': 800,
    'height': 500
}
