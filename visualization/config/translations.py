"""
Translations Configuration - Từ điển dịch thuật tiếng Việt
Chứa tất cả text UI và mapping disaster types
"""

# Từ điển dịch loại thảm họa từ tiếng Anh sang tiếng Việt
DISASTER_VN = {
    'Earthquake': 'Động đất',
    'Flood': 'Lũ lụt',
    'Hurricane': 'Bão',
    'Wildfire': 'Cháy rừng',
    'Drought': 'Hạn hán',
    'Tornado': 'Lốc xoáy',
    'Landslide': 'Sạt lở đất',
    'Extreme Heat': 'Nắng nóng cực đoan',
    'Storm Surge': 'Bão biển',
    'Volcanic Eruption': 'Núi lửa'
}

# Các text UI tiếng Việt
UI_TEXT = {
    # Tiêu đề chính
    'main_title': '🌍 Trực Quan Dữ Liệu Thảm Họa Toàn Cầu 2018-2024',
    'subtitle': 'Phân tích và Trực quan hóa Dữ liệu Thiên tai Thế giới',
    
    # Nhãn trục và tiêu đề biểu đồ
    'disaster_type': 'Loại Thảm Họa',
    'country': 'Quốc Gia',
    'year': 'Năm',
    'month': 'Tháng',
    'count': 'Số lượng',
    'total_events': 'Tổng số sự kiện',
    'casualties': 'Số người thương vong',
    'economic_loss': 'Thiệt hại kinh tế (USD)',
    'severity_index': 'Chỉ số mức độ nghiêm trọng',
    'response_time': 'Thời gian ứng phó (ngày)',
    'recovery_time': 'Thời gian phục hồi (ngày)',
    'affected_population': 'Dân số bị ảnh hưởng',
    
    # Tiêu đề các phần
    'distribution_title': '📊 Phân Bố Loại Thảm Họa',
    'distribution_sunburst': 'Phân Bố Loại Thảm Họa (Sunburst)',
    'distribution_bar': 'Tần Suất Các Loại Thảm Họa',
    
    'map_title': '🌐 Bản Đồ Thảm Họa Toàn Cầu',
    'map_subtitle': 'Phân Bố Địa Lý và Mức Độ Nghiêm Trọng',
    
    'trends_title': '📈 Xu Hướng Thảm Họa Theo Thời Gian',
    'trends_yearly': 'Xu Hướng Hàng Năm Theo Loại Thảm Họa',
    'trends_monthly': 'Xu Hướng Trung Bình Theo Tháng',
    
    'economic_title': '💰 Thiệt Hại Kinh Tế và Thương Vong',
    'economic_loss_chart': 'Top 10 Quốc Gia: Thiệt Hại Kinh Tế',
    'casualties_chart': 'Top 10 Quốc Gia: Số Người Thương Vong',
    
    'top_countries_title': '🏆 Top 10 Quốc Gia Bị Ảnh Hưởng Nhiều Nhất',
    'top_countries_treemap': 'Tổng Số Sự Kiện Theo Quốc Gia',
    'top_countries_donut': 'Phân Bố Tỷ Lệ Top 10 Quốc Gia',
    
    'response_title': '⚡ Hiệu Quả Ứng Phó và Hồi Phục',
    'response_scatter': 'Mối Quan Hệ Thời Gian Ứng Phó - Hồi Phục',
    'response_radar': 'So Sánh Hiệu Suất Ứng Phó Theo Loại Thảm Họa',
    
    'heatmap_title': '🔥 Heatmap: Thảm Họa Theo Tháng và Loại',
    'heatmap_subtitle': 'Phân Bố Theo Mùa Của Các Loại Thảm Họa',
    
    'dashboard_title': '📊 Dashboard Tổng Quan',
    'dashboard_subtitle': 'Tổng Hợp Các Chỉ Số Chính',
    
    # Tên tháng tiếng Việt
    'months': {
        'January': 'Tháng 1',
        'February': 'Tháng 2',
        'March': 'Tháng 3',
        'April': 'Tháng 4',
        'May': 'Tháng 5',
        'June': 'Tháng 6',
        'July': 'Tháng 7',
        'August': 'Tháng 8',
        'September': 'Tháng 9',
        'October': 'Tháng 10',
        'November': 'Tháng 11',
        'December': 'Tháng 12'
    },
    
    # Hover template
    'hover_country': '<b>%{customdata[0]}</b>',
    'hover_disaster': 'Loại: %{customdata[1]}',
    'hover_events': 'Số sự kiện: %{customdata[2]:,}',
    'hover_casualties': 'Thương vong: %{customdata[3]:,}',
    'hover_economic': 'Thiệt hại: $%{customdata[4]:,.0f}',
}

# Định dạng số
NUMBER_FORMAT = {
    'decimal_separator': ',',
    'thousands_separator': '.',
    'currency_symbol': '$',
    'percentage_format': '{:.1f}%'
}
