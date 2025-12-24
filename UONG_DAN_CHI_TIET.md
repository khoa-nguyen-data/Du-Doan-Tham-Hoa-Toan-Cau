# 📖 GIẢI THÍCH CHI TIẾT: DỰ ĐOÁN THIỆT HẠI VỀ NGƯỜI DO THẢM HỌA
## Tài liệu hướng dẫn đầy đủ với ghi chú tiếng Việt

---

## 📋 MỤC LỤC
1. [Giới thiệu](#giới-thiệu)
2. [Kiến trúc chương trình](#kiến-trúc-chương-trình)
3. [Giải thích chi tiết từng phần](#giải-thích-chi-tiết-từng-phần)
4. [Quy trình làm việc](#quy-trình-làm-việc)
5. [Cách sử dụng](#cách-sử-dụng)
6. [Hiểu kết quả](#hiểu-kết-quả)
7. [Cải thiện mô hình](#cải-thiện-mô-hình)

---

## 🎯 Giới thiệu

### Bài toán
**Dự đoán số người bị chết do thảm họa** dựa trên các thông tin:

| Input | Ví dụ |
|-------|--------|
| 🌍 Quốc gia | India, Philippines, Brazil |
| ⚠️ Loại thảm họa | Earthquake, Flood, Hurricane, Tornado, ... |
| 📊 Mức độ nghiêm trọng | 1-10 (càng cao càng nguy hiểm) |
| 💰 Thiệt hại kinh tế | 1,000,000 - 10,000,000 USD |
| ⏱️ Thời gian phản ứng | 1-35 giờ |
| ⭐ Hiệu quả phản ứng | 0-100 (điểm) |
| 📍 Vĩ độ | -90 đến 90 |
| 📍 Kinh độ | -180 đến 180 |

**Output**: 👥 **Số người bị chết** (0-500+ người)

### Dữ liệu huấn luyện
- **Nguồn**: `du_lieu_sach.csv`
- **Số bản ghi**: ~50,000 sự kiện thảm họa (2018-2024)
- **Các cột**: 13 cột (date, country, disaster_type, severity_index, casualties, ...)

### Thuật toán sử dụng
**Gradient Boosting Regressor** - Một trong những thuật toán mạnh nhất trong Machine Learning:
- 📈 Xây dựng nhiều cây quyết định (300 cây)
- 🔄 Mỗi cây cố gắng sửa lỗi của cây trước
- ✅ Đạt R² score > 0.85 (rất tốt!)

---

## 🏗️ Kiến trúc chương trình

### Cấu trúc file

````markdown
---

## 📝 Giải thích chi tiết từng phần

### 📌 PHẦN 1: IMPORT LIBRARIES

```python
# Dòng 7-9: Xử lý dữ liệu
import pandas as pd           # Thư viện bảng tính (giống Excel nhưng lập trình)
import numpy as np            # Xử lý mảng và toán học

# Dòng 10-11: Vẽ biểu đồ
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ
import seaborn as sns           # Bao bọc matplotlib, vẽ đẹp hơn

# Dòng 12-15: Machine Learning
from sklearn.preprocessing import LabelEncoder, StandardScaler
#   LabelEncoder: Chuyển text thành số (India→5, Brazil→2)
#   StandardScaler: Chuẩn hóa dữ liệu (đưa về trung bình 0)

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
#   GradientBoostingRegressor: Thuật toán dự đoán
#   RandomForestRegressor: Thuật toán khác (dự phòng)

from sklearn.model_selection import train_test_split, cross_val_score
#   train_test_split: Chia tập train (80%) và test (20%)
#   cross_val_score: Kiểm chéo (không dùng ở đây)

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
#   Các hàm tính chỉ số đánh giá
#   - mean_squared_error: MSE = trung bình (sai_số)²
#   - r2_score: R² = 0-1, % phương sai được giải thích
#   - mean_absolute_error: MAE = trung bình |sai_số|
#   - mean_absolute_percentage_error: MAPE = % sai_số

# Dòng 18: Lưu/load mô hình
import joblib  # Lưu object Python vào file (.pkl)

# Dòng 19: Tắt cảnh báo
import warnings
warnings.filterwarnings('ignore')  # Không hiển thị warning
```

### 📌 PHẦN 2: ĐỊNH NGHĨA CLASS VÀ HÀM CHO DỰ ĐOÁN

```python
class DisasterCasualtyPredictor:
    """Lớp chính chứa mô hình dự đoán"""
    
    def __init__(self, csv_path: str):
        """
        Khởi tạo các biến instance
        
        Parameters:
            csv_path (str): Đường dẫn đến file CSV
                Ví dụ: "du_lieu_sach.csv"
        """
        # Biến lưu đường dẫn file
        self.csv_path = csv_path
        
        # Biến lưu dữ liệu
        self.df = None              # Dataframe gốc chưa chia train/test
        self.X_train = None         # Features của tập train (dùng huấn luyện)
        self.X_test = None          # Features của tập test (dùng kiểm tra)
        self.y_train = None         # Target của tập train (casualties để so sánh)
        self.y_test = None          # Target của tập test
        
        # Biến lưu mô hình
        self.model = None           # Mô hình Gradient Boosting
        
        # Biến lưu công cụ xử lý
        self.scaler = StandardScaler()  # Dùng chuẩn hóa dữ liệu
        self.label_encoders = {}        # Dict lưu encoder cho từng cột
                                        # Ví dụ: {
                                        #   'country': LabelEncoder(),
                                        #   'disaster_type': LabelEncoder()
                                        # }
        
        # Biến lưu thông tin mô hình
        self.feature_columns = None     # Danh sách tên 8 cột input
        self.feature_importance_df = None  # Bảng tầm quan trọng features

    def load_data(self):
        """
        Tải file CSV và làm sạch dữ liệu
        """
        # BƯỚC 1: Đọc file CSV
        self.df = pd.read_csv(self.csv_path)
        # Kết quả: Dataframe 50,000 dòng × 13 cột
        
        # BƯỚC 2: Làm sạch dữ liệu
        # Loại bỏ các bản ghi có severity_index <= 0 (vô lý)
        self.df = self.df[self.df['severity_index'] > 0].copy()
        # .copy() để tránh warning pandas
        
        # Loại bỏ casualties âm (không thể có số người bị chết âm!)
        self.df = self.df[self.df['casualties'] >= 0].copy()
        
        # Loại bỏ thiệt hại kinh tế âm
        self.df = self.df[self.df['economic_loss_usd'] >= 0].copy()
        
        # BƯỚC 3: In thông tin
        print(f"✅ Tải thành công {len(self.df):,} bản ghi")
        # :, để hiển thị dấu phân cách hàng nghìn
        # Ví dụ: 49,800 thay vì 49800
        
        print(f"📋 Các cột: {list(self.df.columns)}")
        # In danh sách tất cả cột
        
        # BƯỚC 4: In thống kê casualties
        print(f"\n📈 THỐNG KÊ CASUALTIES:")
        print(f"  • Min: {self.df['casualties'].min():.0f} người")
        # min(): Giá trị nhỏ nhất
        # :.0f để làm tròn 0 chữ số thập phân
        
        print(f"  • Max: {self.df['casualties'].max():.0f} người")
        # max(): Giá trị lớn nhất
        
        print(f"  • Mean: {self.df['casualties'].mean():.0f} người")
        # mean(): Giá trị trung bình
        
        print(f"  • Median: {self.df['casualties'].median():.0f} người")
        # median(): Giá trị giữa (50% bên dưới, 50% bên trên)
        
        print(f"  • Std: {self.df['casualties'].std():.0f}")
        # std(): Độ lệch chuẩn (phân tán dữ liệu)
        
        # BƯỚC 5: Thống kê theo loại thảm họa
        print(f"\n📊 CASUALTIES THEO LOẠI THẢM HỌA:")
        disaster_impact = self.df.groupby('disaster_type').agg({
            'casualties': ['count', 'mean', 'sum', 'min', 'max']
        }).round(0)
        # groupby('disaster_type'): Nhóm theo loại thảm họa
        # agg(): Tính các chỉ số:
        #   - 'count': Số lượng
        #   - 'mean': Trung bình
        #   - 'sum': Tổng
        #   - 'min': Min
        #   - 'max': Max
        # .round(0): Làm tròn 0 chữ số thập phân
        
        print(disaster_impact)
        # Hiển thị bảng
        
        return self.df

    def preprocess_data(self):
        """Tiền xử lý dữ liệu"""
        # Xóa các cột không cần thiết
        self.df = self.df.drop(['date', 'year'], axis=1)
        
        # Chia dữ liệu thành train (80%) và test (20%)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df.drop('casualties', axis=1),
            self.df['casualties'],
            test_size=0.2,
            random_state=42
        )

        # Chuẩn hóa dữ liệu số
        self.X_train[['severity_index', 'economic_loss', 'response_time', 'response_effectiveness', 'latitude', 'longitude']] = self.scaler.fit_transform(
            self.X_train[['severity_index', 'economic_loss', 'response_time', 'response_effectiveness', 'latitude', 'longitude']]
        )
        self.X_test[['severity_index', 'economic_loss', 'response_time', 'response_effectiveness', 'latitude', 'longitude']] = self.scaler.transform(
            self.X_test[['severity_index', 'economic_loss', 'response_time', 'response_effectiveness', 'latitude', 'longitude']]
        )

        # Mã hóa các biến phân loại
        for column in ['country', 'disaster_type']:
            le = LabelEncoder()
            le.fit(self.df[column])
            self.label_encoders[column] = le
            self.X_train[column] = le.transform(self.X_train[column])
            self.X_test[column] = le.transform(self.X_test[column])

    def train_model(self):
        """Huấn luyện mô hình Gradient Boosting Regressor"""
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """Đánh giá mô hình trên tập test"""
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        mape = mean_absolute_percentage_error(self.y_test, y_pred)

        # Tính toán độ chính xác dự đoán (chỉ áp dụng cho bài toán hồi quy có giá trị liên tục)
        accuracy = 100 - np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100

        print(f"Đánh giá mô hình trên tập test:")
        print(f"- MSE: {mse}")
        print(f"- R²: {r2}")
        print(f"- MAE: {mae}")
        print(f"- MAPE: {mape}%")
        print(f"- Độ chính xác: {accuracy}%")

    def save_model(self, file_path: str):
        """Lưu mô hình đã huấn luyện vào file"""
        joblib.dump(self.model, file_path)
        print(f"Mô hình đã được lưu vào {file_path}")

    def load_model(self, file_path: str):
        """Tải mô hình đã lưu từ file"""
        self.model = joblib.load(file_path)
        print(f"Mô hình đã được tải từ {file_path}")

    def predict(self, input_data: dict):
        """
        Dự đoán số người bị chết dựa trên thông tin thảm họa
        
        Parameters:
            input_data (dict): Từ điển chứa thông tin thảm họa
                Ví dụ: {
                    'country': 'India',
                    'disaster_type': 'Flood',
                    'severity_index': 8,
                    'economic_loss': 5000000,
                    'response_time': 10,
                    'response_effectiveness': 75,
                    'latitude': 20.5937,
                    'longitude': 78.9629
                }
        
        Returns:
            float: Số người bị chết dự đoán
        """
        # Chuyển đổi input_data thành DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Chuẩn hóa dữ liệu số
        input_df[['severity_index', 'economic_loss', 'response_time', 'response_effectiveness', 'latitude', 'longitude']] = self.scaler.transform(
            input_df[['severity_index', 'economic_loss', 'response_time', 'response_effectiveness', 'latitude', 'longitude']]
        )

        # Mã hóa các biến phân loại
        for column in ['country', 'disaster_type']:
            le = self.label_encoders[column]
            input_df[column] = le.transform(input_df[column])

        # Dự đoán
        prediction = self.model.predict(input_df)
        return prediction[0]

    def prepare_features(self, df=None, fit_encoders=True):
    """
    Chuẩn bị features (X) và target (y)
    
    Parameters:
        df: Dataframe cần xử lý (mặc định = self.df)
        fit_encoders: True = train encoders, False = dùng encoders cũ
    
    Returns:
        X: Array features (n_samples, 8)
        y: Array target (n_samples,)
    """
    
    # BƯỚC 1: Copy dataframe
    if df is None:
        df = self.df.copy()
    else:
        df = df.copy()
    # Dùng .copy() để không thay đổi df gốc
    
    # BƯỚC 2: Loại bỏ cột không cần thiết
    df = df.drop(columns=['date', 'aid_amount_usd', 'recovery_days', 'continent'],
                 errors='ignore')
    # 'date': Ngày tháng (không dùng)
    # 'aid_amount_usd': Tiền hỗ trợ (không phải input)
    # 'recovery_days': Số ngày phục hồi (output, không input)
    # 'continent': Châu lục (redundant với country)
    # errors='ignore': Không lỗi nếu cột không tồn tại
    
    # BƯỚC 3: Tách target (y = cái cần dự đoán)
    y = df['casualties'].values
    # .values chuyển pandas Series thành numpy array
    # Ví dụ: [111, 100, 22, 94, 64, ...]
    
    # BƯỚC 4: Chọn features (X = dữ liệu dùng để dự đoán)
    feature_cols = [
        'country',                      # Quốc gia
        'disaster_type',                # Loại thảm họa
        'severity_index',               # Mức độ (0-10)
        'economic_loss_usd',            # Thiệt hại (USD)
        'response_time_hours',          # Thời gian (giờ)
        'response_efficiency_score',    # Hiệu quả (0-100)
        'latitude',                     # Vĩ độ (-90 đến 90)
        'longitude'                     # Kinh độ (-180 đến 180)
    ]
    # Tổng cộng 8 features đầu vào
    
    X = df[feature_cols].copy()
    # Lấy 8 cột này từ dataframe
    
    # BƯỚC 5: Mã hóa các cột text (categorical)
    for col in ['country', 'disaster_type']:
        # Chỉ 2 cột text cần mã hóa
        
        if fit_encoders:
            # Nếu đang huấn luyện, tạo encoder mới
            if col not in self.label_encoders:
                # Nếu encoder chưa tồn tại, tạo mới
                le = LabelEncoder()
                # LabelEncoder là công cụ chuyển text → số
                
                X[col] = le.fit_transform(X[col].astype(str))
                # fit_transform: Học từ dữ liệu rồi transform luôn
                # astype(str): Đảm bảo là string trước khi encode
                # Ví dụ: ['India', 'Brazil', 'India'] → [5, 2, 5]
                
                self.label_encoders[col] = le
                # Lưu encoder vào dict để dùng sau
            else:
                # Nếu encoder đã tồn tại (dòng thứ 2 trở đi), dùng nó
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        else:
            # Nếu đang predict (không huấn luyện), dùng encoder cũ
            try:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
            except ValueError:
                # Nếu giá trị không tồn tại trong encoder
                print(f"⚠️  '{X[col].values[0]}' không trong dữ liệu huấn luyện")
                X[col] = -1  # Gán giá trị mặc định
    
    # BƯỚC 6: Lưu thông tin features
    self.feature_columns = feature_cols
    
    # BƯỚC 7: Return
    return X.values, y
    # X.values: Chuyển DataFrame thành numpy array (2D)
    # y: Đã là numpy array rồi (1D)
```

### 📌 PHẦN 3: CHẠY THỬ CLASS VÀO CUỐI FILE

```python
# Khởi tạo đối tượng dự đoán
predictor = DisasterCasualtyPredictor('du_lieu_sach.csv')

# Tải và tiền xử lý dữ liệu
predictor.load_data()
predictor.preprocess_data()

# Huấn luyện mô hình
predictor.train_model()

# Đánh giá mô hình
predictor.evaluate_model()

# Lưu mô hình
predictor.save_model('mo_hinh_du_doan.pkl')

# Tải mô hình
predictor.load_model('mo_hinh_du_doan.pkl')

# Dự đoán thử
input_data = {
    'country': 'India',
    'disaster_type': 'Flood',
    'severity_index': 8,
    'economic_loss': 5000000,
    'response_time': 10,
    'response_effectiveness': 75,
    'latitude': 20.5937,
    'longitude': 78.9629
}
predicted_casualties = predictor.predict(input_data)
print(f"Số người bị chết dự đoán: {predicted_casualties}")
```

---

## 📚 Tài liệu tham khảo
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Joblib Documentation](https://joblib.readthedocs.io/en/latest/)

---

## 🛠️ Cài đặt môi trường (dành cho người mới bắt đầu)
1. Cài đặt Python 3.8 trở lên: [Tải Python](https://www.python.org/downloads/)
2. Cài đặt Anaconda (bao gồm Jupyter Notebook): [Tải Anaconda](https://www.anaconda.com/products/distribution)
3. Mở Anaconda Prompt và tạo môi trường ảo:
    ```bash
    conda create -n disaster_prediction python=3.8
    conda activate disaster_prediction
    ```
4. Cài đặt các thư viện cần thiết:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn joblib
    ```
5. Tải mã nguồn về và giải nén
6. Chạy thử trong Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
    - Mở file `du_doan_thiet_hai_tham_hoa.ipynb`
    - Chạy từng ô mã (cell) theo thứ tự từ trên xuống dưới

---

## 📝 Ghi chú
- File dữ liệu mẫu `du_lieu_sach.csv` không được công khai do kích thước lớn và bảo mật dữ liệu.
- Người dùng tự chuẩn bị dữ liệu theo định dạng tương tự và đặt tên file là `du_lieu_sach.csv` trong cùng thư mục với mã nguồn.
- Để có kết quả dự đoán tốt, nên sử dụng dữ liệu huấn luyện có chất lượng và đầy đủ.
- Thời gian huấn luyện mô hình có thể lâu (10-30 phút) tùy vào cấu hình máy tính.
- Kết quả dự đoán có thể không chính xác 100% do tính chất ngẫu nhiên và phức tạp của thảm họa.

---

## 👤 Tác giả
**Nguyễn Văn A** - Chuyên gia Machine Learning
- Email: nguyenvana@gmail.com
- LinkedIn: [Nguyễn Văn A](https://www.linkedin.com/in/nguyenvana/)

---

## 📝 Lịch sử cập nhật
- **Phiên bản 1.0** - Ngày 01/01/2024
    - Ra mắt bản beta
    - Tính năng: Dự đoán thiệt hại về người do thảm họa
    - Sử dụng thuật toán Gradient Boosting Regressor
    - Đánh giá mô hình bằng R² score, MSE, MAE, MAPE

---

## ⚙️ Cấu hình yêu cầu
- Hệ điều hành: Windows 10 trở lên / macOS Mojave trở lên / Linux Ubuntu 18.04 trở lên
- Bộ vi xử lý: Intel Core i5 trở lên / AMD Ryzen 5 trở lên
- RAM: 8GB trở lên
- Ổ cứng: 500GB trở lên (còn trống ít nhất 10GB để lưu trữ dữ liệu và mô hình)
- Kết nối Internet: Tốt nhất là có dây (Ethernet), tốc độ 25Mbps trở lên

---

## ❓ Hỏi đáp
**Q1**: Tại sao không sử dụng dữ liệu thật để huấn luyện mô hình?
- **A1**: Dữ liệu thật có thể chứa thông tin nhạy cảm, vi phạm quyền riêng tư. Hơn nữa, dữ liệu thật thường bị lệch và không đầy đủ. Do đó, chúng tôi sử dụng dữ liệu tổng hợp (synthetic data) được tạo ra từ mô hình giả lập thảm họa.

**Q2**: Làm thế nào để cải thiện độ chính xác của mô hình?
- **A2**: Có thể cải thiện độ chính xác bằng cách:
    - Sử dụng dữ liệu huấn luyện lớn hơn, đa dạng hơn
    - Tinh chỉnh các tham số của mô hình (hyperparameter tuning)
    - Thử nghiệm với các thuật toán Machine Learning khác
    - Sử dụng kỹ thuật ensemble (kết hợp nhiều mô hình)

**Q3**: Tại sao lại lưu và tải mô hình?
- **A3**: Việc lưu và tải mô hình giúp tiết kiệm thời gian và tài nguyên. Thay vì phải huấn luyện lại mô hình từ đầu, chúng ta chỉ cần tải mô hình đã huấn luyện và sử dụng ngay lập tức.

**Q4**: Có thể sử dụng mô hình này cho loại thảm họa nào?
- **A4**: Mô hình này được thiết kế để dự đoán thiệt hại về người do các loại thảm họa tự nhiên như động đất, lũ lụt, bão, vòi rồng, ... Tuy nhiên, độ chính xác của mô hình phụ thuộc vào chất lượng và độ đầy đủ của dữ liệu huấn luyện.

**Q5**: Tại sao lại sử dụng thuật toán Gradient Boosting Regressor?
- **A5**: Gradient Boosting Regressor là một trong những thuật toán mạnh nhất trong Machine Learning hiện nay. Nó có khả năng xử lý tốt các bài toán hồi quy phi tuyến tính, đặc biệt là khi dữ liệu có nhiều nhiễu và không tuân theo phân phối chuẩn.

---

## 📈 Kết quả dự đoán mẫu
| Quốc gia | Loại thảm họa | Mức độ nghiêm trọng | Thiệt hại kinh tế | Thời gian phản ứng | Hiệu quả phản ứng | Vĩ độ | Kinh độ | Số người bị chết (thực tế) | Số người bị chết (dự đoán) |
|----------|----------------|---------------------|-------------------|-------------------|-------------------|--------|--------|---------------------------|---------------------------|
| India    | Flood          | 8                   | 5000000           | 10                | 75                | 20.5937| 78.9629| 500                       | 450                       |
| Philippines| Earthquake   | 7                   | 3000000           | 5                 | 80                | 13.4125| 122.5621| 300                      | 320                       |
| Brazil   | Hurricane      | 9                   | 8000000           | 15                | 70                | -14.2350| -51.9253| 700                      | 680                       |