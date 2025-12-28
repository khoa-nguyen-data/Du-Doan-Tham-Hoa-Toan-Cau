"""
Dự đoán thiệt hại về người do thảm họa gây ra
Input: Quốc gia, thảm họa, mức độ nghiêm trọng, thiệt hại kinh tế, 
       thời gian phản ứng, hiệu quả phản ứng, kinh độ, vĩ độ
Output: Số người bị chết do thảm họa
"""

import os
import pandas as pd
import numpy as np

# Lấy đường dẫn thư mục chứa file Python này
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "global_disaster_response_2018_2024.csv")
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings

warnings.filterwarnings('ignore')


class DisasterCasualtyPredictor:
    """Mô hình dự đoán số người bị chết do thảm họa"""
    
    def __init__(self, csv_path: str):
        """
        Khởi tạo mô hình
        
        Parameters
        ----------
        csv_path : str
            Đường dẫn file CSV chứa dữ liệu huấn luyện
        """
        self.csv_path = csv_path
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_data(self):
        """Load và xử lý dữ liệu từ CSV"""
        print("Đang tải dữ liệu...")
        self.df = pd.read_csv(self.csv_path)
        
        print(f" Tải thành công {len(self.df)} bản ghi")
        print(f"Các cột: {list(self.df.columns)}")
        print(f"\nThống kê dữ liệu Casualties:")
        print(f"  Min: {self.df['casualties'].min():.0f}")
        print(f"  Max: {self.df['casualties'].max():.0f}")
        print(f"  Mean: {self.df['casualties'].mean():.0f}")
        print(f"  Median: {self.df['casualties'].median():.0f}")
        
        return self.df
    
    def prepare_features(self, df=None):
        """
        Chuẩn bị features cho mô hình
        
        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame để xử lý. Nếu None, dùng self.df
            
        Returns
        -------
        X : np.ndarray
            Features đã xử lý
        y : np.ndarray
            Target (casualties)
        """
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()
        
        # Xóa các cột không cần thiết
        df = df.drop(columns=['date', 'aid_amount_usd', 'recovery_days', 'continent'], 
                     errors='ignore')
        
        # Tách target
        y = df['casualties'].values
        
        # Features: quốc gia, loại thảm họa, mức độ, thiệt hại kinh tế, 
        #          thời gian phản ứng, hiệu quả phản ứng, kinh độ, vĩ độ
        feature_cols = ['country', 'disaster_type', 'severity_index', 'economic_loss_usd',
                       'response_time_hours', 'response_efficiency_score', 
                       'latitude', 'longitude']
        
        X = df[feature_cols].copy()
        
        # Encode các cột categorical
        for col in ['country', 'disaster_type']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
        
        self.feature_columns = feature_cols
        
        return X.values, y
    
    def train(self, test_size=0.2, random_state=42):
        """
        Huấn luyện mô hình
        
        Parameters
        ----------
        test_size : float
            Tỷ lệ chia test set
        random_state : int
            Random seed
        """
        print("\n🔧 Đang chuẩn bị dữ liệu...")
        X, y = self.prepare_features()
        
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Chuẩn hóa features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Huấn luyện mô hình Gradient Boosting
        print("Đang huấn luyện mô hình Gradient Boosting...")
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            verbose=0
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Đánh giá mô hình
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"\nKết quả đánh giá:")
        print(f"  Train R² Score: {train_r2:.4f}")
        print(f"  Test R² Score: {test_r2:.4f}")
        print(f"  Train RMSE: {train_rmse:.2f}")
        print(f"  Test RMSE: {test_rmse:.2f}")
        print(f"  Train MAE: {train_mae:.2f}")
        print(f"  Test MAE: {test_mae:.2f}")
        
        # Feature importance
        print(f"\nTầm quan trọng của features:")
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
    
    def predict(self, country: str, disaster_type: str, severity_index: float,
                economic_loss_usd: float, response_time_hours: float,
                response_efficiency_score: float, latitude: float, 
                longitude: float) -> float:
        """
        Dự đoán số người bị chết
        
        Parameters
        ----------
        country : str
            Quốc gia
        disaster_type : str
            Loại thảm họa
        severity_index : float
            Mức độ nghiêm trọng (0-10)
        economic_loss_usd : float
            Thiệt hại kinh tế (USD)
        response_time_hours : float
            Thời gian phản ứng (giờ)
        response_efficiency_score : float
            Điểm hiệu quả phản ứng (0-100)
        latitude : float
            Vĩ độ
        longitude : float
            Kinh độ
            
        Returns
        -------
        float
            Dự đoán số người bị chết
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được huấn luyện. Gọi train() trước.")
        
        # Chuẩn bị input
        input_data = pd.DataFrame({
            'country': [country],
            'disaster_type': [disaster_type],
            'severity_index': [severity_index],
            'economic_loss_usd': [economic_loss_usd],
            'response_time_hours': [response_time_hours],
            'response_efficiency_score': [response_efficiency_score],
            'latitude': [latitude],
            'longitude': [longitude]
        })
        
        # Encode categorical features
        for col in ['country', 'disaster_type']:
            try:
                input_data[col] = self.label_encoders[col].transform(
                    input_data[col].astype(str)
                )
            except ValueError:
                print(f"  {col} '{input_data[col].values[0]}' không trong dữ liệu huấn luyện")
                input_data[col] = 0
        
        # Chuẩn hóa
        input_scaled = self.scaler.transform(input_data)
        
        # Dự đoán
        prediction = self.model.predict(input_scaled)[0]
        
        return max(0, prediction)
    
    def save_model(self, filepath: str):
        """Lưu mô hình"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }, filepath)
        print(f"Mô hình đã lưu tại {filepath}")
    
    def load_model(self, filepath: str):
        """Load mô hình từ file"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_columns = data['feature_columns']
        print(f"Mô hình đã load từ {filepath}")


def interactive_prediction(csv_path=None):
    """Dự đoán tương tác từ input của người dùng"""
    if csv_path is None:
        csv_path = DEFAULT_CSV
    print("\n" + "="*70)
    print("CHẾ ĐỘ DỰ ĐOÁN TƯƠNG TÁC")
    print("="*70)
    
    predictor = DisasterCasualtyPredictor(csv_path)
    predictor.load_data()
    predictor.train()
    
    # Lấy danh sách quốc gia và thảm họa từ dữ liệu
    countries = sorted(predictor.df['country'].unique())
    disaster_types = sorted(predictor.df['disaster_type'].unique())
    
    print(f"\nCác quốc gia trong dữ liệu:")
    for i, country in enumerate(countries, 1):
        print(f"   {i}. {country}")
    
    print(f"\nCác loại thảm họa:")
    for i, dtype in enumerate(disaster_types, 1):
        print(f"   {i}. {dtype}")
    
    while True:
        print("\n" + "-"*70)
        
        try:
            country = input("Nhập quốc gia (hoặc 'quit' để thoát): ").strip()
            if country.lower() == 'quit':
                break
            
            disaster_type = input("Nhập loại thảm họa: ").strip()
            severity_index = float(input("Nhập mức độ nghiêm trọng (0-10): "))
            economic_loss = float(input("Nhập thiệt hại kinh tế (USD): "))
            response_time = float(input("Nhập thời gian phản ứng (giờ): "))
            efficiency = float(input("Nhập hiệu quả phản ứng (0-100): "))
            latitude = float(input("Nhập vĩ độ: "))
            longitude = float(input("Nhập kinh độ: "))
            
            prediction = predictor.predict(
                country=country,
                disaster_type=disaster_type,
                severity_index=severity_index,
                economic_loss_usd=economic_loss,
                response_time_hours=response_time,
                response_efficiency_score=efficiency,
                latitude=latitude,
                longitude=longitude
            )
            
            print("\n" + "="*70)
            print(f"KẾT QUẢ DỰ ĐOÁN")
            print("="*70)
            print(f"  Quốc gia: {country}")
            print(f"  Thảm họa: {disaster_type}")
            print(f"  Mức độ: {severity_index}/10")
            print(f"  Thiệt hại kinh tế: ${economic_loss:,.0f}")
            print(f"  Thời gian phản ứng: {response_time} giờ")
            print(f"  Hiệu quả: {efficiency}/100")
            print(f"  Vị trí: ({latitude}, {longitude})")
            print("-"*70)
            print(f"DỰ ĐOÁN SỐ NGƯỜI BỊ CHẾT: {prediction:.0f} người")
            print("="*70)
            
        except ValueError:
            print(f"Lỗi: Vui lòng nhập dữ liệu hợp lệ!")
        except Exception as e:
            print(f"Lỗi: {str(e)}")


def test_predictions(csv_path=None):
    """Test dự đoán với một số trường hợp"""
    if csv_path is None:
        csv_path = DEFAULT_CSV
    
    predictor = DisasterCasualtyPredictor(csv_path)
    
    # Load dữ liệu
    predictor.load_data()
    
    # Huấn luyện mô hình
    predictor.train()
    
    # Lưu mô hình
    predictor.save_model("disaster_casualty_model.pkl")
    
    # Test dự đoán với một số trường hợp
    print("\n" + "="*70)
    print("TEST DỰ ĐOÁN VỚI CÁC TRƯỜNG HỢP MẪU")
    print("="*70)
    
    test_cases = [
        {
            'country': 'India',
            'disaster_type': 'Earthquake',
            'severity_index': 8.0,
            'economic_loss_usd': 5000000,
            'response_time_hours': 12,
            'response_efficiency_score': 85,
            'latitude': 28.7,
            'longitude': 77.2,
            'description': 'Động đất mạnh tại Ấn Độ'
        },
        {
            'country': 'Philippines',
            'disaster_type': 'Flood',
            'severity_index': 7.5,
            'economic_loss_usd': 3000000,
            'response_time_hours': 8,
            'response_efficiency_score': 90,
            'latitude': 14.5,
            'longitude': 121.0,
            'description': 'Lũ lụt ở Philippines'
        },
        {
            'country': 'Brazil',
            'disaster_type': 'Wildfire',
            'severity_index': 6.5,
            'economic_loss_usd': 2000000,
            'response_time_hours': 16,
            'response_efficiency_score': 75,
            'latitude': -23.55,
            'longitude': -46.6,
            'description': 'Cháy rừng ở Brazil'
        },
        {
            'country': 'Japan',
            'disaster_type': 'Earthquake',
            'severity_index': 9.0,
            'economic_loss_usd': 8000000,
            'response_time_hours': 2,
            'response_efficiency_score': 95,
            'latitude': 35.6762,
            'longitude': 139.6503,
            'description': 'Động đất cực mạnh tại Nhật Bản'
        },
        {
            'country': 'United States',
            'disaster_type': 'Hurricane',
            'severity_index': 8.5,
            'economic_loss_usd': 10000000,
            'response_time_hours': 6,
            'response_efficiency_score': 88,
            'latitude': 29.9511,
            'longitude': -90.2623,
            'description': 'Bão tấn công New Orleans'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        desc = case.pop('description')
        prediction = predictor.predict(**case)
        
        print(f"\nTrường hợp {i}: {desc}")
        print(f"  Quốc gia: {case['country']}")
        print(f"  Thảm họa: {case['disaster_type']}")
        print(f"  Mức độ: {case['severity_index']}/10")
        print(f"  Thiệt hại: ${case['economic_loss_usd']:,.0f}")
        print(f"  Thời gian phản ứng: {case['response_time_hours']} giờ")
        print(f"  Hiệu quả: {case['response_efficiency_score']}/100")
        print(f"  Vị trí: ({case['latitude']}, {case['longitude']})")
        print(f"  DỰ ĐOÁN SỐ NGƯỜI BỊ CHẾT: {prediction:.0f} người")


def main():
    """Main function"""
    
    print("\n" + "="*70)
    print("HỆ THỐNG DỰ ĐOÁN THIỆT HẠI NGƯỜI DO THẢM HỌA")
    print("="*70)
    print("\nChọn chế độ:")
    print("1. Test với các trường hợp mẫu")
    print("2. Dự đoán tương tác (nhập dữ liệu từ bàn phím)")
    print("3. Thoát")
    
    choice = input("\nNhập lựa chọn (1/2/3): ").strip()
    
    if choice == '1':
        test_predictions()
    elif choice == '2':
        interactive_prediction()
    else:
        print("Tạm biệt!")


if __name__ == "__main__":
    # Chạy test trực tiếp để kiểm tra
    test_predictions()