"""
Dự đoán thiệt hại về người do thảm họa gây ra
Input: Quốc gia, thảm họa, mức độ nghiêm trọng, thiệt hại kinh tế, 
       thời gian phản ứng, hiệu quả phản ứng, kinh độ, vĩ độ
Output: Số người bị chết do thảm họa
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
        print("📊 Đang tải dữ liệu...")
        self.df = pd.read_csv(self.csv_path)
        
        print(f"✅ Tải thành công {len(self.df)} bản ghi")
        print(f"Các cột: {list(self.df.columns)}")
        print(f"\nThống kê dữ liệu:")
        print(self.df.describe())
        
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
        
        # Chuẩn hóa features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Huấn luyện mô hình Gradient Boosting
        print("🚀 Đang huấn luyện mô hình Gradient Boosting...")
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
        
        print(f"\n📈 Kết quả đánh giá:")
        print(f"Train R² Score: {train_r2:.4f}")
        print(f"Test R² Score: {test_r2:.4f}")
        print(f"Train RMSE: {train_rmse:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        print(f"Train MAE: {train_mae:.2f}")
        print(f"Test MAE: {test_mae:.2f}")
        
        # Feature importance
        print(f"\n⭐ Tầm quan trọng của features:")
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.to_string(index=False))
        
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
                # Nếu giá trị không trong training set, dùng giá trị unknown
                print(f"⚠️  {col} '{input_data[col].values[0]}' không trong dữ liệu huấn luyện")
                input_data[col] = 0
        
        # Chuẩn hóa
        input_scaled = self.scaler.transform(input_data)
        
        # Dự đoán
        prediction = self.model.predict(input_scaled)[0]
        
        return max(0, prediction)  # Không cho số âm
    
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Dự đoán batch
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame chứa các input cần dự đoán
            
        Returns
        -------
        predictions : np.ndarray
            Mảng dự đoán
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được huấn luyện.")
        
        X, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return np.maximum(predictions, 0)
    
    def save_model(self, filepath: str):
        """Lưu mô hình"""
        joblib.dump(self.model, filepath)
        print(f"✅ Mô hình đã lưu tại {filepath}")
    
    def load_model(self, filepath: str):
        """Load mô hình từ file"""
        self.model = joblib.load(filepath)
        print(f"✅ Mô hình đã load từ {filepath}")


def main():
    """Main function - Demo sử dụng"""
    
    # Khởi tạo
    csv_path = "du_lieu_sach.csv"
    predictor = DisasterCasualtyPredictor(csv_path)
    
    # Load dữ liệu
    predictor.load_data()
    
    # Huấn luyện mô hình
    predictor.train()
    
    # Lưu mô hình
    predictor.save_model("disaster_casualty_model.pkl")
    
    # Test dự đoán với một số trường hợp
    print("\n" + "="*60)
    print("🔮 TEST DỰ ĐOÁN")
    print("="*60)
    
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
            'disaster_type': 'Typhoon',
            'severity_index': 7.5,
            'economic_loss_usd': 3000000,
            'response_time_hours': 8,
            'response_efficiency_score': 90,
            'latitude': 14.5,
            'longitude': 121.0,
            'description': 'Bão lớn tại Philippines'
        },
        {
            'country': 'Brazil',
            'disaster_type': 'Flood',
            'severity_index': 6.5,
            'economic_loss_usd': 2000000,
            'response_time_hours': 16,
            'response_efficiency_score': 75,
            'latitude': -23.55,
            'longitude': -46.6,
            'description': 'Lũ lụt ở Brazil'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        desc = case.pop('description')
        prediction = predictor.predict(**case)
        
        print(f"\n📍 Trường hợp {i}: {desc}")
        print(f"   🌍 Quốc gia: {case['country']}")
        print(f"   ⚠️  Thảm họa: {case['disaster_type']}")
        print(f"   📊 Mức độ: {case['severity_index']}")
        print(f"   💰 Thiệt hại: ${case['economic_loss_usd']:,.0f}")
        print(f"   ⏱️  Thời gian phản ứng: {case['response_time_hours']} giờ")
        print(f"   ⭐ Hiệu quả: {case['response_efficiency_score']}/100")
        print(f"   📍 Vị trí: ({case['latitude']}, {case['longitude']})")
        print(f"   👥 DỰ ĐOÁN SỐ NGƯỜI BỊ CHẾT: {prediction:.0f} người")


if __name__ == "__main__":
    main()