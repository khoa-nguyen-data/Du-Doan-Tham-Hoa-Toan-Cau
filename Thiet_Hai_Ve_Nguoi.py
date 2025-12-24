"""
DỰ ĐOÁN THIỆT HẠI VỀ NGƯỜI DO THẢM HỌA GÂY RA
==================================================
Input: Quốc gia, thảm họa, mức độ nghiêm trọng, thiệt hại kinh tế, 
       thời gian phản ứng, hiệu quả phản ứng, kinh độ, vĩ độ
Output: Số người bị chết do thảm họa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
import joblib
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DisasterCasualtyPredictor:
    """Mô hình dự đoán số người bị chết do thảm họa - Gradient Boosting"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.feature_importance_df = None
        
    def load_data(self):
        """Load và khám phá dữ liệu"""
        print("="*80)
        print("📊 BƯỚC 1: TẢI DỮ LIỆU")
        print("="*80)
        
        self.df = pd.read_csv(self.csv_path)
        print(f"✅ Tải thành công {len(self.df):,} bản ghi")
        print(f"📋 Các cột: {list(self.df.columns)}")
        
        # Làm sạch dữ liệu
        self.df = self.df[self.df['severity_index'] > 0].copy()
        self.df = self.df[self.df['casualties'] >= 0].copy()
        self.df = self.df[self.df['economic_loss_usd'] >= 0].copy()
        
        print(f"✅ Sau làm sạch: {len(self.df):,} bản ghi")
        
        # Thống kê
        print(f"\n📈 THỐNG KÊ CASUALTIES:")
        print(f"  • Min: {self.df['casualties'].min():.0f} người")
        print(f"  • Max: {self.df['casualties'].max():.0f} người")
        print(f"  • Mean: {self.df['casualties'].mean():.0f} người")
        print(f"  • Median: {self.df['casualties'].median():.0f} người")
        print(f"  • Std: {self.df['casualties'].std():.0f}")
        
        # Thống kê theo loại thảm họa
        print(f"\n📊 CASUALTIES THEO LOẠI THẢM HỌA:")
        disaster_impact = self.df.groupby('disaster_type').agg({
            'casualties': ['count', 'mean', 'sum', 'min', 'max']
        }).round(0)
        print(disaster_impact)
        
        return self.df
    
    def prepare_features(self, df=None, fit_encoders=True):
        """Chuẩn bị features với feature engineering"""
        if df is None:
            df = self.df.copy()
        else:
            df = df.copy()
        
        # Drop non-feature columns
        df = df.drop(columns=['date', 'aid_amount_usd', 'recovery_days', 'continent'], 
                     errors='ignore')
        
        # Target variable
        y = df['casualties'].values
        
        # Feature columns
        feature_cols = ['country', 'disaster_type', 'severity_index', 
                       'economic_loss_usd', 'response_time_hours', 
                       'response_efficiency_score', 'latitude', 'longitude']
        
        X = df[feature_cols].copy()
        
        # Encode categorical variables
        for col in ['country', 'disaster_type']:
            if fit_encoders:
                if col not in self.label_encoders:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
            else:
                try:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
                except ValueError as e:
                    print(f"⚠️  {col} không tìm thấy trong dữ liệu huấn luyện")
                    X[col] = -1
        
        self.feature_columns = feature_cols
        return X.values, y
    
    def train(self, test_size=0.2, random_state=42):
        """Huấn luyện mô hình"""
        print("\n" + "="*80)
        print("🔧 BƯỚC 2: CHUẨN BỊ VÀ HUẤN LUYỆN DỮ LIỆU")
        print("="*80)
        
        # Prepare features
        X, y = self.prepare_features(fit_encoders=True)
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\n📊 CHIA DỮ LIỆU:")
        print(f"  • Train set: {len(self.X_train):,} mẫu ({(1-test_size)*100:.0f}%)")
        print(f"  • Test set: {len(self.X_test):,} mẫu ({test_size*100:.0f}%)")
        
        # Scaling
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train Gradient Boosting
        print(f"\n🚀 HUẤN LUYỆN MÔ HÌNH GRADIENT BOOSTING...")
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=random_state,
            verbose=0
        )
        
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate
        return self._evaluate_model()
    
    def _evaluate_model(self):
        """Đánh giá mô hình"""
        print("\n" + "="*80)
        print("📈 BƯỚC 3: ĐÁNH GIÁ MÔ HÌNH")
        print("="*80)
        
        # Predictions
        y_pred_train = self.model.predict(self.X_train_scaled)
        y_pred_test = self.model.predict(self.X_test_scaled)
        
        # Metrics
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        train_mape = mean_absolute_percentage_error(self.y_train, y_pred_train)
        test_mape = mean_absolute_percentage_error(self.y_test, y_pred_test)
        
        print(f"\n📊 KẾT QUẢ ĐÁNH GIÁ:")
        print(f"\n  {'Metric':<30} {'Train':<20} {'Test':<20}")
        print(f"  {'-'*70}")
        print(f"  {'R² Score':<30} {train_r2:<20.4f} {test_r2:<20.4f}")
        print(f"  {'RMSE (người)':<30} {train_rmse:<20.2f} {test_rmse:<20.2f}")
        print(f"  {'MAE (người)':<30} {train_mae:<20.2f} {test_mae:<20.2f}")
        print(f"  {'MAPE (%)':<30} {train_mape*100:<20.2f} {test_mape*100:<20.2f}")
        
        # Feature importance
        self._print_feature_importance()
        
        return {
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_rmse': train_rmse, 'test_rmse': test_rmse,
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_mape': train_mape, 'test_mape': test_mape
        }
    
    def _print_feature_importance(self):
        """In tầm quan trọng feature"""
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n⭐ TẦM QUAN TRỌNG CÁC FEATURES:")
        print(f"  {'Feature':<35} {'Importance':<15}")
        print(f"  {'-'*50}")
        for _, row in importance_df.iterrows():
            print(f"  {row['feature']:<35} {row['importance']:.4f}")
        
        self.feature_importance_df = importance_df
    
    def predict(self, country: str, disaster_type: str, severity_index: float,
                economic_loss_usd: float, response_time_hours: float,
                response_efficiency_score: float, latitude: float, 
                longitude: float) -> float:
        """Dự đoán cho một bản ghi"""
        if self.model is None:
            raise ValueError("❌ Mô hình chưa được huấn luyện!")
        
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
        
        # Encode
        for col in ['country', 'disaster_type']:
            try:
                input_data[col] = self.label_encoders[col].transform(
                    input_data[col].astype(str)
                )
            except ValueError:
                print(f"⚠️  '{input_data[col].values[0]}' không trong dữ liệu huấn luyện")
                input_data[col] = -1
        
        # Scale & predict
        input_scaled = self.scaler.transform(input_data)
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
        print(f"✅ Mô hình đã lưu: {filepath}")
    
    def load_model(self, filepath: str):
        """Load mô hình"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_columns = data['feature_columns']
        print(f"✅ Load mô hình: {filepath}")


def test_model(csv_path="du_lieu_sach.csv"):
    """Test mô hình với các trường hợp mẫu"""
    
    predictor = DisasterCasualtyPredictor(csv_path)
    predictor.load_data()
    predictor.train()
    predictor.save_model("disaster_casualty_model.pkl")
    
    print("\n" + "="*80)
    print("🔮 BƯỚC 4: TEST DỰ ĐOÁN")
    print("="*80)
    
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
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        desc = case.pop('description')
        pred = predictor.predict(**case)
        
        print(f"\n{'='*80}")
        print(f"📍 Trường hợp {i}: {desc}")
        print(f"{'='*80}")
        print(f"  Input:")
        print(f"    🌍 Quốc gia: {case['country']}")
        print(f"    ⚠️  Thảm họa: {case['disaster_type']}")
        print(f"    📊 Mức độ: {case['severity_index']}/10")
        print(f"    💰 Thiệt hại: ${case['economic_loss_usd']:,.0f}")
        print(f"    ⏱️  Thời gian phản ứng: {case['response_time_hours']} giờ")
        print(f"    ⭐ Hiệu quả: {case['response_efficiency_score']}/100")
        print(f"    📍 Vị trí: ({case['latitude']}, {case['longitude']})")
        print(f"  Output:")
        print(f"    👥 DỰ ĐOÁN: {pred:.0f} người bị chết")


def main():
    """Main menu"""
    print("\n" + "="*80)
    print("🌍 HỆ THỐNG DỰ ĐOÁN THIỆT HẠI NGƯỜI DO THẢM HỌA")
    print("="*80)
    print("\n1️⃣  Test mô hình với các trường hợp mẫu")
    print("2️⃣  Dự đoán tương tác")
    print("3️⃣  Thoát")
    
    choice = input("\nChọn (1/2/3): ").strip()
    
    if choice == '1':
        test_model()
    elif choice == '2':
        predictor = DisasterCasualtyPredictor("du_lieu_sach.csv")
        predictor.load_data()
        predictor.train()
        
        countries = sorted(predictor.df['country'].unique())
        disasters = sorted(predictor.df['disaster_type'].unique())
        
        print(f"\n📍 Quốc gia: {', '.join(countries)}")
        print(f"⚠️  Thảm họa: {', '.join(disasters)}")
        
        while True:
            print("\n" + "-"*80)
            try:
                c = input("Quốc gia (hoặc 'q' thoát): ").strip()
                if c.lower() == 'q': break
                d = input("Thảm họa: ").strip()
                s = float(input("Mức độ (0-10): "))
                e = float(input("Thiệt hại (USD): "))
                r = float(input("Thời gian phản ứng (giờ): "))
                f = float(input("Hiệu quả (0-100): "))
                la = float(input("Vĩ độ: "))
                lo = float(input("Kinh độ: "))
                
                pred = predictor.predict(c, d, s, e, r, f, la, lo)
                print(f"\n✅ DỰ ĐOÁN: {pred:.0f} người bị chết")
            except Exception as ex:
                print(f"❌ Lỗi: {ex}")


if __name__ == "__main__":
    main()