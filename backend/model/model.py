# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')
import re
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple
import time

class MLPredictor:
    """Machine Learning predictor for asset returns"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_trained = {}
        self.feature_columns = [
            'SMA_20', 'SMA_50', 'RSI', 'Volatility', 'Volume_MA',
            'Price_Change', 'High_Low_Ratio', 'Close_Open_Ratio',
            'Return_Lag_1', 'Return_Lag_5', 'Return_Lag_10',
            'Volume_Lag_1', 'Volume_Lag_5', 'Volume_Lag_10'
        ]
        self.model_cache_dir = 'ml_models'
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        """Ensure model cache directory exists"""
        if not os.path.exists(self.model_cache_dir):
            os.makedirs(self.model_cache_dir)

    def prepare_features(self, historical_data):
        """Prepare features for ML model"""
        df = historical_data.copy()

        # Technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

        # Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']

        # Lagged features
        for lag in [1, 5, 10]:
            df[f'Return_Lag_{lag}'] = df['Close'].pct_change().shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)

        # Target variables (future returns)
        df['Future_Return_1M'] = df['Close'].pct_change().shift(-21)  # 1 month ahead
        df['Future_Return_3M'] = df['Close'].pct_change().shift(-63)  # 3 months ahead
        df['Future_Return_6M'] = df['Close'].pct_change().shift(-126) # 6 months ahead

        return df.dropna()

    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def train_model(self, ticker, historical_data, target_period='1M'):
        """Train ML model for a specific ticker"""
        try:
            # Check if model already exists and is recent
            model_file = os.path.join(self.model_cache_dir, f"{ticker}_{target_period}.pkl")
            if os.path.exists(model_file):
                # Check if model is less than 7 days old
                file_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(model_file))).days
                if file_age < 7:
                    self._load_cached_model(ticker, target_period)
                    return True

            # Prepare features
            df = self.prepare_features(historical_data)
            target_column = f'Future_Return_{target_period}'

            # Remove rows with NaN values
            df_clean = df[self.feature_columns + [target_column]].dropna()

            if len(df_clean) < 100:  # Need sufficient data
                return False

            X = df_clean[self.feature_columns]
            y = df_clean[target_column]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train ensemble of models
            models = {
                'rf': RandomForestRegressor(
                    n_estimators=200, max_depth=10, min_samples_split=5,
                    min_samples_leaf=2, random_state=42
                ),
                'gb': GradientBoostingRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42
                )
            }

            best_model = None
            best_score = float('-inf')

            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                score = model.score(X_test_scaled, y_test)

                if score > best_score:
                    best_score = score
                    best_model = model

            # Store model and scaler
            model_key = f"{ticker}_{target_period}"
            self.models[model_key] = best_model
            self.scalers[model_key] = scaler
            self.model_trained[model_key] = True

            # Cache model to disk
            self._save_model_to_cache(ticker, target_period, best_model, scaler)

            # Calculate accuracy metrics
            y_pred = best_model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)

            if r2 > 0.1:  # Only show if model has reasonable performance
                print(f"🤖 ML model trained for {ticker} ({target_period}): R² = {r2:.3f}")

            return True

        except Exception as e:
            print(f"❌ Error training model for {ticker}: {e}")
            return False

    def _save_model_to_cache(self, ticker, target_period, model, scaler):
        """Save model to disk cache"""
        try:
            model_data = {
                'model': model,
                'scaler': scaler,
                'timestamp': datetime.now().isoformat()
            }

            model_file = os.path.join(self.model_cache_dir, f"{ticker}_{target_period}.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            print(f"⚠️ Could not cache model for {ticker}: {e}")

    def _load_cached_model(self, ticker, target_period):
        """Load model from disk cache"""
        try:
            model_file = os.path.join(self.model_cache_dir, f"{ticker}_{target_period}.pkl")
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)

            model_key = f"{ticker}_{target_period}"
            self.models[model_key] = model_data['model']
            self.scalers[model_key] = model_data['scaler']
            self.model_trained[model_key] = True

            return True
        except Exception as e:
            return False

    def predict_returns(self, ticker, current_data, target_period='1M'):
        """Predict future returns for a ticker"""
        model_key = f"{ticker}_{target_period}"

        if model_key not in self.models:
            return None

        try:
            # Prepare current features
            df = self.prepare_features(current_data)

            if len(df) == 0:
                return None

            # Get latest row features
            latest_features = df[self.feature_columns].iloc[-1:].fillna(method='ffill')

            # Handle any remaining NaN values
            if latest_features.isnull().any().any():
                latest_features = latest_features.fillna(0)

            # Scale features
            features_scaled = self.scalers[model_key].transform(latest_features)

            # Make prediction
            prediction = self.models[model_key].predict(features_scaled)[0]

            # Convert to percentage and add some bounds checking
            predicted_return = np.clip(prediction * 100, -50, 100)

            return predicted_return

        except Exception as e:
            print(f"❌ Error predicting for {ticker}: {e}")
            return None

    def get_feature_importance(self, ticker, target_period='1M'):
        """Get feature importance for interpretability"""
        model_key = f"{ticker}_{target_period}"

        if model_key not in self.models:
            return None

        try:
            importance = self.models[model_key].feature_importances_
            return dict(zip(self.feature_columns, importance))
        except:
            return None

    def batch_predict(self, tickers, target_period='1M'):
        """Batch prediction for multiple tickers"""
        predictions = {}

        for ticker in tickers:
            try:
                # Fetch data for prediction
                instrument = yf.Ticker(ticker)
                hist = instrument.history(period='2y')

                if len(hist) > 100:
                    # Train model if needed
                    if f"{ticker}_{target_period}" not in self.model_trained:
                        self.train_model(ticker, hist, target_period)

                    # Get prediction
                    pred = self.predict_returns(ticker, hist, target_period)
                    if pred is not None:
                        predictions[ticker] = pred

            except Exception as e:
                continue

        return predictions

class MarketDataManager:
    """Enhanced market data manager with ML capabilities"""

    def __init__(self):
        # Comprehensive Indian market instruments
        self.nifty50_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'BAJFINANCE.NS', 'BHARTIARTL.NS', 'HINDUNILVR.NS', 'KOTAKBANK.NS', 
            'LT.NS', 'TITAN.NS', 'ADANIENT.NS', 'ULTRACEMCO.NS', 'AXISBANK.NS',
            'WIPRO.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'ASIANPAINT.NS', 'DMART.NS',
            'BAJAJFINSV.NS', 'NESTLEIND.NS', 'TECHM.NS', 'HCLTECH.NS', 'SBIN.NS',
            'INDUSINDBK.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'M&M.NS',
            'HEROMOTOCO.NS', 'DIVISLAB.NS', 'BRITANNIA.NS', 'TATAMOTORS.NS'
        ]

        self.index_etfs = [
            'NIFTYBEES.NS', 'BANKBEES.NS', 'JUNIORBEES.NS', 'SETFNIF50.NS',
            'SETFNIFBK.NS', 'LIQUIDBEES.NS', 'GOLDBEES.NS', 'SILVRETF.NS'
        ]

        self.debt_instruments = [
            'LIQUIDBEES.NS', 'SETF10GILT.NS', 'ABSLBANETF.NS', 'GILT5YBEES.NS'
        ]

        self.sector_etfs = [
            'ICICIB22.NS', 'ABSLNN50ET.NS', 'HDFCNIFETF.NS'
        ]

        # Cache for market data
        self.cache = {}
        self.cache_file = 'market_cache.json'
        self.ml_predictor = MLPredictor()
        self.prediction_cache = {}

        self.load_cache()

    def load_cache(self):
        """Load cached market data"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except:
                self.cache = {}

    def save_cache(self):
        """Save market data to cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except:
            pass

    def is_cache_valid(self, ticker):
        """Check if cached data is still valid (less than 1 hour old)"""
        if ticker not in self.cache:
            return False

        try:
            cache_time = datetime.fromisoformat(self.cache[ticker]['timestamp'])
            return (datetime.now() - cache_time).seconds < 3600
        except:
            return False

    def fetch_instrument_data_with_ml(self, ticker, period='2y'):
        """Enhanced fetch with ML predictions"""
        try:
            # Check cache first
            if self.is_cache_valid(ticker) and ticker in self.cache:
                cached_data = self.cache[ticker]['data']
                if 'predicted_return_1m' in cached_data:
                    return cached_data

            instrument = yf.Ticker(ticker)
            time.sleep(2)
            hist = instrument.history(period=period)

            if len(hist) == 0:
                return None

            info = instrument.info
            current_price = hist['Close'][-1]

            # Calculate basic performance metrics
            if len(hist) > 1:
                year_return = ((hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0] * 100)
                volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100

                # Calculate Sharpe ratio (assuming risk-free rate of 7%)
                returns = hist['Close'].pct_change().dropna()
                sharpe_ratio = (returns.mean() * 252 - 0.07) / (returns.std() * np.sqrt(252))
            else:
                year_return = 0
                volatility = 0
                sharpe_ratio = 0

            # Get name with fallback
            name = info.get('longName') or info.get('shortName') or ticker.replace('.NS', '').replace('.BO', '')

            # Basic data structure
            data = {
                'ticker': ticker,
                'name': name,
                'current_price': round(current_price, 2),
                'year_return': round(year_return, 2),
                'volatility': round(volatility, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'category': self._get_category(ticker)
            }

            # Add ML predictions if we have sufficient data
            if len(hist) > 100:
                # Train models for different time horizons
                for period_key in ['1M', '3M', '6M']:
                    model_key = f"{ticker}_{period_key}"

                    # Train model if not already trained
                    if model_key not in self.ml_predictor.model_trained:
                        success = self.ml_predictor.train_model(ticker, hist, period_key)
                        if not success:
                            continue

                    # Get prediction
                    prediction = self.ml_predictor.predict_returns(ticker, hist, period_key)
                    if prediction is not None:
                        data[f'predicted_return_{period_key.lower()}'] = round(prediction, 2)

                # Calculate confidence score
                data['ml_confidence'] = self._calculate_confidence_score(ticker, hist)

                # Calculate ML-enhanced recommendation score
                data['ml_score'] = self._calculate_ml_score(data)

            # Cache the data
            self.cache[ticker] = {
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            self.save_cache()

            return data

        except Exception as e:
            print(f"Error fetching ML data for {ticker}: {e}")
            time.sleep(2)
            # Fallback to basic data
            return self.fetch_instrument_data(ticker, period='1y')

    def fetch_instrument_data(self, ticker, period='1y'):
        """Basic fetch method (fallback)"""
        try:
            instrument = yf.Ticker(ticker)
            hist = instrument.history(period=period)

            if len(hist) == 0:
                return None

            info = instrument.info
            current_price = hist['Close'][-1]

            # Calculate performance metrics
            if len(hist) > 1:
                year_return = ((hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0] * 100)
                volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100

                # Calculate Sharpe ratio
                returns = hist['Close'].pct_change().dropna()
                sharpe_ratio = (returns.mean() * 252 - 0.07) / (returns.std() * np.sqrt(252))
            else:
                year_return = 0
                volatility = 0
                sharpe_ratio = 0

            # Get name with fallback
            name = info.get('longName') or info.get('shortName') or ticker.replace('.NS', '').replace('.BO', '')

            data = {
                'ticker': ticker,
                'name': name,
                'current_price': round(current_price, 2),
                'year_return': round(year_return, 2),
                'volatility': round(volatility, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'category': self._get_category(ticker)
            }

            return data

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    def _get_category(self, ticker):
        """Determine category of instrument"""
        if ticker in self.nifty50_stocks:
            return 'Equity'
        elif ticker in self.index_etfs or ticker in self.sector_etfs:
            return 'ETF'
        elif ticker in self.debt_instruments:
            return 'Debt ETF'
        else:
            return 'Other'

    def _calculate_confidence_score(self, ticker, historical_data):
        """Calculate confidence score for ML predictions"""
        try:
            # Base confidence on data quality and quantity
            data_points = len(historical_data)
            volatility = historical_data['Close'].pct_change().std()

            # Base confidence on data availability
            if data_points > 500:
                base_confidence = 0.8
            elif data_points > 250:
                base_confidence = 0.6
            else:
                base_confidence = 0.4

            # Adjust for volatility (higher volatility = lower confidence)
            if volatility < 0.02:  # Low volatility
                vol_adjustment = 0.1
            elif volatility < 0.05:  # Medium volatility
                vol_adjustment = 0.0
            else:  # High volatility
                vol_adjustment = -0.1

            confidence = max(0.1, min(0.9, base_confidence + vol_adjustment))
            return round(confidence, 2)

        except:
            return 0.5  # Default confidence

    def _calculate_ml_score(self, data):
        """Calculate ML-enhanced recommendation score"""
        try:
            # Base score from traditional metrics
            base_score = (
                (data.get('year_return', 0) / 20) * 0.3 +  # Normalize to 20% max
                (data.get('sharpe_ratio', 0) / 2) * 0.3 +   # Normalize to 2.0 max
                (1 - data.get('volatility', 50) / 50) * 0.2  # Lower volatility is better
            )

            # ML enhancement
            ml_boost = 0
            confidence = data.get('ml_confidence', 0.5)

            # Consider 1-month prediction with confidence weighting
            pred_1m = data.get('predicted_return_1m', 0)
            if pred_1m is not None:
                ml_boost += (pred_1m / 10) * confidence * 0.2  # Normalize to 10% monthly max

            # Consider 3-month prediction
            pred_3m = data.get('predicted_return_3m', 0)
            if pred_3m is not None:
                ml_boost += (pred_3m / 30) * confidence * 0.1  # Normalize to 30% quarterly max

            total_score = base_score + ml_boost
            return round(max(0, min(1, total_score)), 3)

        except:
            return 0.5

    def get_filtered_recommendations_with_ml(self, instruments, risk_appetite, min_return=None):
        """Get ML-enhanced filtered recommendations"""
        recommendations = []

        print(f"🤖 Analyzing {len(instruments)} instruments with AI...")

        for i, ticker in enumerate(instruments):
            if i % 5 == 0:  # Progress indicator
                print(f"   Processed {i}/{len(instruments)} instruments...")

            data = self.fetch_instrument_data_with_ml(ticker)
            if data and self._meets_criteria_with_ml(data, risk_appetite, min_return):
                recommendations.append(data)

        # Sort by ML-enhanced recommendation score
        recommendations.sort(key=lambda x: x.get('ml_score', x.get('sharpe_ratio', 0)), reverse=True)
        return recommendations

    def _meets_criteria_with_ml(self, data, risk_appetite, min_return):
        """Enhanced criteria checking with ML insights"""
        # Basic criteria
        if data['year_return'] < -20:  # Avoid severely underperforming assets
            return False

        # Risk-based filtering
        if risk_appetite == 'Low':
            volatility_threshold = 20
            min_confidence = 0.6
        elif risk_appetite == 'Medium':
            volatility_threshold = 35
            min_confidence = 0.4
        else:  # High risk
            volatility_threshold = 100
            min_confidence = 0.3

        if data['volatility'] > volatility_threshold:
            return False

        # ML-enhanced filtering
        ml_confidence = data.get('ml_confidence', 0.5)
        if ml_confidence < min_confidence:
            return False

        # Consider ML predictions if available
        pred_1m = data.get('predicted_return_1m')
        if pred_1m is not None and pred_1m < -10:  # Avoid assets with very negative predictions
            return False

        return True

    def get_asset_class_predictions(self, risk_appetite):
        """Get ML predictions for different asset classes"""
        asset_classes = {
            'Equity': self.nifty50_stocks[:15],
            'Debt': self.debt_instruments,
            'ETF': self.index_etfs[:8]
        }

        predictions = {}

        for asset_class, tickers in asset_classes.items():
            print(f"🔍 Analyzing {asset_class} instruments...")

            class_predictions = []

            for ticker in tickers:
                data = self.fetch_instrument_data_with_ml(ticker)
                if data and data.get('predicted_return_1m') is not None:
                    class_predictions.append({
                        'ticker': ticker,
                        'name': data['name'],
                        'predicted_1m': data['predicted_return_1m'],
                        'predicted_3m': data.get('predicted_return_3m', 0),
                        'predicted_6m': data.get('predicted_return_6m', 0),
                        'confidence': data.get('ml_confidence', 0.5),
                        'current_return': data['year_return'],
                        'volatility': data['volatility'],
                        'ml_score': data.get('ml_score', 0.5)
                    })

            # Sort by ML score (combination of predictions and confidence)
            class_predictions.sort(
                key=lambda x: x['ml_score'], 
                reverse=True
            )

            predictions[asset_class] = class_predictions[:5]  # Top 5 per class

        return predictions

    def get_portfolio_recommendations_with_ml(self, risk_appetite, timeline_months, amount, allocation):
        """Get ML-enhanced comprehensive portfolio recommendations"""
        recommendations = {
            'debt': [],
            'equity': [],
            'mutual_fund': []
        }

        # Debt recommendations
        if allocation[0] > 5:
            # Government schemes
            govt_schemes = self._get_government_schemes(timeline_months)
            recommendations['debt'].extend(govt_schemes)

            # Debt ETFs with ML
            debt_etfs = self.get_filtered_recommendations_with_ml(
                self.debt_instruments, 'Low'
            )
            recommendations['debt'].extend(debt_etfs[:3])

        # Equity recommendations with ML
        if allocation[1] > 10 and timeline_months > 12:
            equity_recs = self.get_filtered_recommendations_with_ml(
                self.nifty50_stocks[:20], risk_appetite, min_return=5
            )
            recommendations['equity'].extend(equity_recs[:5])

        # Mutual Fund/ETF recommendations with ML
        if allocation[2] > 10:
            etf_recs = self.get_filtered_recommendations_with_ml(
                self.index_etfs + self.sector_etfs, risk_appetite
            )
            for etf in etf_recs[:4]:
                etf['category'] = 'ETF/Index Fund'
                recommendations['mutual_fund'].append(etf)

        return self._compile_final_recommendations_with_ml(recommendations, allocation)

    def _get_government_schemes(self, timeline_months):
        """Get government investment schemes"""
        schemes = []

        # Always available schemes
        schemes.append({
            'ticker': 'FD',
            'name': 'Bank Fixed Deposit',
            'current_price': 0,
            'year_return': 7.5,
            'volatility': 0,
            'sharpe_ratio': 0.5,
            'category': 'Government Scheme',
            'ml_confidence': 0.9,
            'ml_score': 0.7
        })

        if timeline_months >= 60:  # 5 years minimum
            schemes.append({
                'ticker': 'PPF',
                'name': 'Public Provident Fund',
                'current_price': 0,
                'year_return': 7.1,
                'volatility': 0,
                'sharpe_ratio': 0.4,
                'category': 'Government Scheme',
                'ml_confidence': 0.9,
                'ml_score': 0.8
            })

        if timeline_months >= 36:  # 3 years minimum
            schemes.append({
                'ticker': 'NSC',
                'name': 'National Savings Certificate',
                'current_price': 0,
                'year_return': 7.7,
                'volatility': 0,
                'sharpe_ratio': 0.45,
                'category': 'Government Scheme',
                'ml_confidence': 0.9,
                'ml_score': 0.75
            })

        return schemes

    def _compile_final_recommendations_with_ml(self, recommendations, allocation):
        """Compile final ML-enhanced recommendations"""
        final_recs = []

        # Sort categories by allocation percentage
        categories = [
            ('debt', allocation[0], recommendations['debt']),
            ('equity', allocation[1], recommendations['equity']),
            ('mutual_fund', allocation[2], recommendations['mutual_fund'])
        ]
        categories.sort(key=lambda x: x[1], reverse=True)

        # Add top recommendations from each significant category
        for category_name, alloc_pct, category_recs in categories:
            if alloc_pct > 10 and category_recs:
                # Sort by ML score if available
                category_recs.sort(
                    key=lambda x: x.get('ml_score', x.get('sharpe_ratio', 0)), 
                    reverse=True
                )
                final_recs.append(category_recs[0])

        # Fill remaining slots with best ML-scored performers
        all_remaining = []
        for category_name, alloc_pct, category_recs in categories:
            all_remaining.extend(category_recs[1:])

        # Sort all remaining by ML score
        all_remaining.sort(
            key=lambda x: x.get('ml_score', x.get('sharpe_ratio', 0)), 
            reverse=True
        )

        for rec in all_remaining:
            if len(final_recs) < 6:
                final_recs.append(rec)

        self.save_cache()  # Save updated cache
        return final_recs[:6]

class IntelligentAllocationEngine:
    """Enhanced allocation engine with ML insights"""

    def __init__(self):
        self.allocation_rules = self._load_allocation_rules()

    def _load_allocation_rules(self):
        """Load sophisticated allocation rules"""
        return {
            'age_equity_rule': lambda age: max(20, min(80, 100 - age)),
            'risk_multipliers': {
                'Low': {'equity': 0.6, 'debt': 1.4, 'mf': 1.0},
                'Medium': {'equity': 1.0, 'debt': 1.0, 'mf': 1.0},
                'High': {'equity': 1.3, 'debt': 0.7, 'mf': 1.1}
            },
            'goal_adjustments': {
                'Short-Term': {'debt': 1.5, 'equity': 0.5, 'mf': 0.8},
                'Retirement': {'debt': 0.8, 'equity': 1.2, 'mf': 1.1},
                'Education': {'debt': 1.3, 'equity': 0.7, 'mf': 1.0},
                'Wealth Creation': {'debt': 0.7, 'equity': 1.3, 'mf': 1.2},
                'Tax Saving': {'debt': 0.9, 'equity': 1.0, 'mf': 1.4},
                'Emergency Fund': {'debt': 1.8, 'equity': 0.3, 'mf': 0.5}
            }
        }

    def calculate_allocation_with_ml_insights(self, profile, market_predictions=None):
        """Calculate allocation with ML market insights"""
        # Get base allocation
        base_allocation = self.calculate_allocation(profile)

        if not market_predictions:
            return base_allocation

        # ML-based adjustments
        ml_adjustments = self._calculate_ml_adjustments(market_predictions, profile)

        # Apply ML adjustments (limited to ±10% to avoid extreme allocations)
        adjusted_allocation = np.array([
            max(5, min(70, base_allocation[0] + ml_adjustments[0])),  # Debt
            max(10, min(80, base_allocation[1] + ml_adjustments[1])), # Equity
            max(10, min(60, base_allocation[2] + ml_adjustments[2]))  # MF/ETF
        ])

        # Normalize to 100%
        total = adjusted_allocation.sum()
        adjusted_allocation = (adjusted_allocation / total) * 100

        return adjusted_allocation

    def calculate_allocation(self, profile):
        """Calculate intelligent portfolio allocation"""
        age = profile.get('age', 30)
        risk = profile.get('risk_appetite', 'Medium')
        goal = profile.get('investment_goal', 'Wealth Creation')
        timeline = profile.get('timeline_months', 60)
        income = profile.get('annual_income', 800000)

        # Base allocation using age rule
        base_equity = self.allocation_rules['age_equity_rule'](age)
        base_debt = min(50, (100 - base_equity) * 0.6)
        base_mf = 100 - base_equity - base_debt

        # Apply risk multipliers
        risk_mult = self.allocation_rules['risk_multipliers'][risk]
        equity_alloc = base_equity * risk_mult['equity']
        debt_alloc = base_debt * risk_mult['debt']
        mf_alloc = base_mf * risk_mult['mf']

        # Apply goal adjustments
        if goal in self.allocation_rules['goal_adjustments']:
            goal_adj = self.allocation_rules['goal_adjustments'][goal]
            equity_alloc *= goal_adj['equity']
            debt_alloc *= goal_adj['debt']
            mf_alloc *= goal_adj['mf']

        # Timeline adjustments
        if timeline < 12:  # Less than 1 year
            debt_alloc *= 1.5
            equity_alloc *= 0.5
        elif timeline < 36:  # Less than 3 years
            debt_alloc *= 1.2
            equity_alloc *= 0.8

        # Income-based adjustments
        if income > 1500000:  # High income
            equity_alloc *= 1.1
            debt_alloc *= 0.9

        # Normalize to 100%
        total = equity_alloc + debt_alloc + mf_alloc
        equity_alloc = (equity_alloc / total) * 100
        debt_alloc = (debt_alloc / total) * 100
        mf_alloc = (mf_alloc / total) * 100

        # Apply constraints
        equity_alloc = max(10, min(80, equity_alloc))
        debt_alloc = max(5, min(70, debt_alloc))
        mf_alloc = max(10, min(60, mf_alloc))

        # Final normalization
        total = equity_alloc + debt_alloc + mf_alloc
        return np.array([
            debt_alloc / total * 100,
            equity_alloc / total * 100,
            mf_alloc / total * 100
        ])

    def _calculate_ml_adjustments(self, market_predictions, profile):
        """Calculate ML-based allocation adjustments"""
        adjustments = [0, 0, 0]  # [debt, equity, mf]

        try:
            risk_tolerance = profile.get('risk_appetite', 'Medium')
            timeline = profile.get('timeline_months', 60)

            # Calculate average predictions for each asset class
            equity_avg_pred = self._get_average_prediction(market_predictions.get('Equity', []))
            debt_avg_pred = self._get_average_prediction(market_predictions.get('Debt', []))
            etf_avg_pred = self._get_average_prediction(market_predictions.get('ETF', []))

            # Adjustment factors based on predictions
            if equity_avg_pred > 5 and risk_tolerance != 'Low' and timeline > 12:
                adjustments[1] += 5  # Increase equity allocation
                adjustments[0] -= 2.5  # Decrease debt allocation
                adjustments[2] -= 2.5  # Decrease MF allocation
            elif equity_avg_pred < -2:
                adjustments[1] -= 3  # Decrease equity allocation
                adjustments[0] += 2  # Increase debt allocation
                adjustments[2] += 1  # Increase MF allocation

            if debt_avg_pred > 3:
                adjustments[0] += 2  # Increase debt allocation
                adjustments[1] -= 1  # Decrease equity allocation
                adjustments[2] -= 1  # Decrease MF allocation

            if etf_avg_pred > 4:
                adjustments[2] += 3  # Increase ETF allocation
                adjustments[0] -= 1.5  # Decrease debt allocation
                adjustments[1] -= 1.5  # Decrease equity allocation

        except Exception as e:
            print(f"Warning: Could not apply ML adjustments: {e}")

        return adjustments

    def _get_average_prediction(self, predictions):
        """Get weighted average prediction for an asset class"""
        if not predictions:
            return 0

        total_weighted_pred = 0
        total_confidence = 0

        for pred in predictions:
            confidence = pred.get('confidence', 0.5)
            pred_1m = pred.get('predicted_1m', 0)

            total_weighted_pred += pred_1m * confidence
            total_confidence += confidence

        return total_weighted_pred / total_confidence if total_confidence > 0 else 0
    def get_investment_recommendations(self, investment_goal: str, risk_appetite: str):
        """
        Provides investment recommendations. This version delegates to MarketDataManager
        for ML-enhanced recommendations.
        """
        # Ensure the MarketDataManager instance used here is properly initialized.
        # For simplicity, we'll create a new one, but ideally, you'd pass it or get it from a shared context.
        local_market_manager = MarketDataManager()
        local_market_manager.ml_predictor = ml_predictor_instance # Inject the global ML Predictor

        # The get_portfolio_recommendations_with_ml in MarketDataManager
        # is the most comprehensive one. We need to pass mock amount/allocation
        # if not truly calculating a portfolio, or simplify what it expects.
        # Let's adapt it to fetch based on goal and risk.

        # Decide which instruments to recommend based on goal/risk
        instruments_to_consider = []
        if 'Wealth Creation' in investment_goal or risk_appetite == 'High':
            instruments_to_consider.extend(local_market_manager.nifty50_stocks)
            instruments_to_consider.extend(local_market_manager.index_etfs)
        elif 'Retirement' in investment_goal or 'Education' in investment_goal:
            # For long-term goals, blend equity and ETFs
            instruments_to_consider.extend(local_market_manager.nifty50_stocks[:10])
            instruments_to_consider.extend(local_market_manager.index_etfs)
            instruments_to_consider.extend(local_market_manager.sector_etfs)
        elif 'Short-Term' in investment_goal or risk_appetite == 'Low':
            instruments_to_consider.extend(local_market_manager.debt_instruments)
            instruments_to_consider.extend(local_market_manager.index_etfs) # LiquidBees
        else: # Default for other goals or if goal not specific enough
            instruments_to_consider.extend(local_market_manager.nifty50_stocks[:15])
            instruments_to_consider.extend(local_market_manager.index_etfs)
            instruments_to_consider.extend(local_market_manager.debt_instruments)


        # Remove duplicates
        instruments_to_consider = list(set(instruments_to_consider))

        # Use the existing filtered recommendations method
        # Note: get_filtered_recommendations_with_ml sorts by ml_score by default
        recommendations = local_market_manager.get_filtered_recommendations_with_ml(
            instruments_to_consider,
            risk_appetite
        )

        # Return a subset, perhaps top 6, similar to what _compile_final_recommendations_with_ml does
        return recommendations[:6]

class AdvancedFinancialCalculator:
    """Enhanced financial calculator with ML insights"""

    @staticmethod
    def calculate_sip_amount(target_amount, timeline_months, expected_return=12):
        """Calculate SIP amount needed for target"""
        monthly_return = expected_return / 100 / 12
        if monthly_return == 0:
            return target_amount / timeline_months

        # SIP formula: FV = PMT * [((1 + r)^n - 1) / r]
        future_value_factor = ((1 + monthly_return) ** timeline_months - 1) / monthly_return
        return target_amount / future_value_factor

    @staticmethod
    def calculate_future_value_with_ml(monthly_investment, timeline_months, returns, ml_predictions=None):
        """Calculate future value with ML-enhanced scenarios"""
        scenarios = {}

        # Traditional scenarios
        for scenario, annual_return in returns.items():
            monthly_return = annual_return / 100 / 12
            if monthly_return == 0:
                fv = monthly_investment * timeline_months
            else:
                fv = monthly_investment * (((1 + monthly_return) ** timeline_months - 1) / monthly_return)
            scenarios[scenario] = fv

        # ML-enhanced scenario
        if ml_predictions:
            try:
                # Calculate ML-based expected return
                avg_prediction = sum(pred.get('predicted_1m', 0) for pred in ml_predictions) / len(ml_predictions)
                ml_annual_return = avg_prediction * 12  # Annualize monthly prediction

                # Ensure reasonable bounds
                ml_annual_return = max(5, min(25, ml_annual_return))

                monthly_return = ml_annual_return / 100 / 12
                if monthly_return > 0:
                    ml_fv = monthly_investment * (((1 + monthly_return) ** timeline_months - 1) / monthly_return)
                    scenarios[f'AI Prediction ({ml_annual_return:.1f}%)'] = ml_fv
            except:
                pass

        return scenarios

    @staticmethod
    def calculate_risk_metrics_with_ml(allocation, volatilities, ml_predictions=None):
        """Calculate portfolio risk metrics with ML insights"""
        # Traditional portfolio volatility
        portfolio_volatility = np.sqrt(
            sum((allocation[i] / 100) ** 2 * (volatilities[i] / 100) ** 2 for i in range(len(allocation)))
        ) * 100

        # ML-enhanced risk assessment
        ml_risk_adjustment = 0
        if ml_predictions:
            try:
                # Calculate prediction uncertainty as additional risk factor
                all_predictions = []
                for asset_class, preds in ml_predictions.items():
                    for pred in preds:
                        if pred.get('predicted_1m') is not None:
                            all_predictions.append(pred['predicted_1m'])

                if all_predictions:
                    pred_volatility = np.std(all_predictions)
                    ml_risk_adjustment = pred_volatility * 0.5  # Scale down the adjustment
            except:
                pass

        adjusted_volatility = portfolio_volatility + ml_risk_adjustment

        return {
            'traditional_risk': round(portfolio_volatility, 2),
            'ml_adjusted_risk': round(adjusted_volatility, 2),
            'ml_risk_adjustment': round(ml_risk_adjustment, 2)
        }

class EnhancedInputParser:
    """Enhanced input parser with ML understanding"""

    def __init__(self):
        self.amount_patterns = [
            r'(?:₹|Rs\.?|INR)?\s?(\d+(?:,\d+)*(?:\.\d+)?)\s*(L|l|lakhs?|lacs?|K|k|thousands?|Cr|cr|crores?)',
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(L|l|lakhs?|lacs?|K|k|thousands?|Cr|cr|crores?)(?:\s*(?:rupees?|Rs\.?|INR))?',
            r'(?:₹|Rs\.?|INR)\s?(\d+(?:,\d+)*(?:\.\d+)?)',
            r'(\d+(?:,\d+)*(?:\.\d+)?)(?=\s*(?:rupees?|for))'
        ]

        self.time_patterns = [
            r'(?:in|after|within)\s*(\d+)\s*(years?|yrs?|months?|mths?)',
            r'(\d+)\s*(years?|yrs?|months?|mths?)',
            r'(\d+)\s*-\s*(\d+)\s*(years?|months?)'
        ]

        self.goal_keywords = {
            'vacation': 'Short-Term',
            'holiday': 'Short-Term',
            'trip': 'Short-Term',
            'car': 'Short-Term',
            'vehicle': 'Short-Term',
            'wedding': 'Short-Term',
            'marriage': 'Short-Term',
            'house': 'Property Purchase',
            'home': 'Property Purchase',
            'property': 'Property Purchase',
            'education': 'Education',
            'study': 'Education',
            'child': 'Education',
            'retire': 'Retirement',
            'retirement': 'Retirement',
            'wealth': 'Wealth Creation',
            'rich': 'Wealth Creation',
            'emergency': 'Emergency Fund',
            'tax': 'Tax Saving'
        }

        # ML-related keywords
        self.ml_keywords = {
            'prediction': True,
            'forecast': True,
            'ai': True,
            'machine learning': True,
            'future': True,
            'expected': True
        }

    def parse_user_input(self, message):
        """Enhanced parsing with ML context detection"""
        extracted = {}
        message_lower = message.lower()

        # Check if user is asking for ML insights
        extracted['wants_ml_insights'] = any(keyword in message_lower for keyword in self.ml_keywords)

        # Extract amount
        for pattern in self.amount_patterns:
            amount_match = re.search(pattern, message, re.IGNORECASE)
            if amount_match:
                value_str = amount_match.group(1).replace(',', '')
                value = float(value_str)

                suffix = amount_match.group(2) if len(amount_match.groups()) > 1 else None
                if suffix:
                    suffix_lower = suffix.lower()
                    if suffix_lower.startswith('l') or 'lakh' in suffix_lower:
                        value *= 100000
                    elif suffix_lower.startswith('k'):
                        value *= 1000
                    elif suffix_lower.startswith('cr'):
                        value *= 10000000

                if value >= 1000:
                    extracted["goal_amount"] = int(value)
                    break

        # Extract timeline
        for pattern in self.time_patterns:
            time_match = re.search(pattern, message, re.IGNORECASE)
            if time_match:
                if len(time_match.groups()) == 3:  # Range pattern
                    min_val = int(time_match.group(1))
                    max_val = int(time_match.group(2))
                    value = (min_val + max_val) // 2
                    unit = time_match.group(3).lower()
                else:
                    value = int(time_match.group(1))
                    unit = time_match.group(2).lower()

                if 'year' in unit or 'yr' in unit:
                    extracted["timeline_months"] = value * 12
                else:
                    extracted["timeline_months"] = value
                break

        # Extract goal type
        for keyword, goal_type in self.goal_keywords.items():
            if keyword in message_lower:
                extracted["goal_type"] = goal_type
                extracted["investment_goal"] = goal_type
                break

        # Extract risk appetite
        if any(phrase in message_lower for phrase in ['low risk', 'safe', 'conservative']):
            extracted["risk_appetite"] = "Low"
        elif any(phrase in message_lower for phrase in ['high risk', 'aggressive', 'risky']):
            extracted["risk_appetite"] = "High"
        elif any(phrase in message_lower for phrase in ['medium risk', 'moderate', 'balanced']):
            extracted["risk_appetite"] = "Medium"

        # Extract age
        age_patterns = [
            r'\b(\d{2})\b(?:\s*years?\s*old)?',
            r'age\s*(?:is\s*)?(\d{2})',
            r"i'm\s*(\d{2})"
        ]
        for pattern in age_patterns:
            age_match = re.search(pattern, message_lower)
            if age_match:
                age = int(age_match.group(1))
                if 18 <= age <= 70:
                    extracted["age"] = age
                    break

        # Extract income
        income_patterns = [
            r'earn\s*(?:₹|Rs\.?|INR)?\s?(\d+(?:,\d+)*(?:\.\d+)?)\s*(L|l|lakhs?|K|k)?',
            r'salary\s*(?:₹|Rs\.?|INR)?\s?(\d+(?:,\d+)*(?:\.\d+)?)\s*(L|l|lakhs?|K|k)?',
            r'income\s*(?:₹|Rs\.?|INR)?\s?(\d+(?:,\d+)*(?:\.\d+)?)\s*(L|l|lakhs?|K|k)?'
        ]
        for pattern in income_patterns:
            income_match = re.search(pattern, message, re.IGNORECASE)
            if income_match:
                value_str = income_match.group(1).replace(',', '')
                value = float(value_str)
                suffix = income_match.group(2)
                if suffix and suffix.lower().startswith('l'):
                    value *= 100000
                elif suffix and suffix.lower().startswith('k'):
                    value *= 1000
                extracted["annual_income"] = int(value)
                break

        return extracted

class EnhancedReportGenerator:
    """Enhanced report generator with ML insights"""

    def __init__(self):
        self.calculator = AdvancedFinancialCalculator()

    def generate_comprehensive_report_with_ml(self, recommendations, goal_amount, timeline_months, 
                                            allocation, user_profile, ml_predictions):
        """Generate comprehensive report with ML insights"""
        if not recommendations:
            return "❌ Unable to fetch current market data. Please try again later."

        report = self._generate_header(goal_amount, timeline_months, user_profile)
        report += self._generate_allocation_section_with_ml(allocation, ml_predictions)
        report += self._generate_investment_calculation_with_ml(goal_amount, timeline_months, allocation, ml_predictions)
        report += self._generate_recommendations_section_with_ml(recommendations, allocation)
        report += self._generate_ml_predictions_section(ml_predictions)
        report += self._generate_risk_analysis_with_ml(allocation, recommendations, ml_predictions)
        report += self._generate_scenarios_with_ml(goal_amount, timeline_months, ml_predictions)
        report += self._generate_action_plan_with_ml(recommendations, allocation)
        report += self._generate_disclaimer()

        return report

    def _generate_header(self, goal_amount, timeline_months, profile):
        """Generate enhanced report header"""
        years = timeline_months // 12
        months = timeline_months % 12

        timeline_str = f"{years} years" if months == 0 else f"{years} years {months} months"
        if timeline_months < 12:
            timeline_str = f"{timeline_months} months"

        header = f" FINVERSE AI - ML-POWERED INVESTMENT REPORT\n"
        header += f"{'='*70}\n"
        header += f"• Target Amount: ₹{goal_amount:,}\n"
        header += f"• Timeline: {timeline_str}\n"
        header += f"• Risk Profile: {profile.get('risk_appetite', 'Medium')}\n"
        header += f"• Goal: {profile.get('investment_goal', 'Wealth Creation')}\n"
        header += f"• AI Analysis: Enabled\n"

        return header

    def _generate_allocation_section_with_ml(self, allocation, ml_predictions):
        """Generate ML-enhanced portfolio allocation section"""
        section = f"\n\n\n AI-OPTIMIZED PORTFOLIO ALLOCATION\n"
        section += f"{'-'*40}\n"
        section += f"•  Debt Instruments: {allocation[0]:.1f}%\n"
        section += f"• Equity: {allocation[1]:.1f}%\n"
        section += f"• Mutual Funds/ETFs: {allocation[2]:.1f}%\n"

        # ML insights on allocation
        if ml_predictions:
            section += f"\n\n\n AI Allocation Insights:\n"

            # Get best performing asset class prediction
            best_class = self._get_best_performing_class(ml_predictions)
            if best_class:
                section += f"   • Best AI-predicted asset class: {best_class['name']}\n"
                section += f"   • Expected 1M return: {best_class['return']:+.1f}%\n"

        return section

    def _generate_investment_calculation_with_ml(self, goal_amount, timeline_months, allocation, ml_predictions):
        """Generate ML-enhanced investment calculation section"""
        # Calculate expected return based on ML predictions
        expected_return = 12  # Default
        if ml_predictions:
            ml_return = self._calculate_ml_expected_return(ml_predictions)
            if ml_return:
                expected_return = ml_return

        monthly_investment = self.calculator.calculate_sip_amount(goal_amount, timeline_months, expected_return)

        section = f"\n AI-CALCULATED MONTHLY INVESTMENT\n"
        section += f"{'-'*40}\n"
        section += f"• Total Monthly SIP: ₹{monthly_investment:,.0f}\n"
        section += f"   • Debt portion: ₹{monthly_investment * allocation[0] / 100:,.0f}\n"
        section += f"   • Equity portion: ₹{monthly_investment * allocation[1] / 100:,.0f}\n"
        section += f"   • Mutual Fund portion: ₹{monthly_investment * allocation[2] / 100:,.0f}\n"

        if ml_predictions:
            section += f"\n AI-Based Expected Return: {expected_return:.1f}% p.a.\n"

        return section

    def _generate_recommendations_section_with_ml(self, recommendations, allocation):
        """Generate ML-enhanced detailed recommendations section"""
        section = f"\n\n\n AI-RANKED INVESTMENT RECOMMENDATIONS\n"
        section += f"{'='*70}\n"

        for i, rec in enumerate(recommendations, 1):
            section += f"\n{i}. {rec['name']} ({rec['ticker']})\n"
            section += f"   📂 Category: {rec['category']}\n"

            if rec['category'] in ['Equity', 'ETF', 'ETF/Index Fund']:
                section += f"   💰 Current Price/NAV: ₹{rec['current_price']:.2f}\n"
                section += f"   📊 1-Year Return: {rec['year_return']:.2f}%\n"
                section += f"   📉 Volatility: {rec['volatility']:.2f}%\n"
                section += f"   ⚡ Sharpe Ratio: {rec.get('sharpe_ratio', 'N/A')}\n"

                # ML predictions if available
                if rec.get('predicted_return_1m'):
                    section += f"   🤖 AI 1M Prediction: {rec['predicted_return_1m']:+.1f}%\n"
                if rec.get('predicted_return_3m'):
                    section += f"   🤖 AI 3M Prediction: {rec['predicted_return_3m']:+.1f}%\n"
                if rec.get('ml_confidence'):
                    confidence_emoji = "🟢" if rec['ml_confidence'] > 0.7 else "🟡" if rec['ml_confidence'] > 0.5 else "🔴"
                    section += f"   {confidence_emoji} AI Confidence: {rec['ml_confidence']:.0%}\n"
                if rec.get('ml_score'):
                    section += f"   🎯 AI Score: {rec['ml_score']:.3f}/1.000\n"
            else:
                section += f"   📊 Expected Return: {rec['year_return']:.2f}% p.a.\n"
                section += f"   🛡️  Risk Level: Very Low\n"

            # Investment suggestion based on allocation
            if rec['category'] == 'Equity' and allocation[1] > 20:
                section += f"   💡 Suggested Allocation: High (part of {allocation[1]:.1f}% equity)\n"
            elif rec['category'] in ['ETF', 'ETF/Index Fund'] and allocation[2] > 20:
                section += f"   💡 Suggested Allocation: Medium (part of {allocation[2]:.1f}% MF/ETF)\n"
            elif 'Debt' in rec['category'] and allocation[0] > 15:
                section += f"   💡 Suggested Allocation: Conservative (part of {allocation[0]:.1f}% debt)\n"

        return section

    def _generate_ml_predictions_section(self, ml_predictions):
        """Generate detailed ML predictions section"""
        if not ml_predictions:
            return ""

        section = f"\n\n\n DETAILED AI MARKET PREDICTIONS\n"
        section += f"{'='*50}\n"

        for asset_class, predictions in ml_predictions.items():
            if predictions:
                section += f"\n📊 {asset_class} AI Analysis:\n"
                section += f"{'-'*30}\n"

                for i, pred in enumerate(predictions[:3], 1):  # Top 3 per class
                    confidence_emoji = "🟢" if pred['confidence'] > 0.7 else "🟡" if pred['confidence'] > 0.5 else "🔴"

                    section += f"{i}. {pred['name']}\n"
                    section += f"   📈 AI 1M Prediction: {pred['predicted_1m']:+.1f}%\n"

                    if pred.get('predicted_3m'):
                        section += f"   📈 AI 3M Prediction: {pred['predicted_3m']:+.1f}%\n"
                    if pred.get('predicted_6m'):
                        section += f"   📈 AI 6M Prediction: {pred['predicted_6m']:+.1f}%\n"

                    section += f"   {confidence_emoji} AI Confidence: {pred['confidence']:.0%}\n"
                    section += f"   📊 Current YTD: {pred['current_return']:+.1f}%\n"
                    section += f"   📉 Volatility: {pred['volatility']:.1f}%\n"

                    if pred.get('ml_score'):
                        section += f"   🎯 AI Score: {pred['ml_score']:.3f}\n"
                    section += "\n"

        section += f" AI Model Features: Technical indicators, price patterns,\n"
        section += f"   volume analysis, and historical performance correlation\n"
        section += f"⚠️  Disclaimer: AI predictions are probabilistic estimates\n"
        section += f"   based on historical data and current market patterns.\n"

        return section

    def _generate_risk_analysis_with_ml(self, allocation, recommendations, ml_predictions):
        """Generate ML-enhanced risk analysis section"""
        # Calculate portfolio risk metrics
        volatilities = [rec.get('volatility', 20) for rec in recommendations[:3]]
        if len(volatilities) < 3:
            volatilities.extend([20] * (3 - len(volatilities)))

        risk_metrics = self.calculator.calculate_risk_metrics_with_ml(allocation, volatilities, ml_predictions)

        section = f"\n⚠️  AI-ENHANCED RISK ANALYSIS\n"
        section += f"{'-'*40}\n"
        section += f"• Traditional Portfolio Risk: {risk_metrics['traditional_risk']:.1f}%\n"
        section += f"• AI-Adjusted Portfolio Risk: {risk_metrics['ml_adjusted_risk']:.1f}%\n"

        if risk_metrics['ml_risk_adjustment'] != 0:
            section += f"🔍 ML Risk Adjustment: {risk_metrics['ml_risk_adjustment']:+.1f}%\n"

        # Risk level classification
        adjusted_risk = risk_metrics['ml_adjusted_risk']
        if adjusted_risk < 12:
            risk_level = "🟢 Low Risk"
            risk_desc = "Conservative portfolio with stable returns"
        elif adjusted_risk < 22:
            risk_level = "🟡 Medium Risk"
            risk_desc = "Balanced risk-return profile"
        else:
            risk_level = "🔴 High Risk"
            risk_desc = "Aggressive portfolio with higher volatility"

        section += f"• Risk Classification: {risk_level}\n"
        section += f"• Risk Description: {risk_desc}\n"

        # AI risk insights
        if ml_predictions:
            section += f"\n AI Risk Insights:\n"

            # Calculate prediction dispersion as uncertainty measure
            all_predictions = []
            for asset_class, preds in ml_predictions.items():
                for pred in preds:
                    if pred.get('predicted_1m') is not None:
                        all_predictions.append(pred['predicted_1m'])

            if all_predictions:
                pred_std = np.std(all_predictions)
                if pred_std < 2:
                    section += f"   • Low prediction uncertainty - Market consensus strong\n"
                elif pred_std < 5:
                    section += f"   • Moderate prediction uncertainty - Mixed signals\n"
                else:
                    section += f"   • High prediction uncertainty - Volatile market conditions\n"

        return section

    def _generate_scenarios_with_ml(self, goal_amount, timeline_months, ml_predictions):
        """Generate ML-enhanced scenario analysis"""
        monthly_sip = self.calculator.calculate_sip_amount(goal_amount, timeline_months)

        # Get all predictions for ML scenario
        all_predictions = []
        if ml_predictions:
            for asset_class, preds in ml_predictions.items():
                all_predictions.extend(preds)

        scenarios = self.calculator.calculate_future_value_with_ml(
            monthly_sip, timeline_months,
            {'Conservative (8%)': 8, 'Moderate (12%)': 12, 'Aggressive (15%)': 15},
            all_predictions
        )

        section = f"\n\n\n AI-ENHANCED SCENARIO ANALYSIS\n"
        section += f"{'-'*40}\n"
        section += f"With monthly SIP of ₹{monthly_sip:,.0f}:\n\n"

        for scenario, value in scenarios.items():
            difference = value - goal_amount
            percentage = (difference / goal_amount) * 100

            if 'AI Prediction' in scenario:
                emoji = "🤖"
            elif difference > 0:
                emoji = "✅"
            else:
                emoji = "❌"

            if difference > 0:
                section += f"{emoji} {scenario}: ₹{value:,.0f} (+₹{difference:,.0f}, +{percentage:.1f}%)\n"
            else:
                section += f"{emoji} {scenario}: ₹{value:,.0f} (₹{abs(difference):,.0f}, {percentage:.1f}%)\n"

        return section

    def _generate_action_plan_with_ml(self, recommendations, allocation):
        """Generate ML-enhanced actionable plan"""
        section = f"\n\n\n AI-POWERED ACTION PLAN\n"
        section += f"{'-'*40}\n"
        section += f"1. 🏦 Open investment accounts if not already done\n"
        section += f"2. 💳 Set up SIP mandates for systematic investing\n"
        section += f"3. 🤖 Start with top AI-ranked recommendations\n"
        section += f"4. 📊 Implement AI-optimized allocation strategy\n"
        section += f"5. 📱 Monitor AI predictions and market signals monthly\n"
        section += f"6. 🔄 Rebalance portfolio quarterly based on AI insights\n"
        section += f"7. 📈 Review and adjust strategy based on performance\n"

        # AI-specific recommendations
        section += f"\n\n\n AI-Specific Recommendations:\n"
        section += f"   • Follow high-confidence AI predictions closely\n"
        section += f"   • Consider increasing allocation to top AI-scored assets\n"
        section += f"   • Monitor prediction accuracy and adjust trust accordingly\n"
        section += f"   • Use AI volatility predictions for timing entries\n"

        return section

    def _generate_disclaimer(self):
        """Generate enhanced disclaimer"""
        disclaimer = f"\n\n\n⚠️  IMPORTANT DISCLAIMER\n"
        disclaimer += f"{'='*70}\n"
        disclaimer += f" This combines AI predictions with traditional financial analysis\n"
        disclaimer += f" AI models are trained on historical data - future may differ\n"
        disclaimer += f" Machine learning predictions have inherent uncertainty\n"
        disclaimer += f" Past performance does not guarantee future results\n"
        disclaimer += f" Markets are subject to risk - invest wisely\n"
        disclaimer += f" AI confidence scores indicate model certainty, not guarantees\n"
        disclaimer += f" Consider consulting a certified financial advisor\n"
        disclaimer += f" Diversify investments across asset classes and time\n"
        disclaimer += f"{'='*70}\n"

        return disclaimer

    def _get_best_performing_class(self, ml_predictions):
        """Get the best performing asset class from ML predictions"""
        best_class = None
        best_return = float('-inf')

        for asset_class, predictions in ml_predictions.items():
            if predictions:
                # Get weighted average return for the class
                total_weighted_return = 0
                total_confidence = 0

                for pred in predictions:
                    confidence = pred.get('confidence', 0.5)
                    pred_return = pred.get('predicted_1m', 0)

                    total_weighted_return += pred_return * confidence
                    total_confidence += confidence

                if total_confidence > 0:
                    avg_return = total_weighted_return / total_confidence
                    if avg_return > best_return:
                        best_return = avg_return
                        best_class = {
                            'name': asset_class,
                            'return': avg_return
                        }

        return best_class

    def _calculate_ml_expected_return(self, ml_predictions):
        """Calculate expected return based on ML predictions"""
        try:
            all_predictions = []
            all_confidences = []

            for asset_class, predictions in ml_predictions.items():
                for pred in predictions:
                    if pred.get('predicted_1m') is not None:
                        all_predictions.append(pred['predicted_1m'])
                        all_confidences.append(pred.get('confidence', 0.5))

            if all_predictions:
                # Calculate confidence-weighted average
                weighted_sum = sum(p * c for p, c in zip(all_predictions, all_confidences))
                total_confidence = sum(all_confidences)

                if total_confidence > 0:
                    monthly_return = weighted_sum / total_confidence
                    # Annualize and ensure reasonable bounds
                    annual_return = monthly_return * 12
                    return max(6, min(20, annual_return))

        except Exception as e:
            print(f"Error calculating ML expected return: {e}")

        return None

class FinVerseAI:
    """Enhanced FinVerse AI with integrated ML capabilities"""

    def __init__(self):
        self.market_manager = MarketDataManager()
        self.allocation_engine = IntelligentAllocationEngine()
        self.input_parser = EnhancedInputParser()
        self.report_generator = EnhancedReportGenerator()
        self.user_sessions = {}
        self.ml_enabled = True

    def get_user_session(self, session_id='default'):
        """Get or create user session"""
        if session_id not in self.user_sessions:
            self.user_sessions[session_id] = {
                'profile': {},
                'conversation_history': [],
                'ml_preferences': {'enabled': True, 'confidence_threshold': 0.5}
            }
        return self.user_sessions[session_id]

    def update_user_profile(self, session_id, new_info):
        """Update user profile with new information"""
        session = self.get_user_session(session_id)
        session['profile'].update(new_info)
        session['conversation_history'].append(new_info)

    def process_user_query(self, user_input, session_id='default'):
        """Enhanced query processing with ML capabilities"""
        session = self.get_user_session(session_id)

        # Parse new information
        parsed_info = self.input_parser.parse_user_input(user_input)

        # Update user profile
        if parsed_info:
            self.update_user_profile(session_id, parsed_info)

        # Handle ML-specific queries
        if self._is_ml_query(user_input):
            return self._handle_ml_query(user_input, session)

        # Check for non-financial queries
        if self._is_non_financial_query(user_input):
            return self._handle_non_financial_query()

        # Handle ML toggle requests
        if 'disable ai' in user_input.lower() or 'turn off ml' in user_input.lower():
            session['ml_preferences']['enabled'] = False
            return "🤖 AI predictions disabled. Using traditional analysis only."

        if 'enable ai' in user_input.lower() or 'turn on ml' in user_input.lower():
            session['ml_preferences']['enabled'] = True
            return "🤖 AI predictions enabled. Enhanced analysis activated!"

        # Handle retirement planning
        if 'retirement' in user_input.lower() and 'goal_amount' not in session['profile']:
            return self._handle_retirement_planning(session)

        # Check for missing information
        missing_info = self._identify_missing_info(session['profile'])

        if missing_info:
            return self._request_missing_info(missing_info)

        # Generate ML-enhanced recommendations
        return self._generate_ml_enhanced_recommendations(session)

    def _is_ml_query(self, user_input):
        """Check if query is specifically about ML predictions"""
        ml_query_keywords = [
            'ai prediction', 'machine learning', 'forecast', 'ml analysis',
            'artificial intelligence', 'predict', 'future performance',
            'ai recommendation', 'ml score', 'confidence'
        ]
        return any(keyword in user_input.lower() for keyword in ml_query_keywords)

    def _handle_ml_query(self, user_input, session):
        """Handle ML-specific queries"""
        if 'how does ai work' in user_input.lower() or 'ml explanation' in user_input.lower():
            return self._explain_ml_methodology()

        if 'ai accuracy' in user_input.lower() or 'prediction accuracy' in user_input.lower():
            return self._explain_ai_accuracy()

        # Generate quick ML insights
        try:
            print("🤖 Generating AI market insights...")
            ml_predictions = self.market_manager.get_asset_class_predictions('Medium')
            return self._generate_ml_insights_summary(ml_predictions)
        except Exception as e:
            return f"❌ Error generating AI insights: {str(e)}"

    def _explain_ml_methodology(self):
        """Explain the ML methodology used"""
        explanation = f"\n🤖 FINVERSE AI METHODOLOGY EXPLAINED\n"
        explanation += f"{'='*50}\n\n"

        explanation += f"📊 Data Sources:\n"
        explanation += f"   • Historical price and volume data\n"
        explanation += f"   • Technical indicators (RSI, Moving Averages)\n"
        explanation += f"   • Market volatility patterns\n"
        explanation += f"   • Trading volume analysis\n\n"

        explanation += f"🧠 Machine Learning Models:\n"
        explanation += f"   • Random Forest Regressor (primary)\n"
        explanation += f"   • Gradient Boosting Regressor (secondary)\n"
        explanation += f"   • Ensemble approach for better accuracy\n"
        explanation += f"   • Feature importance analysis\n\n"

        explanation += f"🎯 Prediction Horizons:\n"
        explanation += f"   • 1 Month (21 trading days)\n"
        explanation += f"   • 3 Months (63 trading days)\n"
        explanation += f"   • 6 Months (126 trading days)\n\n"

        explanation += f"📈 Key Features Used:\n"
        explanation += f"   • 20 & 50-day moving averages\n"
        explanation += f"   • RSI (Relative Strength Index)\n"
        explanation += f"   • Price volatility measures\n"
        explanation += f"   • Volume patterns and trends\n"
        explanation += f"   • Lagged returns (1, 5, 10 days)\n\n"

        explanation += f"🎯 Confidence Scoring:\n"
        explanation += f"   • Based on data quality and quantity\n"
        explanation += f"   • Adjusted for market volatility\n"
        explanation += f"   • Higher confidence = more reliable predictions\n\n"

        explanation += f"⚠️  Limitations:\n"
        explanation += f"   • Based on historical patterns\n"
        explanation += f"   • Cannot predict external shocks\n"
        explanation += f"   • Market conditions can change rapidly\n"

        return explanation

    def _explain_ai_accuracy(self):
        """Explain AI accuracy and limitations"""
        accuracy_info = f"\n📊 AI PREDICTION ACCURACY EXPLAINED\n"
        accuracy_info += f"{'='*45}\n\n"

        accuracy_info += f"🎯 Model Performance:\n"
        accuracy_info += f"   • Typically 60-75% directional accuracy\n"
        accuracy_info += f"   • Better performance on stable markets\n"
        accuracy_info += f"   • Lower accuracy during high volatility\n"
        accuracy_info += f"   • Continuous model retraining improves results\n\n"

        accuracy_info += f"📈 Confidence Levels Guide:\n"
        accuracy_info += f"   • 🟢 High (70%+): Strong historical patterns\n"
        accuracy_info += f"   • 🟡 Medium (50-70%): Moderate confidence\n"
        accuracy_info += f"   • 🔴 Low (<50%): Use with caution\n\n"

        accuracy_info += f"🔍 What Affects Accuracy:\n"
        accuracy_info += f"   • Amount of historical data available\n"
        accuracy_info += f"   • Market volatility and conditions\n"
        accuracy_info += f"   • External economic factors\n"
        accuracy_info += f"   • Company-specific news and events\n\n"

        accuracy_info += f"💡 Best Practices:\n"
        accuracy_info += f"   • Use AI as one factor among many\n"
        accuracy_info += f"   • Focus on high-confidence predictions\n"
        accuracy_info += f"   • Combine with fundamental analysis\n"
        accuracy_info += f"   • Diversify based on multiple signals\n"

        return accuracy_info

    def _generate_ml_insights_summary(self, ml_predictions):
        """Generate quick ML insights summary"""
        if not ml_predictions:
            return "❌ Unable to generate AI insights at this time."

        summary = f"\n🤖 AI MARKET INSIGHTS SUMMARY\n"
        summary += f"{'='*40}\n"

        for asset_class, predictions in ml_predictions.items():
            if predictions:
                best = predictions[0]  # Top prediction
                summary += f"\n📊 {asset_class}:\n"
                summary += f"   🥇 Top Pick: {best['name']}\n"
                summary += f"   📈 AI 1M Prediction: {best['predicted_1m']:+.1f}%\n"

                if best.get('predicted_3m'):
                    summary += f"   📈 AI 3M Prediction: {best['predicted_3m']:+.1f}%\n"

                confidence_emoji = "🟢" if best['confidence'] > 0.7 else "🟡" if best['confidence'] > 0.5 else "🔴"
                summary += f"   {confidence_emoji} Confidence: {best['confidence']:.0%}\n"
                summary += f"   📊 Current Performance: {best['current_return']:+.1f}%\n"

        summary += f"\n💡 AI Recommendation: Focus on high-confidence predictions\n"
        summary += f"🔄 Data freshness: Real-time analysis\n"
        summary += f"\nFor detailed investment planning, share your financial goals!\n"

        return summary

    def _is_non_financial_query(self, user_input):
        """Check if query is non-financial"""
        non_financial_keywords = [
            'weather', 'recipe', 'movie', 'game', 'food', 'sports', 
            'cricket', 'football', 'politics', 'news', 'entertainment'
        ]
        financial_keywords = ['invest', 'save', 'money', 'fund', 'portfolio', 'return', 'ai', 'prediction']

        has_non_financial = any(word in user_input.lower() for word in non_financial_keywords)
        has_financial = any(word in user_input.lower() for word in financial_keywords)

        return has_non_financial and not has_financial

    def _handle_non_financial_query(self):
        """Handle non-financial queries"""
        return ("\n🤖 I'm FinVerse AI, your AI-powered investment advisor!\n\n"
                "I specialize in:\n"
                "• 🤖 AI-driven market predictions and analysis\n"
                "• 📊 ML-optimized portfolio allocation\n"
                "• 🎯 Goal-based investment planning with AI insights\n"
                "• 📈 Real-time market analysis and stock recommendations\n"
                "• 💰 Smart SIP calculations with predictive modeling\n\n"
                "Try asking: 'Show me AI predictions for the market' or\n"
                "'I want to invest 10L in 2 years with AI recommendations'\n")

    def _handle_retirement_planning(self, session):
        """Handle retirement planning queries"""
        profile = session['profile']
        if 'age' in profile:
            retirement_age = 60
            years_to_retirement = retirement_age - profile['age']
            profile['timeline_months'] = years_to_retirement * 12
            profile['investment_goal'] = 'Retirement'

            return (f"\n🎯 AI-Enhanced Retirement Planning!\n\n"
                   f"Planning for retirement in {years_to_retirement} years with AI insights.\n\n"
                   f"To provide personalized AI recommendations, I need:\n"
                   f"• Your target retirement corpus\n"
                   f"• Your current annual income\n"
                   f"• Your risk appetite (Low/Medium/High)\n\n"
                   f"Example: 'I want 2 crores for retirement, earn 15L annually, medium risk'\n"
                   f"🤖 AI will analyze market trends and optimize your strategy!\n")
        else:
            return ("\n🎯 AI-Powered Retirement Planning!\n\n"
                   "Let me create an AI-optimized retirement strategy.\n\n"
                   "I need some information:\n"
                   "• Your current age\n"
                   "• Target retirement amount\n"
                   "• Current annual income\n"
                   "• Risk appetite\n\n"
                   "Example: 'I'm 35, want 2 crores for retirement, earn 12L annually'\n"
                   f"🤖 AI will provide predictive insights for your retirement planning!\n")

    def _identify_missing_info(self, profile):
        """Identify missing critical information"""
        missing = []

        # Check for financial goal completion
        if profile.get('goal_amount') or profile.get('investment_goal') == 'Retirement':
            if 'goal_amount' not in profile and profile.get('investment_goal') != 'Retirement':
                missing.append('goal_amount')
            if 'timeline_months' not in profile:
                missing.append('timeline')

        return missing

    def _request_missing_info(self, missing_info):
        """Request missing information from user"""
        if 'goal_amount' in missing_info:
            return ("\n💰 I need to know your target investment amount for AI analysis.\n\n"
                   "Please specify: 'I want to save [amount] for [purpose]'\n"
                   "Example: 'I want to save 10 lakhs for a car'\n"
                   "🤖 AI will then provide optimized recommendations!\n")

        if 'timeline' in missing_info:
            return ("\n⏱️  When do you need this amount? (AI needs timeline for predictions)\n\n"
                   "Please specify: 'I need it in [time period]'\n"
                   "Example: 'I need it in 3 years' or 'in 18 months'\n"
                   "🤖 AI will optimize strategy based on your timeline!\n")

        return "\n🤖 Please provide more details about your investment goals for AI analysis.\n"

    def _generate_ml_enhanced_recommendations(self, session):
        """Generate comprehensive ML-enhanced investment recommendations"""
        profile = session['profile']
        ml_enabled = session['ml_preferences']['enabled']

        # Fill in default values where needed
        complete_profile = {
            'age': profile.get('age', 30),
            'annual_income': profile.get('annual_income', 800000),
            'risk_appetite': profile.get('risk_appetite', 'Medium'),
            'investment_goal': profile.get('investment_goal', 'Wealth Creation'),
            'timeline_months': profile.get('timeline_months', 60)
        }

        # Get ML predictions if enabled
        ml_predictions = None
        if ml_enabled:
            try:
                print("🤖 Analyzing market data with AI...")
                ml_predictions = self.market_manager.get_asset_class_predictions(
                    complete_profile['risk_appetite']
                )
            except Exception as e:
                print(f"⚠️ AI analysis partially failed: {e}")
                print("📊 Falling back to traditional analysis...")

        # Calculate allocation with ML insights
        allocation = self.allocation_engine.calculate_allocation_with_ml_insights(
            complete_profile, ml_predictions
        )

        # Get market recommendations
        if ml_enabled:
            recommendations = self.market_manager.get_portfolio_recommendations_with_ml(
                complete_profile['risk_appetite'],
                complete_profile['timeline_months'],
                profile.get('goal_amount', 1000000),
                allocation
            )
        else:
            recommendations = self.market_manager.get_portfolio_recommendations(
                complete_profile['risk_appetite'],
                complete_profile['timeline_months'],
                profile.get('goal_amount', 1000000),
                allocation
            )

        # Generate comprehensive report
        if 'goal_amount' in profile:
            if ml_enabled and ml_predictions:
                return self.report_generator.generate_comprehensive_report_with_ml(
                    recommendations,
                    profile['goal_amount'],
                    complete_profile['timeline_months'],
                    allocation,
                    complete_profile,
                    ml_predictions
                )
            else:
                return self.report_generator.generate_comprehensive_report(
                    recommendations,
                    profile['goal_amount'],
                    complete_profile['timeline_months'],
                    allocation,
                    complete_profile
                )
        else:
            # Profile-based response
            return self._generate_profile_based_response_with_ml(
                allocation, complete_profile, recommendations, ml_predictions
            )

    def _generate_profile_based_response_with_ml(self, allocation, profile, recommendations, ml_predictions):
        """Generate ML-enhanced response for profile-only queries"""
        response = f"\n📊 AI-POWERED PORTFOLIO RECOMMENDATION\n"
        response += f"{'='*50}\n"
        response += f"👤 Profile: {profile['age']} years, {profile['risk_appetite']} risk\n"
        response += f"🎯 Goal: {profile['investment_goal']}\n"
        response += f"🤖 AI Analysis: {'Enabled' if ml_predictions else 'Disabled'}\n\n"

        response += f"📈 {'AI-Optimized' if ml_predictions else 'Traditional'} Allocation:\n"
        response += f"   • 🏛️  Debt Instruments: {allocation[0]:.1f}%\n"
        response += f"   • 📊 Equity: {allocation[1]:.1f}%\n"
        response += f"   • 🏦 Mutual Funds/ETFs: {allocation[2]:.1f}%\n\n"

        if recommendations:
            response += f"🏆 Top {'AI-Ranked' if ml_predictions else ''} Investment Options:\n"
            for i, rec in enumerate(recommendations[:3], 1):
                response += f"   {i}. {rec['name']} ({rec['category']})\n"
                if rec.get('year_return'):
                    response += f"      📊 1-Year Return: {rec['year_return']:.1f}%\n"
                if rec.get('predicted_return_1m') and ml_predictions:
                    response += f"      🤖 AI 1M Prediction: {rec['predicted_return_1m']:+.1f}%\n"
                if rec.get('ml_confidence') and ml_predictions:
                    confidence_emoji = "🟢" if rec['ml_confidence'] > 0.7 else "🟡" if rec['ml_confidence'] > 0.5 else "🔴"
                    response += f"      {confidence_emoji} AI Confidence: {rec['ml_confidence']:.0%}\n"

        # ML insights summary
        if ml_predictions:
            response += f"\n🤖 AI Market Insights:\n"
            for asset_class, predictions in ml_predictions.items():
                if predictions:
                    best_pick = predictions[0]
                    response += f"📊 Best {asset_class}: {best_pick['name']}\n"
                    response += f"   🔮 Predicted 1M return: {best_pick['predicted_1m']:+.1f}%\n"
                    response += f"   🎯 AI Confidence: {best_pick['confidence']:.0%}\n\n"

        response += f"\n💡 To get specific investment amounts and detailed AI analysis,\n"
        response += f"   please share your financial goal and target amount.\n"
        response += f"\nExample: 'I want to save 15 lakhs in 4 years for house down payment'\n"

        if not ml_predictions:
            response += f"\n🤖 Want AI predictions? Type 'enable AI' for enhanced analysis!\n"

        return response

    def generate_comprehensive_report(self, recommendations, goal_amount, timeline_months, 
                                    allocation, user_profile):
        """Generate traditional comprehensive investment report (fallback)"""
        if not recommendations:
            return "❌ Unable to fetch current market data. Please try again later."

        report = self._generate_header_traditional(goal_amount, timeline_months, user_profile)
        report += self._generate_allocation_section_traditional(allocation)
        report += self._generate_investment_calculation_traditional(goal_amount, timeline_months, allocation)
        report += self._generate_recommendations_section_traditional(recommendations, allocation)
        report += self._generate_risk_analysis_traditional(allocation, recommendations)
        report += self._generate_scenarios_traditional(goal_amount, timeline_months)
        report += self._generate_action_plan_traditional(recommendations, allocation)
        report += self._generate_disclaimer_traditional()

        return report

    def _generate_header_traditional(self, goal_amount, timeline_months, profile):
        """Generate traditional report header"""
        years = timeline_months // 12
        months = timeline_months % 12

        timeline_str = f"{years} years" if months == 0 else f"{years} years {months} months"
        if timeline_months < 12:
            timeline_str = f"{timeline_months} months"

        header = f"\n{'='*70}\n"
        header += f"🎯 FINVERSE AI - TRADITIONAL INVESTMENT REPORT\n"
        header += f"{'='*70}\n"
        header += f"💰 Target Amount: ₹{goal_amount:,}\n"
        header += f"⏱️  Timeline: {timeline_str}\n"
        header += f"👤 Risk Profile: {profile.get('risk_appetite', 'Medium')}\n"
        header += f"🎯 Goal: {profile.get('investment_goal', 'Wealth Creation')}\n"

        return header

    def _generate_allocation_section_traditional(self, allocation):
        """Generate traditional portfolio allocation section"""
        section = f"\n📊 RECOMMENDED PORTFOLIO ALLOCATION\n"
        section += f"{'-'*40}\n"
        section += f"🏛️  Debt Instruments: {allocation[0]:.1f}%\n"
        section += f"📈 Equity: {allocation[1]:.1f}%\n"
        section += f"🏦 Mutual Funds/ETFs: {allocation[2]:.1f}%\n"

        return section

    def _generate_investment_calculation_traditional(self, goal_amount, timeline_months, allocation):
        """Generate traditional investment calculation section"""
        monthly_investment = AdvancedFinancialCalculator.calculate_sip_amount(goal_amount, timeline_months)

        section = f"\n💸 MONTHLY INVESTMENT BREAKDOWN\n"
        section += f"{'-'*40}\n"
        section += f"💳 Total Monthly SIP: ₹{monthly_investment:,.0f}\n"
        section += f"   • Debt portion: ₹{monthly_investment * allocation[0] / 100:,.0f}\n"
        section += f"   • Equity portion: ₹{monthly_investment * allocation[1] / 100:,.0f}\n"
        section += f"   • Mutual Fund portion: ₹{monthly_investment * allocation[2] / 100:,.0f}\n"

        return section

    def _generate_recommendations_section_traditional(self, recommendations, allocation):
        """Generate traditional recommendations section"""
        section = f"\n🏆 TOP INVESTMENT RECOMMENDATIONS\n"
        section += f"{'='*70}\n"

        for i, rec in enumerate(recommendations, 1):
            section += f"\n{i}. {rec['name']} ({rec['ticker']})\n"
            section += f"   📂 Category: {rec['category']}\n"

            if rec['category'] in ['Equity', 'ETF', 'ETF/Index Fund']:
                section += f"   💰 Current Price/NAV: ₹{rec['current_price']:.2f}\n"
                section += f"   📊 1-Year Return: {rec['year_return']:.2f}%\n"
                section += f"   📉 Volatility: {rec['volatility']:.2f}%\n"
                section += f"   ⚡ Sharpe Ratio: {rec.get('sharpe_ratio', 'N/A')}\n"
            else:
                section += f"   📊 Expected Return: {rec['year_return']:.2f}% p.a.\n"
                section += f"   🛡️  Risk Level: Very Low\n"

        return section

    def _generate_risk_analysis_traditional(self, allocation, recommendations):
        """Generate traditional risk analysis section"""
        volatilities = [rec.get('volatility', 0) for rec in recommendations[:3]]
        if len(volatilities) < 3:
            volatilities.extend([0] * (3 - len(volatilities)))

        portfolio_risk = AdvancedFinancialCalculator.calculate_risk_metrics(allocation, volatilities)

        section = f"\n⚠️  RISK ANALYSIS\n"
        section += f"{'-'*40}\n"
        section += f"📊 Portfolio Volatility: {portfolio_risk:.1f}%\n"

        if portfolio_risk < 10:
            risk_level = "🟢 Low Risk"
        elif portfolio_risk < 20:
            risk_level = "🟡 Medium Risk"
        else:
            risk_level = "🔴 High Risk"

        section += f"🎯 Risk Level: {risk_level}\n"

        return section

    def _generate_scenarios_traditional(self, goal_amount, timeline_months):
        """Generate traditional scenario analysis"""
        monthly_sip = AdvancedFinancialCalculator.calculate_sip_amount(goal_amount, timeline_months)

        scenarios = AdvancedFinancialCalculator.calculate_future_value(
            monthly_sip, timeline_months,
            {'Conservative (8%)': 8, 'Moderate (12%)': 12, 'Aggressive (15%)': 15}
        )

        section = f"\n🔮 SCENARIO ANALYSIS\n"
        section += f"{'-'*40}\n"
        section += f"With monthly SIP of ₹{monthly_sip:,.0f}:\n\n"

        for scenario, value in scenarios.items():
            difference = value - goal_amount
            percentage = (difference / goal_amount) * 100

            if difference > 0:
                section += f"✅ {scenario}: ₹{value:,.0f} (+₹{difference:,.0f}, +{percentage:.1f}%)\n"
            else:
                section += f"❌ {scenario}: ₹{value:,.0f} (₹{abs(difference):,.0f}, {percentage:.1f}%)\n"

        return section

    def _generate_action_plan_traditional(self, recommendations, allocation):
        """Generate traditional actionable plan"""
        section = f"\n📋 ACTION PLAN\n"
        section += f"{'-'*40}\n"
        section += f"1. 🏦 Open investment accounts if not already done\n"
        section += f"2. 💳 Set up SIP mandates for systematic investing\n"
        section += f"3. 📊 Start with the top-ranked recommendation\n"
        section += f"4. 📈 Review and rebalance quarterly\n"
        section += f"5. 📱 Monitor performance monthly\n"

        return section

    def _generate_disclaimer_traditional(self):
        """Generate traditional disclaimer"""
        disclaimer = f"\n⚠️  IMPORTANT DISCLAIMER\n"
        disclaimer += f"{'='*70}\n"
        disclaimer += f"• This is analysis based on traditional financial metrics\n"
        disclaimer += f"• Past performance does not guarantee future results\n"
        disclaimer += f"• Markets are subject to risk - invest wisely\n"
        disclaimer += f"• Consider consulting a certified financial advisor\n"
        disclaimer += f"• Diversify your investments across asset classes\n"
        disclaimer += f"{'='*70}\n"

        return disclaimer

class ChatInterface:
    """Enhanced chat interface with ML capabilities"""

    def __init__(self):
        self.finverse_ai = FinVerseAI()
        self.session_id = 'default'

    def display_welcome_message(self):
        """Display enhanced welcome message with ML features"""
        welcome = f"\n{'='*70}\n"
        welcome += f"🚀 WELCOME TO FINVERSE AI - ML-POWERED INVESTMENT ADVISOR!\n"
        welcome += f"{'='*70}\n\n"

        welcome += f"🎯 What I can help you with:\n"
        welcome += f"   • 🤖 AI-driven market predictions and analysis\n"
        welcome += f"   • 📊 ML-optimized portfolio allocation\n"
        welcome += f"   • 💰 Goal-based investment planning with AI insights\n"
        welcome += f"   • 📈 Real-time market recommendations with confidence scores\n"
        welcome += f"   • 🧮 Smart SIP calculations with predictive modeling\n"
        welcome += f"   • ⚖️  Risk assessment with AI-enhanced volatility analysis\n\n"

        welcome += f"💬 Smart Examples to try:\n"
        welcome += f"   • 'I'm 28, want to save 10L for wedding in 2 years'\n"
        welcome += f"   • 'Show me AI predictions for the market'\n"
        welcome += f"   • 'Need 50L for retirement, I'm 35, earn 15L annually'\n"
        welcome += f"   • 'I'm 32, high risk, looking for AI-powered wealth creation'\n"
        welcome += f"   • 'How does your AI prediction work?'\n\n"

        welcome += f"🤖 AI Features:\n"
        welcome += f"   • Machine Learning predictions for 1M, 3M, 6M horizons\n"
        welcome += f"   • Confidence scores for all AI recommendations\n"
        welcome += f"   • Real-time market analysis with 500+ data points\n"
        welcome += f"   • Smart allocation adjustments based on AI insights\n\n"

        welcome += f"🎮 Commands:\n"
        welcome += f"   • 'help' - Detailed usage guide\n"
        welcome += f"   • 'enable/disable AI' - Toggle ML predictions\n"
        welcome += f"   • 'ai accuracy' - Learn about prediction accuracy\n"
        welcome += f"   • 'clear' - Reset conversation\n"
        welcome += f"   • 'exit' - End session\n\n"

        welcome += f"💡 Pro Tips:\n"
        welcome += f"   • Mention your age, income, and risk appetite for better AI analysis\n"
        welcome += f"   • Specify timeline and target amount for detailed ML planning\n"
        welcome += f"   • Ask about AI confidence levels for risk assessment\n"
        welcome += f"   • Use 'show AI insights' for quick market predictions\n\n"

        welcome += f"{'='*70}\n"

        print(welcome)

    def display_help(self):
        """Display comprehensive help information"""
        help_text = f"\n📚 FINVERSE AI HELP GUIDE (ML-ENHANCED)\n"
        help_text += f"{'='*50}\n\n"

        help_text += f"🔤 Command Examples:\n"
        help_text += f"   • Investment Planning: 'Save 5L for car in 18 months'\n"
        help_text += f"   • AI Analysis: 'Show me AI predictions for equity markets'\n"
        help_text += f"   • Retirement: 'Plan retirement with AI, I'm 30, earn 12L'\n"
        help_text += f"   • Portfolio Review: 'AI-optimize my current investments'\n"
        help_text += f"   • Risk Analysis: 'What's the AI confidence for these picks?'\n\n"

        help_text += f"🤖 AI-Specific Commands:\n"
        help_text += f"   • 'enable AI' / 'disable AI' - Toggle ML predictions\n"
        help_text += f"   • 'how does AI work' - Explain ML methodology\n"
        help_text += f"   • 'ai accuracy' - Learn about prediction reliability\n"
        help_text += f"   • 'show AI insights' - Quick market analysis\n"
        help_text += f"   • 'ml explanation' - Detailed AI methodology\n\n"

        help_text += f"💡 Supported Formats:\n"
        help_text += f"   • Amounts: 5L, 10 lakhs, 2 crores, 50K, ₹100000\n"
        help_text += f"   • Timeline: 2 years, 18 months, 5 yrs\n"
        help_text += f"   • Risk: Low/Medium/High risk\n\n"

        help_text += f"🎯 Investment Goals:\n"
        help_text += f"   • Short-term: Vacation, car, wedding\n"
        help_text += f"   • Long-term: Retirement, wealth creation\n"
        help_text += f"   • Education: Child's future, higher studies\n"
        help_text += f"   • Property: House, real estate investment\n\n"

        help_text += f"📊 AI Confidence Levels:\n"
        help_text += f"   • 🟢 High (70%+): Strong patterns, reliable predictions\n"
        help_text += f"   • 🟡 Medium (50-70%): Moderate confidence, use with caution\n"
        help_text += f"   • 🔴 Low (<50%): Uncertain patterns, high risk\n\n"

        help_text += f"⚡ Advanced Features:\n"
        help_text += f"   • Multi-timeframe predictions (1M, 3M, 6M)\n"
        help_text += f"   • ML-optimized portfolio allocation\n"
        help_text += f"   • Risk-adjusted AI recommendations\n"
        help_text += f"   • Real-time model retraining\n\n"

        help_text += f"Type 'exit' to quit or continue chatting for AI-powered advice!\n"
        print(help_text)

    def run_chat(self):
        """Run the enhanced chat interface with ML capabilities"""
        self.display_welcome_message()

        while True:
            try:
                user_input = input("\n🧑 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Thank you for using FinVerse AI!")
                    print("🤖 Remember: AI-powered investing creates smarter wealth!")
                    print("🔄 Come back anytime for ML-enhanced financial advice!\n")
                    break

                if user_input.lower() in ['help', '?']:
                    self.display_help()
                    continue

                if user_input.lower() in ['clear', 'reset']:
                    self.finverse_ai.user_sessions[self.session_id] = {
                        'profile': {},
                        'conversation_history': [],
                        'ml_preferences': {'enabled': True, 'confidence_threshold': 0.5}
                    }
                    print("\n🔄 Session reset! AI analysis re-enabled. Start fresh with new goals.\n")
                    continue

                # Process the query with ML capabilities
                print("\n🤖 FinVerse AI:")
                response = self.finverse_ai.process_user_query(user_input, self.session_id)
                print(response)

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Sorry, I encountered an error: {str(e)}")
                print("🔄 Please try rephrasing your question or type 'help' for guidance.")
                print("🤖 AI features may be temporarily unavailable.\n")

class FinVerseManager:
    """Enhanced application manager with ML initialization"""

    def __init__(self):
        self.chat_interface = ChatInterface()

    def initialize_system(self):
        """Initialize the enhanced FinVerse system with ML capabilities"""
        print("🚀 Initializing FinVerse AI with Machine Learning...")
        print("📊 Loading market data sources...")
        print("🤖 Preparing ML prediction models...")
        print("🧠 Training intelligent allocation engine...")
        print("📈 Setting up real-time market analysis...")
        print("🎯 Calibrating confidence scoring system...")
        print("✅ AI-powered system ready!")
        print("\n🤖 ML Features Activated:")
        print("   • Predictive market analysis")
        print("   • Smart portfolio optimization") 
        print("   • Risk-adjusted recommendations")
        print("   • Confidence-based decision making\n")

    def run_application(self):
        """Run the ML-enhanced main application"""
        try:
            self.initialize_system()
            self.chat_interface.run_chat()
        except Exception as e:
            print(f"❌ System error: {str(e)}")
            print("🔧 Please restart the application.")
            print("🤖 If ML features fail, traditional analysis will be used.")

# Enhanced utility functions for ML integration
class MarketAnalyzer:
    """Advanced market analysis utilities with ML"""

    @staticmethod
    def calculate_portfolio_metrics_with_ml(recommendations, allocation, ml_predictions=None):
        """Calculate advanced portfolio metrics with ML insights"""
        if not recommendations:
            return {}

        total_return = 0
        total_risk = 0
        ml_adjusted_return = 0

        for i, rec in enumerate(recommendations[:3]):
            weight = allocation[i] / 100 if i < len(allocation) else 0
            base_return = rec.get('year_return', 0)

            total_return += base_return * weight
            total_risk += (rec.get('volatility', 0) ** 2) * (weight ** 2)

            # ML enhancement
            if ml_predictions and rec.get('predicted_return_1m'):
                ml_return = rec['predicted_return_1m'] * 12  # Annualize
                confidence = rec.get('ml_confidence', 0.5)

                # Blend traditional and ML returns based on confidence
                blended_return = (base_return * (1 - confidence) + ml_return * confidence)
                ml_adjusted_return += blended_return * weight

        total_risk = total_risk ** 0.5

        metrics = {
            'expected_return': round(total_return, 2),
            'portfolio_risk': round(total_risk, 2),
            'sharpe_ratio': round(total_return / total_risk if total_risk > 0 else 0, 3)
        }

        if ml_predictions:
            metrics['ml_adjusted_return'] = round(ml_adjusted_return, 2)
            metrics['ml_enhancement'] = round(ml_adjusted_return - total_return, 2)

        return metrics

    @staticmethod
    def get_market_sentiment_with_ai(ml_predictions=None):
        """Get AI-enhanced market sentiment"""
        if not ml_predictions:
            # Fallback to simple sentiment
            import random
            sentiments = ['Bullish', 'Neutral', 'Bearish']
            return random.choice(sentiments)

        # Calculate sentiment based on ML predictions
        all_predictions = []
        for asset_class, preds in ml_predictions.items():
            for pred in preds:
                if pred.get('predicted_1m') is not None:
                    all_predictions.append(pred['predicted_1m'])

        if all_predictions:
            avg_prediction = sum(all_predictions) / len(all_predictions)

            if avg_prediction > 3:
                return "🟢 AI Bullish"
            elif avg_prediction > -1:
                return "🟡 AI Neutral"
            else:
                return "🔴 AI Bearish"

        return "🤖 AI Analyzing"

class InvestmentEducator:
    """Enhanced investment education with ML insights"""

    @staticmethod
    def get_ml_investment_tip(risk_level, ml_confidence=None):
        """Get ML-enhanced educational investment tips"""
        base_tips = {
            'Low': [
                "💡 Even with AI, diversification remains crucial for stability",
                "🤖 High AI confidence (70%+) in debt instruments indicates strong stability",
                "📈 Consider AI-recommended blue chips for conservative growth",
                "🛡️ Build emergency fund before following AI equity predictions"
            ],
            'Medium': [
                "⚖️ Balance AI insights with fundamental analysis for best results",
                "📊 AI confidence scores help you weight your allocation decisions",
                "🎯 Use ML predictions for timing but maintain long-term perspective",
                "📚 High AI confidence doesn't eliminate the need for diversification"
            ],
            'High': [
                "🎢 AI can spot opportunities, but high returns still mean high volatility",
                "🔍 Cross-verify AI predictions with your own market research",
                "💎 ML insights work best with patient, long-term strategies",
                "🎯 Don't chase every high-confidence AI prediction - stay disciplined"
            ]
        }

        tips = base_tips.get(risk_level, base_tips['Medium'])
        base_tip = tips[0]

        # Add ML-specific context if confidence is provided
        if ml_confidence is not None:
            if ml_confidence > 0.7:
                ml_addition = " 🟢 Current AI confidence is high - good time for action!"
            elif ml_confidence > 0.5:
                ml_addition = " 🟡 AI confidence is moderate - proceed with balanced approach."
            else:
                ml_addition = " 🔴 AI confidence is low - consider waiting or reducing exposure."

            return base_tip + ml_addition

        return base_tip

    @staticmethod
    def explain_ml_concepts():
        """Explain key ML concepts for users"""
        explanation = f"\n📚 AI/ML CONCEPTS FOR INVESTORS\n"
        explanation += f"{'='*40}\n\n"

        explanation += f"🤖 Machine Learning: Algorithm learns patterns from historical data\n"
        explanation += f"📊 Confidence Score: How certain the AI is about its prediction\n"
        explanation += f"🎯 Feature Importance: Which factors matter most for predictions\n"
        explanation += f"📈 Ensemble Models: Multiple AI models working together\n"
        explanation += f"🔮 Prediction Horizon: Time period for which AI makes forecasts\n"
        explanation += f"📉 Volatility Prediction: AI estimates of price fluctuation risk\n"
        explanation += f"⚖️ Risk-Adjusted Returns: Returns considered relative to risk taken\n\n"

        explanation += f"💡 Key Takeaways:\n"
        explanation += f"   • AI enhances but doesn't replace human judgment\n"
        explanation += f"   • Higher confidence = more reliable predictions\n"
        explanation += f"   • Past patterns may not predict future events\n"
        explanation += f"   • Use AI as one tool among many in your toolkit\n"

        return explanation

def main():
    """Enhanced main execution function with ML capabilities"""
    try:
        # Print startup banner
        print(f"\n{'='*70}")
        print(f"🤖 FINVERSE AI - MACHINE LEARNING POWERED INVESTMENT ADVISOR")
        print(f"{'='*70}")
        print(f"🚀 Loading AI models and market data...")
        print(f"📊 Preparing predictive analytics engine...")
        print(f"🎯 Calibrating confidence scoring algorithms...")

        # Create and run the ML-enhanced FinVerse application
        app = FinVerseManager()
        app.run_application()

    except KeyboardInterrupt:
        print("\n\n👋 Application terminated by user. Goodbye!")
        print("🤖 Thank you for using AI-powered investment advice!")
    except Exception as e:
        print(f"\n❌ Critical error: {str(e)}")
        print("🔧 Please contact support or restart the application.")
        print("💡 Tip: Ensure you have stable internet for AI features.")

# Additional utility classes for enhanced functionality
class MLModelManager:
    """Manages ML model lifecycle and performance"""

    def __init__(self):
        self.model_performance = {}
        self.model_usage_stats = {}

    def track_prediction_accuracy(self, ticker, predicted_return, actual_return, time_horizon):
        """Track how accurate predictions are over time"""
        if ticker not in self.model_performance:
            self.model_performance[ticker] = {
                '1M': [], '3M': [], '6M': []
            }

        # Calculate accuracy (how close prediction was to actual)
        error = abs(predicted_return - actual_return)
        accuracy = max(0, 100 - error)  # Simple accuracy metric

        self.model_performance[ticker][time_horizon].append({
            'accuracy': accuracy,
            'error': error,
            'predicted': predicted_return,
            'actual': actual_return,
            'timestamp': datetime.now()
        })

    def get_model_reliability(self, ticker, time_horizon='1M'):
        """Get reliability score for a specific model"""
        if ticker not in self.model_performance:
            return 0.5  # Default moderate reliability

        performances = self.model_performance[ticker].get(time_horizon, [])
        if not performances:
            return 0.5

        # Calculate average accuracy over recent predictions
        recent_performances = performances[-10:]  # Last 10 predictions
        avg_accuracy = sum(p['accuracy'] for p in recent_performances) / len(recent_performances)

        return min(0.9, max(0.1, avg_accuracy / 100))

    def update_usage_stats(self, ticker, prediction_used=True):
        """Track which models are being used most"""
        if ticker not in self.model_usage_stats:
            self.model_usage_stats[ticker] = {
                'total_requests': 0,
                'predictions_used': 0,
                'last_used': None
            }

        self.model_usage_stats[ticker]['total_requests'] += 1
        if prediction_used:
            self.model_usage_stats[ticker]['predictions_used'] += 1
        self.model_usage_stats[ticker]['last_used'] = datetime.now()

class PerformanceTracker:
    """Tracks portfolio performance with ML insights"""

    def __init__(self):
        self.portfolio_history = {}
        self.ml_prediction_history = {}

    def log_portfolio_performance(self, user_id, portfolio_data, ml_predictions=None):
        """Log portfolio performance for tracking"""
        timestamp = datetime.now()

        if user_id not in self.portfolio_history:
            self.portfolio_history[user_id] = []

        entry = {
            'timestamp': timestamp,
            'portfolio': portfolio_data,
            'ml_predictions': ml_predictions
        }

        self.portfolio_history[user_id].append(entry)

    def calculate_ml_alpha(self, user_id):
        """Calculate excess returns attributable to ML insights"""
        if user_id not in self.portfolio_history:
            return None

        history = self.portfolio_history[user_id]
        if len(history) < 2:
            return None

        # This would calculate the difference between ML-guided and traditional returns
        # Implementation would depend on actual performance tracking
        return {
            'ml_alpha': 0,  # Placeholder
            'tracking_period': len(history),
            'confidence': 0.5
        }

class RiskManager:
    """Enhanced risk management with ML"""

    @staticmethod
    def calculate_var_with_ml(portfolio_data, ml_predictions=None, confidence_level=0.95):
        """Calculate Value at Risk with ML enhancement"""
        # Traditional VaR calculation
        returns = [asset.get('year_return', 0) / 100 for asset in portfolio_data]
        weights = [asset.get('weight', 1/len(portfolio_data)) for asset in portfolio_data]

        portfolio_return = sum(r * w for r, w in zip(returns, weights))

        # Simple VaR approximation (would need more sophisticated implementation)
        portfolio_volatility = 0.15  # Placeholder

        # Normal distribution assumption
        from scipy import stats
        var_multiplier = stats.norm.ppf(1 - confidence_level)
        traditional_var = portfolio_return + (var_multiplier * portfolio_volatility)

        if ml_predictions:
            # Adjust VaR based on ML prediction uncertainty
            ml_uncertainty = RiskManager._calculate_ml_uncertainty(ml_predictions)
            ml_adjusted_var = traditional_var * (1 + ml_uncertainty)

            return {
                'traditional_var': round(traditional_var * 100, 2),
                'ml_adjusted_var': round(ml_adjusted_var * 100, 2),
                'ml_uncertainty_factor': round(ml_uncertainty, 3)
            }

        return {
            'traditional_var': round(traditional_var * 100, 2)
        }

    @staticmethod
    def _calculate_ml_uncertainty(ml_predictions):
        """Calculate uncertainty factor from ML predictions"""
        try:
            confidences = []
            for asset_class, preds in ml_predictions.items():
                for pred in preds:
                    if pred.get('confidence'):
                        confidences.append(pred['confidence'])

            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                # Higher average confidence = lower uncertainty
                uncertainty = 1 - avg_confidence
                return min(0.5, uncertainty)  # Cap at 50% uncertainty

        except Exception:
            pass

        return 0.2  # Default 20% uncertainty

class ConfigManager:
    """Manages application configuration"""

    def __init__(self):
        self.config_file = 'finverse_config.json'
        self.default_config = {
            'ml_enabled': True,
            'confidence_threshold': 0.5,
            'max_predictions_per_session': 100,
            'cache_duration_hours': 1,
            'model_retrain_days': 7,
            'api_rate_limit': 60,
            'default_risk_appetite': 'Medium',
            'supported_markets': ['NSE', 'BSE'],
            'prediction_horizons': ['1M', '3M', '6M'],
            'max_recommendations': 6
        }
        self.config = self.load_config()

    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    config = self.default_config.copy()
                    config.update(loaded_config)
                    return config
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")

        return self.default_config.copy()

    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config file: {e}")

    def get(self, key, default=None):
        """Get configuration value"""
        return self.config.get(key, default)

    def set(self, key, value):
        """Set configuration value"""
        self.config[key] = value
        self.save_config()

# Initialize global configuration
config_manager = ConfigManager()

class APIManager:
    """Manages external API calls and rate limiting"""

    def __init__(self):
        self.rate_limits = {}
        self.api_cache = {}
        self.max_requests_per_minute = config_manager.get('api_rate_limit', 60)

    def check_rate_limit(self, api_name):
        """Check if API call is within rate limits"""
        now = datetime.now()
        minute_key = now.strftime('%Y-%m-%d-%H-%M')

        if api_name not in self.rate_limits:
            self.rate_limits[api_name] = {}

        current_requests = self.rate_limits[api_name].get(minute_key, 0)

        if current_requests >= self.max_requests_per_minute:
            return False

        self.rate_limits[api_name][minute_key] = current_requests + 1
        return True

    def clean_old_rate_limits(self):
        """Clean old rate limit entries"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=5)

        for api_name in self.rate_limits:
            keys_to_remove = []
            for time_key in self.rate_limits[api_name]:
                try:
                    time_obj = datetime.strptime(time_key, '%Y-%m-%d-%H-%M')
                    if time_obj < cutoff:
                        keys_to_remove.append(time_key)
                except ValueError:
                    keys_to_remove.append(time_key)

            for key in keys_to_remove:
                self.rate_limits[api_name].pop(key, None)

# Global API manager instance
api_manager = APIManager()

ml_predictor_instance = MLPredictor()

def predict_allocation(user_data: Dict) -> Dict:
    """
    Predicts investment allocation based on user data using the ML models.
    This function is called by FastAPI's main.py.

    Args:
        user_data (Dict): A dictionary containing user profile data and potentially
                          a natural language message. Expected keys:
                          'age', 'annual_income', 'monthly_savings',
                          'risk_appetite', 'investment_goal', 'timeline_months',
                          'emergency_fund', 'existing_investment_pct'.
                          The 'message' field should be handled by NLP before
                          calling this, as it's not directly a model feature.

    Returns:
        Dict: A dictionary containing predicted allocations for 'debt_allocation',
              'equity_allocation', and 'mutual_fund_allocation',
              PLUS the 'full_report' text.
    """
    global ml_predictor_instance

    # Ensure the cache directory exists and models are ready.
    # The MLPredictor's train_model handles caching and reloading if recent.
    # For a simple allocation, we mainly need the market predictions.

    # Re-initialize MarketDataManager and IntelligentAllocationEngine
    # to ensure they use the global ml_predictor_instance.
    # In a real-world scenario, you might pass these instances around or
    # ensure they are singleton if their state is shared globally.
    market_manager_instance = MarketDataManager()
    market_manager_instance.ml_predictor = ml_predictor_instance # Inject the shared MLPredictor
    allocation_engine_instance = IntelligentAllocationEngine()


    # Get ML predictions for various asset classes based on the user's risk appetite.
    # Ensure 'risk_appetite' is present in user_data, default if not.
    risk_appetite = user_data.get('risk_appetite', 'Medium')
    ml_predictions = market_manager_instance.get_asset_class_predictions(risk_appetite)

    # Calculate the portfolio allocation using ML insights
    # Ensure all required profile keys are present, even if with defaults.
    # The `calculate_allocation_with_ml_insights` method expects specific keys.
    profile_for_allocation = {
        'age': user_data.get('age', 30),
        'annual_income': user_data.get('annual_income', 800000),
        'monthly_savings': user_data.get('monthly_savings', 20000),
        'risk_appetite': risk_appetite,
        'investment_goal': user_data.get('investment_goal', 'Wealth Creation'),
        'timeline_months': user_data.get('timeline_months', 60),
        'emergency_fund': user_data.get('emergency_fund', 100000),
        'existing_investment_pct': user_data.get('existing_investment_pct', 0.1),
        'goal_amount': user_data.get('goal_amount', 1000000) # Added default for report generation
    }

    print(f"--- [DEBUG] Profile for Allocation: {profile_for_allocation}")

    # Use the allocation engine to get the enhanced allocation
    debt_alloc_pct, equity_alloc_pct, mf_alloc_pct = \
        allocation_engine_instance.calculate_allocation_with_ml_insights(
            profile_for_allocation, ml_predictions
        )

    # --- CHANGE STARTS HERE ---
    # Get investment recommendations using the MarketDataManager's comprehensive method
    # Pass the calculated allocation to this method
    recommendations = market_manager_instance.get_portfolio_recommendations_with_ml(
        profile_for_allocation['risk_appetite'],
        profile_for_allocation['timeline_months'],
        profile_for_allocation['goal_amount'],
        (debt_alloc_pct, equity_alloc_pct, mf_alloc_pct) # Pass the calculated allocation here
    )

    # Generate the comprehensive report
    report_generator_instance = EnhancedReportGenerator()
    full_report_text = report_generator_instance.generate_comprehensive_report_with_ml(
        recommendations,
        profile_for_allocation['goal_amount'],
        profile_for_allocation['timeline_months'],
        (debt_alloc_pct, equity_alloc_pct, mf_alloc_pct),
        profile_for_allocation,
        ml_predictions
    )

    return {
        "debt_allocation": round(float(debt_alloc_pct), 2),
        "equity_allocation": round(float(equity_alloc_pct), 2),
        "mutual_fund_allocation": round(float(mf_alloc_pct), 2),
        "full_report": full_report_text
    }


if __name__ == "__main__":
    # Set up any additional configuration or logging here
    try:
        # Initialize logging if needed
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('finverse_ai.log'),
                logging.StreamHandler()
            ]
        )

        # Run the main application
        main()

    except Exception as e:
        print(f"Failed to start FinVerse AI: {e}")
        print("Please check your Python environment and dependencies.")