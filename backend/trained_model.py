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

class MarketDataManager:
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

        cache_time = datetime.fromisoformat(self.cache[ticker]['timestamp'])
        return (datetime.now() - cache_time).seconds < 3600

    def fetch_instrument_data(self, ticker, period='1y'):
        """Fetch data for a single instrument with caching"""
        if self.is_cache_valid(ticker):
            return self.cache[ticker]['data']

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

                # Calculate Sharpe ratio (assuming risk-free rate of 7%)
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

            # Cache the data
            self.cache[ticker] = {
                'data': data,
                'timestamp': datetime.now().isoformat()
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

    def get_filtered_recommendations(self, instruments, risk_appetite, min_return=None):
        """Get filtered recommendations based on risk appetite"""
        recommendations = []

        for ticker in instruments:
            data = self.fetch_instrument_data(ticker)
            if data and self._meets_criteria(data, risk_appetite, min_return):
                recommendations.append(data)

        # Sort by risk-adjusted returns (Sharpe ratio)
        recommendations.sort(key=lambda x: x.get('sharpe_ratio', 0), reverse=True)
        return recommendations

    def _meets_criteria(self, data, risk_appetite, min_return):
        """Check if instrument meets investment criteria"""
        if data['year_return'] < -20:  # Avoid severely underperforming assets
            return False

        if risk_appetite == 'Low':
            return data['volatility'] < 20 and data['year_return'] > 0
        elif risk_appetite == 'Medium':
            return data['volatility'] < 35
        else:  # High risk
            return True

    def get_portfolio_recommendations(self, risk_appetite, timeline_months, amount, allocation):
        """Get comprehensive portfolio recommendations"""
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

            # Debt ETFs
            debt_etfs = self.get_filtered_recommendations(
                self.debt_instruments, 'Low'
            )
            recommendations['debt'].extend(debt_etfs[:2])

        # Equity recommendations
        if allocation[1] > 10 and timeline_months > 12:
            equity_recs = self.get_filtered_recommendations(
                self.nifty50_stocks[:20], risk_appetite, min_return=5
            )
            recommendations['equity'].extend(equity_recs[:3])

        # Mutual Fund/ETF recommendations
        if allocation[2] > 10:
            etf_recs = self.get_filtered_recommendations(
                self.index_etfs + self.sector_etfs, risk_appetite
            )
            for etf in etf_recs[:3]:
                etf['category'] = 'ETF/Index Fund'
                recommendations['mutual_fund'].append(etf)

        return self._compile_final_recommendations(recommendations, allocation)

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
            'category': 'Government Scheme'
        })

        if timeline_months >= 60:  # 5 years minimum
            schemes.append({
                'ticker': 'PPF',
                'name': 'Public Provident Fund',
                'current_price': 0,
                'year_return': 7.1,
                'volatility': 0,
                'sharpe_ratio': 0.4,
                'category': 'Government Scheme'
            })

        if timeline_months >= 36:  # 3 years minimum
            schemes.append({
                'ticker': 'NSC',
                'name': 'National Savings Certificate',
                'current_price': 0,
                'year_return': 7.7,
                'volatility': 0,
                'sharpe_ratio': 0.45,
                'category': 'Government Scheme'
            })

        return schemes

    def _compile_final_recommendations(self, recommendations, allocation):
        """Compile final recommendations based on allocation"""
        final_recs = []

        # Sort categories by allocation percentage
        categories = [
            ('debt', allocation[0], recommendations['debt']),
            ('equity', allocation[1], recommendations['equity']),
            ('mutual_fund', allocation[2], recommendations['mutual_fund'])
        ]
        categories.sort(key=lambda x: x[1], reverse=True)

        # Add recommendations from each significant category
        for category_name, alloc_pct, category_recs in categories:
            if alloc_pct > 10 and category_recs:
                final_recs.append(category_recs[0])

        # Fill remaining slots with best performers
        for category_name, alloc_pct, category_recs in categories:
            for rec in category_recs[1:]:
                if len(final_recs) < 5:
                    final_recs.append(rec)

        self.save_cache()  # Save updated cache
        return final_recs[:5]

class IntelligentAllocationEngine:
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

class AdvancedFinancialCalculator:
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
    def calculate_future_value(monthly_investment, timeline_months, returns):
        """Calculate future value with different return scenarios"""
        scenarios = {}
        for scenario, annual_return in returns.items():
            monthly_return = annual_return / 100 / 12
            if monthly_return == 0:
                fv = monthly_investment * timeline_months
            else:
                fv = monthly_investment * (((1 + monthly_return) ** timeline_months - 1) / monthly_return)
            scenarios[scenario] = fv
        return scenarios

    @staticmethod
    def calculate_risk_metrics(allocation, volatilities):
        """Calculate portfolio risk metrics"""
        portfolio_volatility = np.sqrt(
            sum((allocation[i] / 100) ** 2 * (volatilities[i] / 100) ** 2 for i in range(len(allocation)))
        ) * 100
        return portfolio_volatility

class EnhancedInputParser:
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

    def parse_user_input(self, message):
        """Parse user input for financial parameters"""
        extracted = {}
        message_lower = message.lower()

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
    def __init__(self):
        self.calculator = AdvancedFinancialCalculator()

    def generate_comprehensive_report(self, recommendations, goal_amount, timeline_months, 
                                    allocation, user_profile):
        """Generate a comprehensive investment report"""
        if not recommendations:
            return "❌ Unable to fetch current market data. Please try again later."

        report = self._generate_header(goal_amount, timeline_months, user_profile)
        report += self._generate_allocation_section(allocation)
        report += self._generate_investment_calculation(goal_amount, timeline_months, allocation)
        report += self._generate_recommendations_section(recommendations, allocation)
        report += self._generate_risk_analysis(allocation, recommendations)
        report += self._generate_scenarios(goal_amount, timeline_months)
        report += self._generate_action_plan(recommendations, allocation)
        report += self._generate_disclaimer()

        return report

    def _generate_header(self, goal_amount, timeline_months, profile):
        """Generate report header"""
        years = timeline_months // 12
        months = timeline_months % 12

        timeline_str = f"{years} years" if months == 0 else f"{years} years {months} months"
        if timeline_months < 12:
            timeline_str = f"{timeline_months} months"

        header = f"\n{'='*70}\n"
        header += f"🎯 FINVERSE AI - PERSONALIZED INVESTMENT REPORT\n"
        header += f"{'='*70}\n"
        header += f"💰 Target Amount: ₹{goal_amount:,}\n"
        header += f"⏱️  Timeline: {timeline_str}\n"
        header += f"👤 Risk Profile: {profile.get('risk_appetite', 'Medium')}\n"
        header += f"🎯 Goal: {profile.get('investment_goal', 'Wealth Creation')}\n"

        return header

    def _generate_allocation_section(self, allocation):
        """Generate portfolio allocation section"""
        section = f"\n📊 RECOMMENDED PORTFOLIO ALLOCATION\n"
        section += f"{'-'*40}\n"
        section += f"🏛️  Debt Instruments: {allocation[0]:.1f}%\n"
        section += f"📈 Equity: {allocation[1]:.1f}%\n"
        section += f"🏦 Mutual Funds/ETFs: {allocation[2]:.1f}%\n"

        return section

    def _generate_investment_calculation(self, goal_amount, timeline_months, allocation):
        """Generate investment calculation section"""
        monthly_investment = self.calculator.calculate_sip_amount(goal_amount, timeline_months)

        section = f"\n💸 MONTHLY INVESTMENT BREAKDOWN\n"
        section += f"{'-'*40}\n"
        section += f"💳 Total Monthly SIP: ₹{monthly_investment:,.0f}\n"
        section += f"   • Debt portion: ₹{monthly_investment * allocation[0] / 100:,.0f}\n"
        section += f"   • Equity portion: ₹{monthly_investment * allocation[1] / 100:,.0f}\n"
        section += f"   • Mutual Fund portion: ₹{monthly_investment * allocation[2] / 100:,.0f}\n"

        return section

    def _generate_recommendations_section(self, recommendations, allocation):
        """Generate detailed recommendations section"""
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

            # Investment suggestion based on allocation
            if rec['category'] == 'Equity' and allocation[1] > 20:
                section += f"   💡 Suggested Allocation: High (part of {allocation[1]:.1f}% equity)\n"
            elif rec['category'] in ['ETF', 'ETF/Index Fund'] and allocation[2] > 20:
                section += f"   💡 Suggested Allocation: Medium (part of {allocation[2]:.1f}% MF/ETF)\n"
            elif 'Debt' in rec['category'] and allocation[0] > 15:
                section += f"   💡 Suggested Allocation: Conservative (part of {allocation[0]:.1f}% debt)\n"

        return section

    def _generate_risk_analysis(self, allocation, recommendations):
        """Generate risk analysis section"""
        # Calculate portfolio volatility
        volatilities = [rec.get('volatility', 0) for rec in recommendations[:3]]
        if len(volatilities) < 3:
            volatilities.extend([0] * (3 - len(volatilities)))

        portfolio_risk = self.calculator.calculate_risk_metrics(allocation, volatilities)

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

    def _generate_scenarios(self, goal_amount, timeline_months):
        """Generate scenario analysis"""
        monthly_sip = self.calculator.calculate_sip_amount(goal_amount, timeline_months)

        scenarios = self.calculator.calculate_future_value(
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

    def _generate_action_plan(self, recommendations, allocation):
        """Generate actionable plan"""
        section = f"\n📋 ACTION PLAN\n"
        section += f"{'-'*40}\n"
        section += f"1. 🏦 Open investment accounts if not already done\n"
        section += f"2. 💳 Set up SIP mandates for systematic investing\n"
        section += f"3. 📊 Start with the top-ranked recommendation\n"
        section += f"4. 📈 Review and rebalance quarterly\n"
        section += f"5. 📱 Monitor performance monthly\n"

        return section

    def _generate_disclaimer(self):
        """Generate disclaimer"""
        disclaimer = f"\n⚠️  IMPORTANT DISCLAIMER\n"
        disclaimer += f"{'='*70}\n"
        disclaimer += f"• This is AI-generated advice based on market data analysis\n"
        disclaimer += f"• Past performance does not guarantee future results\n"
        disclaimer += f"• Markets are subject to risk - invest wisely\n"
        disclaimer += f"• Consider consulting a certified financial advisor\n"
        disclaimer += f"• Diversify your investments across asset classes\n"
        disclaimer += f"{'='*70}\n"

        return disclaimer

class FinVerseAI:
    def __init__(self):
        self.market_manager = MarketDataManager()
        self.allocation_engine = IntelligentAllocationEngine()
        self.input_parser = EnhancedInputParser()
        self.report_generator = EnhancedReportGenerator()
        self.user_sessions = {}

    def get_user_session(self, session_id='default'):
        """Get or create user session"""
        if session_id not in self.user_sessions:
            self.user_sessions[session_id] = {
                'profile': {},
                'conversation_history': []
            }
        return self.user_sessions[session_id]

    def update_user_profile(self, session_id, new_info):
        """Update user profile with new information"""
        session = self.get_user_session(session_id)
        session['profile'].update(new_info)
        session['conversation_history'].append(new_info)

    def process_user_query(self, user_input, session_id='default'):
        """Process user query and generate response"""
        session = self.get_user_session(session_id)

        # Parse new information
        parsed_info = self.input_parser.parse_user_input(user_input)

        # Update user profile
        if parsed_info:
            self.update_user_profile(session_id, parsed_info)

        # Check for non-financial queries
        if self._is_non_financial_query(user_input):
            return self._handle_non_financial_query()

        # Handle retirement planning
        if 'retirement' in user_input.lower() and 'goal_amount' not in session['profile']:
            return self._handle_retirement_planning(session)

        # Check for missing information
        missing_info = self._identify_missing_info(session['profile'])

        if missing_info:
            return self._request_missing_info(missing_info)

        # Generate recommendations
        return self._generate_recommendations(session)

    def _is_non_financial_query(self, user_input):
        """Check if query is non-financial"""
        non_financial_keywords = [
            'weather', 'recipe', 'movie', 'game', 'food', 'sports', 
            'cricket', 'football', 'politics', 'news', 'entertainment'
        ]
        financial_keywords = ['invest', 'save', 'money', 'fund', 'portfolio', 'return']

        has_non_financial = any(word in user_input.lower() for word in non_financial_keywords)
        has_financial = any(word in user_input.lower() for word in financial_keywords)

        return has_non_financial and not has_financial

    def _handle_non_financial_query(self):
        """Handle non-financial queries"""
        return ("\n🤖 I'm FinVerse AI, your personal investment advisor!\n\n"
                "I specialize in:\n"
                "• 📊 Portfolio allocation recommendations\n"
                "• 🎯 Goal-based investment planning\n"
                "• 📈 Market analysis and stock recommendations\n"
                "• 💰 SIP calculations and wealth planning\n\n"
                "What financial goals would you like to discuss today?\n")

    def _handle_retirement_planning(self, session):
        """Handle retirement planning queries"""
        profile = session['profile']
        if 'age' in profile:
            retirement_age = 60
            years_to_retirement = retirement_age - profile['age']
            profile['timeline_months'] = years_to_retirement * 12
            profile['investment_goal'] = 'Retirement'

            return (f"\n🎯 Planning for retirement in {years_to_retirement} years!\n\n"
                   f"To provide personalized recommendations, I need to know:\n"
                   f"• Your target retirement corpus\n"
                   f"• Your current annual income\n"
                   f"• Your risk appetite (Low/Medium/High)\n\n"
                   f"Example: 'I want 2 crores for retirement, earn 15L annually, medium risk'\n")
        else:
            return ("\n🎯 Great! Let's plan your retirement.\n\n"
                   "I need some information:\n"
                   "• Your current age\n"
                   "• Target retirement amount\n"
                   "• Current annual income\n"
                   "• Risk appetite\n\n"
                   "Example: 'I'm 35, want 2 crores for retirement, earn 12L annually'\n")

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
            return ("\n💰 I need to know your target investment amount.\n\n"
                   "Please specify: 'I want to save [amount] for [purpose]'\n"
                   "Example: 'I want to save 10 lakhs for a car'\n")

        if 'timeline' in missing_info:
            return ("\n⏱️  When do you need this amount?\n\n"
                   "Please specify: 'I need it in [time period]'\n"
                   "Example: 'I need it in 3 years' or 'in 18 months'\n")
            return "\n🤖 Please provide more details about your investment goals.\n"

    def _generate_recommendations(self, session):
        """Generate comprehensive investment recommendations"""
        profile = session['profile']

        # Fill in default values where needed
        complete_profile = {
            'age': profile.get('age', 30),
            'annual_income': profile.get('annual_income', 800000),
            'risk_appetite': profile.get('risk_appetite', 'Medium'),
            'investment_goal': profile.get('investment_goal', 'Wealth Creation'),
            'timeline_months': profile.get('timeline_months', 60)
        }

        # Calculate allocation
        allocation = self.allocation_engine.calculate_allocation(complete_profile)

        # Get market recommendations
        recommendations = self.market_manager.get_portfolio_recommendations(
            complete_profile['risk_appetite'],
            complete_profile['timeline_months'],
            profile.get('goal_amount', 1000000),
            allocation
        )

        # Generate comprehensive report
        if 'goal_amount' in profile:
            return self.report_generator.generate_comprehensive_report(
                recommendations,
                profile['goal_amount'],
                complete_profile['timeline_months'],
                allocation,
                complete_profile
            )
        else:
            # Just show allocation for profile-based queries
            return self._generate_profile_based_response(allocation, complete_profile, recommendations)

    def _generate_profile_based_response(self, allocation, profile, recommendations):
        """Generate response for profile-only queries"""
        response = f"\n📊 PERSONALIZED PORTFOLIO RECOMMENDATION\n"
        response += f"{'='*50}\n"
        response += f"👤 Profile: {profile['age']} years, {profile['risk_appetite']} risk\n"
        response += f"🎯 Goal: {profile['investment_goal']}\n\n"

        response += f"📈 Recommended Allocation:\n"
        response += f"   • 🏛️  Debt Instruments: {allocation[0]:.1f}%\n"
        response += f"   • 📊 Equity: {allocation[1]:.1f}%\n"
        response += f"   • 🏦 Mutual Funds/ETFs: {allocation[2]:.1f}%\n\n"

        if recommendations:
            response += f"🏆 Top Investment Options:\n"
            for i, rec in enumerate(recommendations[:3], 1):
                response += f"   {i}. {rec['name']} ({rec['category']})\n"
                if rec.get('year_return'):
                    response += f"      📊 1-Year Return: {rec['year_return']:.1f}%\n"

        response += f"\n💡 To get specific investment amounts and detailed analysis,\n"
        response += f"   please share your financial goal and target amount.\n"
        response += f"\nExample: 'I want to save 15 lakhs in 4 years for house down payment'\n"

        return response

class ChatInterface:
    def __init__(self):
        self.finverse_ai = FinVerseAI()
        self.session_id = 'default'

    def display_welcome_message(self):
        """Display enhanced welcome message"""
        welcome = f"\n{'='*70}\n"
        welcome += f"🚀 WELCOME TO FINVERSE AI - YOUR SMART INVESTMENT ADVISOR!\n"
        welcome += f"{'='*70}\n\n"

        welcome += f"🎯 What I can help you with:\n"
        welcome += f"   • 📊 Personalized portfolio allocation\n"
        welcome += f"   • 💰 Goal-based investment planning\n"
        welcome += f"   • 📈 Real-time market recommendations\n"
        welcome += f"   • 🧮 SIP calculations and projections\n"
        welcome += f"   • ⚖️  Risk assessment and optimization\n\n"

        welcome += f"💬 Smart Examples to try:\n"
        welcome += f"   • 'I'm 28, want to save 10L for wedding in 2 years'\n"
        welcome += f"   • 'Need 50L for retirement, I'm 35, earn 15L annually'\n"
        welcome += f"   • 'Want to invest 25K monthly, medium risk appetite'\n"
        welcome += f"   • 'I'm 32, high risk, looking for wealth creation options'\n\n"

        welcome += f"🤖 Pro Tips:\n"
        welcome += f"   • Mention your age, income, and risk appetite for better advice\n"
        welcome += f"   • Specify timeline and target amount for detailed planning\n"
        welcome += f"   • Ask follow-up questions to refine recommendations\n\n"

        welcome += f"Type 'help' for more options or 'exit' to quit.\n"
        welcome += f"{'='*70}\n"

        print(welcome)

    def display_help(self):
        """Display help information"""
        help_text = f"\n📚 FINVERSE AI HELP GUIDE\n"
        help_text += f"{'='*40}\n\n"

        help_text += f"🔤 Command Examples:\n"
        help_text += f"   • Investment Planning: 'Save 5L for car in 18 months'\n"
        help_text += f"   • Retirement: 'Plan retirement, I'm 30, earn 12L'\n"
        help_text += f"   • Profile Setup: 'I'm 25, medium risk, software engineer'\n"
        help_text += f"   • Portfolio Review: 'Review my current investments'\n\n"

        help_text += f"💡 Supported Formats:\n"
        help_text += f"   • Amounts: 5L, 10 lakhs, 2 crores, 50K, ₹100000\n"
        help_text += f"   • Timeline: 2 years, 18 months, 5 yrs\n"
        help_text += f"   • Risk: Low/Medium/High risk\n\n"

        help_text += f"🎯 Investment Goals:\n"
        help_text += f"   • Short-term: Vacation, car, wedding\n"
        help_text += f"   • Long-term: Retirement, wealth creation\n"
        help_text += f"   • Education: Child's future, higher studies\n"
        help_text += f"   • Property: House, real estate investment\n\n"

        help_text += f"Type 'exit' to quit or continue chatting for investment advice!\n"
        print(help_text)

    def run_chat(self):
        """Run the main chat interface"""
        self.display_welcome_message()

        while True:
            try:
                user_input = input("\n🧑 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Thank you for using FinVerse AI!")
                    print("💫 Remember: Smart investing today creates wealth tomorrow!")
                    print("🔄 Come back anytime for personalized financial advice!\n")
                    break

                if user_input.lower() in ['help', '?']:
                    self.display_help()
                    continue

                if user_input.lower() in ['clear', 'reset']:
                    self.finverse_ai.user_sessions[self.session_id] = {
                        'profile': {},
                        'conversation_history': []
                    }
                    print("\n🔄 Session reset! You can start fresh with new goals.\n")
                    continue

                # Process the query
                print("\n🤖 FinVerse AI:")
                response = self.finverse_ai.process_user_query(user_input, self.session_id)
                print(response)

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Sorry, I encountered an error: {str(e)}")
                print("🔄 Please try rephrasing your question or type 'help' for guidance.\n")

class FinVerseManager:
    """Main application manager"""

    def __init__(self):
        self.chat_interface = ChatInterface()

    def initialize_system(self):
        """Initialize the FinVerse system"""
        print("🚀 Initializing FinVerse AI...")
        print("📊 Loading market data sources...")
        print("🧠 Preparing intelligent allocation engine...")
        print("✅ System ready!\n")

    def run_application(self):
        """Run the main application"""
        try:
            self.initialize_system()
            self.chat_interface.run_chat()
        except Exception as e:
            print(f"❌ System error: {str(e)}")
            print("🔧 Please restart the application.")

# Advanced utility functions
class MarketAnalyzer:
    """Advanced market analysis utilities"""

    @staticmethod
    def calculate_portfolio_metrics(recommendations, allocation):
        """Calculate advanced portfolio metrics"""
        if not recommendations:
            return {}

        total_return = 0
        total_risk = 0

        for i, rec in enumerate(recommendations[:3]):
            weight = allocation[i] / 100 if i < len(allocation) else 0
            total_return += rec.get('year_return', 0) * weight
            total_risk += (rec.get('volatility', 0) ** 2) * (weight ** 2)

        total_risk = total_risk ** 0.5

        return {
            'expected_return': round(total_return, 2),
            'portfolio_risk': round(total_risk, 2),
            'sharpe_ratio': round(total_return / total_risk if total_risk > 0 else 0, 3)
        }

    @staticmethod
    def get_market_sentiment():
        """Get simplified market sentiment"""
        # This could be enhanced with real market sentiment APIs
        import random
        sentiments = ['Bullish', 'Neutral', 'Bearish']
        return random.choice(sentiments)

class InvestmentEducator:
    """Provide investment education and tips"""

    @staticmethod
    def get_investment_tip(risk_level):
        """Get educational investment tips"""
        tips = {
            'Low': [
                "💡 Diversification is key even in low-risk portfolios",
                "🏦 Consider laddering your FDs for better liquidity",
                "📈 Even conservative investors should have some equity exposure",
                "🛡️ Build your emergency fund before investing"
            ],
            'Medium': [
                "⚖️ Balance is crucial - don't put all eggs in one basket",
                "📊 Regular portfolio rebalancing helps maintain your target allocation",
                "🎯 SIP investing helps average out market volatility",
                "📚 Stay informed but don't let news drive your decisions"
            ],
            'High': [
                "🎢 High returns come with high volatility - stay invested for long term",
                "🔍 Research thoroughly before investing in individual stocks",
                "💎 Have patience - wealth creation takes time",
                "🎯 Don't try to time the market - time in market beats timing the market"
            ]
        }
        return tips.get(risk_level, tips['Medium'])[0]

def main():
    """Main execution function"""
    try:
        # Create and run the FinVerse application
        app = FinVerseManager()
        app.run_application()

    except KeyboardInterrupt:
        print("\n\n👋 Application terminated by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Critical error: {str(e)}")
        print("🔧 Please contact support or restart the application.")

if __name__ == "__main__":
    main()