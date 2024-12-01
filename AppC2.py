import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import hashlib
import base64
from cryptography.fernet import Fernet
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from sklearn.preprocessing import PolynomialFeatures

# Security configurations
def generate_key():
    """Generate a key and save it to a file"""
    if not os.path.exists('secret.key'):
        key = Fernet.generate_key()
        with open('secret.key', 'wb') as key_file:
            key_file.write(key)
    return open('secret.key', 'rb').read()

def get_cipher():
    """Get the cipher object using the stored key"""
    key = generate_key()
    return Fernet(key)

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def save_users_data():
    """Save users data to an encrypted JSON file"""
    cipher = get_cipher()
    
    # Convert expenses DataFrame to serializable format
    expenses_dict = st.session_state.expenses.copy()
    expenses_dict['Date'] = expenses_dict['Date'].astype(str)  # Convert dates to strings
    
    data = {
        'users': st.session_state.users,
        'expenses': expenses_dict.to_dict('records')
    }
    
    encrypted_data = cipher.encrypt(json.dumps(data).encode())
    
    # Create secure directory if it doesn't exist
    secure_dir = Path('secure_data')
    secure_dir.mkdir(exist_ok=True)
    
    with open(secure_dir / 'user_data.encrypted', 'wb') as f:
        f.write(encrypted_data)

def load_users_data():
    """Load users data from encrypted file"""
    try:
        cipher = get_cipher()
        secure_dir = Path('secure_data')
        with open(secure_dir / 'user_data.encrypted', 'rb') as f:
            encrypted_data = f.read()
            decrypted_data = cipher.decrypt(encrypted_data)
            data = json.loads(decrypted_data)
            st.session_state.users = data['users']
            
            # Convert loaded data to DataFrame and handle dates
            expenses_df = pd.DataFrame(data['expenses'])
            if not expenses_df.empty and 'Date' in expenses_df.columns:
                expenses_df['Date'] = pd.to_datetime(expenses_df['Date']).dt.date
            st.session_state.expenses = expenses_df
            
    except (FileNotFoundError, Exception) as e:
        # Initialize with correct columns and empty DataFrame
        st.session_state.users = {}
        st.session_state.expenses = pd.DataFrame({
            'Date': [],
            'Category': [],
            'Amount': [],
            'Description': [],
            'Username': []
        })

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    if 'user_logged_in' not in st.session_state:
        st.session_state.user_logged_in = False
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    load_users_data()

def login_user(username, password):
    """Login with hashed password verification"""
    hashed_password = hash_password(password)
    if (username in st.session_state.users and 
        st.session_state.users[username]['password'] == hashed_password):
        st.session_state.user_logged_in = True
        st.session_state.current_user = username
        return True
    return False

def register_user(username, password):
    """Register with password hashing"""
    if username not in st.session_state.users:
        hashed_password = hash_password(password)
        st.session_state.users[username] = {
            'password': hashed_password,
            'income': 0,
            'created_at': datetime.now().isoformat()
        }
        save_users_data()
        return True
    return False

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    return True, "Password is strong"

def update_income(amount):
    st.session_state.users[st.session_state.current_user]['income'] = amount

def add_expense(date, category, amount, description):
    """Add new expense with username"""
    new_expense = pd.DataFrame({
        'Date': [str(date)],  # Convert date to string when adding
        'Category': [category],
        'Amount': [amount],
        'Description': [description],
        'Username': [st.session_state.current_user]
    })
    st.session_state.expenses = pd.concat([st.session_state.expenses, new_expense], ignore_index=True)
    save_users_data()  # Save after adding expense

def get_user_expenses():
    """Get expenses for current user with validation"""
    # Initialize expenses DataFrame if it doesn't exist
    if not hasattr(st.session_state, 'expenses'):
        st.session_state.expenses = pd.DataFrame({
            'Date': [],
            'Category': [],
            'Amount': [],
            'Description': [],
            'Username': []
        })
    
    # Ensure all required columns exist
    required_columns = ['Date', 'Category', 'Amount', 'Description', 'Username']
    for col in required_columns:
        if col not in st.session_state.expenses.columns:
            st.session_state.expenses[col] = []
    
    user_expenses = st.session_state.expenses[st.session_state.expenses['Username'] == st.session_state.current_user]
    
    # Convert Date column to datetime if it exists and has values
    if not user_expenses.empty and 'Date' in user_expenses.columns:
        user_expenses['Date'] = pd.to_datetime(user_expenses['Date']).dt.date
    
    return user_expenses

def get_total_expenses():
    """Safely calculate total expenses"""
    try:
        user_expenses = get_user_expenses()
        if user_expenses.empty:
            return 0
        return user_expenses['Amount'].sum()
    except Exception as e:
        st.error(f"Error calculating total expenses: {str(e)}")
        return 0

def calculate_remaining_budget():
    """Calculate remaining budget with error handling"""
    try:
        user_expenses = get_user_expenses()
        if user_expenses.empty or 'Amount' not in user_expenses.columns:
            total_expenses = 0
        else:
            total_expenses = user_expenses['Amount'].sum()
        
        monthly_income = st.session_state.users[st.session_state.current_user]['income']
        return monthly_income - total_expenses
    except Exception as e:
        st.error(f"Error calculating budget: {str(e)}")
        return 0

def format_indian_currency(amount):
    """Format amount in Indian currency style (with commas)"""
    s = str(int(amount))
    result = s[-3:]
    s = s[:-3]
    while s:
        result = s[-2:] + ',' + result if len(s) > 2 else s + ',' + result
        s = s[:-2]
    return f"₹{result}.{str(amount % 1)[2:4] if amount % 1 else '00'}"

def calculate_financial_health_score():
    """
    Calculate a comprehensive financial health score (0-100) based on multiple factors
    """
    user_expenses = get_user_expenses()
    monthly_income = st.session_state.users[st.session_state.current_user]['income']
    
    scores = {}
    weights = {
        'savings_rate': 0.25,
        'expense_stability': 0.20,
        'budget_adherence': 0.20,
        'emergency_fund': 0.15,
        'expense_diversity': 0.10,
        'bill_regularity': 0.10
    }
    
    # 1. Savings Rate Score (25%)
    if monthly_income > 0:
        total_expenses = user_expenses['Amount'].sum() if not user_expenses.empty else 0
        savings_rate = (monthly_income - total_expenses) / monthly_income
        scores['savings_rate'] = min(100, (savings_rate / 0.3) * 100)  # 30% savings rate = perfect score
    else:
        scores['savings_rate'] = 0

    # 2. Expense Stability Score (20%)
    if not user_expenses.empty and len(user_expenses) > 3:
        monthly_expenses = user_expenses.groupby(pd.to_datetime(user_expenses['Date']).dt.strftime('%Y-%m'))['Amount'].sum()
        cv = monthly_expenses.std() / monthly_expenses.mean()  # Coefficient of variation
        scores['expense_stability'] = max(0, 100 * (1 - cv))
    else:
        scores['expense_stability'] = 50  # Neutral score for new users

    # 3. Budget Adherence Score (20%)
    budget_categories = {
        'Food': 0.3,
        'Transport': 0.15,
        'Entertainment': 0.1,
        'Utilities': 0.25,
        'Other': 0.2
    }
    
    if not user_expenses.empty:
        category_spending = user_expenses.groupby('Category')['Amount'].sum() / user_expenses['Amount'].sum()
        budget_variance = sum(abs(category_spending.get(cat, 0) - target) for cat, target in budget_categories.items())
        scores['budget_adherence'] = max(0, 100 * (1 - budget_variance))
    else:
        scores['budget_adherence'] = 50

    # 4. Emergency Fund Score (15%)
    # Assuming 6 months of expenses is ideal for emergency fund
    if not user_expenses.empty:
        monthly_avg_expense = user_expenses['Amount'].sum() / len(user_expenses['Date'].unique())
        emergency_fund_ratio = monthly_income / (monthly_avg_expense * 6)
        scores['emergency_fund'] = min(100, emergency_fund_ratio * 100)
    else:
        scores['emergency_fund'] = 0

    # 5. Expense Diversity Score (10%)
    if not user_expenses.empty:
        category_counts = user_expenses['Category'].value_counts()
        diversity_score = (len(category_counts) / len(budget_categories)) * 100
        scores['expense_diversity'] = diversity_score
    else:
        scores['expense_diversity'] = 0

    # 6. Bill Regularity Score (10%)
    if not user_expenses.empty:
        monthly_regularity = user_expenses.groupby(
            [pd.to_datetime(user_expenses['Date']).dt.strftime('%Y-%m'), 'Category']
        ).size().unstack(fill_value=0)
        regularity_score = (monthly_regularity > 0).mean().mean() * 100
        scores['bill_regularity'] = regularity_score
    else:
        scores['bill_regularity'] = 0

    # Calculate weighted final score
    final_score = sum(score * weights[metric] for metric, score in scores.items())
    
    return {
        'final_score': final_score,
        'component_scores': scores,
        'weights': weights
    }

def display_financial_health_dashboard():
    """
    Display the financial health score and insights
    """
    st.header("🏆 Financial Health Score")
    
    health_data = calculate_financial_health_score()
    final_score = health_data['final_score']
    scores = health_data['component_scores']
    
    # Display overall score with gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = final_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Overall Financial Health"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "red"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    st.plotly_chart(fig)
    
    # Display component scores
    st.subheader("Score Components")
    cols = st.columns(3)
    for idx, (metric, score) in enumerate(scores.items()):
        with cols[idx % 3]:
            st.metric(
                label=metric.replace('_', ' ').title(),
                value=f"{score:.1f}%",
                delta=f"Weight: {health_data['weights'][metric]*100}%"
            )
    
    # Financial Health Insights
    st.subheader("💡 Personalized Insights")
    
    if final_score >= 80:
        st.success("🌟 Excellent financial health! Keep up the great work!")
    elif final_score >= 60:
        st.info("📈 Good financial health, with room for improvement.")
    else:
        st.warning("⚠️ Your financial health needs attention.")
    
    # Detailed recommendations based on lowest scoring components
    st.write("### Recommendations for Improvement")
    
    lowest_scores = sorted(scores.items(), key=lambda x: x[1])[:3]
    for metric, score in lowest_scores:
        if metric == 'savings_rate' and score < 70:
            st.write("- 💰 **Improve Savings Rate**: Consider the 50/30/20 rule - 50% needs, 30% wants, 20% savings")
        elif metric == 'expense_stability' and score < 70:
            st.write("- 📊 **Enhance Expense Stability**: Create a monthly budget and stick to it")
        elif metric == 'budget_adherence' and score < 70:
            st.write("- 📌 **Better Budget Adherence**: Track expenses daily and adjust spending in overbudget categories")
        elif metric == 'emergency_fund' and score < 70:
            st.write("- 🏦 **Build Emergency Fund**: Aim to save 6 months of expenses for emergencies")
        elif metric == 'expense_diversity' and score < 70:
            st.write("- 🎯 **Diversify Expenses**: Ensure you're allocating money across all important categories")
        elif metric == 'bill_regularity' and score < 70:
            st.write("- 📅 **Improve Bill Regularity**: Set up automatic payments for regular bills")

    # Historical Score Tracking
    if 'financial_health_history' not in st.session_state:
        st.session_state.financial_health_history = []
    
    # Add current score to history (limit to last 10 scores)
    current_date = datetime.now().strftime('%Y-%m-%d')
    st.session_state.financial_health_history.append((current_date, final_score))
    st.session_state.financial_health_history = st.session_state.financial_health_history[-10:]
    
    # Display historical trend
    if len(st.session_state.financial_health_history) > 1:
        st.subheader("📈 Score History")
        history_df = pd.DataFrame(st.session_state.financial_health_history, columns=['Date', 'Score'])
        fig = px.line(history_df, x='Date', y='Score', title='Financial Health Score Trend')
        st.plotly_chart(fig)

def remove_expense(index):
    """Remove an expense by index"""
    try:
        # Get user expenses
        user_expenses = get_user_expenses()
        
        # Get the index in the main DataFrame
        main_index = st.session_state.expenses[
            st.session_state.expenses['Username'] == st.session_state.current_user
        ].index[index]
        
        # Remove the expense
        st.session_state.expenses = st.session_state.expenses.drop(main_index)
        
        # Save the updated data
        save_users_data()
        return True
    except Exception as e:
        st.error(f"Error removing expense: {str(e)}")
        return False

def forecast_expenses():
    """
    Forecast future expenses using historical data and machine learning
    """
    df = get_user_expenses()
    if len(df) < 10:  # Need sufficient data for forecasting
        return None, None
    
    try:
        # Prepare time series data
        df['Date'] = pd.to_datetime(df['Date'])
        daily_expenses = df.groupby('Date')['Amount'].sum().reset_index()
        
        # Create features (days from start)
        X = (daily_expenses['Date'] - daily_expenses['Date'].min()).dt.days.values.reshape(-1, 1)
        y = daily_expenses['Amount'].values
        
        # Fit polynomial regression for better trend capture
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Generate future dates for prediction
        last_date = daily_expenses['Date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
        future_X = np.array((future_dates - daily_expenses['Date'].min()).days).reshape(-1, 1)
        future_X_poly = poly.transform(future_X)
        
        # Make predictions
        predictions = model.predict(future_X_poly)
        
        return future_dates, predictions
        
    except Exception as e:
        st.error(f"Error in expense forecasting: {str(e)}")
        return None, None

def optimize_savings_goal():
    """
    Optimize savings goals based on income, spending patterns, and machine learning
    """
    user_expenses = get_user_expenses()
    monthly_income = st.session_state.users[st.session_state.current_user]['income']
    
    if user_expenses.empty:
        return None
    
    try:
        # Calculate monthly statistics
        user_expenses['Date'] = pd.to_datetime(user_expenses['Date'])
        monthly_expenses = user_expenses.groupby(user_expenses['Date'].dt.strftime('%Y-%m')).agg({
            'Amount': ['sum', 'mean', 'std']
        }).reset_index()
        
        # Calculate optimal savings based on expense stability and income
        if len(monthly_expenses) > 1:
            mean_expense = monthly_expenses['Amount']['mean'].mean()
            std_expense = monthly_expenses['Amount']['std'].mean()
            
            # Calculate expense stability (higher is better)
            expense_stability = float(1 - (std_expense / mean_expense)) if mean_expense > 0 else 0
            
            # Base saving rate adjusted by stability
            base_saving_rate = 0.2  # 20% base saving rate
            optimal_saving_rate = float(base_saving_rate * (1 + expense_stability))
            
            # Keep between 10% and 40%
            optimal_saving_rate = min(max(float(optimal_saving_rate), 0.1), 0.4)
            optimal_monthly_saving = monthly_income * optimal_saving_rate
            
            return {
                'optimal_rate': optimal_saving_rate,
                'monthly_target': optimal_monthly_saving,
                'expense_stability': expense_stability
            }
        else:
            # Default values for new users
            return {
                'optimal_rate': 0.2,  # Default 20% saving rate
                'monthly_target': monthly_income * 0.2,
                'expense_stability': 0.5
            }
            
    except Exception as e:
        st.error(f"Error calculating savings goal: {str(e)}")
        return None

# Add this to your main dashboard
def display_ai_insights():
    st.header("🤖 AI-Powered Insights")
    
    # Expense Forecasting
    with st.expander("📈 Expense Forecast (Next 30 Days)"):
        future_dates, predictions = forecast_expenses()
        if future_dates is not None:
            fig = go.Figure()
            
            # Historical data
            historical = get_user_expenses()
            historical['Date'] = pd.to_datetime(historical['Date'])
            daily_historical = historical.groupby('Date')['Amount'].sum().reset_index()
            
            fig.add_trace(go.Scatter(
                x=daily_historical['Date'],
                y=daily_historical['Amount'],
                name='Historical Expenses',
                line=dict(color='blue')
            ))
            
            # Forecasted data
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title='Expense Forecast',
                xaxis_title='Date',
                yaxis_title='Amount (₹)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig)
            
            # Show key metrics
            avg_forecast = np.mean(predictions)
            max_forecast = np.max(predictions)
            st.info(f"📊 Average forecasted daily expense: ₹{avg_forecast:,.2f}")
            st.warning(f"⚠️ Peak forecasted daily expense: ₹{max_forecast:,.2f}")
        else:
            st.info("Need more expense data for accurate forecasting (minimum 10 entries)")

    # Savings Goal Optimization
    with st.expander("💰 Smart Savings Goals"):
        savings_insights = optimize_savings_goal()
        if savings_insights:
            optimal_rate = savings_insights['optimal_rate']
            monthly_target = savings_insights['monthly_target']
            stability = savings_insights['expense_stability']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Recommended Savings Rate", f"{optimal_rate:.1%}")
                st.metric("Monthly Savings Target", f"₹{monthly_target:,.2f}")
            
            with col2:
                st.metric("Expense Stability Score", f"{stability:.2f}")
                
            # Generate personalized advice
            if stability < 0.3:
                st.warning("🎯 Your expenses vary significantly. Consider setting up a budget for more consistent spending.")
            elif stability < 0.7:
                st.info("📈 Your spending is moderately stable. Focus on maintaining regular savings.")
            else:
                st.success("🌟 You have very stable spending habits! Great for long-term financial planning.")
            
            # Progress towards savings goal
            current_month = datetime.now().strftime('%Y-%m')
            month_expenses = get_user_expenses()[
                pd.to_datetime(get_user_expenses()['Date']).dt.strftime('%Y-%m') == current_month
            ]['Amount'].sum()
            
            current_savings = st.session_state.users[st.session_state.current_user]['income'] - month_expenses
            savings_progress = current_savings / monthly_target
            
            st.progress(min(float(savings_progress), 1.0))
            st.write(f"Current month's savings progress: {savings_progress:.1%} of target")
        else:
            st.info("Add some expenses to get personalized savings recommendations!")

st.title('Personal Expense Tracker')

# Login/Register Section
if not st.session_state.user_logged_in:
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.header("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if login_user(login_username, login_password):
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials!")

    with tab2:
        st.header("Register")
        reg_username = st.text_input("Username", key="reg_username")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_password_confirm = st.text_input("Confirm Password", type="password", key="reg_password_confirm")
        
        if st.button("Register"):
            if reg_password != reg_password_confirm:
                st.error("Passwords do not match!")
            else:
                is_valid, message = validate_password(reg_password)
                if not is_valid:
                    st.error(message)
                elif register_user(reg_username, reg_password):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username already exists!")

else:
    st.write(f"Welcome, {st.session_state.current_user}!")
    
    # Income Section
    with st.expander("Update Monthly Income"):
        current_income = st.session_state.users[st.session_state.current_user]['income']
        new_income = st.number_input('Set Monthly Income', 
                                   min_value=0.0, 
                                   value=float(current_income), 
                                   format="%.2f")
        if st.button('Update Income'):
            update_income(new_income)
            st.success('Income updated!')

    # Expense Section
    with st.sidebar:
        st.header('Add Expense')
        date = st.date_input('Date')
        category = st.selectbox('Category', ['Food', 'Transport', 'Entertainment', 'Utilities', 'Other'])
        amount = st.number_input('Amount', min_value=0.0, format="%.2f")
        description = st.text_input('Description')
        if st.button('Add'):
            add_expense(date, category, amount, description)
            st.success('Expense added!')

    # Budget Overview
    st.header('Budget Overview')
    remaining_budget = calculate_remaining_budget()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Monthly Income", format_indian_currency(st.session_state.users[st.session_state.current_user]['income']))
    with col2:
        total_expenses = get_total_expenses()
        st.metric("Total Expenses", format_indian_currency(total_expenses))
    with col3:
        st.metric("Remaining Budget", format_indian_currency(remaining_budget))

    # Display User's Expenses
    st.header('Your Expenses')
    user_expenses = get_user_expenses()
    if not user_expenses.empty and 'Amount' in user_expenses.columns:
        # Create columns for the expense list
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Display expenses with index
            st.dataframe(
                user_expenses[['Date', 'Category', 'Amount', 'Description']]
                .style.format({'Amount': '₹{:,.2f}'.format})
            )
        
        with col2:
            # Add remove expense section
            st.write("Remove Expense")
            expense_index = st.number_input(
                "Enter expense row number to remove (0 to {})".format(len(user_expenses)-1),
                min_value=0,
                max_value=len(user_expenses)-1 if len(user_expenses) > 0 else 0,
                value=0
            )
            if st.button("Remove Selected Expense"):
                if remove_expense(expense_index):
                    st.success("Expense removed successfully!")
                    st.rerun()
                else:
                    st.error("Failed to remove expense")
    else:
        st.info("No expenses recorded yet!")

    # Optional: Add a clear all expenses button
    if not user_expenses.empty:
        if st.button("Clear All Expenses", type="secondary"):
            if st.button("Are you sure? This cannot be undone!", type="primary"):
                st.session_state.expenses = st.session_state.expenses[
                    st.session_state.expenses['Username'] != st.session_state.current_user
                ]
                save_users_data()
                st.success("All expenses cleared!")
                st.rerun()

    # Visualization
    st.header('Expense Analysis')
    if not user_expenses.empty:
        fig, ax = plt.subplots()
        sns.barplot(data=user_expenses, x='Category', y='Amount', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Logout Button
    if st.sidebar.button('Logout'):
        st.session_state.user_logged_in = False
        st.session_state.current_user = None
        st.rerun()

    # Financial Health Dashboard
    if st.session_state.user_logged_in:
        display_financial_health_dashboard()

    # AI Insights
    if st.session_state.user_logged_in:
        display_ai_insights()