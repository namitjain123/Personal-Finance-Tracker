import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime
import sqlite3
import hashlib
import sys
import os
from db_config import DB_PATH
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

DB_PATH = resource_path('expense_tracker.db')

class ExpenseTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Expense Tracker")
        self.root.geometry("800x600")
        
        self.current_user = None
        
        # Create main container
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initially show login frame
        self.show_login_frame()

    def get_db_connection(self):
        """Get database connection"""
        try:
            connection = sqlite3.connect(DB_PATH)
            return connection
        except Exception as e:
            messagebox.showerror("Database Error", str(e))
            return None

    def show_login_frame(self):
        # Clear main container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Login frame
        login_frame = ttk.LabelFrame(self.main_container, text="Login", padding="20")
        login_frame.grid(row=0, column=0, padx=10, pady=10)
        
        # Username
        ttk.Label(login_frame, text="Username:").grid(row=0, column=0, pady=5)
        self.username_var = tk.StringVar()
        username_entry = ttk.Entry(login_frame, textvariable=self.username_var)
        username_entry.grid(row=0, column=1, pady=5)
        
        # Password
        ttk.Label(login_frame, text="Password:").grid(row=1, column=0, pady=5)
        self.password_var = tk.StringVar()
        password_entry = ttk.Entry(login_frame, textvariable=self.password_var, show="*")
        password_entry.grid(row=1, column=1, pady=5)
        
        # Buttons
        ttk.Button(login_frame, text="Login", command=self.login).grid(row=2, column=0, pady=10)
        ttk.Button(login_frame, text="Register", command=self.show_register_frame).grid(row=2, column=1, pady=10)

    def show_register_frame(self):
        # Clear main container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Register frame
        register_frame = ttk.LabelFrame(self.main_container, text="Register", padding="20")
        register_frame.grid(row=0, column=0, padx=10, pady=10)
        
        # Username
        ttk.Label(register_frame, text="Username:").grid(row=0, column=0, pady=5)
        self.reg_username_var = tk.StringVar()
        username_entry = ttk.Entry(register_frame, textvariable=self.reg_username_var)
        username_entry.grid(row=0, column=1, pady=5)
        
        # Password
        ttk.Label(register_frame, text="Password:").grid(row=1, column=0, pady=5)
        self.reg_password_var = tk.StringVar()
        password_entry = ttk.Entry(register_frame, textvariable=self.reg_password_var, show="*")
        password_entry.grid(row=1, column=1, pady=5)
        
        # Buttons
        ttk.Button(register_frame, text="Register", command=self.register).grid(row=2, column=0, pady=10)
        ttk.Button(register_frame, text="Back to Login", command=self.show_login_frame).grid(row=2, column=1, pady=10)

    def show_main_frame(self):
        # Clear main container
        for widget in self.main_container.winfo_children():
            widget.destroy()
        
        # Main frame
        main_frame = ttk.Frame(self.main_container)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Welcome message
        ttk.Label(main_frame, text=f"Welcome, {self.current_user}!").grid(row=0, column=0, pady=10)
        
        # Add Expense Frame
        expense_frame = ttk.LabelFrame(main_frame, text="Add Expense", padding="10")
        expense_frame.grid(row=1, column=0, padx=10, pady=5, sticky=(tk.W, tk.E))
        
        # Amount
        ttk.Label(expense_frame, text="Amount:").grid(row=0, column=0, pady=5)
        self.amount_var = tk.StringVar()
        ttk.Entry(expense_frame, textvariable=self.amount_var).grid(row=0, column=1, pady=5)
        
        # Category
        ttk.Label(expense_frame, text="Category:").grid(row=1, column=0, pady=5)
        self.category_var = tk.StringVar()
        categories = ['Food', 'Transport', 'Entertainment', 'Utilities', 'Other']
        ttk.Combobox(expense_frame, textvariable=self.category_var, values=categories).grid(row=1, column=1, pady=5)
        
        # Description
        ttk.Label(expense_frame, text="Description:").grid(row=2, column=0, pady=5)
        self.description_var = tk.StringVar()
        ttk.Entry(expense_frame, textvariable=self.description_var).grid(row=2, column=1, pady=5)
        
        # Add button
        ttk.Button(expense_frame, text="Add Expense", command=self.add_expense).grid(row=3, column=0, columnspan=2, pady=10)
        
        # Income Management Section
        income_frame = ttk.LabelFrame(main_frame, text="Income Management", padding="10")
        income_frame.grid(row=2, column=0, padx=10, pady=5, sticky=(tk.W, tk.E))

        # Current Income Display
        self.current_income = self.get_user_income()
        ttk.Label(income_frame, text=f"Current Monthly Income: ₹{self.current_income:.2f}").grid(row=0, column=0, pady=5)
        
        # Update Income
        ttk.Label(income_frame, text="New Income:").grid(row=1, column=0, pady=5)
        self.new_income_var = tk.StringVar()
        ttk.Entry(income_frame, textvariable=self.new_income_var).grid(row=1, column=1, pady=5)
        ttk.Button(income_frame, text="Update Income", command=self.update_income).grid(row=1, column=2, pady=5)

        # Expenses List
        expenses_frame = ttk.LabelFrame(main_frame, text="Your Expenses", padding="10")
        expenses_frame.grid(row=3, column=0, padx=10, pady=5, sticky=(tk.W, tk.E))
        
        # Create Treeview
        self.expenses_tree = ttk.Treeview(expenses_frame, columns=('Date', 'Category', 'Amount', 'Description'), show='headings')
        self.expenses_tree.heading('Date', text='Date')
        self.expenses_tree.heading('Category', text='Category')
        self.expenses_tree.heading('Amount', text='Amount')
        self.expenses_tree.heading('Description', text='Description')
        self.expenses_tree.grid(row=0, column=0, pady=5)
        
        # Add Remove Button
        remove_button = ttk.Button(expenses_frame, text="Remove Selected", command=self.remove_expense)
        remove_button.grid(row=1, column=0, pady=5)
        
        # Add a Clear All button
        clear_button = ttk.Button(expenses_frame, text="Clear All", command=self.clear_all_expenses)
        clear_button.grid(row=2, column=0, pady=5)
        
        # Load expenses
        self.load_expenses()

        # Add Analytics Button
        ttk.Button(main_frame, text="Show Analytics", command=self.show_insights_window).grid(row=4, column=0, pady=10)
        
        # Logout button
        ttk.Button(main_frame, text="Logout", command=self.logout).grid(row=5, column=0, pady=10)

    def login(self):
        username = self.username_var.get()
        password = self.password_var.get()
        
        try:
            connection = self.get_db_connection()
            cursor = connection.cursor()
            
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            cursor.execute(
                "SELECT * FROM users WHERE username = ? AND password = ?",
                (username, hashed_password)
            )
            
            if cursor.fetchone():
                self.current_user = username
                self.show_main_frame()
            else:
                messagebox.showerror("Error", "Invalid credentials!")
                
        except Exception as e:
            messagebox.showerror("Database Error", str(e))
        finally:
            if connection:
                connection.close()

    def register(self):
        username = self.reg_username_var.get()
        password = self.reg_password_var.get()
        
        try:
            connection = self.get_db_connection()
            cursor = connection.cursor()
            
            # Check if username exists
            cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                messagebox.showerror("Error", "Username already exists!")
                return
            
            # Insert new user
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            cursor.execute(
                "INSERT INTO users (username, password, income) VALUES (?, ?, ?)",
                (username, hashed_password, 0)
            )
            connection.commit()
            
            messagebox.showinfo("Success", "Registration successful! Please login.")
            self.show_login_frame()
            
        except Exception as e:
            messagebox.showerror("Database Error", str(e))
        finally:
            if connection:
                connection.close()

    def add_expense(self):
        amount = self.amount_var.get()
        category = self.category_var.get()
        description = self.description_var.get()
        
        try:
            amount = float(amount)
            if amount <= 0:
                messagebox.showerror("Error", "Amount must be greater than 0!")
                return
                
            connection = self.get_db_connection()
            cursor = connection.cursor()
            
            cursor.execute(
                """INSERT INTO expenses (username, date, category, amount, description)
                VALUES (?, ?, ?, ?, ?)""",
                (self.current_user, datetime.now().date(), category, amount, description)
            )
            connection.commit()
            
            # Clear fields
            self.amount_var.set("")
            self.category_var.set("")
            self.description_var.set("")
            
            # Reload expenses
            self.load_expenses()
            
        except ValueError:
            messagebox.showerror("Error", "Invalid amount!")
        except Exception as e:
            messagebox.showerror("Database Error", str(e))
        finally:
            if connection:
                connection.close()

    def load_expenses(self):
        # Clear existing items
        for item in self.expenses_tree.get_children():
            self.expenses_tree.delete(item)
            
        try:
            connection = self.get_db_connection()
            cursor = connection.cursor()
            
            cursor.execute(
                """SELECT date, category, amount, description 
                FROM expenses WHERE username = ?
                ORDER BY date DESC""",
                (self.current_user,)
            )
            
            for expense in cursor.fetchall():
                self.expenses_tree.insert('', 'end', values=expense)
                
        except Exception as e:
            messagebox.showerror("Database Error", str(e))
        finally:
            if connection:
                connection.close()

    def logout(self):
        self.current_user = None
        self.show_login_frame()

    def get_user_income(self):
        try:
            connection = self.get_db_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT income FROM users WHERE username = ?", (self.current_user,))
            result = cursor.fetchone()
            return float(result[0]) if result else 0.0
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get income: {str(e)}")
            return 0.0
        finally:
            if connection:
                connection.close()

    def update_income(self):
        try:
            new_income = float(self.new_income_var.get())
            if new_income < 0:
                messagebox.showerror("Error", "Income cannot be negative!")
                return

            # Get current income first
            current_income = self.get_user_income()
            # Add new income to current income
            total_income = current_income + new_income

            connection = self.get_db_connection()
            cursor = connection.cursor()
            cursor.execute(
                "UPDATE users SET income = ? WHERE username = ?",
                (total_income, self.current_user)
            )
            connection.commit()
            messagebox.showinfo("Success", f"Income updated successfully!\nPrevious Income: ₹{current_income:.2f}\nAdded Income: ₹{new_income:.2f}\nTotal Income: ₹{total_income:.2f}")
            self.new_income_var.set("")  # Clear the input field
            self.show_main_frame()  # Refresh the main frame
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update income: {str(e)}")
        finally:
            if connection:
                connection.close()

    def show_insights_window(self):
        insights_window = tk.Toplevel(self.root)
        insights_window.title("AI-Powered Expense Insights")
        insights_window.geometry("1000x800")

        notebook = ttk.Notebook(insights_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=5)

        # Existing tabs
        category_frame = ttk.Frame(notebook)
        notebook.add(category_frame, text='Category Distribution')
        self.show_category_distribution(category_frame)

        trend_frame = ttk.Frame(notebook)
        notebook.add(trend_frame, text='Monthly Trend')
        self.show_monthly_trend(trend_frame)

        budget_frame = ttk.Frame(notebook)
        notebook.add(budget_frame, text='Budget Analysis')
        self.show_budget_analysis(budget_frame)

        # New AI Analysis tabs
        forecast_frame = ttk.Frame(notebook)
        notebook.add(forecast_frame, text='Expense Forecast')
        self.show_expense_forecast(forecast_frame)

        pattern_frame = ttk.Frame(notebook)
        notebook.add(pattern_frame, text='Spending Patterns')
        self.show_spending_patterns(pattern_frame)

        recommendation_frame = ttk.Frame(notebook)
        notebook.add(recommendation_frame, text='Smart Recommendations')
        self.show_recommendations(recommendation_frame)

    def show_category_distribution(self, frame):
        try:
            connection = self.get_db_connection()
            cursor = connection.cursor()
            cursor.execute("""
                SELECT category, SUM(amount) as total
                FROM expenses
                WHERE username = ?
                GROUP BY category
            """, (self.current_user,))
            data = cursor.fetchall()

            if not data:
                ttk.Label(frame, text="No expenses recorded yet!").pack(pady=20)
                return

            categories = [row[0] for row in data]
            amounts = [row[1] for row in data]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(amounts, labels=categories, autopct='%1.1f%%')
            ax.set_title('Expense Distribution by Category')

            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load category distribution: {str(e)}")
        finally:
            if connection:
                connection.close()

    def show_monthly_trend(self, frame):
        try:
            connection = self.get_db_connection()
            cursor = connection.cursor()
            cursor.execute("""
                SELECT strftime('%Y-%m', date) as month, SUM(amount) as total
                FROM expenses
                WHERE username = ?
                GROUP BY month
                ORDER BY month
            """, (self.current_user,))
            data = cursor.fetchall()

            if not data:
                ttk.Label(frame, text="No expenses recorded yet!").pack(pady=20)
                return

            months = [row[0] for row in data]
            amounts = [row[1] for row in data]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(months, amounts, marker='o')
            ax.set_title('Monthly Expense Trend')
            ax.set_xlabel('Month')
            ax.set_ylabel('Total Expenses (₹)')
            plt.xticks(rotation=45)

            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load monthly trend: {str(e)}")
        finally:
            if connection:
                connection.close()

    def show_budget_analysis(self, frame):
        income = self.get_user_income()
        total_expenses = self.get_total_expenses()
        remaining = income - total_expenses

        # Create a frame for text information
        info_frame = ttk.Frame(frame)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(info_frame, text=f"Monthly Income: ₹{income:.2f}").pack()
        ttk.Label(info_frame, text=f"Total Expenses: ₹{total_expenses:.2f}").pack()
        ttk.Label(info_frame, text=f"Remaining Budget: ₹{remaining:.2f}").pack()

        # Create budget progress bar
        if income > 0:
            progress = (total_expenses / income) * 100
            progress_var = tk.DoubleVar(value=min(progress, 100))
            ttk.Progressbar(
                info_frame, 
                variable=progress_var,
                maximum=100,
                length=300,
                mode='determinate'
            ).pack(pady=10)

            # Add warning if over budget
            if progress > 100:
                ttk.Label(
                    info_frame,
                    text="WARNING: Over Budget!",
                    foreground='red'
                ).pack()

    def get_total_expenses(self):
        try:
            connection = self.get_db_connection()
            cursor = connection.cursor()
            cursor.execute("""
                SELECT SUM(amount)
                FROM expenses
                WHERE username = ?
            """, (self.current_user,))
            result = cursor.fetchone()
            return float(result[0]) if result[0] else 0.0
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get total expenses: {str(e)}")
            return 0.0
        finally:
            if connection:
                connection.close()

    def show_expense_forecast(self, frame):
        """Show AI-powered expense forecasting"""
        try:
            connection = self.get_db_connection()
            cursor = connection.cursor()
            
            # Get historical daily expenses
            cursor.execute("""
                SELECT date, SUM(amount) as daily_total
                FROM expenses
                WHERE username = ?
                GROUP BY date
                ORDER BY date
            """, (self.current_user,))
            
            data = cursor.fetchall()
            
            if len(data) < 10:
                ttk.Label(frame, text="Need more expense data for forecasting (minimum 10 days)").pack(pady=20)
                return

            # Prepare data for forecasting
            dates = [datetime.strptime(row[0], '%Y-%m-%d') for row in data]
            amounts = [float(row[1]) for row in data]
            
            # Create features (days from start)
            X = np.array([(date - dates[0]).days for date in dates]).reshape(-1, 1)
            y = np.array(amounts)

            # Fit polynomial regression
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)

            # Generate future dates for prediction
            future_days = 30
            future_x = np.array(range(X[-1][0] + 1, X[-1][0] + future_days + 1)).reshape(-1, 1)
            future_x_poly = poly.transform(future_x)
            predictions = model.predict(future_x_poly)

            # Plot results
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot historical data
            ax.plot(dates, amounts, 'b-', label='Historical Expenses')
            
            # Plot predictions
            future_dates = [dates[-1] + timedelta(days=i+1) for i in range(future_days)]
            ax.plot(future_dates, predictions, 'r--', label='Forecast')
            
            ax.set_title('Expense Forecast (Next 30 Days)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Amount (₹)')
            ax.legend()
            plt.xticks(rotation=45)

            # Add to frame
            canvas = FigureCanvasTkAgg(fig, frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Add summary statistics
            stats_frame = ttk.Frame(frame)
            stats_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(stats_frame, 
                     text=f"Predicted Average Daily Expense: ₹{np.mean(predictions):.2f}").pack()
            ttk.Label(stats_frame, 
                     text=f"Predicted Maximum Daily Expense: ₹{np.max(predictions):.2f}").pack()
            ttk.Label(stats_frame, 
                     text=f"Predicted Total for Next 30 Days: ₹{np.sum(predictions):.2f}").pack()

        except Exception as e:
            ttk.Label(frame, text=f"Error generating forecast: {str(e)}").pack(pady=20)
        finally:
            if connection:
                connection.close()

    def show_spending_patterns(self, frame):
        """Show AI analysis of spending patterns"""
        try:
            connection = self.get_db_connection()
            cursor = connection.cursor()
            
            # Get category-wise spending patterns
            cursor.execute("""
                SELECT 
                    category,
                    COUNT(*) as frequency,
                    AVG(amount) as avg_amount,
                    MAX(amount) as max_amount,
                    MIN(amount) as min_amount
                FROM expenses
                WHERE username = ?
                GROUP BY category
            """, (self.current_user,))
            
            patterns = cursor.fetchall()
            
            if not patterns:
                ttk.Label(frame, text="No expense data available for pattern analysis").pack(pady=20)
                return

            # Create Treeview for patterns
            tree = ttk.Treeview(frame, columns=('Category', 'Frequency', 'Average', 'Maximum', 'Minimum'), show='headings')
            tree.heading('Category', text='Category')
            tree.heading('Frequency', text='Frequency')
            tree.heading('Average', text='Average Amount')
            tree.heading('Maximum', text='Maximum Amount')
            tree.heading('Minimum', text='Minimum Amount')
            
            for pattern in patterns:
                tree.insert('', 'end', values=(
                    pattern[0],
                    pattern[1],
                    f"₹{pattern[2]:.2f}",
                    f"₹{pattern[3]:.2f}",
                    f"₹{pattern[4]:.2f}"
                ))
            
            tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

            # Add insights text
            insights_frame = ttk.LabelFrame(frame, text="Key Insights", padding="10")
            insights_frame.pack(fill=tk.X, padx=10, pady=5)

            # Find most frequent category
            most_frequent = max(patterns, key=lambda x: x[1])
            ttk.Label(insights_frame, 
                     text=f"Most frequent expense category: {most_frequent[0]} ({most_frequent[1]} times)").pack()

            # Find highest average spending category
            highest_avg = max(patterns, key=lambda x: x[2])
            ttk.Label(insights_frame, 
                     text=f"Highest average spending: {highest_avg[0]} (₹{highest_avg[2]:.2f})").pack()

        except Exception as e:
            ttk.Label(frame, text=f"Error analyzing patterns: {str(e)}").pack(pady=20)
        finally:
            if connection:
                connection.close()

    def show_recommendations(self, frame):
        """Show AI-powered recommendations"""
        try:
            income = self.get_user_income()
            total_expenses = self.get_total_expenses()
            
            # Create recommendations frame
            recommendations_frame = ttk.LabelFrame(frame, text="Smart Recommendations", padding="10")
            recommendations_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

            # Calculate metrics
            expense_ratio = (total_expenses / income * 100) if income > 0 else 0
            
            # Get category distribution
            connection = self.get_db_connection()
            cursor = connection.cursor()
            cursor.execute("""
                SELECT category, SUM(amount) as total
                FROM expenses
                WHERE username = ?
                GROUP BY category
            """, (self.current_user,))
            categories = cursor.fetchall()
            
            # Generate recommendations
            recommendations = []
            
            # Budget recommendations
            if expense_ratio > 90:
                recommendations.append("⚠️ Warning: You're spending more than 90% of your income!")
            elif expense_ratio > 70:
                recommendations.append("⚠️ Consider reducing expenses to save more of your income.")
            
            # Category-specific recommendations
            if categories:
                total_spent = sum(cat[1] for cat in categories)
                for category, amount in categories:
                    category_percentage = (amount / total_spent * 100)
                    if category_percentage > 40:
                        recommendations.append(
                            f"📊 Your {category} expenses ({category_percentage:.1f}%) seem high. "
                            "Consider setting a budget for this category."
                        )
            
            # Savings recommendation
            if income > 0:
                savings_rate = ((income - total_expenses) / income * 100)
                if savings_rate < 20:
                    recommendations.append(
                        "💰 Your savings rate is below recommended 20%. "
                        "Try to increase savings for financial security."
                    )
            
            # Display recommendations
            for i, rec in enumerate(recommendations, 1):
                ttk.Label(recommendations_frame, text=f"{i}. {rec}", wraplength=400).pack(pady=5, anchor='w')
            
            if not recommendations:
                ttk.Label(recommendations_frame, 
                         text="Great job! Your spending patterns look healthy.").pack(pady=20)

            # Add summary metrics
            metrics_frame = ttk.LabelFrame(frame, text="Financial Metrics", padding="10")
            metrics_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(metrics_frame, text=f"Expense to Income Ratio: {expense_ratio:.1f}%").pack()
            if income > 0:
                ttk.Label(metrics_frame, text=f"Savings Rate: {((income - total_expenses) / income * 100):.1f}%").pack()

        except Exception as e:
            ttk.Label(frame, text=f"Error generating recommendations: {str(e)}").pack(pady=20)
        finally:
            if connection:
                connection.close()

    def remove_expense(self):
        """Remove selected expense from the treeview and database"""
        # Get selected item
        selected_item = self.expenses_tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select an expense to remove")
            return
        
        # Confirm deletion
        if not messagebox.askyesno("Confirm", "Are you sure you want to remove this expense?"):
            return
        
        try:
            # Get the values of selected item
            item_values = self.expenses_tree.item(selected_item)['values']
            date, category, amount, description = item_values
            
            connection = self.get_db_connection()
            cursor = connection.cursor()
            
            # Delete from database
            cursor.execute("""
                DELETE FROM expenses 
                WHERE username = ? AND date = ? AND category = ? AND amount = ? AND description = ?
            """, (self.current_user, date, category, amount, description))
            
            connection.commit()
            
            # Remove from treeview
            self.expenses_tree.delete(selected_item)
            messagebox.showinfo("Success", "Expense removed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove expense: {str(e)}")
        finally:
            if connection:
                connection.close()

    def clear_all_expenses(self):
        """Clear all expenses for the current user"""
        if not messagebox.askyesno("Confirm", "Are you sure you want to remove ALL expenses? This cannot be undone!"):
            return
        
        try:
            connection = self.get_db_connection()
            cursor = connection.cursor()
            
            # Delete all expenses for current user
            cursor.execute("DELETE FROM expenses WHERE username = ?", (self.current_user,))
            connection.commit()
            
            # Clear treeview
            for item in self.expenses_tree.get_children():
                self.expenses_tree.delete(item)
            
            messagebox.showinfo("Success", "All expenses cleared!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear expenses: {str(e)}")
        finally:
            if connection:
                connection.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = ExpenseTrackerApp(root)
    root.mainloop() 