import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
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
from PIL import Image
import io
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
import ttkbootstrap as ttk  # Replace tkinter.ttk with ttkbootstrap
from ttkbootstrap.constants import *  # Import ttkbootstrap constants
from ttkbootstrap.style import Style

METALLIC_GREY = "#71797E"  # For text and borders
OCEAN_BLUE = "#0077BE"     # For buttons and accents
LIGHT_GREY = "#F5F5F5"     # For hover effects
WHITE = "#FFFFFF"          # For background

class ExpenseTrackerApp:
    def __init__(self, root):
        # Create custom theme
        self.style = Style()
        
        # Configure base button style
        self.style.configure(
            "Custom.TButton",
            background=OCEAN_BLUE,
            foreground=WHITE,
            bordercolor=OCEAN_BLUE,
            focuscolor=OCEAN_BLUE,
            font=('Helvetica', 10)
        )
        
        # Use style map for button hover
        self.style.map("Custom.TButton",
            background=[("active", LIGHT_GREY)],
            foreground=[("active", OCEAN_BLUE)],
            bordercolor=[("active", OCEAN_BLUE)]
        )
        
        # Update frame style to white background
        self.style.configure(
            "Custom.TFrame",
            background=WHITE
        )
        
        # Update label style
        self.style.configure(
            "Custom.TLabel",
            background=WHITE,
            foreground=METALLIC_GREY
        )
        
        # Update entry style
        self.style.configure(
            "Custom.TEntry",
            fieldbackground=WHITE,
            foreground=METALLIC_GREY
        )
        
        # Update treeview style
        self.style.configure(
            "Custom.Treeview",
            background=WHITE,
            fieldbackground=WHITE,
            foreground=METALLIC_GREY,
            selectbackground=OCEAN_BLUE,
            selectforeground=WHITE
        )
        
        # Update labelframe style
        self.style.configure(
            "Custom.TLabelframe",
            background=WHITE,
            foreground=METALLIC_GREY
        )
        
        # Update labelframe label style
        self.style.configure(
            "Custom.TLabelframe.Label",
            background=WHITE,
            foreground=METALLIC_GREY,
            font=('Helvetica', 10, 'bold')
        )
        
        # Update treeview heading style
        self.style.configure(
            "Custom.Treeview.Heading",
            background=OCEAN_BLUE,
            foreground=WHITE,
            relief="flat"
        )
        
        # Apply theme to root window
        self.root = root
        self.root.title("💰 Expense Tracker")
        self.root.geometry("800x600")
        self.root.configure(bg=WHITE)  # Set root background to white
        
        # Main container with white background
        self.main_container = ttk.Frame(
            self.root,
            padding="20",
            style="Custom.TFrame"
        )
        self.main_container.grid(
            row=0,
            column=0,
            sticky=(tk.W, tk.E, tk.N, tk.S),
            padx=20,
            pady=20
        )
        
        # Configure grid weights for responsive design
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        # Initialize database if it doesn't exist
        self.initialize_database()
        
        self.current_user = None
        
        # Initially show login frame
        self.show_login_frame()

    def show_login_frame(self):
        """Show the login frame with custom styling"""
        self.fade_out_widgets()
        
        # Login frame with custom styling
        login_frame = ttk.Frame(self.main_container, padding="20", style="Custom.TFrame")
        login_frame.grid(row=0, column=0, padx=20, pady=20)
        
        # Welcome header
        ttk.Label(
            login_frame,
            text="Welcome Back!",
            font=('Helvetica', 24, 'bold'),
            style="Custom.TLabel"
        ).grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Username
        ttk.Label(login_frame, text="👤 Username:", style="Custom.TLabel").grid(row=1, column=0, pady=5)
        self.username_var = tk.StringVar()
        username_entry = ttk.Entry(
            login_frame,
            width=30,
            style="Custom.TEntry",
            textvariable=self.username_var
        )
        username_entry.grid(row=1, column=1, pady=5)
        
        # Password
        ttk.Label(login_frame, text="🔒 Password:", style="Custom.TLabel").grid(row=2, column=0, pady=5)
        self.password_var = tk.StringVar()
        password_entry = ttk.Entry(
            login_frame,
            show="•",
            width=30,
            style="Custom.TEntry",
            textvariable=self.password_var
        )
        password_entry.grid(row=2, column=1, pady=5)
        
        # Button frame
        button_frame = ttk.Frame(login_frame, style="Custom.TFrame")
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        # Login button
        login_btn = ttk.Button(
            button_frame,
            text="Login",
            command=self.login,
            style="Custom.TButton",
            width=15
        )
        login_btn.pack(side='left', padx=5)
        
        # Register button
        register_btn = ttk.Button(
            button_frame,
            text="Register",
            command=self.show_register_frame,
            style="Custom.TButton",
            width=15
        )
        register_btn.pack(side='left', padx=5)

    def show_main_frame(self):
        """Show main frame with custom styling"""
        self.fade_out_widgets()
        
        # Create scrollable frame
        canvas = tk.Canvas(self.main_container, bg=WHITE, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, style="Custom.TFrame")
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Welcome message
        ttk.Label(
            scrollable_frame,
            text=f"Welcome, {self.current_user}!",
            font=('Helvetica', 16, 'bold'),
            style="Custom.TLabel"
        ).pack(pady=10)
        
        # Features section
        features_frame = ttk.LabelFrame(
            scrollable_frame,
            text="✨ Features",
            padding=10,
            style="Custom.TLabelframe"
        )
        features_frame.pack(fill='x', padx=10, pady=5)
        
        # Feature buttons with styling
        feature_buttons = [
            ("💳 Add Expense", self.show_add_expense_frame),
            ("💰 Income Management", self.show_income_management),
            ("📊 View Expenses", self.show_expenses_list),
            ("📈 Analytics", self.show_insights_window),
            ("🔄 Recurring Expense", self.add_recurring_expense),
            ("🎯 Budget Goals", self.set_budget_goals),
            ("📤 Export", self.export_expenses),
            ("➗ Split Expense", self.split_expense),
            ("📅 Budget Calendar", self.show_budget_calendar)
        ]
        
        for i, (text, command) in enumerate(feature_buttons):
            row = i // 3
            col = i % 3
            btn = ttk.Button(
                features_frame,
                text=text,
                command=command,
                style="Custom.TButton",
                width=20
            )
            btn.grid(row=row, column=col, padx=5, pady=5)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Logout button
        logout_btn = ttk.Button(
            scrollable_frame,
            text="Logout",
            command=self.logout,
            style="Custom.TButton",
            width=20
        )
        logout_btn.pack(pady=20)

    def show_register_frame(self):
        """Show the register frame with custom styling"""
        self.fade_out_widgets()
        
        # Register frame with custom styling
        register_frame = ttk.Frame(self.main_container, padding="20", style="Custom.TFrame")
        register_frame.grid(row=0, column=0, padx=20, pady=20)
        
        # Registration header
        ttk.Label(
            register_frame,
            text="Create Account",
            font=('Helvetica', 24, 'bold'),
            style="Custom.TLabel"
        ).grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Username
        ttk.Label(register_frame, text="👤 Username:", style="Custom.TLabel").grid(row=1, column=0, pady=5)
        self.reg_username_var = tk.StringVar()
        username_entry = ttk.Entry(
            register_frame,
            width=30,
            style="Custom.TEntry",
            textvariable=self.reg_username_var
        )
        username_entry.grid(row=1, column=1, pady=5)
        
        # Password
        ttk.Label(register_frame, text="🔒 Password:", style="Custom.TLabel").grid(row=2, column=0, pady=5)
        self.reg_password_var = tk.StringVar()
        password_entry = ttk.Entry(
            register_frame,
            show="•",
            width=30,
            style="Custom.TEntry",
            textvariable=self.reg_password_var
        )
        password_entry.grid(row=2, column=1, pady=5)
        
        # Button frame
        button_frame = ttk.Frame(register_frame, style="Custom.TFrame")
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        # Register button
        register_btn = ttk.Button(
            button_frame,
            text="Register",
            command=self.register,
            style="Custom.TButton",
            width=15
        )
        register_btn.pack(side='left', padx=5)
        
        # Back to Login button
        back_btn = ttk.Button(
            button_frame,
            text="Back to Login",
            command=self.show_login_frame,
            style="Custom.TButton",
            width=15
        )
        back_btn.pack(side='left', padx=5)

    def fade_out_widgets(self):
        """Clear all widgets from main container"""
        for widget in self.main_container.winfo_children():
            widget.destroy()

    def initialize_database(self):
        """Initialize database with required tables if they don't exist"""
        try:
            connection = sqlite3.connect(DB_PATH)
            cursor = connection.cursor()
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL,
                    income REAL DEFAULT 0
                )
            ''')
            
            # Create expenses table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS expenses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    date DATE,
                    category TEXT,
                    amount REAL,
                    description TEXT,
                    FOREIGN KEY (username) REFERENCES users(username)
                )
            ''')
            
            connection.commit()
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to initialize database: {str(e)}")
        finally:
            if connection:
                connection.close()

    def get_db_connection(self):
        """Get database connection"""
        try:
            connection = sqlite3.connect(DB_PATH)
            return connection
        except Exception as e:
            messagebox.showerror("Database Error", str(e))
            return None

    def login(self):
        """Handle login"""
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
        """Handle registration"""
        username = self.reg_username_var.get()
        password = self.reg_password_var.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Please fill in all fields!")
            return
        
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

    def logout(self):
        """Handle logout"""
        self.current_user = None
        self.show_login_frame()

    def show_add_expense_frame(self):
        """Show the add expense section"""
        self.fade_out_widgets()
        
        expense_frame = ttk.LabelFrame(
            self.main_container,
            text="💳 Add Expense",
            padding="15",
            style="Custom.TLabelframe"
        )
        expense_frame.pack(fill='x', padx=10, pady=5)
        
        # Amount
        ttk.Label(expense_frame, text="Amount:", style="Custom.TLabel").pack(pady=5)
        self.amount_var = tk.StringVar()
        ttk.Entry(expense_frame, textvariable=self.amount_var, style="Custom.TEntry").pack(pady=5)
        
        # Category
        ttk.Label(expense_frame, text="Category:", style="Custom.TLabel").pack(pady=5)
        self.category_var = tk.StringVar()
        categories = ['Food', 'Transport', 'Entertainment', 'Utilities', 'Other']
        ttk.Combobox(expense_frame, textvariable=self.category_var, values=categories).pack(pady=5)
        
        # Description
        ttk.Label(expense_frame, text="Description:", style="Custom.TLabel").pack(pady=5)
        self.description_var = tk.StringVar()
        ttk.Entry(expense_frame, textvariable=self.description_var, style="Custom.TEntry").pack(pady=5)
        
        # Date
        ttk.Label(expense_frame, text="Date:", style="Custom.TLabel").pack(pady=5)
        self.date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(expense_frame, textvariable=self.date_var, style="Custom.TEntry").pack(pady=5)
        
        # Buttons
        button_frame = ttk.Frame(expense_frame, style="Custom.TFrame")
        button_frame.pack(pady=10)
        
        ttk.Button(
            button_frame,
            text="Add Expense",
            command=self.add_expense,
            style="Custom.TButton"
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame,
            text="Back",
            command=self.show_main_frame,
            style="Custom.TButton"
        ).pack(side='left', padx=5)

    def add_expense(self):
        """Add a new expense to the database"""
        try:
            amount = float(self.amount_var.get())
            category = self.category_var.get()
            description = self.description_var.get()
            date = self.date_var.get()
            
            if amount <= 0:
                messagebox.showerror("Error", "Amount must be greater than 0")
                return
            
            if not category:
                messagebox.showerror("Error", "Please select a category")
                return
            
            connection = self.get_db_connection()
            cursor = connection.cursor()
            
            cursor.execute("""
                INSERT INTO expenses (username, date, category, amount, description)
                VALUES (?, ?, ?, ?, ?)
            """, (self.current_user, date, category, amount, description))
            
            connection.commit()
            messagebox.showinfo("Success", "Expense added successfully!")
            
            # Clear the fields
            self.amount_var.set("")
            self.category_var.set("")
            self.description_var.set("")
            self.date_var.set(datetime.now().strftime("%Y-%m-%d"))
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid amount")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add expense: {str(e)}")
        finally:
            if connection:
                connection.close()

    def show_expenses_list(self):
        """Show the list of expenses"""
        self.fade_out_widgets()
        
        expenses_frame = ttk.LabelFrame(
            self.main_container,
            text="📊 Your Expenses",
            padding="15",
            style="Custom.TLabelframe"
        )
        expenses_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create Treeview
        self.expenses_tree = ttk.Treeview(
            expenses_frame,
            columns=('Date', 'Category', 'Amount', 'Description'),
            show='headings',
            style="Custom.Treeview"
        )
        
        # Configure columns
        self.expenses_tree.heading('Date', text='Date')
        self.expenses_tree.heading('Category', text='Category')
        self.expenses_tree.heading('Amount', text='Amount')
        self.expenses_tree.heading('Description', text='Description')
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(expenses_frame, orient="vertical", command=self.expenses_tree.yview)
        self.expenses_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.expenses_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Load expenses
        self.load_expenses()
        
        # Buttons frame
        button_frame = ttk.Frame(self.main_container, style="Custom.TFrame")
        button_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(
            button_frame,
            text="Remove Selected",
            command=self.remove_expense,
            style="Custom.TButton"
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame,
            text="Clear All",
            command=self.clear_all_expenses,
            style="Custom.TButton"
        ).pack(side='left', padx=5)
        
        ttk.Button(
            button_frame,
            text="Back",
            command=self.show_main_frame,
            style="Custom.TButton"
        ).pack(side='right', padx=5)

    def load_expenses(self):
        """Load expenses from database into treeview"""
        # Clear existing items
        for item in self.expenses_tree.get_children():
            self.expenses_tree.delete(item)
        
        try:
            connection = self.get_db_connection()
            cursor = connection.cursor()
            
            cursor.execute("""
                SELECT date, category, amount, description 
                FROM expenses 
                WHERE username = ?
                ORDER BY date DESC
            """, (self.current_user,))
            
            for row in cursor.fetchall():
                self.expenses_tree.insert('', 'end', values=row)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load expenses: {str(e)}")
        finally:
            if connection:
                connection.close()

    def remove_expense(self):
        """Remove selected expense"""
        selected_item = self.expenses_tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select an expense to remove")
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to remove this expense?"):
            try:
                connection = self.get_db_connection()
                cursor = connection.cursor()
                
                # Get the values of the selected item
                values = self.expenses_tree.item(selected_item)['values']
                
                cursor.execute("""
                    DELETE FROM expenses 
                    WHERE username = ? AND date = ? AND category = ? AND amount = ? AND description = ?
                """, (self.current_user, values[0], values[1], values[2], values[3]))
                
                connection.commit()
                self.expenses_tree.delete(selected_item)
                messagebox.showinfo("Success", "Expense removed successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to remove expense: {str(e)}")
            finally:
                if connection:
                    connection.close()

    def show_income_management(self):
        """Show income management section"""
        self.fade_out_widgets()
        
        income_frame = ttk.LabelFrame(
            self.main_container,
            text="💰 Income Management",
            padding="15",
            style="Custom.TLabelframe"
        )
        income_frame.pack(fill='x', padx=10, pady=5)
        
        # Current Income Display
        current_income = self.get_user_income()
        ttk.Label(
            income_frame,
            text=f"Current Monthly Income: ₹{current_income:.2f}",
            style="Custom.TLabel",
            font=('Helvetica', 12, 'bold')
        ).pack(pady=10)
        
        # New Income Entry
        ttk.Label(income_frame, text="New Monthly Income:", style="Custom.TLabel").pack(pady=5)
        self.new_income_var = tk.StringVar()
        ttk.Entry(
            income_frame,
            textvariable=self.new_income_var,
            style="Custom.TEntry"
        ).pack(pady=5)
        
        # Update Button
        ttk.Button(
            income_frame,
            text="Update Income",
            command=self.update_income,
            style="Custom.TButton"
        ).pack(pady=10)
        
        # Back Button
        ttk.Button(
            income_frame,
            text="Back",
            command=self.show_main_frame,
            style="Custom.TButton"
        ).pack(pady=5)

    def get_user_income(self):
        """Get current user's income"""
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
        """Update user's income"""
        try:
            new_income = float(self.new_income_var.get())
            if new_income < 0:
                messagebox.showerror("Error", "Income cannot be negative")
                return
            
            connection = self.get_db_connection()
            cursor = connection.cursor()
            cursor.execute(
                "UPDATE users SET income = ? WHERE username = ?",
                (new_income, self.current_user)
            )
            connection.commit()
            
            messagebox.showinfo("Success", "Income updated successfully!")
            self.show_income_management()  # Refresh the frame
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid amount")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update income: {str(e)}")
        finally:
            if connection:
                connection.close()

    def show_insights_window(self):
        """Show analytics and insights with AI recommendations"""
        self.fade_out_widgets()
        
        insights_frame = ttk.LabelFrame(
            self.main_container,
            text="📈 Analytics & Insights",
            padding="15",
            style="Custom.TLabelframe"
        )
        insights_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create tabs
        notebook = ttk.Notebook(insights_frame)
        notebook.pack(fill='both', expand=True, pady=10)
        
        # Category Distribution Tab
        category_frame = ttk.Frame(notebook, style="Custom.TFrame")
        notebook.add(category_frame, text="Category Distribution")
        self.plot_category_distribution(category_frame)
        
        # Monthly Trend Tab
        trend_frame = ttk.Frame(notebook, style="Custom.TFrame")
        notebook.add(trend_frame, text="Monthly Trend")
        self.plot_monthly_trend(trend_frame)
        
        # AI Recommendations Tab
        ai_frame = ttk.Frame(notebook, style="Custom.TFrame")
        notebook.add(ai_frame, text="AI Recommendations")
        self.show_ai_recommendations(ai_frame)
        
        # Back Button
        ttk.Button(
            self.main_container,
            text="Back",
            command=self.show_main_frame,
            style="Custom.TButton"
        ).pack(pady=10)

    def plot_category_distribution(self, parent_frame):
        """Plot expense distribution by category"""
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
                ttk.Label(
                    parent_frame,
                    text="No expenses to show",
                    style="Custom.TLabel"
                ).pack(pady=20)
                return
            
            categories = [row[0] for row in data]
            amounts = [row[1] for row in data]
            
            # Create pie chart
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(amounts, labels=categories, autopct='%1.1f%%')
            ax.set_title("Expense Distribution by Category")
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, parent_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create chart: {str(e)}")
        finally:
            if connection:
                connection.close()

    def plot_monthly_trend(self, parent_frame):
        """Plot monthly expense trend"""
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
                ttk.Label(
                    parent_frame,
                    text="No expenses to show",
                    style="Custom.TLabel"
                ).pack(pady=20)
                return
            
            months = [row[0] for row in data]
            amounts = [row[1] for row in data]
            
            # Create line chart
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(months, amounts, marker='o')
            ax.set_title("Monthly Expense Trend")
            ax.set_xlabel("Month")
            ax.set_ylabel("Total Expenses (₹)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, parent_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create chart: {str(e)}")
        finally:
            if connection:
                connection.close()

    def show_ai_recommendations(self, parent_frame):
        """Show AI-based spending recommendations with smaller dataset requirements"""
        try:
            connection = self.get_db_connection()
            cursor = connection.cursor()
            
            # Get recent expenses (last 10 entries)
            cursor.execute("""
                SELECT date, category, amount
                FROM expenses
                WHERE username = ?
                ORDER BY date DESC
                LIMIT 10
            """, (self.current_user,))
            
            expenses = cursor.fetchall()
            
            if len(expenses) < 5:  # Minimum 5 entries required
                ttk.Label(
                    parent_frame,
                    text="Please add at least 5 expenses for personalized recommendations.",
                    style="Custom.TLabel"
                ).pack(pady=20)
                return
            
            # Create recommendations frame
            recommendations_frame = ttk.LabelFrame(
                parent_frame,
                text="💡 Smart Recommendations",
                padding=10,
                style="Custom.TLabelframe"
            )
            recommendations_frame.pack(fill='x', padx=10, pady=5)
            
            # Calculate basic statistics
            total_spent = sum(amount for _, _, amount in expenses)
            avg_per_expense = total_spent / len(expenses)
            
            # Get category breakdown
            category_totals = {}
            for _, category, amount in expenses:
                category_totals[category] = category_totals.get(category, 0) + amount
            
            # Find highest spending category
            highest_category = max(category_totals.items(), key=lambda x: x[1])
            
            # Display recent spending summary
            ttk.Label(
                recommendations_frame,
                text=f"Recent Spending Analysis (Last {len(expenses)} transactions)",
                style="Custom.TLabel",
                font=('Helvetica', 10, 'bold')
            ).pack(anchor='w', pady=5)
            
            ttk.Label(
                recommendations_frame,
                text=f"Total spent: ₹{total_spent:.2f}",
                style="Custom.TLabel"
            ).pack(anchor='w', pady=2)
            
            # Category-specific recommendations
            ttk.Label(
                recommendations_frame,
                text="\nCategory Insights:",
                style="Custom.TLabel",
                font=('Helvetica', 10, 'bold')
            ).pack(anchor='w', pady=5)
            
            for category, amount in category_totals.items():
                percentage = (amount / total_spent) * 100
                ttk.Label(
                    recommendations_frame,
                    text=f"{category}: ₹{amount:.2f} ({percentage:.1f}%)",
                    style="Custom.TLabel"
                ).pack(anchor='w', pady=1)
            
            # Smart recommendations based on patterns
            ttk.Label(
                recommendations_frame,
                text="\nPersonalized Recommendations:",
                style="Custom.TLabel",
                font=('Helvetica', 10, 'bold')
            ).pack(anchor='w', pady=5)
            
            # High expense alerts
            high_expenses = [
                (date, category, amount) 
                for date, category, amount in expenses 
                if amount > avg_per_expense * 1.5
            ]
            if high_expenses:
                ttk.Label(
                    recommendations_frame,
                    text="⚠️ High Expense Alert:",
                    style="Custom.TLabel",
                    foreground='red'
                ).pack(anchor='w', pady=2)
                for date, category, amount in high_expenses:
                    ttk.Label(
                        recommendations_frame,
                        text=f"• {category} on {date}: ₹{amount:.2f}",
                        style="Custom.TLabel"
                    ).pack(anchor='w', pady=1)
            
            # Category-specific advice
            if highest_category[1] > total_spent * 0.4:
                ttk.Label(
                    recommendations_frame,
                    text=f"\n💡 Tip: Your {highest_category[0]} expenses are quite high "
                         f"({(highest_category[1]/total_spent*100):.1f}% of total)",
                    style="Custom.TLabel"
                ).pack(anchor='w', pady=2)
            
            # Budget recommendations
            monthly_income = self.get_user_income()
            if monthly_income > 0:
                recommended_budget = {
                    'Essential': monthly_income * 0.5,  # 50% for essentials
                    'Savings': monthly_income * 0.3,    # 30% for savings
                    'Lifestyle': monthly_income * 0.2   # 20% for lifestyle
                }
                
                ttk.Label(
                    recommendations_frame,
                    text="\n📊 Recommended Monthly Budget:",
                    style="Custom.TLabel",
                    font=('Helvetica', 10, 'bold')
                ).pack(anchor='w', pady=5)
                
                for category, amount in recommended_budget.items():
                    ttk.Label(
                        recommendations_frame,
                        text=f"{category}: ₹{amount:.2f}",
                        style="Custom.TLabel"
                    ).pack(anchor='w', pady=1)
            
            # Quick tips based on spending patterns
            tips = [
                "Try using cash for discretionary spending - it helps control impulse purchases",
                "Consider meal planning to reduce food expenses",
                "Look for subscriptions you might not be using",
                "Set aside money for emergencies",
                "Track daily expenses to identify unnecessary spending"
            ]
            
            ttk.Label(
                recommendations_frame,
                text="\n✨ Quick Tips:",
                style="Custom.TLabel",
                font=('Helvetica', 10, 'bold')
            ).pack(anchor='w', pady=5)
            
            for tip in tips[:3]:  # Show only 3 random tips
                ttk.Label(
                    recommendations_frame,
                    text=f"• {tip}",
                    style="Custom.TLabel"
                ).pack(anchor='w', pady=1)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate recommendations: {str(e)}")
        finally:
            if connection:
                connection.close()

    def add_recurring_expense(self):
        """Add recurring expenses (monthly, weekly, etc.)"""
        recurring_window = tk.Toplevel(self.root)
        recurring_window.title("Add Recurring Expense")
        recurring_window.geometry("400x400")
        recurring_window.configure(bg=WHITE)
        
        frame = ttk.LabelFrame(
            recurring_window,
            text="🔄 Recurring Expense",
            padding="15",
            style="Custom.TLabelframe"
        )
        frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Amount
        ttk.Label(frame, text="Amount:", style="Custom.TLabel").pack(pady=5)
        amount_var = tk.StringVar()
        ttk.Entry(frame, textvariable=amount_var, style="Custom.TEntry").pack(pady=5)
        
        # Description
        ttk.Label(frame, text="Description:", style="Custom.TLabel").pack(pady=5)
        desc_var = tk.StringVar()
        ttk.Entry(frame, textvariable=desc_var, style="Custom.TEntry").pack(pady=5)
        
        # Frequency
        ttk.Label(frame, text="Frequency:", style="Custom.TLabel").pack(pady=5)
        freq_var = tk.StringVar()
        frequencies = ['Weekly', 'Monthly', 'Yearly']
        ttk.Combobox(frame, textvariable=freq_var, values=frequencies).pack(pady=5)
        
        def save_recurring():
            try:
                amount = float(amount_var.get())
                desc = desc_var.get()
                freq = freq_var.get()
                
                if amount <= 0:
                    messagebox.showerror("Error", "Amount must be greater than 0")
                    return
                    
                if not freq:
                    messagebox.showerror("Error", "Please select frequency")
                    return
                
                connection = self.get_db_connection()
                cursor = connection.cursor()
                
                next_date = datetime.now()
                if freq == 'Weekly':
                    next_date += timedelta(days=7)
                elif freq == 'Monthly':
                    next_date = next_date.replace(month=next_date.month + 1)
                else:  # Yearly
                    next_date = next_date.replace(year=next_date.year + 1)
                
                cursor.execute("""
                    INSERT INTO recurring_expenses 
                    (username, amount, frequency, description, next_date)
                    VALUES (?, ?, ?, ?, ?)
                """, (self.current_user, amount, freq, desc, next_date.strftime('%Y-%m-%d')))
                
                connection.commit()
                messagebox.showinfo("Success", "Recurring expense added!")
                recurring_window.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", str(e))
            finally:
                if connection:
                    connection.close()
        
        ttk.Button(
            frame,
            text="Save",
            command=save_recurring,
            style="Custom.TButton"
        ).pack(pady=10)

    def set_budget_goals(self):
        """Set budget goals for different categories"""
        goals_window = tk.Toplevel(self.root)
        goals_window.title("Budget Goals")
        goals_window.geometry("400x500")
        goals_window.configure(bg=WHITE)
        
        frame = ttk.LabelFrame(
            goals_window,
            text="🎯 Budget Goals",
            padding="15",
            style="Custom.TLabelframe"
        )
        frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Category selection
        ttk.Label(frame, text="Category:", style="Custom.TLabel").pack(pady=5)
        category_var = tk.StringVar()
        categories = ['Food', 'Transport', 'Entertainment', 'Utilities', 'Other']
        ttk.Combobox(frame, textvariable=category_var, values=categories).pack(pady=5)
        
        # Amount
        ttk.Label(frame, text="Monthly Budget:", style="Custom.TLabel").pack(pady=5)
        amount_var = tk.StringVar()
        ttk.Entry(frame, textvariable=amount_var, style="Custom.TEntry").pack(pady=5)
        
        def save_goal():
            try:
                amount = float(amount_var.get())
                category = category_var.get()
                
                if amount <= 0:
                    messagebox.showerror("Error", "Amount must be greater than 0")
                    return
                    
                if not category:
                    messagebox.showerror("Error", "Please select category")
                    return
                
                connection = self.get_db_connection()
                cursor = connection.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO budget_goals (username, category, amount)
                    VALUES (?, ?, ?)
                """, (self.current_user, category, amount))
                
                connection.commit()
                messagebox.showinfo("Success", "Budget goal set!")
                display_goals()
                
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid amount")
            except Exception as e:
                messagebox.showerror("Error", str(e))
            finally:
                if connection:
                    connection.close()
        
        # Goals list
        goals_list = ttk.Frame(frame, style="Custom.TFrame")
        goals_list.pack(fill='both', expand=True, pady=10)
        
        def display_goals():
            for widget in goals_list.winfo_children():
                widget.destroy()
                
            try:
                connection = self.get_db_connection()
                cursor = connection.cursor()
                cursor.execute("""
                    SELECT category, amount 
                    FROM budget_goals 
                    WHERE username = ?
                """, (self.current_user,))
                
                for category, amount in cursor.fetchall():
                    goal_frame = ttk.Frame(goals_list, style="Custom.TFrame")
                    goal_frame.pack(fill='x', pady=2)
                    
                    ttk.Label(
                        goal_frame,
                        text=f"{category}: ₹{amount:.2f}",
                        style="Custom.TLabel"
                    ).pack(side='left')
                    
                    def delete_goal(cat=category):
                        if messagebox.askyesno("Confirm", f"Delete budget goal for {cat}?"):
                            try:
                                cursor.execute("""
                                    DELETE FROM budget_goals 
                                    WHERE username = ? AND category = ?
                                """, (self.current_user, cat))
                                connection.commit()
                                display_goals()
                            except Exception as e:
                                messagebox.showerror("Error", str(e))
                    
                    ttk.Button(
                        goal_frame,
                        text="Delete",
                        command=delete_goal,
                        style="Custom.TButton"
                    ).pack(side='right')
                    
            except Exception as e:
                messagebox.showerror("Error", str(e))
            finally:
                if connection:
                    connection.close()
        
        ttk.Button(
            frame,
            text="Set Goal",
            command=save_goal,
            style="Custom.TButton"
        ).pack(pady=10)
        
        display_goals()

    def export_expenses(self):
        """Export expenses to Excel/PDF"""
        export_window = tk.Toplevel(self.root)
        export_window.title("Export Expenses")
        export_window.geometry("300x200")
        export_window.configure(bg=WHITE)
        
        frame = ttk.LabelFrame(
            export_window,
            text="📤 Export Options",
            padding="15",
            style="Custom.TLabelframe"
        )
        frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        def export_to_excel():
            try:
                connection = self.get_db_connection()
                query = """
                    SELECT date, category, amount, description 
                    FROM expenses 
                    WHERE username = ?
                    ORDER BY date DESC
                """
                df = pd.read_sql_query(query, connection, params=(self.current_user,))
                
                filename = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx")]
                )
                
                if filename:
                    df.to_excel(filename, index=False)
                    messagebox.showinfo("Success", "Expenses exported to Excel!")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")
            finally:
                if connection:
                    connection.close()
        
        def export_to_pdf():
            try:
                connection = self.get_db_connection()
                cursor = connection.cursor()
                
                cursor.execute("""
                    SELECT date, category, amount, description 
                    FROM expenses 
                    WHERE username = ?
                    ORDER BY date DESC
                """, (self.current_user,))
                
                data = cursor.fetchall()
                if not data:
                    messagebox.showwarning("Warning", "No expenses to export")
                    return
                
                filename = filedialog.asksaveasfilename(
                    defaultextension=".pdf",
                    filetypes=[("PDF files", "*.pdf")]
                )
                
                if filename:
                    doc = SimpleDocTemplate(filename, pagesize=letter)
                    elements = []
                    
                    # Add title
                    elements.append(Table(
                        [["Expense Report"]],
                        style=TableStyle([
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, -1), 16),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
                        ])
                    ))
                    
                    # Add expense data
                    table_data = [["Date", "Category", "Amount", "Description"]] + data
                    t = Table(table_data)
                    t.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ]))
                    elements.append(t)
                    
                    doc.build(elements)
                    messagebox.showinfo("Success", "Expenses exported to PDF!")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")
            finally:
                if connection:
                    connection.close()
        
        ttk.Button(
            frame,
            text="Export to Excel",
            command=export_to_excel,
            style="Custom.TButton"
        ).pack(pady=10)
        
        ttk.Button(
            frame,
            text="Export to PDF",
            command=export_to_pdf,
            style="Custom.TButton"
        ).pack(pady=10)

    def split_expense(self):
        """Split expenses between multiple people"""
        split_window = tk.Toplevel(self.root)
        split_window.title("Split Expense")
        split_window.geometry("500x600")
        split_window.configure(bg=WHITE)
        
        frame = ttk.LabelFrame(
            split_window,
            text="➗ Split Expense",
            padding="15",
            style="Custom.TLabelframe"
        )
        frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Amount
        ttk.Label(frame, text="Total Amount:", style="Custom.TLabel").pack(pady=5)
        amount_var = tk.StringVar()
        ttk.Entry(frame, textvariable=amount_var, style="Custom.TEntry").pack(pady=5)
        
        # People frame
        people_frame = ttk.Frame(frame, style="Custom.TFrame")
        people_frame.pack(fill='x', pady=10)
        
        people = []  # List to store (name_entry, share_entry) tuples
        
        def add_person():
            person_frame = ttk.Frame(people_frame, style="Custom.TFrame")
            person_frame.pack(fill='x', pady=2)
            
            ttk.Label(person_frame, text="Name:", style="Custom.TLabel").pack(side='left', padx=5)
            name_entry = ttk.Entry(person_frame, style="Custom.TEntry")
            name_entry.pack(side='left', padx=5)
            
            ttk.Label(person_frame, text="Share:", style="Custom.TLabel").pack(side='left', padx=5)
            share_entry = ttk.Entry(person_frame, width=10, style="Custom.TEntry")
            share_entry.pack(side='left', padx=5)
            share_entry.insert(0, "1")  # Default share
            
            def remove_person():
                people.remove((name_entry, share_entry))
                person_frame.destroy()
            
            ttk.Button(
                person_frame,
                text="Remove",
                command=remove_person,
                style="Custom.TButton"
            ).pack(side='left', padx=5)
            
            people.append((name_entry, share_entry))
        
        # Add at least two people
        add_person()
        add_person()
        
        ttk.Button(
            frame,
            text="Add Person",
            command=add_person,
            style="Custom.TButton"
        ).pack(pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(frame, text="Split Results", style="Custom.TLabelframe")
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        def calculate_split():
            for widget in results_frame.winfo_children():
                widget.destroy()
            
            try:
                total = float(amount_var.get())
                if total <= 0:
                    messagebox.showerror("Error", "Amount must be greater than 0")
                    return
                
                shares_list = []
                names_list = []
                
                for name_entry, share_entry in people:
                    name = name_entry.get().strip()
                    share = share_entry.get().strip()
                    
                    if not name:
                        messagebox.showerror("Error", "Please enter all names")
                        return
                    if not share:
                        messagebox.showerror("Error", "Please enter all shares")
                        return
                    
                    try:
                        share_value = float(share)
                        if share_value <= 0:
                            messagebox.showerror("Error", f"Share for {name} must be greater than 0")
                            return
                        shares_list.append(share_value)
                        names_list.append(name)
                    except ValueError:
                        messagebox.showerror("Error", f"Invalid share value for {name}")
                        return
                
                total_shares = sum(shares_list)
                
                # Display results
                ttk.Label(
                    results_frame,
                    text="Split Results:",
                    style="Custom.TLabel",
                    font=('Helvetica', 10, 'bold')
                ).pack(anchor='w', pady=5)
                
                for name, share in zip(names_list, shares_list):
                    amount = (share / total_shares) * total
                    percentage = (share / total_shares) * 100
                    result_text = f"{name}: ₹{amount:.2f} ({percentage:.1f}%)"
                    ttk.Label(
                        results_frame,
                        text=result_text,
                        style="Custom.TLabel"
                    ).pack(anchor='w')
                    
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid amount")
        
        ttk.Button(
            frame,
            text="Calculate Split",
            command=calculate_split,
            style="Custom.TButton"
        ).pack(pady=10)

    def show_budget_calendar(self):
        """Show expenses in a calendar view"""
        calendar_window = tk.Toplevel(self.root)
        calendar_window.title("Budget Calendar")
        calendar_window.geometry("800x600")
        calendar_window.configure(bg=WHITE)
        
        frame = ttk.LabelFrame(
            calendar_window,
            text="📅 Budget Calendar",
            padding="15",
            style="Custom.TLabelframe"
        )
        frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Month navigation
        nav_frame = ttk.Frame(frame, style="Custom.TFrame")
        nav_frame.pack(fill='x', pady=10)
        
        current_date = datetime.now()
        current_month = current_date.month
        current_year = current_date.year
        
        month_label = ttk.Label(
            nav_frame,
            text=current_date.strftime("%B %Y"),
            style="Custom.TLabel",
            font=('Helvetica', 12, 'bold')
        )
        month_label.pack(side='left', padx=10)
        
        def update_calendar(offset):
            nonlocal current_month, current_year
            if offset == 1:  # Next month
                if current_month == 12:
                    current_month = 1
                    current_year += 1
                else:
                    current_month += 1
            else:  # Previous month
                if current_month == 1:
                    current_month = 12
                    current_year -= 1
                else:
                    current_month -= 1
            
            month_label.config(text=f"{datetime(current_year, current_month, 1).strftime('%B %Y')}")
            display_calendar()
        
        ttk.Button(
            nav_frame,
            text="← Previous",
            command=lambda: update_calendar(-1),
            style="Custom.TButton"
        ).pack(side='left', padx=5)
        
        ttk.Button(
            nav_frame,
            text="Next →",
            command=lambda: update_calendar(1),
            style="Custom.TButton"
        ).pack(side='left', padx=5)
        
        # Calendar frame
        calendar_frame = ttk.Frame(frame, style="Custom.TFrame")
        calendar_frame.pack(fill='both', expand=True, pady=10)
        
        # Day headers
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for i, day in enumerate(days):
            ttk.Label(
                calendar_frame,
                text=day,
                style="Custom.TLabel",
                font=('Helvetica', 10, 'bold')
            ).grid(row=0, column=i, padx=2, pady=2)
        
        def display_calendar():
            # Clear previous calendar
            for widget in calendar_frame.winfo_children():
                if int(widget.grid_info()['row']) > 0:
                    widget.destroy()
            
            # Get expenses for the month
            try:
                connection = self.get_db_connection()
                cursor = connection.cursor()
                cursor.execute("""
                    SELECT date, SUM(amount) as total
                    FROM expenses
                    WHERE username = ? 
                    AND strftime('%Y-%m', date) = ?
                    GROUP BY date
                """, (self.current_user, f"{current_year}-{current_month:02d}"))
                
                expenses = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Calculate calendar dates
                first_day = datetime(current_year, current_month, 1)
                if current_month == 12:
                    last_day = datetime(current_year + 1, 1, 1) - timedelta(days=1)
                else:
                    last_day = datetime(current_year, current_month + 1, 1) - timedelta(days=1)
                
                start_weekday = first_day.weekday()
                
                # Create calendar grid
                day = 1
                for week in range(6):
                    for weekday in range(7):
                        if (week == 0 and weekday < start_weekday) or day > last_day.day:
                            ttk.Label(calendar_frame, text="", style="Custom.TLabel").grid(
                                row=week+1, column=weekday, padx=2, pady=2
                            )
                        else:
                            date_str = f"{current_year}-{current_month:02d}-{day:02d}"
                            day_frame = ttk.Frame(calendar_frame, style="Custom.TFrame")
                            day_frame.grid(row=week+1, column=weekday, sticky='nsew', padx=1, pady=1)
                            
                            # Day number
                            ttk.Label(
                                day_frame,
                                text=str(day),
                                style="Custom.TLabel",
                                font=('Helvetica', 10, 'bold')
                            ).pack()
                            
                            # Expense amount if exists
                            if date_str in expenses:
                                amount_label = ttk.Label(
                                    day_frame,
                                    text=f"₹{expenses[date_str]:.2f}",
                                    style="Custom.TLabel"
                                )
                                amount_label.pack()
                                
                                # Color code based on amount
                                if expenses[date_str] > 1000:
                                    amount_label.configure(foreground='red')
                                else:
                                    amount_label.configure(foreground='green')
                            
                            day += 1
                            
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load calendar: {str(e)}")
            finally:
                if connection:
                    connection.close()
        
        # Display initial calendar
        display_calendar()

    def clear_all_expenses(self):
        """Clear all expenses for the current user"""
        if messagebox.askyesno("Confirm", "Are you sure you want to delete all expenses? This cannot be undone."):
            try:
                connection = self.get_db_connection()
                cursor = connection.cursor()
                
                cursor.execute("""
                    DELETE FROM expenses 
                    WHERE username = ?
                """, (self.current_user,))
                
                connection.commit()
                messagebox.showinfo("Success", "All expenses cleared successfully!")
                
                # Refresh the expenses list
                self.load_expenses()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear expenses: {str(e)}")
            finally:
                if connection:
                    connection.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = ExpenseTrackerApp(root)
    root.mainloop()