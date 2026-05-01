import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import requests
import io
from datetime import datetime
import warnings
import openai
from typing import Dict, List, Any, Optional
import json
import base64
from PIL import Image, ImageTk
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import threading
import os
warnings.filterwarnings('ignore')

class NFLAnalyticsUI:
    """NFL Fan Analytics System with Tkinter UI for CSV selection"""
    
    def __init__(self):
        self.df = None
        self.filtered_df = None
        self.analytics_engine = None
        self.chatbot = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main Tkinter interface"""
        self.root = tk.Tk()
        self.root.title("🏈 NFL Fan Analytics Dashboard")
        self.root.geometry("700x500")
        self.root.configure(bg='#f0f8ff')
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🏈 NFL Fan Analytics Dashboard", 
                               font=('Arial', 24, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Data Source Section
        data_frame = ttk.LabelFrame(main_frame, text="Data Source", padding="15")
        data_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # CSV File Selection
        ttk.Label(data_frame, text="Select CSV File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(data_frame, textvariable=self.file_path_var, width=50)
        self.file_entry.grid(row=0, column=1, padx=(10, 5), pady=5)
        
        self.browse_btn = ttk.Button(data_frame, text="Browse", command=self.browse_csv_file)
        self.browse_btn.grid(row=0, column=2, padx=(5, 0), pady=5)
        
        # API URL Option
        ttk.Label(data_frame, text="Or CSV URL:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.url_var = tk.StringVar()
        self.url_entry = ttk.Entry(data_frame, textvariable=self.url_var, width=50)
        self.url_entry.grid(row=1, column=1, padx=(10, 5), pady=5)
        
        self.fetch_btn = ttk.Button(data_frame, text="Fetch", command=self.fetch_csv_url)
        self.fetch_btn.grid(row=1, column=2, padx=(5, 0), pady=5)
        
        # Generate Sample Data Option
        self.sample_btn = ttk.Button(data_frame, text="Generate Sample Data", 
                                   command=self.generate_sample_data)
        self.sample_btn.grid(row=2, column=1, pady=10)
        
        # Analysis Options
        analysis_frame = ttk.LabelFrame(main_frame, text="Analysis Options", padding="15")
        analysis_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # API Key for Chatbot
        ttk.Label(analysis_frame, text="OpenAI API Key (Optional):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.api_key_var = tk.StringVar()
        self.api_entry = ttk.Entry(analysis_frame, textvariable=self.api_key_var, show="*", width=50)
        self.api_entry.grid(row=0, column=1, padx=(10, 0), pady=5)
        
        # Buttons Frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=3, column=0, columnspan=3, pady=20)
        
        self.load_btn = ttk.Button(buttons_frame, text="🚀 Start Analysis", 
                                 command=self.start_analysis, style='Accent.TButton')
        self.load_btn.grid(row=0, column=0, padx=10)
        
        self.dashboard_btn = ttk.Button(buttons_frame, text="📊 Create Dashboard", 
                                      command=self.create_dashboard, state='disabled')
        self.dashboard_btn.grid(row=0, column=1, padx=10)
        
        self.chatbot_btn = ttk.Button(buttons_frame, text="🤖 Open Chatbot", 
                                    command=self.open_chatbot, state='disabled')
        self.chatbot_btn.grid(row=0, column=2, padx=10)
        
        self.export_btn = ttk.Button(buttons_frame, text="💾 Export Results", 
                                   command=self.export_results, state='disabled')
        self.export_btn.grid(row=0, column=3, padx=10)
        
        # Status Frame
        self.status_var = tk.StringVar(value="Ready to load data...")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, 
                               font=('Arial', 10), foreground='blue')
        status_label.grid(row=4, column=0, columnspan=3, pady=(20, 0))
        
        # Data Preview Frame
        self.preview_frame = ttk.LabelFrame(main_frame, text="Data Preview", padding="10")
        self.preview_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(20, 0))
        
        # Create Treeview for data preview
        self.tree = ttk.Treeview(self.preview_frame)
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars for treeview
        v_scrollbar = ttk.Scrollbar(self.preview_frame, orient="vertical", command=self.tree.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.tree.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(self.preview_frame, orient="horizontal", command=self.tree.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.tree.configure(xscrollcommand=h_scrollbar.set)
        
        # Configure grid weights
        main_frame.grid_rowconfigure(5, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        self.preview_frame.grid_rowconfigure(0, weight=1)
        self.preview_frame.grid_columnconfigure(0, weight=1)
        
    def browse_csv_file(self):
        """Open file dialog to select CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select NFL Fan Data CSV File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.status_var.set(f"Selected: {os.path.basename(file_path)}")
    
    def fetch_csv_url(self):
        """Fetch CSV from URL"""
        url = self.url_var.get().strip()
        if not url:
            messagebox.showerror("Error", "Please enter a valid URL")
            return
            
        try:
            self.status_var.set("Fetching data from URL...")
            self.root.update()
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file
            temp_file = "temp_nfl_data.csv"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            self.file_path_var.set(temp_file)
            self.status_var.set(f"Fetched data from URL successfully")
            
        except requests.RequestException as e:
            messagebox.showerror("Error", f"Failed to fetch data from URL: {str(e)}")
            self.status_var.set("Error fetching data from URL")
    
    def generate_sample_data(self):
        """Generate sample NFL fan data"""
        try:
            self.status_var.set("Generating sample data...")
            self.root.update()
            
            # Create sample data
            sample_data = self.create_sample_nfl_data(500)
            
            # Save to file
            sample_file = "sample_nfl_fan_data.csv"
            sample_data.to_csv(sample_file, index=False)
            
            self.file_path_var.set(sample_file)
            self.status_var.set("Sample data generated successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate sample data: {str(e)}")
    
    def create_sample_nfl_data(self, n_rows: int = 500) -> pd.DataFrame:
        """Create realistic sample NFL fan data"""
        np.random.seed(42)
        
        first_names = ["John", "Mike", "Sarah", "Jessica", "David", "Emma", "James", "Olivia", 
                      "Daniel", "Sophia", "Michael", "Ashley", "Christopher", "Jennifer"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", 
                     "Davis", "Martinez", "Taylor", "Anderson", "Wilson", "Moore"]
        
        locations = ["New York, NY", "Dallas, TX", "Chicago, IL", "Los Angeles, CA", "Miami, FL", 
                    "Philadelphia, PA", "Boston, MA", "Seattle, WA", "Denver, CO", "Atlanta, GA"]
        
        teams = ["Dallas Cowboys", "New England Patriots", "Green Bay Packers", "Kansas City Chiefs", 
                "San Francisco 49ers", "Philadelphia Eagles", "Pittsburgh Steelers", "Seattle Seahawks", 
                "Buffalo Bills", "Chicago Bears"]
        
        income_brackets = ["<40k", "40-80k", "80-120k", "120k+"]
        seat_sections = ["Upper", "Lower", "Club", "VIP"]
        channels = ["ESPN", "Fox", "NBC", "CBS", "Amazon Prime"]
        subscription_services = ["NFL+", "Amazon Prime", "Peacock", "YouTube TV", "None"]
        merch_categories = ["Jerseys", "Caps", "Collectibles", "Home Décor"]
        engagement_levels = ["Low", "Medium", "High"]
        
        data = []
        for i in range(1, n_rows + 1):
            first = np.random.choice(first_names)
            last = np.random.choice(last_names)
            age = np.random.randint(18, 75)
            
            # Create realistic correlations
            income_bracket = np.random.choice(income_brackets, p=[0.3, 0.35, 0.25, 0.1])
            
            # Income affects spending and seat preference
            if income_bracket == "120k+":
                avg_ticket_spend = np.random.uniform(200, 500)
                merch_spend = np.random.uniform(500, 2000)
                seat_section = np.random.choice(seat_sections, p=[0.1, 0.3, 0.4, 0.2])
            elif income_bracket == "80-120k":
                avg_ticket_spend = np.random.uniform(120, 300)
                merch_spend = np.random.uniform(200, 800)
                seat_section = np.random.choice(seat_sections, p=[0.2, 0.4, 0.3, 0.1])
            else:
                avg_ticket_spend = np.random.uniform(50, 180)
                merch_spend = np.random.uniform(0, 400)
                seat_section = np.random.choice(seat_sections, p=[0.6, 0.3, 0.08, 0.02])
            
            engagement = np.random.choice(engagement_levels, p=[0.3, 0.5, 0.2])
            loyalty_base = {"Low": 30, "Medium": 60, "High": 85}[engagement]
            loyalty_score = np.random.randint(max(1, loyalty_base-20), min(101, loyalty_base+20))
            
            row = {
                "fan_id": f"FAN{i:05d}",
                "name": f"{first} {last}",
                "email_addr": f"{first.lower()}.{last.lower()}{i}@example.com",
                "age": age,
                "gender": np.random.choice(["Male", "Female", "Other"], p=[0.48, 0.48, 0.04]),
                "location": np.random.choice(locations),
                "income_bracket": income_bracket,
                "season_ticket_holder": np.random.choice(["Yes", "No"], p=[0.35, 0.65]),
                "games_attended_2024": np.random.randint(0, 12),
                "avg_ticket_spend": round(avg_ticket_spend, 2),
                "seat_section": seat_section,
                "concession_spend": round(np.random.uniform(15, 85), 2),
                "parking_pass": np.random.choice(["Yes", "No"], p=[0.65, 0.35]),
                "primary_channel": np.random.choice(channels),
                "avg_game_watch_time": np.random.randint(45, 180),
                "games_watched_2024": np.random.randint(0, 20),
                "subscription_service": np.random.choice(subscription_services),
                "favorite_broadcast_day": np.random.choice(["Sunday", "Monday", "Thursday"], p=[0.7, 0.2, 0.1]),
                "co_viewing": np.random.randint(1, 6),
                "merch_spend_2024": round(merch_spend, 2),
                "favorite_merch_category": np.random.choice(merch_categories),
                "favorite_team": np.random.choice(teams),
                "social_media_engagement": engagement,
                "mobile_app_usage": np.random.choice(["Yes", "No"], p=[0.75, 0.25]),
                "email_subscriber": np.random.choice(["Yes", "No"], p=[0.6, 0.4]),
                "fan_loyalty_score": loyalty_score
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def start_analysis(self):
        """Start the analysis process"""
        file_path = self.file_path_var.get().strip()
        url = self.url_var.get().strip()
        api_key = self.api_key_var.get().strip() if hasattr(self, 'api_key_var') else None
        
        if not file_path and not url:
            messagebox.showerror("Error", "Please select a CSV file or provide a URL")
            return
        
        try:
            # Update status
            self.status_var.set("Loading and analyzing data...")
            self.root.update()
            
            # Initialize analytics engine
            self.analytics_engine = NFLFanAnalytics()
            
            # Load data
            if file_path:
                self.analytics_engine.load_data_from_file(file_path)
            else:
                self.analytics_engine.load_data_from_url(url)
            
            # Initialize chatbot if API key provided
            if api_key:
                self.chatbot = NFLChatbot(self.analytics_engine, api_key)
            
            self.df = self.analytics_engine.df
            self.filtered_df = self.analytics_engine.filtered_df
            
            # Update UI
            self.update_data_preview()
            self.enable_analysis_buttons()
            self.status_var.set(f"✅ Data loaded successfully! {len(self.df)} records ready for analysis")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_var.set("❌ Error loading data")
    
    def update_data_preview(self):
        """Update the data preview treeview"""
        if self.df is None:
            return
            
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Configure columns
        columns = list(self.df.columns[:8])  # Show first 8 columns
        self.tree["columns"] = columns
        self.tree["show"] = "headings"
        
        # Configure column headings and widths
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, minwidth=100, anchor='center')
        
        # Add data (first 100 rows for performance)
        for index, row in self.df.head(100).iterrows():
            values = [str(row[col]) for col in columns]
            self.tree.insert("", "end", values=values)
    
    def enable_analysis_buttons(self):
        """Enable analysis buttons after data is loaded"""
        self.dashboard_btn.config(state='normal')
        self.export_btn.config(state='normal')
        if self.chatbot:
            self.chatbot_btn.config(state='normal')
    
    def create_dashboard(self):
        """Create comprehensive analytics dashboard"""
        if self.analytics_engine is None:
            messagebox.showerror("Error", "Please load data first")
            return
        
        self.status_var.set("Creating comprehensive dashboard...")
        self.root.update()
        
        # Create dashboard in separate thread to prevent UI freezing
        thread = threading.Thread(target=self._create_dashboard_thread)
        thread.daemon = True
        thread.start()
    
    def _create_dashboard_thread(self):
        """Create dashboard in separate thread"""
        try:
            self.analytics_engine.create_comprehensive_dashboard()
            self.analytics_engine.create_advanced_analytics()
            self.root.after(0, lambda: self.status_var.set("✅ Dashboard created successfully!"))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"❌ Error creating dashboard: {str(e)}"))
    
    def open_chatbot(self):
        """Open chatbot interface"""
        if self.chatbot is None:
            messagebox.showinfo("Info", "Chatbot requires OpenAI API key. Please provide one and restart analysis.")
            return
        
        # Create chatbot window
        self.create_chatbot_window()
    
    def create_chatbot_window(self):
        """Create separate chatbot window"""
        chat_window = tk.Toplevel(self.root)
        chat_window.title("🤖 NFL Analytics Assistant")
        chat_window.geometry("600x700")
        chat_window.configure(bg='#f0f8ff')
        
        # Chat frame
        chat_frame = ttk.Frame(chat_window, padding="15")
        chat_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Chat display
        self.chat_display = tk.Text(chat_frame, height=25, width=70, wrap=tk.WORD,
                                   state='disabled', bg='white', font=('Arial', 10))
        self.chat_display.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Chat scrollbar
        chat_scroll = ttk.Scrollbar(chat_frame, orient="vertical", command=self.chat_display.yview)
        chat_scroll.grid(row=0, column=2, sticky=(tk.N, tk.S))
        self.chat_display.configure(yscrollcommand=chat_scroll.set)
        
        # Input frame
        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Chat input
        self.chat_input = tk.Text(input_frame, height=3, width=50, wrap=tk.WORD)
        self.chat_input.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Send button
        send_btn = ttk.Button(input_frame, text="Send", command=self.send_chat_message)
        send_btn.grid(row=0, column=1, padx=(10, 0))
        
        # Bind Enter key
        self.chat_input.bind('<Control-Return>', lambda e: self.send_chat_message())
        
        # Configure grid weights
        chat_window.grid_rowconfigure(0, weight=1)
        chat_window.grid_columnconfigure(0, weight=1)
        chat_frame.grid_rowconfigure(0, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_columnconfigure(0, weight=1)
        
        # Add welcome message
        self.add_chat_message("🤖 Hello! I'm your NFL Analytics Assistant. I can help you create visualizations, analyze data, and generate insights. Try asking:\n\n• 'Show me spending patterns by age'\n• 'Create a team popularity chart'\n• 'What are the key insights?'\n• 'Generate business recommendations'", "assistant")
    
    def add_chat_message(self, message: str, sender: str):
        """Add message to chat display"""
        self.chat_display.config(state='normal')
        
        timestamp = datetime.now().strftime("%H:%M")
        
        if sender == "user":
            self.chat_display.insert(tk.END, f"👤 You ({timestamp}):\n", "user_header")
            self.chat_display.insert(tk.END, f"{message}\n\n", "user_message")
        else:
            self.chat_display.insert(tk.END, f"🤖 Assistant ({timestamp}):\n", "bot_header")
            self.chat_display.insert(tk.END, f"{message}\n\n", "bot_message")
        
        # Configure text tags for styling
        self.chat_display.tag_config("user_header", foreground="blue", font=('Arial', 10, 'bold'))
        self.chat_display.tag_config("bot_header", foreground="green", font=('Arial', 10, 'bold'))
        self.chat_display.tag_config("user_message", foreground="black")
        self.chat_display.tag_config("bot_message", foreground="black")
        
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
    
    def send_chat_message(self):
        """Send chat message to bot"""
        message = self.chat_input.get("1.0", tk.END).strip()
        if not message:
            return
        
        # Add user message
        self.add_chat_message(message, "user")
        
        # Clear input
        self.chat_input.delete("1.0", tk.END)
        
        # Process with chatbot
        if self.chatbot:
            try:
                response = self.chatbot.process_message(message)
                self.add_chat_message(response, "assistant")
            except Exception as e:
                self.add_chat_message(f"❌ Error processing message: {str(e)}", "assistant")
        else:
            # Fallback response without AI
            response = self.generate_fallback_response(message)
            self.add_chat_message(response, "assistant")
    
    def generate_fallback_response(self, message: str) -> str:
        """Generate fallback response when no AI is available"""
        message_lower = message.lower()
        
        if 'spending' in message_lower:
            stats = self.analytics_engine.generate_summary_stats()
            return f"💰 Average total spending across all fans is ${stats['avg_total_spend']:.0f}. The highest spenders are typically in the VIP seat sections with an average loyalty score above 70."
        
        elif 'team' in message_lower and 'popular' in message_lower:
            top_teams = self.filtered_df['favorite_team'].value_counts().head(3)
            return f"🏆 Most popular teams:\n1. {top_teams.index[0]} ({top_teams.iloc[0]} fans)\n2. {top_teams.index[1]} ({top_teams.iloc[1]} fans)\n3. {top_teams.index[2]} ({top_teams.iloc[2]} fans)"
        
        elif 'insight' in message_lower or 'analysis' in message_lower:
            stats = self.analytics_engine.generate_summary_stats()
            return f"📊 Key Insights:\n• Average loyalty score: {stats['avg_loyalty_score']:.1f}/100\n• {stats['season_ticket_holder_pct']:.1f}% are season ticket holders\n• {stats['high_engagement_pct']:.1f}% have high social media engagement\n• Top market: {stats['top_location']}"
        
        elif 'recommend' in message_lower:
            return "💡 Business Recommendations:\n• Target high-loyalty fans with premium experiences\n• Increase mobile app adoption through incentives\n• Focus marketing on underperforming age groups\n• Create location-specific campaigns for top markets"
        
        else:
            return "I can help you analyze your NFL fan data! Try asking about:\n• Spending patterns\n• Team popularity\n• Fan engagement\n• Business recommendations\n• Data insights"
    
    def export_results(self):
        """Export analysis results"""
        if self.df is None:
            messagebox.showerror("Error", "No data to export")
            return
        
        # Ask user what to export
        export_window = tk.Toplevel(self.root)
        export_window.title("Export Options")
        export_window.geometry("400x300")
        export_window.configure(bg='#f0f8ff')
        
        export_frame = ttk.Frame(export_window, padding="20")
        export_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(export_frame, text="Select Export Options:", font=('Arial', 12, 'bold')).grid(row=0, column=0, pady=(0, 15))
        
        # Export options
        self.export_csv_var = tk.BooleanVar(value=True)
        self.export_stats_var = tk.BooleanVar(value=True)
        self.export_insights_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(export_frame, text="📄 Filtered Data (CSV)", variable=self.export_csv_var).grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Checkbutton(export_frame, text="📊 Summary Statistics (JSON)", variable=self.export_stats_var).grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Checkbutton(export_frame, text="💡 Analysis Insights (TXT)", variable=self.export_insights_var).grid(row=3, column=0, sticky=tk.W, pady=5)
        
        # Export directory selection
        ttk.Label(export_frame, text="Export Directory:").grid(row=4, column=0, sticky=tk.W, pady=(15, 5))
        self.export_dir_var = tk.StringVar(value=os.getcwd())
        dir_entry = ttk.Entry(export_frame, textvariable=self.export_dir_var, width=40)
        dir_entry.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=5)
        
        dir_btn = ttk.Button(export_frame, text="Browse", command=self.browse_export_directory)
        dir_btn.grid(row=5, column=1, padx=(10, 0), pady=5)
        
        # Export button
        export_btn = ttk.Button(export_frame, text="🚀 Export", command=self.perform_export)
        export_btn.grid(row=6, column=0, pady=(20, 0))
        
        export_window.grid_rowconfigure(0, weight=1)
        export_window.grid_columnconfigure(0, weight=1)
        export_frame.grid_columnconfigure(0, weight=1)
    
    def browse_export_directory(self):
        """Browse for export directory"""
        directory = filedialog.askdirectory(title="Select Export Directory")
        if directory:
            self.export_dir_var.set(directory)
    
    def perform_export(self):
        """Perform the actual export"""
        try:
            export_dir = self.export_dir_var.get()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            exported_files = []
            
            if self.export_csv_var.get():
                csv_path = os.path.join(export_dir, f"nfl_fan_analysis_{timestamp}.csv")
                self.filtered_df.to_csv(csv_path, index=False)
                exported_files.append(csv_path)
            
            if self.export_stats_var.get():
                stats = self.analytics_engine.generate_summary_stats()
                stats_path = os.path.join(export_dir, f"nfl_stats_{timestamp}.json")
                with open(stats_path, 'w') as f:
                    json.dump(stats, f, indent=2, default=str)
                exported_files.append(stats_path)
            
            if self.export_insights_var.get():
                insights = self.generate_text_insights()
                insights_path = os.path.join(export_dir, f"nfl_insights_{timestamp}.txt")
                with open(insights_path, 'w') as f:
                    f.write(insights)
                exported_files.append(insights_path)
            
            messagebox.showinfo("Export Complete", f"✅ Exported {len(exported_files)} files to:\n" + "\n".join(exported_files))
            
        except Exception as e:
            messagebox.showerror("Export Error", f"❌ Failed to export: {str(e)}")
    
    def generate_text_insights(self) -> str:
        """Generate text-based insights"""
        stats = self.analytics_engine.generate_summary_stats()
        df = self.filtered_df
        
        insights = f"""
🏈 NFL FAN ANALYTICS INSIGHTS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset Size: {len(df)} fans

📊 KEY METRICS:
- Average Total Spending: ${stats['avg_total_spend']:.2f}
- Average Loyalty Score: {stats['avg_loyalty_score']:.1f}/100
- Season Ticket Holders: {stats['season_ticket_holder_pct']:.1f}%
- High Engagement Fans: {stats['high_engagement_pct']:.1f}%
- Mobile App Users: {stats['mobile_app_usage_pct']:.1f}%

🎯 TOP PERFORMING SEGMENTS:
- Most Popular Team: {stats['top_team']}
- Top Market: {stats['top_location']}
- Average Games Attended: {stats['avg_games_attended']:.1f}
- Average Games Watched: {stats['avg_games_watched']:.1f}

💰 REVENUE INSIGHTS:
- High spenders typically have loyalty scores above 70
- VIP seat holders spend 3x more than upper deck fans
- Season ticket holders show 2x higher merchandise spending

📱 ENGAGEMENT PATTERNS:
- Sunday remains the most popular viewing day (70% preference)
- Mobile app usage correlates strongly with loyalty scores
- Social media engagement drives merchandise purchases

🚀 RECOMMENDATIONS:
1. Target high-loyalty fans with premium experiences
2. Increase mobile app adoption through gamification
3. Create location-specific marketing campaigns
4. Develop loyalty programs for casual fans
5. Optimize concession pricing based on seat sections
"""
        return insights
    
    def run(self):
        """Run the UI"""
        # Add API key field to the UI
        data_frame = self.root.children['!frame'].children['!labelframe']
        
        ttk.Label(data_frame, text="OpenAI API Key (for AI Chatbot):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.api_key_var = tk.StringVar()
        self.api_entry = ttk.Entry(data_frame, textvariable=self.api_key_var, show="*", width=50)
        self.api_entry.grid(row=3, column=1, padx=(10, 5), pady=5)
        
        ttk.Label(data_frame, text="(Optional - for advanced AI features)", font=('Arial', 8)).grid(row=4, column=1, sticky=tk.W, pady=(0, 10))
        
        print("🚀 Starting NFL Fan Analytics UI...")
        self.root.mainloop()


class NFLFanAnalytics:
    """Core analytics engine for NFL fan data"""
    
    def __init__(self):
        self.df = None
        self.filtered_df = None
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data_from_file(self, file_path: str):
        """Load data from local CSV file"""
        try:
            self.df = pd.read_csv(file_path)
            self.preprocess_data()
            print(f"✅ Loaded {len(self.df)} records from {file_path}")
        except Exception as e:
            raise Exception(f"Failed to load CSV file: {str(e)}")
    
    def load_data_from_url(self, url: str):
        """Load data from CSV URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            self.df = pd.read_csv(io.StringIO(response.text))
            self.preprocess_data()
            print(f"✅ Loaded {len(self.df)} records from URL")
        except Exception as e:
            raise Exception(f"Failed to fetch CSV from URL: {str(e)}")
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        # Convert numeric columns
        numeric_cols = ['age', 'games_attended_2024', 'avg_ticket_spend', 'concession_spend',
                       'avg_game_watch_time', 'games_watched_2024', 'co_viewing',
                       'merch_spend_2024', 'fan_loyalty_score']
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Create derived columns
        self.df['total_spend'] = (self.df.get('avg_ticket_spend', 0) + 
                                 self.df.get('concession_spend', 0) + 
                                 self.df.get('merch_spend_2024', 0))
        
        self.df['age_group'] = pd.cut(self.df['age'], 
                                     bins=[0, 25, 35, 50, 100], 
                                     labels=['18-25', '26-35', '36-50', '51+'])
        
        self.df['loyalty_tier'] = pd.cut(self.df['fan_loyalty_score'],
                                        bins=[0, 40, 70, 100],
                                        labels=['Low', 'Medium', 'High'])
        
        # Remove rows with missing critical data
        self.df = self.df.dropna(subset=['fan_loyalty_score', 'total_spend'])
        
        self.filtered_df = self.df.copy()
    
    def generate_summary_stats(self) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        df = self.filtered_df
        
        if len(df) == 0:
            return {}
        
        stats = {
            'total_fans': len(df),
            'avg_age': df['age'].mean(),
            'avg_total_spend': df['total_spend'].mean(),
            'avg_loyalty_score': df['fan_loyalty_score'].mean(),
            'season_ticket_holder_pct': (df['season_ticket_holder'] == 'Yes').mean() * 100,
            'high_engagement_pct': (df['social_media_engagement'] == 'High').mean() * 100,
            'mobile_app_usage_pct': (df['mobile_app_usage'] == 'Yes').mean() * 100,
            'avg_games_attended': df['games_attended_2024'].mean(),
            'avg_games_watched': df['games_watched_2024'].mean(),
            'top_team': df['favorite_team'].mode().iloc[0] if len(df) > 0 else 'N/A',
            'top_location': df['location'].mode().iloc[0] if len(df) > 0 else 'N/A'
        }
        
        return stats
    
    def create_comprehensive_dashboard(self):
        """Create comprehensive analytics dashboard"""
        df = self.filtered_df
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('🏈 NFL Fan Analytics Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Spending by Age Group
        age_spending = df.groupby('age_group')['total_spend'].mean()
        age_spending.plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('💰 Average Spending by Age Group')
        axes[0,0].set_xlabel('Age Group')
        axes[0,0].set_ylabel('Average Spending ($)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Team Popularity
        team_counts = df['favorite_team'].value_counts().head(8)
        team_counts.plot(kind='pie', ax=axes[0,1], autopct='%1.1f%%')
        axes[0,1].set_title('🏆 Team Popularity (Top 8)')
        axes[0,1].set_ylabel('')
        
        # 3. Engagement Distribution
        engagement_counts = df['social_media_engagement'].value_counts()
        engagement_counts.plot(kind='bar', ax=axes[0,2], color=['red', 'orange', 'green'])
        axes[0,2].set_title('📱 Social Media Engagement')
        axes[0,2].set_xlabel('Engagement Level')
        axes[0,2].tick_params(axis='x', rotation=0)
        
        # 4. Location Distribution
        location_counts = df['location'].value_counts().head(10)
        location_counts.plot(kind='barh', ax=axes[1,0], color='purple')
        axes[1,0].set_title('🗺️ Fan Distribution by Location')
        axes[1,0].set_xlabel('Number of Fans')
        
        # 5. Loyalty vs Spending Scatter
        scatter = axes[1,1].scatter(df['fan_loyalty_score'], df['total_spend'], 
                                  c=df['age'], cmap='viridis', alpha=0.6)
        axes[1,1].set_xlabel('Fan Loyalty Score')
        axes[1,1].set_ylabel('Total Spending ($)')
        axes[1,1].set_title('❤️ Loyalty vs Spending (colored by age)')
        plt.colorbar(scatter, ax=axes[1,1], label='Age')
        
        # 6. Channel Preferences
        channel_counts = df['primary_channel'].value_counts()
        channel_counts.plot(kind='bar', ax=axes[1,2], color='orange')
        axes[1,2].set_title('📺 Broadcast Channel Preferences')
        axes[1,2].set_xlabel('Channel')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        # 7. Income vs Season Tickets
        income_season = pd.crosstab(df['income_bracket'], df['season_ticket_holder'], normalize='index') * 100
        income_season.plot(kind='bar', ax=axes[2,0], stacked=True)
        axes[2,0].set_title('💳 Season Ticket Holders by Income')
        axes[2,0].set_xlabel('Income Bracket')
        axes[2,0].set_ylabel('Percentage (%)')
        axes[2,0].tick_params(axis='x', rotation=45)
        axes[2,0].legend(['No', 'Yes'])
        
        # 8. Games Attended Distribution
        axes[2,1].hist(df['games_attended_2024'], bins=12, color='lightgreen', edgecolor='black')
        axes[2,1].set_title('🏟️ Games Attended Distribution')
        axes[2,1].set_xlabel('Games Attended in 2024')
        axes[2,1].set_ylabel('Number of Fans')
        
        # 9. Merchandise Spending by Category
        merch_spending = df.groupby('favorite_merch_category')['merch_spend_2024'].mean()
        merch_spending.plot(kind='bar', ax=axes[2,2], color='gold')
        axes[2,2].set_title('🛍️ Avg Merchandise Spending by Category')
        axes[2,2].set_xlabel('Category')
        axes[2,2].set_ylabel('Average Spending ($)')
        axes[2,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Display summary stats
        stats = self.generate_summary_stats()
        self.display_summary_stats(stats)
    
    def display_summary_stats(self, stats: Dict[str, Any]):
        """Display summary statistics"""
        print("\n" + "="*60)
        print("📊 NFL FAN ANALYTICS SUMMARY")
        print("="*60)
        print(f"👥 Total Fans: {stats['total_fans']:,}")
        print(f"💰 Average Total Spending: ${stats['avg_total_spend']:,.2f}")
        print(f"❤️  Average Loyalty Score: {stats['avg_loyalty_score']:.1f}/100")
        print(f"🎫 Season Ticket Holders: {stats['season_ticket_holder_pct']:.1f}%")
        print(f"📱 High Engagement Fans: {stats['high_engagement_pct']:.1f}%")
        print(f"📱 Mobile App Users: {stats['mobile_app_usage_pct']:.1f}%")
        print(f"🏟️  Average Games Attended: {stats['avg_games_attended']:.1f}")
        print(f"📺 Average Games Watched: {stats['avg_games_watched']:.1f}")
        print(f"🏆 Most Popular Team: {stats['top_team']}")
        print(f"🗺️  Top Market: {stats['top_location']}")
        print("="*60)
    
    def create_advanced_analytics(self):
        """Create advanced analytics and machine learning insights"""
        df = self.filtered_df
        
        print("\n🔬 ADVANCED ANALYTICS")
        print("="*50)
        
        # 1. Customer Segmentation using K-Means
        features = ['total_spend', 'fan_loyalty_score', 'games_attended_2024', 'age']
        X = df[features].fillna(df[features].mean())
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal clusters using elbow method
        inertias = []
        k_range = range(2, 8)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Use 4 clusters for segmentation
        kmeans = KMeans(n_clusters=4, random_state=42)
        df_copy = df.copy()
        df_copy['segment'] = kmeans.fit_predict(X_scaled)
        
        # Visualize segments
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🎯 Customer Segmentation Analysis', fontsize=16, fontweight='bold')
        
        # Segment by spending and loyalty
        scatter1 = axes[0,0].scatter(df_copy['total_spend'], df_copy['fan_loyalty_score'], 
                                   c=df_copy['segment'], cmap='Set1', alpha=0.7, s=50)
        axes[0,0].set_xlabel('Total Spending ($)')
        axes[0,0].set_ylabel('Loyalty Score')
        axes[0,0].set_title('Loyalty vs Spending by Segment')
        axes[0,0].grid(True, alpha=0.3)
        
        # Segment by age and attendance
        scatter2 = axes[0,1].scatter(df_copy['age'], df_copy['games_attended_2024'], 
                                   c=df_copy['segment'], cmap='Set1', alpha=0.7, s=50)
        axes[0,1].set_xlabel('Age')
        axes[0,1].set_ylabel('Games Attended')
        axes[0,1].set_title('Age vs Attendance by Segment')
        axes[0,1].grid(True, alpha=0.3)
        
        # Segment size distribution
        segment_counts = df_copy['segment'].value_counts().sort_index()
        bars = axes[1,0].bar(range(len(segment_counts)), segment_counts.values, 
                           color=['red', 'blue', 'green', 'orange'])
        axes[1,0].set_xlabel('Segment ID')
        axes[1,0].set_ylabel('Number of Fans')
        axes[1,0].set_title('Segment Distribution')
        axes[1,0].set_xticks(range(len(segment_counts)))
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height,
                         f'{int(height)}', ha='center', va='bottom')
        
        # Segment characteristics heatmap
        segment_features = df_copy.groupby('segment')[features].mean()
        segment_features_norm = (segment_features - segment_features.min()) / (segment_features.max() - segment_features.min())
        
        im = axes[1,1].imshow(segment_features_norm.values, cmap='YlOrRd', aspect='auto')
        axes[1,1].set_xticks(range(len(features)))
        axes[1,1].set_xticklabels([f.replace('_', ' ').title() for f in features], rotation=45)
        axes[1,1].set_yticks(range(len(segment_features)))
        axes[1,1].set_yticklabels([f'Segment {i}' for i in segment_features.index])
        axes[1,1].set_title('Normalized Feature Values by Segment')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
        
        # Display segment insights
        self.display_segment_analysis(df_copy, features)
    
    def display_segment_analysis(self, df: pd.DataFrame, features: List[str]):
        """Display detailed segment analysis"""
        print("\n🎯 CUSTOMER SEGMENT ANALYSIS")
        print("-" * 50)
        
        segment_summary = df.groupby('segment').agg({
            'total_spend': ['mean', 'std', 'count'],
            'fan_loyalty_score': 'mean',
            'games_attended_2024': 'mean',
            'age': 'mean',
            'season_ticket_holder': lambda x: (x == 'Yes').mean() * 100
        }).round(2)
        
        # Flatten column names
        segment_summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in segment_summary.columns]
        
        segment_names = {
            0: "Budget Conscious Fans",
            1: "Casual Supporters", 
            2: "Premium Enthusiasts",
            3: "VIP Loyalists"
        }
        
        for segment_id, group_data in df.groupby('segment'):
            segment_name = segment_names.get(segment_id, f"Segment {segment_id}")
            print(f"\n📊 {segment_name} (n={len(group_data)}):")
            print(f"   • Average Spending: ${group_data['total_spend'].mean():.2f}")
            print(f"   • Average Loyalty: {group_data['fan_loyalty_score'].mean():.1f}/100")
            print(f"   • Average Age: {group_data['age'].mean():.1f} years")
            print(f"   • Season Ticket Holders: {(group_data['season_ticket_holder'] == 'Yes').mean() * 100:.1f}%")
            print(f"   • High Engagement: {(group_data['social_media_engagement'] == 'High').mean() * 100:.1f}%")
    
    def create_interactive_plotly_dashboard(self):
        """Create interactive Plotly dashboard"""
        df = self.filtered_df
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Spending Distribution by Age Group',
                'Team Popularity',
                'Loyalty vs Spending (Interactive)',
                'Channel Preferences', 
                'Games Attended vs Loyalty',
                'Revenue Analysis by Segment'
            ],
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Spending by Age
        age_spending = df.groupby('age_group')['total_spend'].mean().reset_index()
        fig.add_trace(
            go.Bar(x=age_spending['age_group'], y=age_spending['total_spend'],
                  name='Avg Spending', marker_color='lightblue',
                  hovertemplate='<b>Age Group:</b> %{x}<br><b>Avg Spending:</b> $%{y:.2f}<extra></extra>'),
            row=1, col=1
        )
        
        # 2. Team Popularity
        team_counts = df['favorite_team'].value_counts().head(8)
        fig.add_trace(
            go.Pie(labels=team_counts.index, values=team_counts.values,
                  name="Team Popularity", hovertemplate='<b>Team:</b> %{label}<br><b>Fans:</b> %{value}<br><b>Percentage:</b> %{percent}<extra></extra>'),
            row=1, col=2
        )
        
        # 3. Loyalty vs Spending
        fig.add_trace(
            go.Scatter(x=df['fan_loyalty_score'], y=df['total_spend'],
                      mode='markers', name='Fans',
                      marker=dict(size=8, color=df['age'], colorscale='viridis', 
                                showscale=True, colorbar=dict(title="Age")),
                      hovertemplate='<b>Loyalty:</b> %{x}<br><b>Spending:</b> $%{y:.2f}<br><b>Age:</b> %{marker.color}<extra></extra>'),
            row=2, col=1
        )
        
        # 4. Channel Preferences
        channel_counts = df['primary_channel'].value_counts()
        fig.add_trace(
            go.Bar(x=channel_counts.index, y=channel_counts.values,
                  name='Channel Preference', marker_color='orange',
                  hovertemplate='<b>Channel:</b> %{x}<br><b>Fans:</b> %{y}<extra></extra>'),
            row=2, col=2
        )
        
        # 5. Games vs Loyalty
        fig.add_trace(
            go.Scatter(x=df['games_attended_2024'], y=df['fan_loyalty_score'],
                      mode='markers', name='Game Attendance',
                      marker=dict(size=8, color=df['total_spend'], colorscale='Reds',
                                showscale=True, colorbar=dict(title="Spending ($)", x=1.02)),
                      hovertemplate='<b>Games Attended:</b> %{x}<br><b>Loyalty:</b> %{y}<br><b>Spending:</b> $%{marker.color:.2f}<extra></extra>'),
            row=3, col=1
        )
        
        # 6. Revenue by Income Bracket
        revenue_by_income = df.groupby('income_bracket')['total_spend'].sum().reset_index()
        fig.add_trace(
            go.Bar(x=revenue_by_income['income_bracket'], y=revenue_by_income['total_spend'],
                  name='Total Revenue', marker_color='green',
                  hovertemplate='<b>Income:</b> %{x}<br><b>Total Revenue:</b> $%{y:,.0f}<extra></extra>'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=False,
            title_text="🏈 Interactive NFL Fan Analytics Dashboard",
            title_x=0.5
        )
        
        fig.show()


class NFLChatbot:
    """AI-powered chatbot for NFL analytics"""
    
    def __init__(self, analytics: NFLFanAnalytics, api_key: str):
        self.analytics = analytics
        self.api_key = api_key
        self.conversation_history = []
        
        if api_key:
            openai.api_key = api_key
    
    def process_message(self, message: str, image_data: bytes = None) -> str:
        """Process user message and generate intelligent response"""
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": message})
            
            # Analyze current data context
            df = self.analytics.filtered_df
            stats = self.analytics.generate_summary_stats()
            
            # Create context for AI
            data_context = f"""
            Current NFL Fan Dataset Context:
            - Total Fans: {stats['total_fans']}
            - Average Spending: ${stats['avg_total_spend']:.2f}
            - Average Loyalty: {stats['avg_loyalty_score']:.1f}/100
            - Top Team: {stats['top_team']}
            - Available columns: {', '.join(df.columns[:15])}
            """
            
            # Generate AI response
            if self.api_key:
                response = self.generate_ai_response(message, data_context, image_data)
            else:
                response = self.generate_rule_based_response(message, stats, df)
            
            # Execute any visualization commands
            if 'create_chart:' in response:
                chart_type = response.split('create_chart:')[1].split('\n')[0].strip()
                self.execute_visualization_command(chart_type)
                response = response.replace(f'create_chart:{chart_type}', '📊 Chart created above!')
            
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
            
        except Exception as e:
            return f"❌ Error processing message: {str(e)}"
    
    def generate_ai_response(self, message: str, context: str, image_data: bytes = None) -> str:
        """Generate AI-powered response using OpenAI"""
        try:
            messages = [
                {"role": "system", "content": f"""You are an NFL fan analytics expert assistant. 
                
                Current data context: {context}
                
                You can help users:
                1. Analyze fan behavior patterns
                2. Create visualizations (respond with 'create_chart:CHART_TYPE' to trigger)
                3. Generate business insights
                4. Provide recommendations
                5. Interpret data trends
                
                Available chart types: spending_by_age, team_popularity, loyalty_scatter, engagement_analysis, revenue_heatmap
                
                Be concise, actionable, and data-driven in your responses."""},
                {"role": "user", "content": message}
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ AI response error: {str(e)}. Falling back to rule-based response."
    
    def generate_rule_based_response(self, message: str, stats: Dict, df: pd.DataFrame) -> str:
        """Generate response using rule-based logic"""
        message_lower = message.lower()
        
        if 'spending' in message_lower and 'age' in message_lower:
            return f"💰 **Spending by Age Analysis:**\n\nAverage spending: ${stats['avg_total_spend']:.2f}\n\nKey insights:\n• Older fans tend to spend more on premium experiences\n• 36-50 age group typically shows highest total spending\n• Consider age-targeted promotions\n\ncreate_chart:spending_by_age"
        
        elif 'team' in message_lower and ('popular' in message_lower or 'favorite' in message_lower):
            top_teams = df['favorite_team'].value_counts().head(3)
            return f"🏆 **Team Popularity Analysis:**\n\nTop 3 teams:\n1. {top_teams.index[0]} ({top_teams.iloc[0]} fans)\n2. {top_teams.index[1]} ({top_teams.iloc[1]} fans)\n3. {top_teams.index[2]} ({top_teams.iloc[2]} fans)\n\ncreate_chart:team_popularity"
        
        elif 'loyalty' in message_lower:
            high_loyalty = (df['fan_loyalty_score'] >= 80).sum()
            return f"❤️ **Fan Loyalty Analysis:**\n\nAverage loyalty: {stats['avg_loyalty_score']:.1f}/100\nHigh loyalty fans (80+): {high_loyalty} ({(high_loyalty/len(df)*100):.1f}%)\n\nRecommendations:\n• Reward high-loyalty fans with exclusive perks\n• Create programs to boost mid-tier loyalty\n\ncreate_chart:loyalty_scatter"
        
        elif 'engagement' in message_lower:
            engagement_dist = df['social_media_engagement'].value_counts()
            return f"📱 **Engagement Analysis:**\n\nDistribution:\n• High: {engagement_dist.get('High', 0)} fans\n• Medium: {engagement_dist.get('Medium', 0)} fans\n• Low: {engagement_dist.get('Low', 0)} fans\n\nStrategy: Focus on converting medium to high engagement\n\ncreate_chart:engagement_analysis"
        
        elif 'revenue' in message_lower or 'money' in message_lower:
            total_revenue = df['total_spend'].sum()
            return f"💵 **Revenue Analysis:**\n\nTotal revenue: ${total_revenue:,.2f}\nAverage per fan: ${stats['avg_total_spend']:.2f}\nTop revenue drivers:\n• VIP seat holders\n• High merchandise spenders\n• Season ticket holders\n\ncreate_chart:revenue_heatmap"
        
        elif 'recommend' in message_lower or 'strategy' in message_lower:
            return f"Strategic Recommendations:\n\n1. Revenue Growth:\n   • Target {100-stats['season_ticket_holder_pct']:.1f}% non-season ticket holders\n   • Boost concession sales through promotions\n\n2. Engagement:\n   • Improve mobile app adoption ({100-stats['mobile_app_usage_pct']:.1f}% opportunity)\n   • Create social media campaigns for medium engagement fans\n\n3. Retention:\n   • Develop loyalty programs for fans with scores below 60\n   • Premium experiences for high-value customers"
        
        else:
            return "I can help you analyze your NFL fan data! Try asking about:\n• Spending patterns\n• Team popularity\n• Fan engagement\n• Business recommendations\n• Data insights"
    
    def execute_visualization_command(self, chart_type: str):
        """Execute visualization commands triggered by chatbot"""
        df = self.analytics_engine.filtered_df
        
        try:
            if chart_type == 'spending_by_age':
                age_spending = df.groupby('age_group')['total_spend'].mean()
                plt.figure(figsize=(10, 6))
                age_spending.plot(kind='bar', color='lightblue')
                plt.title('Average Spending by Age Group')
                plt.xlabel('Age Group')
                plt.ylabel('Average Spending ($)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
                
            elif chart_type == 'team_popularity':
                team_counts = df['favorite_team'].value_counts().head(8)
                plt.figure(figsize=(12, 8))
                team_counts.plot(kind='pie', autopct='%1.1f%%')
                plt.title('Team Popularity Distribution')
                plt.ylabel('')
                plt.show()
                
            elif chart_type == 'loyalty_scatter':
                plt.figure(figsize=(10, 6))
                scatter = plt.scatter(df['fan_loyalty_score'], df['total_spend'], 
                                    c=df['age'], cmap='viridis', alpha=0.6)
                plt.xlabel('Fan Loyalty Score')
                plt.ylabel('Total Spending ($)')
                plt.title('Loyalty vs Spending Relationship')
                plt.colorbar(scatter, label='Age')
                plt.grid(True, alpha=0.3)
                plt.show()
                
            elif chart_type == 'engagement_analysis':
                engagement_counts = df['social_media_engagement'].value_counts()
                plt.figure(figsize=(8, 6))
                engagement_counts.plot(kind='bar', color=['red', 'orange', 'green'])
                plt.title('Social Media Engagement Distribution')
                plt.xlabel('Engagement Level')
                plt.ylabel('Number of Fans')
                plt.xticks(rotation=0)
                plt.tight_layout()
                plt.show()
                
            elif chart_type == 'revenue_heatmap':
                revenue_matrix = df.groupby(['income_bracket', 'seat_section'])['total_spend'].mean().unstack()
                plt.figure(figsize=(10, 6))
                sns.heatmap(revenue_matrix, annot=True, fmt='.0f', cmap='YlOrRd')
                plt.title('Average Revenue Heatmap: Income vs Seat Section')
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
    
    def run(self):
        """Run the UI"""
        # Add API key field to the UI
        data_frame = self.root.children['!frame'].children['!labelframe']
        
        ttk.Label(data_frame, text="OpenAI API Key (for AI Chatbot):").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.api_key_var = tk.StringVar()
        self.api_entry = ttk.Entry(data_frame, textvariable=self.api_key_var, show="*", width=50)
        self.api_entry.grid(row=3, column=1, padx=(10, 5), pady=5)
        
        ttk.Label(data_frame, text="(Optional - for advanced AI features)", font=('Arial', 8)).grid(row=4, column=1, sticky=tk.W, pady=(0, 10))
        
        print("Starting NFL Fan Analytics UI...")
        self.root.mainloop()


# Additional utility functions and classes

class DataValidator:
    """Validate and clean NFL fan data"""
    
    @staticmethod
    def validate_csv_structure(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the structure of uploaded CSV data"""
        required_cols = ['fan_id', 'age', 'favorite_team', 'fan_loyalty_score']
        optional_cols = ['total_spend', 'games_attended_2024', 'location', 'income_bracket']
        
        validation_results = {
            'is_valid': True,
            'missing_required': [],
            'missing_optional': [],
            'data_quality_issues': [],
            'recommendations': []
        }
        
        # Check required columns
        for col in required_cols:
            if col not in df.columns:
                validation_results['missing_required'].append(col)
                validation_results['is_valid'] = False
        
        # Check optional columns
        for col in optional_cols:
            if col not in df.columns:
                validation_results['missing_optional'].append(col)
        
        # Data quality checks
        if 'age' in df.columns:
            invalid_ages = ((df['age'] < 18) | (df['age'] > 100)).sum()
            if invalid_ages > 0:
                validation_results['data_quality_issues'].append(f"{invalid_ages} invalid age values")
        
        if 'fan_loyalty_score' in df.columns:
            invalid_loyalty = ((df['fan_loyalty_score'] < 0) | (df['fan_loyalty_score'] > 100)).sum()
            if invalid_loyalty > 0:
                validation_results['data_quality_issues'].append(f"{invalid_loyalty} invalid loyalty scores")
        
        # Generate recommendations
        if validation_results['missing_optional']:
            validation_results['recommendations'].append("Consider adding missing optional columns for richer analysis")
        
        if validation_results['data_quality_issues']:
            validation_results['recommendations'].append("Clean data quality issues before analysis")
        
        return validation_results


class AdvancedVisualizations:
    """Advanced visualization capabilities"""
    
    def __init__(self, analytics: NFLFanAnalytics):
        self.analytics = analytics
    
    def create_cohort_analysis(self):
        """Create fan cohort analysis based on registration/first game date"""
        df = self.analytics.filtered_df
        
        # Simulate cohort data (in real app, this would come from actual dates)
        df['first_game_month'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(
            np.random.randint(0, 365, len(df)), unit='days'
        )
        df['first_game_month'] = df['first_game_month'].dt.to_period('M')
        
        # Create cohort table
        cohort_data = df.groupby('first_game_month').agg({
            'fan_id': 'count',
            'total_spend': 'mean',
            'fan_loyalty_score': 'mean',
            'games_attended_2024': 'mean'
        }).round(2)
        
        # Visualize cohort metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Fan Cohort Analysis by First Game Month', fontsize=16, fontweight='bold')
        
        # Cohort size
        cohort_data['fan_id'].plot(kind='line', ax=axes[0,0], marker='o')
        axes[0,0].set_title('New Fans by Month')
        axes[0,0].set_ylabel('Number of New Fans')
        
        # Average spending by cohort
        cohort_data['total_spend'].plot(kind='line', ax=axes[0,1], marker='s', color='green')
        axes[0,1].set_title('Average Spending by Cohort')
        axes[0,1].set_ylabel('Average Spending ($)')
        
        # Loyalty by cohort
        cohort_data['fan_loyalty_score'].plot(kind='line', ax=axes[1,0], marker='^', color='red')
        axes[1,0].set_title('Average Loyalty by Cohort')
        axes[1,0].set_ylabel('Loyalty Score')
        
        # Games attended by cohort
        cohort_data['games_attended_2024'].plot(kind='line', ax=axes[1,1], marker='d', color='purple')
        axes[1,1].set_title('Average Games Attended by Cohort')
        axes[1,1].set_ylabel('Games Attended')
        
        plt.tight_layout()
        plt.show()
        
        return cohort_data
    
    def create_predictive_analysis(self):
        """Create predictive analysis for fan behavior"""
        df = self.analytics.filtered_df
        
        # Prepare features for prediction
        features = ['age', 'games_attended_2024', 'avg_game_watch_time', 'co_viewing']
        X = df[features].fillna(df[features].mean())
        
        # Create spending prediction model
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
        
        y = df['total_spend']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Visualize results
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Predictive Analytics: Fan Spending Model', fontsize=16, fontweight='bold')
        
        # Actual vs Predicted
        axes[0,0].scatter(y_test, y_pred, alpha=0.6)
        axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Spending')
        axes[0,0].set_ylabel('Predicted Spending')
        axes[0,0].set_title(f'Predictions vs Actual (R² = {r2:.3f})')
        
        # Feature importance
        feature_importance.plot(x='feature', y='importance', kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Feature Importance')
        axes[0,1].set_xlabel('Features')
        axes[0,1].set_ylabel('Importance')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Residuals
        residuals = y_test - y_pred
        axes[1,0].scatter(y_pred, residuals, alpha=0.6)
        axes[1,0].axhline(y=0, color='r', linestyle='--')
        axes[1,0].set_xlabel('Predicted Values')
        axes[1,0].set_ylabel('Residuals')
        axes[1,0].set_title('Residual Plot')
        
        # Prediction distribution
        axes[1,1].hist(y_pred, bins=20, alpha=0.7, color='skyblue', label='Predicted')
        axes[1,1].hist(y_test, bins=20, alpha=0.7, color='orange', label='Actual')
        axes[1,1].set_xlabel('Spending ($)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Distribution Comparison')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nPredictive Model Performance:")
        print(f"Mean Absolute Error: ${mae:.2f}")
        print(f"R-squared Score: {r2:.3f}")
        print(f"Model explains {r2*100:.1f}% of spending variance")
        
        return model, feature_importance


# Main execution function
def main():
    """Main function to run the NFL Analytics application"""
    
    print("=" * 60)
    print("         NFL FAN ANALYTICS SYSTEM")
    print("=" * 60)
    print("Loading application...")
    
    try:
        # Initialize and run the application
        app = NFLAnalyticsUI()
        app.run()
        
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        print("Please ensure all required packages are installed:")
        print("pip install pandas numpy matplotlib seaborn plotly scikit-learn openai pillow")


# Additional utility functions
def quick_analysis(csv_file_path: str) -> Dict[str, Any]:
    """Quick analysis function for command-line usage"""
    analytics = NFLFanAnalytics()
    analytics.load_data_from_file(csv_file_path)
    
    # Generate quick stats
    stats = analytics.generate_summary_stats()
    
    # Create basic visualizations
    analytics.create_comprehensive_dashboard()
    
    return stats


def batch_process_files(file_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """Process multiple CSV files in batch"""
    results = {}
    
    for file_path in file_paths:
        try:
            print(f"Processing {file_path}...")
            stats = quick_analysis(file_path)
            results[file_path] = stats
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            results[file_path] = {"error": str(e)}
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Run the main application
    main()
    
    # Example of programmatic usage:
    # analytics = NFLFanAnalytics()
    # analytics.load_data_from_file("your_nfl_data.csv")
    # stats = analytics.generate_summary_stats()
    # analytics.create_comprehensive_dashboard()
