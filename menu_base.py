"""Shared application lifecycle and top-level menu flow."""

from .competitor_analyzer import CompetitorAnalyzer
from .data_loader import WCADataLoader


class WCAMenuBase:
    def __init__(self):
            self.loader = WCADataLoader()
            self.data = None
            self.competitor_analyzer = None


    def initialize(self):
            """Load data and initialize analyzers"""
            print("\n" + "="*60)
            print("🚀 WCA DATA ANALYSIS AND PREDICTION SUITE")
            print("="*60)

            self.data = self.loader.load_all_files()
            self.data = self.loader.preprocess_data()

            self.competitor_analyzer = CompetitorAnalyzer(self.data)

            print("\n✅ System ready!")


    def show_main_menu(self):
            """Display main menu"""
            print("\n" + "="*60)
            print("📊 MAIN MENU")
            print("="*60)
            print("1. 🔍 Analyze Competitor by WCA ID")
            print("2. 📈 Global Statistics & Trends")
            print("3. 🌍 Country Performance Analysis")
            print("4. 🎯 Event Performance Analysis")
            print("5. 🤖 Predictive Models")
            print("6. ❌ Exit")
            print("-"*60)


    def run(self):
            """Run the menu system"""
            self.initialize()

            while True:
                self.show_main_menu()

                try:
                    choice = input("👉 Enter your choice (1-6): ").strip()

                    if choice == '1':
                        self.analyze_competitor_menu()
                    elif choice == '2':
                        self.global_statistics_menu()
                    elif choice == '3':
                        self.country_analysis_menu()
                    elif choice == '4':
                        self.event_analysis_menu()
                    elif choice == '5':
                        self.predictive_models_menu()
                    elif choice == '6':
                        print("\n👋 Thank you for using WCA Analysis Suite! Goodbye!")
                        break
                    else:
                        print("\n❌ Invalid choice. Please enter 1-6.")

                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {str(e)}")


    def analyze_competitor_menu(self):
            """Competitor analysis menu"""
            print("\n" + "="*60)
            print("🔍 COMPETITOR ANALYSIS")
            print("="*60)

            wca_id = input("Enter WCA ID (e.g., 2003BELL01): ").strip().upper()

            if not wca_id:
                print("❌ No WCA ID entered.")
                return

            self.competitor_analyzer.analyze_competitor(wca_id)

            input("\nPress Enter to continue...")
