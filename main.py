"""
WCA Data Analysis and Prediction Suite - MENU DRIVEN
Complete analysis with point labeling, podium pie charts, and legends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class WCADataLoader:
    """Load and preprocess all WCA data files"""
   
    def __init__(self, data_path='./'):
        self.data_path = data_path
        self.data = {}
       
    def load_all_files(self):
        """Load all TSV files"""
        print("\n📂 Loading WCA data files...")
       
        file_configs = {
            'persons': {'file': 'WCA_export_persons.tsv', 'sep': '\t', 'encoding': 'utf-8'},
            'results': {'file': 'WCA_export_results.tsv', 'sep': '\t', 'encoding': 'utf-8'},
            'ranks_average': {'file': 'WCA_export_ranks_average.tsv', 'sep': '\t', 'encoding': 'utf-8'},
            'events': {'file': 'WCA_export_events.tsv', 'sep': '\t', 'encoding': 'utf-8'},
            'countries': {'file': 'WCA_export_countries.tsv', 'sep': '\t', 'encoding': 'utf-8'},
            'continents': {'file': 'WCA_export_continents.tsv', 'sep': '\t', 'encoding': 'utf-8'},
            'round_types': {'file': 'WCA_export_round_types.tsv', 'sep': '\t', 'encoding': 'utf-8'},
            'scrambles': {'file': 'WCA_export_scrambles.tsv', 'sep': '\t', 'encoding': 'utf-8'},
            'result_attempts': {'file': 'WCA_export_result_attempts.tsv', 'sep': '\t', 'encoding': 'utf-8'},
            'competitions': {'file': 'WCA_export_competitions.tsv', 'sep': '\t', 'encoding': 'utf-8'},
            'championships': {'file': 'WCA_export_championships.tsv', 'sep': '\t', 'encoding': 'utf-8'},
            'formats': {'file': 'WCA_export_formats.tsv', 'sep': '\t', 'encoding': 'utf-8'}
        }
       
        for key, config in file_configs.items():
            try:
                df = pd.read_csv(
                    f"{self.data_path}{config['file']}",
                    sep=config['sep'],
                    encoding=config['encoding'],
                    low_memory=False,
                    on_bad_lines='skip'
                )
                self.data[key] = df
                print(f"  ✅ Loaded {key}: {len(df):,} records")
            except Exception as e:
                print(f"  ❌ Error loading {key}: {str(e)}")
                self.data[key] = pd.DataFrame()
       
        return self.data
   
    def preprocess_data(self):
        """Clean and preprocess the loaded data"""
        print("\n🔄 Preprocessing data...")
       
        if 'results' in self.data and not self.data['results'].empty:
            # Convert time fields from centiseconds to seconds
            for col in ['best', 'average']:
                if col in self.data['results'].columns:
                    self.data['results'][f'{col}_seconds'] = self.data['results'][col] / 100
                    # Handle DNF, DNS etc.
                    mask = self.data['results'][col] > 99999999
                    self.data['results'].loc[mask, f'{col}_seconds'] = np.nan
                    self.data['results'].loc[self.data['results'][f'{col}_seconds'] < 0, f'{col}_seconds'] = np.nan
           
            # Extract year from competition IDs
            if 'competition_id' in self.data['results'].columns:
                # Extract 4-digit year using string slicing
                self.data['results']['year'] = self.data['results']['competition_id'].str.extract(r'(\d{4})').astype(float)
                self.data['results'].loc[self.data['results']['year'] < 2000, 'year'] = np.nan
       
        # Create event name mapping
        if 'events' in self.data and not self.data['events'].empty:
            self.event_names = dict(zip(self.data['events']['id'], self.data['events']['name']))
        else:
            self.event_names = {}
       
        print("✅ Preprocessing complete!")
        return self.data


# ============================================================================
# COMPETITOR ANALYSIS
# ============================================================================

class CompetitorAnalyzer:
    """Analyze individual competitor performance"""
   
    def __init__(self, data):
        self.data = data
        self.event_names = {}
        if 'events' in data and not data['events'].empty:
            self.event_names = dict(zip(data['events']['id'], data['events']['name']))
   
    def find_competitor(self, wca_id):
        """Find competitor by WCA ID"""
        if 'persons' not in self.data or self.data['persons'].empty:
            print("❌ Persons data not available")
            return None
       
        # Find the competitor
        competitor = self.data['persons'][
            (self.data['persons']['wca_id'] == wca_id) &
            (self.data['persons']['sub_id'] == 1)
        ]
       
        if competitor.empty:
            return None
       
        return competitor.iloc[0]
   
    def get_competitor_results(self, wca_id):
        """Get all results for a competitor"""
        if 'results' not in self.data or self.data['results'].empty:
            return pd.DataFrame()
       
        results = self.data['results'][
            (self.data['results']['person_id'] == wca_id)
        ].copy()
       
        if not results.empty and 'year' in results.columns:
            results = results.sort_values('year')
       
        return results
   
    def get_competitor_rankings(self, wca_id, event_id=None):
        """Get all rankings for a competitor from competitions"""
        if 'results' not in self.data or self.data['results'].empty:
            return pd.DataFrame()
       
        results = self.get_competitor_results(wca_id)
       
        if event_id:
            results = results[results['event_id'] == event_id]
       
        # Get rankings if available
        rankings = []
        for _, row in results.iterrows():
            ranking = {
                'competition_id': row['competition_id'],
                'event_id': row['event_id'],
                'year': row['year'] if 'year' in row else None,
                'round_type_id': row['round_type_id'] if 'round_type_id' in row else None,
                'pos': row['pos'] if 'pos' in row else None,  # Position in round
                'best': row['best_seconds'] if 'best_seconds' in row else None,
                'average': row['average_seconds'] if 'average_seconds' in row else None
            }
            rankings.append(ranking)
       
        return pd.DataFrame(rankings)
   
    def get_country_results(self, country_id, event_id=None, year=None):
        """Get results from a specific country"""
        if 'results' not in self.data or self.data['results'].empty:
            return pd.DataFrame()
       
        # Check if country_id column exists
        country_col = None
        for col in ['country_id', 'person_country_id']:
            if col in self.data['results'].columns:
                country_col = col
                break
       
        if country_col is None:
            return pd.DataFrame()
       
        results = self.data['results'][self.data['results'][country_col] == country_id]
       
        if event_id:
            results = results[results['event_id'] == event_id]
       
        if year:
            results = results[results['year'] == year]
       
        if 'average_seconds' in results.columns:
            results = results[results['average_seconds'].notna() &
                            (results['average_seconds'] > 0) &
                            (results['average_seconds'] < 600)]
       
        return results
   
    def calculate_improvement_rate(self, results, event_id, use_best=False):
        """Calculate improvement rate for specific event"""
        if use_best and 'best_seconds' in results.columns:
            time_col = 'best_seconds'
        else:
            time_col = 'average_seconds'
       
        event_results = results[results['event_id'] == event_id].sort_values('year')
        event_results = event_results[event_results[time_col].notna() & (event_results[time_col] > 0)]
       
        if len(event_results) < 2:
            return 0, 0, 0, 0
       
        times = event_results[time_col].values
        years = event_results['year'].values
       
        # Linear regression for improvement rate
        if len(times) >= 3:
            z = np.polyfit(years, times, 1)
            improvement_rate = -z[0]  # Negative slope means improvement
            r_squared = np.corrcoef(years, times)[0, 1] ** 2
        else:
            # Simple rate between first and last
            first_time = times[0]
            last_time = times[-1]
            years_diff = years[-1] - years[0]
            if years_diff == 0:
                improvement_rate = 0
            else:
                improvement_rate = (first_time - last_time) / years_diff
            r_squared = 0
       
        first_time = times[0]
        last_time = times[-1]
       
        return improvement_rate, first_time, last_time, r_squared
   
    def predict_future_time(self, results, event_id, years_ahead=1, use_best=False):
        """Predict future time based on improvement trend"""
        if use_best and 'best_seconds' in results.columns:
            time_col = 'best_seconds'
        else:
            time_col = 'average_seconds'
       
        event_results = results[results['event_id'] == event_id].sort_values('year')
        event_results = event_results[event_results[time_col].notna() & (event_results[time_col] > 0)]
       
        if len(event_results) < 2:
            return event_results[time_col].iloc[-1] if len(event_results) > 0 else None, 0
       
        years = event_results['year'].values
        times = event_results[time_col].values
       
        # Use linear regression for prediction
        if len(times) >= 3:
            z = np.polyfit(years, times, 1)
            last_year = years[-1]
            future_year = last_year + years_ahead
            predicted_time = z[0] * future_year + z[1]
           
            # Calculate confidence (based on R²)
            r_squared = np.corrcoef(years, times)[0, 1] ** 2
            confidence = min(100, r_squared * 100)
        else:
            # Simple linear projection
            improvement_rate = (times[0] - times[-1]) / (years[-1] - years[0]) if years[-1] > years[0] else 0
            predicted_time = times[-1] - (improvement_rate * years_ahead)
            confidence = 50  # Default confidence for limited data
       
        return max(0.1, predicted_time), confidence
   
    def predict_podium_probability(self, wca_id, event_id='333'):
        """Calculate probability of podium based on historical podium finishes"""
        # Get competitor info
        competitor = self.find_competitor(wca_id)
        if competitor is None:
            return None, None, None, None
       
        # Get all rankings for this competitor in this event
        rankings = self.get_competitor_rankings(wca_id, event_id)
       
        if rankings.empty or 'pos' not in rankings.columns:
            return 0, "No ranking data available", 0, 0
       
        # Filter out NaN positions and convert to numeric
        rankings = rankings[rankings['pos'].notna()]
        if rankings.empty:
            return 0, "No valid ranking data", 0, 0
       
        rankings['pos'] = pd.to_numeric(rankings['pos'], errors='coerce')
        rankings = rankings.dropna(subset=['pos'])
       
        if rankings.empty:
            return 0, "No valid ranking data", 0, 0
       
        # Calculate podium finishes (positions 1-3)
        rankings['podium'] = rankings['pos'] <= 3
       
        # Calculate podium percentage
        total_competitions = len(rankings)
        podium_count = rankings['podium'].sum()
        podium_percentage = (podium_count / total_competitions) * 100
       
        # Calculate weighted by recency (more recent competitions matter more)
        if 'year' in rankings.columns and rankings['year'].notna().any():
            rankings['year'] = pd.to_numeric(rankings['year'], errors='coerce')
            rankings = rankings.dropna(subset=['year'])
           
            if not rankings.empty:
                current_year = datetime.now().year
                rankings['recency_weight'] = 1 / (current_year - rankings['year'] + 1)
                weighted_podium = np.average(rankings['podium'], weights=rankings['recency_weight'])
                weighted_percentage = weighted_podium * 100
            else:
                weighted_percentage = podium_percentage
        else:
            weighted_percentage = podium_percentage
       
        # Calculate improvement trend
        avg_position_over_time = rankings.groupby('year')['pos'].mean().reset_index() if 'year' in rankings.columns else None
       
        # Predict future improvement
        if avg_position_over_time is not None and len(avg_position_over_time) >= 2:
            years = avg_position_over_time['year'].values
            positions = avg_position_over_time['pos'].values
           
            # Linear regression on positions
            z = np.polyfit(years, positions, 1)
            position_trend = z[0]  # Negative means improving position
           
            # Adjust probability based on trend
            if position_trend < -0.5:  # Improving significantly
                trend_factor = 1.2
            elif position_trend < -0.2:  # Improving moderately
                trend_factor = 1.1
            elif position_trend > 0.5:  # Getting worse
                trend_factor = 0.8
            elif position_trend > 0.2:  # Getting slightly worse
                trend_factor = 0.9
            else:
                trend_factor = 1.0
           
            adjusted_probability = min(100, weighted_percentage * trend_factor)
        else:
            adjusted_probability = weighted_percentage
       
        # Get average podium time for reference
        podium_times = rankings[rankings['podium']]['average'].dropna() if 'average' in rankings.columns else None
        if podium_times is not None and not podium_times.empty:
            avg_podium_time = podium_times.mean()
        else:
            avg_podium_time = None
       
        return adjusted_probability, podium_count, total_competitions, avg_podium_time
   
    def plot_event_performance(self, results, event_id, competitor_name):
        """Create separate window for single and average performance for an event with labeled points"""
        event_name = self.event_names.get(event_id, event_id)
        event_results = results[results['event_id'] == event_id].sort_values('year')
       
        if event_results.empty:
            return
       
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{competitor_name} - {event_name} Performance Analysis', fontsize=14, fontweight='bold')
       
        # Filter valid times
        if 'best_seconds' in event_results.columns:
            best_times = event_results[event_results['best_seconds'].notna() &
                                      (event_results['best_seconds'] > 0) &
                                      (event_results['best_seconds'] < 600)].copy()
        else:
            best_times = pd.DataFrame()
       
        if 'average_seconds' in event_results.columns:
            avg_times = event_results[event_results['average_seconds'].notna() &
                                     (event_results['average_seconds'] > 0) &
                                     (event_results['average_seconds'] < 600)].copy()
        else:
            avg_times = pd.DataFrame()
       
        # Plot 1: Single/Best times with labeled points
        if not best_times.empty:
            years_best = best_times['year'].values
            times_best = best_times['best_seconds'].values
           
            # Plot line
            ax1.plot(years_best, times_best, '-', linewidth=1.5, color='#2E86AB', alpha=0.5)
           
            # Plot points with numbers
            for i, (year, time) in enumerate(zip(years_best, times_best)):
                ax1.plot(year, time, 'o', markersize=8, color='#2E86AB',
                        markeredgecolor='white', markeredgewidth=1)
                ax1.annotate(str(i+1), (year, time), textcoords="offset points",
                            xytext=(0,10), ha='center', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
           
            # Add trend line for best times
            if len(times_best) >= 3:
                z_best = np.polyfit(years_best, times_best, 1)
                p_best = np.poly1d(z_best)
                ax1.plot(years_best, p_best(years_best), '--', color='#2E86AB',
                        alpha=0.5, label=f'Trend: {-z_best[0]:.2f}s/year')
           
            # Highlight best ever
            best_idx = np.argmin(times_best)
            ax1.plot(years_best[best_idx], times_best[best_idx], '*', markersize=15,
                    color='gold', markeredgecolor='black', markeredgewidth=1,
                    label=f'PB: {times_best[best_idx]:.2f}s')
           
            # Add improvement rate text
            impr_rate, first, last, r2 = self.calculate_improvement_rate(results, event_id, use_best=True)
            ax1.text(0.02, 0.98, f'Improvement: {impr_rate:.2f}s/year\nR²: {r2:.2f}',
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
       
        ax1.set_xlabel('Year', fontsize=11)
        ax1.set_ylabel('Time (seconds)', fontsize=11)
        ax1.set_title('Single/Best Times (Numbered Points)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
       
        # Create legend for best times points
        if not best_times.empty:
            legend_text = "Best Times:\n"
            for i, (_, row) in enumerate(best_times.iterrows()):
                comp_name = row.get('competition_id', 'Unknown')[:10]
                year = int(row['year']) if pd.notna(row['year']) else 'Unknown'
                time = row['best_seconds']
                legend_text += f"{i+1}. {comp_name} ({year}): {time:.2f}s\n"
           
            ax1.text(1.02, 0.98, legend_text, transform=ax1.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
       
        # Plot 2: Average times with labeled points
        if not avg_times.empty:
            years_avg = avg_times['year'].values
            times_avg = avg_times['average_seconds'].values
           
            # Plot line
            ax2.plot(years_avg, times_avg, '-', linewidth=1.5, color='#A23B72', alpha=0.5)
           
            # Plot points with numbers
            for i, (year, time) in enumerate(zip(years_avg, times_avg)):
                ax2.plot(year, time, 's', markersize=8, color='#A23B72',
                        markeredgecolor='white', markeredgewidth=1)
                ax2.annotate(str(i+1), (year, time), textcoords="offset points",
                            xytext=(0,10), ha='center', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
           
            # Add trend line for average times
            if len(times_avg) >= 3:
                z_avg = np.polyfit(years_avg, times_avg, 1)
                p_avg = np.poly1d(z_avg)
                ax2.plot(years_avg, p_avg(years_avg), '--', color='#A23B72',
                        alpha=0.5, label=f'Trend: {-z_avg[0]:.2f}s/year')
           
            # Highlight best average
            best_avg_idx = np.argmin(times_avg)
            ax2.plot(years_avg[best_avg_idx], times_avg[best_avg_idx], '*', markersize=15,
                    color='gold', markeredgecolor='black', markeredgewidth=1,
                    label=f'Best Avg: {times_avg[best_avg_idx]:.2f}s')
           
            # Add improvement rate text
            impr_rate, first, last, r2 = self.calculate_improvement_rate(results, event_id, use_best=False)
            ax2.text(0.02, 0.98, f'Improvement: {impr_rate:.2f}s/year\nR²: {r2:.2f}',
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
       
        ax2.set_xlabel('Year', fontsize=11)
        ax2.set_ylabel('Time (seconds)', fontsize=11)
        ax2.set_title('Average Times (Numbered Points)', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
       
        # Create legend for average times points
        if not avg_times.empty:
            legend_text = "Average Times:\n"
            for i, (_, row) in enumerate(avg_times.iterrows()):
                comp_name = row.get('competition_id', 'Unknown')[:10]
                year = int(row['year']) if pd.notna(row['year']) else 'Unknown'
                time = row['average_seconds']
                legend_text += f"{i+1}. {comp_name} ({year}): {time:.2f}s\n"
           
            ax2.text(1.02, 0.98, legend_text, transform=ax2.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8))
       
        plt.tight_layout()
        plt.show()
   
    def plot_podium_pie_charts(self, wca_id, event_id, competitor_name):
        """Create pie charts with legends for podium history and next event probability"""
        event_name = self.event_names.get(event_id, event_id)
       
        # Get podium data for this specific event
        prob, podium_count, total_comps, avg_podium_time = self.predict_podium_probability(wca_id, event_id)
       
        if not isinstance(prob, (int, float)) or total_comps == 0:
            return
       
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{competitor_name} - {event_name} Podium Analysis', fontsize=14, fontweight='bold')
       
        # Pie Chart 1: Historical Podium Finishes with Legend
        labels1 = ['🏆 Podium Finishes', '❌ Non-Podium']
        sizes1 = [podium_count, total_comps - podium_count]
        colors1 = ['#2E86AB', '#A23B72']
        explode1 = (0.1, 0)  # Explode the podium slice
       
        wedges1, texts1, autotexts1 = ax1.pie(sizes1, explode=explode1, colors=colors1,
                                              autopct='%1.1f%%', startangle=90, shadow=True,
                                              textprops={'fontsize': 10})
       
        # Add legend for first pie chart
        ax1.legend(wedges1, labels1, title="Podium History", loc="center left",
                   bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
       
        for autotext in autotexts1:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
       
        ax1.set_title('Historical Podium Performance', fontweight='bold', fontsize=11)
        ax1.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
       
        # Add text box with details
        podium_percentage = (podium_count / total_comps) * 100 if total_comps > 0 else 0
        ax1.text(0, -1.4, f"📊 Statistics:\n• Total Competitions: {total_comps}\n• Podium Finishes: {podium_count}\n• Podium Rate: {podium_percentage:.1f}%",
                ha='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.9))
       
        # Pie Chart 2: Next Event Podium Probability with Legend
        next_podium_prob = prob
        next_non_podium = 100 - next_podium_prob
       
        labels2 = ['✅ Podium Chance', '⬜ Non-Podium Chance']
        sizes2 = [next_podium_prob, next_non_podium]
        colors2 = ['#F18F01', '#C73E1D']
        explode2 = (0.1, 0)
       
        wedges2, texts2, autotexts2 = ax2.pie(sizes2, explode=explode2, colors=colors2,
                                              autopct='%1.1f%%', startangle=90, shadow=True,
                                              textprops={'fontsize': 10})
       
        # Add legend for second pie chart
        ax2.legend(wedges2, labels2, title="Next Event Prediction", loc="center left",
                   bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
       
        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
       
        ax2.set_title('Next Event Podium Prediction', fontweight='bold', fontsize=11)
        ax2.axis('equal')
       
        # Get improvement trend
        results = self.get_competitor_results(wca_id)
        best_impr, _, _, _ = self.calculate_improvement_rate(results, event_id, use_best=True)
        avg_impr, _, _, _ = self.calculate_improvement_rate(results, event_id, use_best=False)
        future_best, best_conf = self.predict_future_time(results, event_id, years_ahead=1, use_best=True)
       
        # Add text box with prediction details
        trend_text = f"🔮 Prediction Details:\n• Probability: {next_podium_prob:.1f}%\n"
        if best_impr > 0:
            trend_text += f"• Improvement: {best_impr:.2f}s/year\n"
        if future_best is not None:
            trend_text += f"• Predicted Best: {future_best:.2f}s"
        else:
            trend_text += "• Insufficient data for time prediction"
       
        ax2.text(0, -1.4, trend_text, ha='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.9))
       
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make room for legends
        plt.show()
   
    def plot_podium_summary(self, wca_id, podium_predictions, competitor_name):
        """Create separate window for overall podium summary with legends"""
        if not podium_predictions:
            return
       
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{competitor_name} - Overall Podium Analysis', fontsize=14, fontweight='bold')
       
        # Podium probability bar chart with better styling
        events = [p['event'] for p in podium_predictions]
        probs = [p['probability'] for p in podium_predictions]
       
        # Sort by probability
        sorted_idx = np.argsort(probs)
        events = [events[i] for i in sorted_idx]
        probs = [probs[i] for i in sorted_idx]
       
        # Create color map based on probability
        colors = ['#FF6B6B' if p < 30 else '#FFD93D' if p < 60 else '#6BCB77' for p in probs]
       
        bars = ax1.barh(events, probs, color=colors, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Podium Probability (%)', fontsize=11)
        ax1.set_title('Podium Probability by Event', fontweight='bold', fontsize=12)
        ax1.set_xlim(0, 100)
       
        # Add value labels
        for bar, prob in zip(bars, probs):
            ax1.text(prob + 1, bar.get_y() + bar.get_height()/2, f'{prob:.1f}%',
                    va='center', fontweight='bold', fontsize=9)
       
        # Add legend for probability colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#6BCB77', label='High (≥60%)'),
            Patch(facecolor='#FFD93D', label='Medium (30-60%)'),
            Patch(facecolor='#FF6B6B', label='Low (<30%)')
        ]
        ax1.legend(handles=legend_elements, title='Probability Level',
                   loc='lower right', fontsize=8)
       
        ax1.grid(True, alpha=0.3, axis='x')
       
        # Podium finishes pie chart for top event with legend
        if podium_predictions:
            top_event = max(podium_predictions, key=lambda x: x['probability'])
            labels2 = ['🏆 Podium Finishes', '❌ Non-Podium']
            sizes2 = [top_event['podium_count'], top_event['total_comps'] - top_event['podium_count']]
            colors2 = ['#2E86AB', '#A23B72']
            explode2 = (0.1, 0)
           
            wedges2, texts2, autotexts2 = ax2.pie(sizes2, explode=explode2, colors=colors2,
                                                  autopct='%1.1f%%', startangle=90, shadow=True,
                                                  textprops={'fontsize': 10})
           
            # Add legend for the pie chart
            ax2.legend(wedges2, labels2, title=f"{top_event['event'][:20]}",
                       loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
           
            for autotext in autotexts2:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
           
            ax2.set_title(f"Podium History - {top_event['event']}", fontweight='bold', fontsize=11)
            ax2.axis('equal')
           
            # Add text with counts
            podium_percentage = (top_event['podium_count'] / top_event['total_comps']) * 100
            ax2.text(0, -1.3, f"📊 Statistics:\n• Total: {top_event['total_comps']}\n• Podium: {top_event['podium_count']}\n• Rate: {podium_percentage:.1f}%",
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.9))
       
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make room for legends
        plt.show()
   
    def analyze_competitor(self, wca_id):
        """Complete analysis for a competitor"""
        print(f"\n{'='*60}")
        print(f"🔍 ANALYZING COMPETITOR: {wca_id}")
        print('='*60)
       
        # Find competitor
        competitor = self.find_competitor(wca_id)
        if competitor is None:
            print(f"❌ Competitor with WCA ID '{wca_id}' not found!")
            return
       
        competitor_name = competitor['name']
       
        # Display basic info
        print(f"\n📋 COMPETITOR INFORMATION:")
        print(f"   Name: {competitor_name}")
        print(f"   Gender: {competitor['gender']}")
        print(f"   Country: {competitor['country_id']}")
        print(f"   WCA ID: {competitor['wca_id']}")
       
        # Get results
        results = self.get_competitor_results(wca_id)
       
        if results.empty:
            print("\n❌ No competition results found for this competitor")
            return
       
        print(f"\n📊 COMPETITION STATISTICS:")
        print(f"   Total competitions: {results['competition_id'].nunique()}")
        print(f"   Events participated: {results['event_id'].nunique()}")
        print(f"   Total solves: {len(results)}")
        print(f"   First competition: {int(results['year'].min()) if results['year'].notna().any() else 'Unknown'}")
        print(f"   Latest competition: {int(results['year'].max()) if results['year'].notna().any() else 'Unknown'}")
        print(f"   Years active: {int(results['year'].max() - results['year'].min()) if results['year'].notna().any() else 'Unknown'}")
       
        # List all events participated
        participated_events = results['event_id'].unique()
        print(f"\n🎯 EVENTS PARTICIPATED ({len(participated_events)}):")
        for event_id in sorted(participated_events):
            event_name = self.event_names.get(event_id, event_id)
            event_count = len(results[results['event_id'] == event_id])
            print(f"   • {event_name} ({event_id}): {event_count} results")
       
        # Analyze by event
        print(f"\n📊 PERFORMANCE SUMMARY BY EVENT:")
       
        event_analysis = []
        for event_id in participated_events:
            event_name = self.event_names.get(event_id, event_id)
           
            # Best times analysis
            best_improvement, best_first, best_last, best_r2 = self.calculate_improvement_rate(results, event_id, use_best=True)
           
            # Average times analysis
            avg_improvement, avg_first, avg_last, avg_r2 = self.calculate_improvement_rate(results, event_id, use_best=False)
           
            event_results = results[results['event_id'] == event_id]
           
            # Get best and average times
            best_time = event_results['best_seconds'].min() if 'best_seconds' in event_results.columns and not event_results['best_seconds'].isna().all() else None
            avg_time = event_results['average_seconds'].mean() if 'average_seconds' in event_results.columns and not event_results['average_seconds'].isna().all() else None
           
            # Predict future times
            future_best, best_conf = self.predict_future_time(results, event_id, years_ahead=1, use_best=True)
            future_avg, avg_conf = self.predict_future_time(results, event_id, years_ahead=1, use_best=False)
           
            event_analysis.append({
                'event_id': event_id,
                'event_name': event_name,
                'num_results': len(event_results),
                'best_time': best_time,
                'avg_time': avg_time,
                'best_improvement': best_improvement,
                'avg_improvement': avg_improvement,
                'best_r2': best_r2,
                'avg_r2': avg_r2,
                'future_best': future_best,
                'future_avg': future_avg,
                'best_conf': best_conf,
                'avg_conf': avg_conf
            })
       
        # Sort events by number of results
        event_df = pd.DataFrame(event_analysis)
        if not event_df.empty:
            event_df = event_df.sort_values('num_results', ascending=False)
           
            # Create a summary table
            print("\n" + "-"*110)
            print(f"{'Event':<25} {'Results':<8} {'Best':<12} {'Avg':<12} {'Best Impr':<15} {'Avg Impr':<15} {'Best R²':<8}")
            print("-"*110)
           
            for _, row in event_df.iterrows():
                best_str = f"{row['best_time']:.2f}s" if pd.notna(row['best_time']) else "N/A"
                avg_str = f"{row['avg_time']:.2f}s" if pd.notna(row['avg_time']) else "N/A"
                best_impr_str = f"{row['best_improvement']:.2f}s/y" if pd.notna(row['best_improvement']) and row['best_improvement'] != 0 else "stable"
                avg_impr_str = f"{row['avg_improvement']:.2f}s/y" if pd.notna(row['avg_improvement']) and row['avg_improvement'] != 0 else "stable"
                best_r2_str = f"{row['best_r2']:.2f}" if pd.notna(row['best_r2']) else "N/A"
               
                print(f"{row['event_name'][:25]:<25} {row['num_results']:<8} {best_str:<12} {avg_str:<12} {best_impr_str:<15} {avg_impr_str:<15} {best_r2_str:<8}")
           
            print("-"*110)
       
        # PODIUM PREDICTION SECTION
        print(f"\n{'='*60}")
        print(f"🏆 PODIUM PREDICTION ANALYSIS")
        print('='*60)
       
        print("\nBased on historical podium finishes and improvement trends:")
       
        podium_predictions = []
        for event_id in participated_events:
            prob, podium_count, total_comps, avg_podium_time = self.predict_podium_probability(wca_id, event_id)
           
            if isinstance(prob, (int, float)) and prob > 0 and total_comps > 0:
                event_name = self.event_names.get(event_id, event_id)
               
                # Get improvement rate for this event
                event_data = event_df[event_df['event_id'] == event_id].iloc[0] if not event_df[event_df['event_id'] == event_id].empty else None
               
                if event_data is not None:
                    best_impr = event_data['best_improvement']
                    avg_impr = event_data['avg_improvement']
                   
                    # Predict future time
                    future_best = event_data['future_best']
                    future_avg = event_data['future_avg']
                else:
                    best_impr = 0
                    avg_impr = 0
                    future_best = None
                    future_avg = None
               
                # Create progress bar
                bar_length = 30
                filled = int(prob / 100 * bar_length)
                bar = '█' * filled + '░' * (bar_length - filled)
               
                print(f"\n   📌 {event_name}:")
                print(f"     [{bar}] {prob:.1f}% podium probability")
                print(f"     • Podium finishes: {podium_count}/{total_comps} competitions ({podium_count/total_comps*100:.1f}%)")
               
                if avg_podium_time is not None:
                    print(f"     • Average podium time: {avg_podium_time:.2f}s")
               
                if best_impr > 0:
                    impr_symbol = "📈" if best_impr > 0 else "📉"
                    print(f"     • Improvement rate: {best_impr:.2f}s/year (best), {avg_impr:.2f}s/year (avg) {impr_symbol}")
               
                if future_best is not None and future_avg is not None and future_best > 0:
                    print(f"     • Predicted next year: {future_best:.2f}s (best), {future_avg:.2f}s (avg)")
               
                podium_predictions.append({
                    'event': event_name,
                    'event_id': event_id,
                    'probability': prob,
                    'podium_count': podium_count,
                    'total_comps': total_comps,
                    'avg_podium_time': avg_podium_time
                })
       
        # GENERATE SEPARATE GRAPHS FOR EACH EVENT PARTICIPATED
        print(f"\n{'='*60}")
        print(f"📊 GENERATING PERFORMANCE GRAPHS")
        print('='*60)
        print(f"Opening separate windows for each event participated...")
        print(f"Each point is numbered - check the legend boxes for details!")
       
        # Create performance graphs for each event
        for event_id in participated_events:
            event_name = self.event_names.get(event_id, event_id)
            print(f"   • Plotting {event_name} performance...")
            self.plot_event_performance(results, event_id, competitor_name)
       
        # Create podium pie charts for each event
        print(f"\n{'='*60}")
        print(f"🥧 GENERATING PODIUM PIE CHARTS")
        print('='*60)
        print(f"Opening separate windows with podium analysis for each event...")
       
        for event_id in participated_events:
            event_name = self.event_names.get(event_id, event_id)
            print(f"   • Plotting {event_name} podium charts...")
            self.plot_podium_pie_charts(wca_id, event_id, competitor_name)
       
        # Create overall podium summary
        if podium_predictions:
            print(f"   • Plotting overall podium summary...")
            self.plot_podium_summary(wca_id, podium_predictions, competitor_name)
       
        # Improvement summary
        print(f"\n{'='*60}")
        print(f"📈 IMPROVEMENT SUMMARY")
        print('='*60)
       
        # Find most improved event
        if not event_df.empty:
            # Filter events with improvement data
            valid_best = event_df[event_df['best_improvement'].notna() & (event_df['best_improvement'] > 0)]
            valid_avg = event_df[event_df['avg_improvement'].notna() & (event_df['avg_improvement'] > 0)]
           
            if not valid_best.empty:
                best_improved = valid_best.loc[valid_best['best_improvement'].idxmax()]
                print(f"\n⚡ FASTEST IMPROVING EVENT (BEST TIMES):")
                print(f"   {best_improved['event_name']}: {best_improved['best_improvement']:.2f}s/year improvement (R²: {best_improved['best_r2']:.2f})")
                if pd.notna(best_improved['best_time']) and best_improved['future_best'] is not None:
                    print(f"   Best time: {best_improved['best_time']:.2f}s → Predicted: {best_improved['future_best']:.2f}s")
           
            if not valid_avg.empty:
                avg_improved = valid_avg.loc[valid_avg['avg_improvement'].idxmax()]
                print(f"\n⚡ FASTEST IMPROVING EVENT (AVERAGE TIMES):")
                print(f"   {avg_improved['event_name']}: {avg_improved['avg_improvement']:.2f}s/year improvement (R²: {avg_improved['avg_r2']:.2f})")
                if pd.notna(avg_improved['avg_time']) and avg_improved['future_avg'] is not None:
                    print(f"   Avg time: {avg_improved['avg_time']:.2f}s → Predicted: {avg_improved['future_avg']:.2f}s")
       
        return event_df, podium_predictions


# ============================================================================
# MAIN MENU SYSTEM
# ============================================================================

class WCAMenu:
    """Main menu system for WCA analysis"""
   
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
   
    def global_statistics_menu(self):
        """Global statistics menu"""
        print("\n" + "="*60)
        print("📈 GLOBAL STATISTICS & TRENDS")
        print("="*60)
       
        if self.data['results'].empty:
            print("❌ No results data available")
            return
       
        results = self.data['results']
       
        # Basic stats
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   Total results: {len(results):,}")
        print(f"   Unique competitors: {results['person_id'].nunique():,}")
        print(f"   Unique competitions: {results['competition_id'].nunique():,}")
        print(f"   Year range: {int(results['year'].min()) if results['year'].notna().any() else 'N/A'} - {int(results['year'].max()) if results['year'].notna().any() else 'N/A'}")
       
        # Event distribution
        if 'events' in self.data and not self.data['events'].empty:
            print(f"\n🎯 TOP EVENTS BY PARTICIPATION:")
            event_counts = results['event_id'].value_counts().head(5)
            for event_id, count in event_counts.items():
                event_name = self.competitor_analyzer.event_names.get(event_id, event_id)
                print(f"   {event_name}: {count:,} results")
       
        # Plot global trends
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Global WCA Statistics', fontsize=14, fontweight='bold')
       
        # 1. Participation over time
        yearly_comp = results.groupby('year')['competition_id'].nunique()
        yearly_comp.plot(ax=axes[0,0], marker='o', color='#2E86AB')
        axes[0,0].set_title('Competitions per Year')
        axes[0,0].set_xlabel('Year')
        axes[0,0].set_ylabel('Number of Competitions')
        axes[0,0].grid(True, alpha=0.3)
       
        # 2. Average times over time (3x3)
        if 'average_seconds' in results.columns:
            threeby3 = results[results['event_id'] == '333']
            if not threeby3.empty:
                yearly_avg = threeby3.groupby('year')['average_seconds'].median()
                yearly_avg.plot(ax=axes[0,1], marker='o', color='#A23B72')
                axes[0,1].set_title('Median 3x3 Time Over Years')
                axes[0,1].set_xlabel('Year')
                axes[0,1].set_ylabel('Time (seconds)')
                axes[0,1].grid(True, alpha=0.3)
       
        # 3. Top countries
        if 'country_id' in results.columns:
            top_countries = results['country_id'].value_counts().head(10)
            top_countries.plot(kind='bar', ax=axes[1,0], color='#F18F01')
            axes[1,0].set_title('Top 10 Countries by Participation')
            axes[1,0].set_xlabel('Country')
            axes[1,0].set_ylabel('Number of Results')
            axes[1,0].tick_params(axis='x', rotation=45)
       
        # 4. Time distribution
        if 'average_seconds' in results.columns:
            times = results[results['average_seconds'] < 60]['average_seconds']
            axes[1,1].hist(times, bins=50, edgecolor='black', alpha=0.7, color='#C73E1D')
            axes[1,1].set_title('Distribution of Solve Times (<60s)')
            axes[1,1].set_xlabel('Time (seconds)')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].axvline(times.median(), color='red', linestyle='--',
                             label=f'Median: {times.median():.2f}s')
            axes[1,1].legend()
       
        plt.tight_layout()
        plt.show()
       
        input("\nPress Enter to continue...")
   
    def country_analysis_menu(self):
        """Country analysis menu"""
        print("\n" + "="*60)
        print("🌍 COUNTRY PERFORMANCE ANALYSIS")
        print("="*60)
       
        if 'countries' not in self.data or self.data['countries'].empty:
            print("❌ Countries data not available")
            return
       
        # Show available countries
        print("\n📋 TOP 20 COUNTRIES BY PARTICIPATION:")
        if 'results' in self.data and 'country_id' in self.data['results'].columns:
            country_counts = self.data['results']['country_id'].value_counts().head(20)
            for i, (country_id, count) in enumerate(country_counts.items(), 1):
                country_name = self.data['countries'][self.data['countries']['id'] == country_id]['name'].values
                country_name = country_name[0] if len(country_name) > 0 else country_id
                print(f"   {i}. {country_name}: {count:,} results")
       
        country_id = input("\nEnter country code (e.g., USA, Germany, China): ").strip()
       
        if not country_id:
            return
       
        # Find country
        country_match = self.data['countries'][
            (self.data['countries']['id'].str.contains(country_id, case=False, na=False)) |
            (self.data['countries']['name'].str.contains(country_id, case=False, na=False))
        ]
       
        if country_match.empty:
            print(f"❌ Country '{country_id}' not found")
            return
       
        country = country_match.iloc[0]
        print(f"\n📊 ANALYZING: {country['name']} ({country['id']})")
       
        # Get country results
        country_results = self.competitor_analyzer.get_country_results(country['id'])
       
        if country_results.empty:
            print("❌ No results found for this country")
            return
       
        print(f"\n   Total results: {len(country_results):,}")
        print(f"   Unique competitors: {country_results['person_id'].nunique():,}")
        print(f"   Unique competitions: {country_results['competition_id'].nunique():,}")
       
        # Best events
        print(f"\n🎯 BEST PERFORMANCES BY EVENT:")
        for event_id in ['333', '222', '444', '555', '666', '777']:
            event_results = country_results[country_results['event_id'] == event_id]
            if not event_results.empty and 'average_seconds' in event_results.columns:
                best_time = event_results['average_seconds'].min()
                event_name = self.competitor_analyzer.event_names.get(event_id, event_id)
                print(f"   {event_name}: {best_time:.2f}s")
       
        # Plot country performance
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{country["name"]} Performance Analysis', fontsize=14, fontweight='bold')
       
        # 1. Top competitors in country
        top_comp = country_results.groupby('person_id')['average_seconds'].min().nsmallest(10)
        axes[0,0].barh(range(len(top_comp)), top_comp.values, color='#2E86AB')
        axes[0,0].set_yticks(range(len(top_comp)))
        axes[0,0].set_yticklabels([f"Competitor {i+1}" for i in range(len(top_comp))])
        axes[0,0].set_xlabel('Best Time (seconds)')
        axes[0,0].set_title(f'Top 10 Competitors')
        axes[0,0].grid(True, alpha=0.3)
       
        # 2. Event participation
        event_participation = country_results['event_id'].value_counts().head(8)
        axes[0,1].bar(event_participation.index, event_participation.values, color='#A23B72')
        axes[0,1].set_xlabel('Event')
        axes[0,1].set_ylabel('Number of Results')
        axes[0,1].set_title('Event Participation')
        axes[0,1].tick_params(axis='x', rotation=45)
       
        # 3. Yearly participation
        yearly_part = country_results.groupby('year')['person_id'].nunique()
        axes[1,0].plot(yearly_part.index, yearly_part.values, marker='o', color='#F18F01', linewidth=2)
        axes[1,0].set_xlabel('Year')
        axes[1,0].set_ylabel('Active Competitors')
        axes[1,0].set_title('Active Competitors Over Time')
        axes[1,0].grid(True, alpha=0.3)
       
        # 4. Time distribution for main event
        main_event = country_results[country_results['event_id'] == '333']
        if not main_event.empty:
            axes[1,1].hist(main_event['average_seconds'], bins=30, edgecolor='black', alpha=0.7, color='#C73E1D')
            axes[1,1].set_xlabel('Time (seconds)')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].set_title('3x3 Time Distribution')
            axes[1,1].axvline(main_event['average_seconds'].median(), color='red',
                             linestyle='--', label=f'Median: {main_event["average_seconds"].median():.2f}s')
            axes[1,1].legend()
       
        plt.tight_layout()
        plt.show()
       
        input("\nPress Enter to continue...")
   
    def event_analysis_menu(self):
        """Event analysis menu"""
        print("\n" + "="*60)
        print("🎯 EVENT PERFORMANCE ANALYSIS")
        print("="*60)
       
        if 'events' not in self.data or self.data['events'].empty:
            print("❌ Events data not available")
            return
       
        # Show available events
        print("\n📋 AVAILABLE EVENTS:")
        for _, row in self.data['events'].iterrows():
            print(f"   {row['id']}: {row['name']}")
       
        event_id = input("\nEnter event ID (e.g., 333, 222, 444): ").strip()
       
        if not event_id:
            return
       
        event_name = self.competitor_analyzer.event_names.get(event_id, event_id)
       
        # Get event results
        if 'results' not in self.data:
            print("❌ Results data not available")
            return
       
        event_results = self.data['results'][
            (self.data['results']['event_id'] == event_id) &
            (self.data['results']['average_seconds'].notna()) &
            (self.data['results']['average_seconds'] < 300)
        ]
       
        if event_results.empty:
            print(f"❌ No results found for event {event_name}")
            return
       
        print(f"\n📊 {event_name} STATISTICS:")
        print(f"   Total results: {len(event_results):,}")
        print(f"   Unique competitors: {event_results['person_id'].nunique():,}")
        print(f"   World Record: {event_results['average_seconds'].min():.2f}s")
        print(f"   Median time: {event_results['average_seconds'].median():.2f}s")
        print(f"   Average time: {event_results['average_seconds'].mean():.2f}s")
        print(f"   Standard deviation: {event_results['average_seconds'].std():.2f}s")
       
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"\n📈 PERCENTILES:")
        for p in percentiles:
            print(f"   {p}th percentile: {event_results['average_seconds'].quantile(p/100):.2f}s")
       
        # Plot distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{event_name} Analysis', fontsize=14, fontweight='bold')
       
        # 1. Time distribution
        axes[0,0].hist(event_results['average_seconds'], bins=50, edgecolor='black', alpha=0.7, color='#2E86AB')
        axes[0,0].set_xlabel('Time (seconds)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Time Distribution')
        axes[0,0].axvline(event_results['average_seconds'].median(), color='red',
                         linestyle='--', label=f'Median: {event_results["average_seconds"].median():.2f}s')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
       
        # 2. Box plot by year
        if 'year' in event_results.columns:
            recent_years = event_results[event_results['year'] >= 2015]
            if not recent_years.empty:
                recent_years.boxplot(column='average_seconds', by='year', ax=axes[0,1])
                axes[0,1].set_title('Times by Year')
                axes[0,1].set_xlabel('Year')
                axes[0,1].set_ylabel('Time (seconds)')
                axes[0,1].tick_params(axis='x', rotation=45)
       
        # 3. Performance over time
        if 'year' in event_results.columns:
            yearly_stats = event_results.groupby('year')['average_seconds'].agg(['mean', 'median', 'min'])
            axes[1,0].plot(yearly_stats.index, yearly_stats['mean'], 'o-', label='Mean', linewidth=2, color='#2E86AB')
            axes[1,0].plot(yearly_stats.index, yearly_stats['median'], 's-', label='Median', linewidth=2, color='#A23B72')
            axes[1,0].plot(yearly_stats.index, yearly_stats['min'], '^-', label='Best', linewidth=2, color='#F18F01')
            axes[1,0].set_xlabel('Year')
            axes[1,0].set_ylabel('Time (seconds)')
            axes[1,0].set_title('Performance Over Time')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
       
        # 4. Top countries for this event
        if 'country_id' in event_results.columns:
            country_stats = event_results.groupby('country_id').agg({
                'average_seconds': 'median',
                'person_id': 'nunique'
            }).reset_index()
            top_countries = country_stats.nsmallest(10, 'average_seconds')
           
            # Get country names
            if 'countries' in self.data and not self.data['countries'].empty:
                country_names = []
                for cid in top_countries['country_id']:
                    name = self.data['countries'][self.data['countries']['id'] == cid]['name'].values
                    country_names.append(name[0][:15] + '...' if len(name[0]) > 15 else name[0] if len(name) > 0 else cid)
            else:
                country_names = top_countries['country_id']
           
            axes[1,1].barh(range(len(top_countries)), top_countries['average_seconds'], color='#C73E1D')
            axes[1,1].set_yticks(range(len(top_countries)))
            axes[1,1].set_yticklabels(country_names)
            axes[1,1].set_xlabel('Median Time (seconds)')
            axes[1,1].set_title('Top 10 Countries by Median Time')
            axes[1,1].grid(True, alpha=0.3)
       
        plt.tight_layout()
        plt.show()
       
        input("\nPress Enter to continue...")
   
    def predictive_models_menu(self):
        """Predictive models menu"""
        print("\n" + "="*60)
        print("🤖 PREDICTIVE MODELS")
        print("="*60)
       
        print("\n1. Forecast World Records (Linear Regression)")
        print("2. Cluster Competitors by Performance (K-Means)")
        print("3. Back to Main Menu")
       
        choice = input("\nEnter choice: ").strip()
       
        if choice == '1':
            self.forecast_world_records()
        elif choice == '2':
            self.cluster_competitors()
   
    def forecast_world_records(self):
        """Forecast world records using linear regression"""
        event_id = input("Enter event ID to forecast (default: 333): ").strip() or '333'
       
        if 'results' not in self.data or self.data['results'].empty:
            print("❌ No results data available")
            return
       
        event_results = self.data['results'][
            (self.data['results']['event_id'] == event_id) &
            (self.data['results']['best_seconds'].notna()) &
            (self.data['results']['best_seconds'] > 0) &
            (self.data['results']['best_seconds'] < 300) &
            (self.data['results']['year'].notna()) &
            (self.data['results']['year'] >= 2003)
        ]
       
        if event_results.empty:
            print(f"❌ No data for event {event_id}")
            return
       
        # Calculate yearly best and world record progression
        yearly_best = event_results.groupby('year')['best_seconds'].min().reset_index()
        yearly_best = yearly_best.sort_values('year')
        yearly_best['wr'] = yearly_best['best_seconds'].cummin()
       
        # Get the actual world record holders
        wr_progression = []
        current_wr = float('inf')
        for _, row in yearly_best.iterrows():
            if row['best_seconds'] < current_wr:
                current_wr = row['best_seconds']
                wr_progression.append(row)
       
        wr_df = pd.DataFrame(wr_progression)
       
        if len(wr_df) >= 3:
            # Linear regression on WR progression
            X = wr_df['year'].values.reshape(-1, 1)
            y = wr_df['best_seconds'].values
           
            model = LinearRegression()
            model.fit(X, y)
           
            last_year = int(wr_df['year'].max())
            future_years = np.arange(last_year + 1, last_year + 11)
            forecast = model.predict(future_years.reshape(-1, 1))
           
            # Calculate confidence intervals
            residuals = y - model.predict(X)
            ci = 1.96 * np.std(residuals)
           
            print(f"\n📈 WORLD RECORD FORECAST FOR {self.competitor_analyzer.event_names.get(event_id, event_id)}:")
            print(f"   Current WR: {wr_df['best_seconds'].iloc[-1]:.2f}s ({int(wr_df['year'].iloc[-1])})")
            print(f"   Trend: {model.coef_[0]:.3f}s per year improvement")
            print(f"   R² Score: {r2_score(y, model.predict(X)):.3f}")
            print("\n   Forecast:")
            for year, pred in zip(future_years, forecast):
                print(f"     {int(year)}: {pred:.2f}s ±{ci:.2f}s")
           
            # Plot the forecast
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f'{self.competitor_analyzer.event_names.get(event_id, event_id)} World Record Forecast', fontsize=14, fontweight='bold')
           
            # Plot 1: Historical WR progression
            axes[0].plot(wr_df['year'], wr_df['best_seconds'], 'o-', linewidth=2, markersize=8, label='Historical WR', color='#2E86AB')
            axes[0].plot(future_years, forecast, 'r--', linewidth=2, label='Forecast', color='#A23B72')
            axes[0].fill_between(future_years, forecast - ci, forecast + ci, alpha=0.2, color='#F18F01', label='95% CI')
            axes[0].set_xlabel('Year')
            axes[0].set_ylabel('Time (seconds)')
            axes[0].set_title('WR Progression & Forecast')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].invert_yaxis()  # Lower times are better
           
            # Plot 2: Distribution of top times
            top_times = event_results.nsmallest(100, 'best_seconds')
            axes[1].hist(top_times['best_seconds'], bins=20, edgecolor='black', alpha=0.7, color='#C73E1D')
            axes[1].axvline(wr_df['best_seconds'].iloc[-1], color='#2E86AB', linewidth=2, label=f'Current WR: {wr_df["best_seconds"].iloc[-1]:.2f}s')
            axes[1].axvline(forecast[0], color='#A23B72', linestyle='--', linewidth=2, label=f'Forecast {future_years[0]}: {forecast[0]:.2f}s')
            axes[1].set_xlabel('Time (seconds)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Distribution of Top 100 Times')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
           
            plt.tight_layout()
            plt.show()
        else:
            print("❌ Not enough WR progression data for forecasting")
   
    def cluster_competitors(self):
        """Cluster competitors by performance"""
        event_id = input("Enter event ID to cluster (default: 333): ").strip() or '333'
       
        if 'results' not in self.data or self.data['results'].empty:
            print("❌ No results data available")
            return
       
        event_results = self.data['results'][
            (self.data['results']['event_id'] == event_id) &
            (self.data['results']['average_seconds'].notna()) &
            (self.data['results']['average_seconds'] < 60)
        ]
       
        if event_results.empty:
            print(f"❌ No data for event {event_id}")
            return
       
        # Calculate competitor statistics
        comp_stats = event_results.groupby('person_id').agg({
            'average_seconds': ['mean', 'std', 'min', 'count']
        }).round(2)
       
        comp_stats.columns = ['avg_time', 'std_time', 'best_time', 'num_solves']
        comp_stats = comp_stats[comp_stats['num_solves'] >= 5].copy()
       
        if len(comp_stats) < 20:
            print("❌ Not enough competitors with sufficient solves for clustering")
            return
       
        # Prepare features for clustering
        features = ['avg_time', 'std_time', 'best_time']
        X = comp_stats[features].fillna(0)
       
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
       
        # Determine optimal number of clusters (elbow method)
        inertias = []
        K_range = range(2, 8)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
       
        optimal_k = 3  # Default
       
        # Perform clustering with optimal_k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        comp_stats['cluster'] = kmeans.fit_predict(X_scaled)
       
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Competitor Clustering - {self.competitor_analyzer.event_names.get(event_id, event_id)}', fontsize=14, fontweight='bold')
       
        # Try PCA for visualization
        try:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
           
            # Plot 1: PCA visualization
            scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                                     c=comp_stats['cluster'], cmap='viridis',
                                     s=50, alpha=0.6)
            axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            axes[0].set_title('PCA Visualization')
            plt.colorbar(scatter, ax=axes[0], label='Cluster')
            axes[0].grid(True, alpha=0.3)
        except:
            # If PCA fails, just show avg_time vs std_time
            scatter = axes[0].scatter(comp_stats['avg_time'], comp_stats['std_time'],
                                     c=comp_stats['cluster'], cmap='viridis',
                                     s=50, alpha=0.6)
            axes[0].set_xlabel('Average Time (seconds)')
            axes[0].set_ylabel('Standard Deviation (seconds)')
            axes[0].set_title('Avg Time vs Std Dev')
            plt.colorbar(scatter, ax=axes[0], label='Cluster')
            axes[0].grid(True, alpha=0.3)
       
        # Plot 2: Cluster characteristics
        cluster_summary = comp_stats.groupby('cluster')[['avg_time', 'std_time', 'best_time', 'num_solves']].mean()
       
        x = np.arange(len(cluster_summary))
        width = 0.2
       
        axes[1].bar(x - width*1.5, cluster_summary['avg_time'], width, label='Avg Time', alpha=0.8, color='#2E86AB')
        axes[1].bar(x - width/2, cluster_summary['std_time'], width, label='Std Dev', alpha=0.8, color='#A23B72')
        axes[1].bar(x + width/2, cluster_summary['best_time'], width, label='Best Time', alpha=0.8, color='#F18F01')
        axes[1].bar(x + width*1.5, cluster_summary['num_solves']/10, width, label='Solves/10', alpha=0.8, color='#C73E1D')
       
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Time (seconds)')
        axes[1].set_title('Cluster Characteristics')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f'Cluster {int(i)}' for i in cluster_summary.index])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
       
        plt.tight_layout()
        plt.show()
       
        print("\n📊 CLUSTER ANALYSIS:")
        for i in range(optimal_k):
            cluster_data = comp_stats[comp_stats['cluster'] == i]
            print(f"\n   Cluster {i}: {len(cluster_data)} competitors")
            print(f"     Avg Time: {cluster_data['avg_time'].mean():.2f}s ± {cluster_data['std_time'].mean():.2f}s")
            print(f"     Best Time: {cluster_data['best_time'].mean():.2f}s")
            print(f"     Avg Solves: {cluster_data['num_solves'].mean():.0f}")


if __name__ == "__main__":
    menu = WCAMenu()
    menu.run()
