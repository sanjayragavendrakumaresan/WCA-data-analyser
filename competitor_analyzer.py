"""Competitor-level analysis, prediction, and plotting."""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CompetitorAnalyzer:
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
