"""Interactive predictive-model and clustering menus."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


class PredictiveMenuMixin:
    def predictive_models_menu(self):
            """Predictive models menu"""
            print("\n" + "="*60)
            print(" PREDICTIVE MODELS")
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
                print(" No results data available")
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
                print(f" No data for event {event_id}")
                return

                                                                
            yearly_best = event_results.groupby('year')['best_seconds'].min().reset_index()
            yearly_best = yearly_best.sort_values('year')
            yearly_best['wr'] = yearly_best['best_seconds'].cummin()

                                                 
            wr_progression = []
            current_wr = float('inf')
            for _, row in yearly_best.iterrows():
                if row['best_seconds'] < current_wr:
                    current_wr = row['best_seconds']
                    wr_progression.append(row)

            wr_df = pd.DataFrame(wr_progression)

            if len(wr_df) >= 3:
                                                     
                X = wr_df['year'].values.reshape(-1, 1)
                y = wr_df['best_seconds'].values

                model = LinearRegression()
                model.fit(X, y)

                last_year = int(wr_df['year'].max())
                future_years = np.arange(last_year + 1, last_year + 11)
                forecast = model.predict(future_years.reshape(-1, 1))

                                                
                residuals = y - model.predict(X)
                ci = 1.96 * np.std(residuals)

                print(f"\n WORLD RECORD FORECAST FOR {self.competitor_analyzer.event_names.get(event_id, event_id)}:")
                print(f"   Current WR: {wr_df['best_seconds'].iloc[-1]:.2f}s ({int(wr_df['year'].iloc[-1])})")
                print(f"   Trend: {model.coef_[0]:.3f}s per year improvement")
                print(f"   R^2 Score: {r2_score(y, model.predict(X)):.3f}")
                print("\n   Forecast:")
                for year, pred in zip(future_years, forecast):
                    print(f"     {int(year)}: {pred:.2f}s +/-{ci:.2f}s")

                                   
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                fig.suptitle(f'{self.competitor_analyzer.event_names.get(event_id, event_id)} World Record Forecast', fontsize=14, fontweight='bold')

                                                   
                axes[0].plot(wr_df['year'], wr_df['best_seconds'], 'o-', linewidth=2, markersize=8, label='Historical WR', color='#2E86AB')
                axes[0].plot(future_years, forecast, 'r--', linewidth=2, label='Forecast', color='#A23B72')
                axes[0].fill_between(future_years, forecast - ci, forecast + ci, alpha=0.2, color='#F18F01', label='95% CI')
                axes[0].set_xlabel('Year')
                axes[0].set_ylabel('Time (seconds)')
                axes[0].set_title('WR Progression & Forecast')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                axes[0].invert_yaxis()                          

                                                   
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
                print(" Not enough WR progression data for forecasting")


    def cluster_competitors(self):
            """Cluster competitors by performance"""
            event_id = input("Enter event ID to cluster (default: 333): ").strip() or '333'

            if 'results' not in self.data or self.data['results'].empty:
                print(" No results data available")
                return

            event_results = self.data['results'][
                (self.data['results']['event_id'] == event_id) &
                (self.data['results']['average_seconds'].notna()) &
                (self.data['results']['average_seconds'] < 60)
            ]

            if event_results.empty:
                print(f" No data for event {event_id}")
                return

                                             
            comp_stats = event_results.groupby('person_id').agg({
                'average_seconds': ['mean', 'std', 'min', 'count']
            }).round(2)

            comp_stats.columns = ['avg_time', 'std_time', 'best_time', 'num_solves']
            comp_stats = comp_stats[comp_stats['num_solves'] >= 5].copy()

            if len(comp_stats) < 20:
                print(" Not enough competitors with sufficient solves for clustering")
                return

                                             
            features = ['avg_time', 'std_time', 'best_time']
            X = comp_stats[features].fillna(0)

                                
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

                                                                 
            inertias = []
            K_range = range(2, 8)
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)

            optimal_k = 3           

                                               
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            comp_stats['cluster'] = kmeans.fit_predict(X_scaled)

                           
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f'Competitor Clustering - {self.competitor_analyzer.event_names.get(event_id, event_id)}', fontsize=14, fontweight='bold')

                                       
            try:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)

                                           
                scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                                         c=comp_stats['cluster'], cmap='viridis',
                                         s=50, alpha=0.6)
                axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
                axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
                axes[0].set_title('PCA Visualization')
                plt.colorbar(scatter, ax=axes[0], label='Cluster')
                axes[0].grid(True, alpha=0.3)
            except:
                                                              
                scatter = axes[0].scatter(comp_stats['avg_time'], comp_stats['std_time'],
                                         c=comp_stats['cluster'], cmap='viridis',
                                         s=50, alpha=0.6)
                axes[0].set_xlabel('Average Time (seconds)')
                axes[0].set_ylabel('Standard Deviation (seconds)')
                axes[0].set_title('Avg Time vs Std Dev')
                plt.colorbar(scatter, ax=axes[0], label='Cluster')
                axes[0].grid(True, alpha=0.3)

                                             
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

            print("\n CLUSTER ANALYSIS:")
            for i in range(optimal_k):
                cluster_data = comp_stats[comp_stats['cluster'] == i]
                print(f"\n   Cluster {i}: {len(cluster_data)} competitors")
                print(f"     Avg Time: {cluster_data['avg_time'].mean():.2f}s +/- {cluster_data['std_time'].mean():.2f}s")
                print(f"     Best Time: {cluster_data['best_time'].mean():.2f}s")
                print(f"     Avg Solves: {cluster_data['num_solves'].mean():.0f}")
