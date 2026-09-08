"""Interactive global, country, and event statistics menus."""

import matplotlib.pyplot as plt
import pandas as pd


class StatisticsMenuMixin:
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
