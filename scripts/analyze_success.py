import pandas as pd
import sys

try:
    df = pd.read_csv(r'benchmark_results_20251127_144229/benchmark_results.csv')
    
    # Define categories
    def get_category(contamination):
        s = str(contamination)
        if s == 'none':
            return 'Pure'
        elif any(x in s for x in ['eeg', 'ecg', 'respiratory']):
            return 'Biomedical'
        else:
            return 'Contaminated'
            
    df['category'] = df['contamination'].apply(get_category)
    
    with open('success_analysis.txt', 'w') as f:
        f.write("--- Category Breakdown ---\n")
        f.write(df.groupby('category')['convergence_flag'].mean().to_string())
        f.write("\n\n--- Contamination Breakdown ---\n")
        f.write(df.groupby('contamination')['convergence_flag'].mean().to_string())
        
    print("Analysis saved to success_analysis.txt")
    
except Exception as e:
    print(f"Error: {e}")
