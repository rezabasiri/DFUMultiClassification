"""Compare all solutions"""
import pandas as pd
import os

if not os.path.exists('results.csv'):
    print("❌ No results.csv found! Run tests first.")
    exit(1)

results = pd.read_csv('results.csv')

print("="*80)
print("RF IMPROVEMENT COMPARISON")
print("="*80)

# Sort by Kappa (best first)
results_sorted = results.sort_values('kappa', ascending=False)

print(f"\n{'Method':<15} {'Acc':>6} {'Kappa':>6} {'F1_I':>6} {'F1_P':>6} {'F1_R':>6} {'Macro_F1':>8}")
print("-"*80)

for _, row in results_sorted.iterrows():
    print(f"{row['method']:<15} {row['accuracy']:>6.3f} {row['kappa']:>6.3f} "
          f"{row['f1_I']:>6.3f} {row['f1_P']:>6.3f} {row['f1_R']:>6.3f} {row['macro_f1']:>8.3f}")

print("\n" + "="*80)

# Best method
best = results_sorted.iloc[0]
print(f"\n✅ BEST METHOD: {best['method']}")
print(f"   Kappa: {best['kappa']:.3f} (vs baseline: {results[results['method']=='baseline']['kappa'].values[0]:.3f})")
print(f"   Accuracy: {best['accuracy']:.3f}")
print(f"   Weakest class F1: {min(best['f1_I'], best['f1_P'], best['f1_R']):.3f}")

# Recommendations
baseline_kappa = results[results['method']=='baseline']['kappa'].values[0]
if best['kappa'] > baseline_kappa + 0.05:
    print(f"\n✅ RECOMMEND: Use '{best['method']}' - significant improvement!")
elif best['kappa'] > baseline_kappa:
    print(f"\n⚠️  RECOMMEND: Use '{best['method']}' - marginal improvement")
else:
    print(f"\n❌ NO IMPROVEMENT: Stick with baseline")

print("\n" + "="*80)
