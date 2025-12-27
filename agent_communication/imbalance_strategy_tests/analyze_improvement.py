#!/usr/bin/env python3
"""
Analyze percentage improvement from baseline for each strategy.
"""

# Results from V4 test
results = [
    {"strategy": "1. Baseline", "f1_macro": 0.3198, "f1_weighted": 0.4890, "min_f1": 0.0247, "kappa": 0.0461, "balanced_acc": 0.3533},
    {"strategy": "2. RandomOverSampler", "f1_macro": 0.3905, "f1_weighted": 0.4666, "min_f1": 0.2212, "kappa": 0.0894, "balanced_acc": 0.4073},
    {"strategy": "3. RandomUnderSampler", "f1_macro": 0.3720, "f1_weighted": 0.4328, "min_f1": 0.2333, "kappa": 0.0845, "balanced_acc": 0.4086},
    {"strategy": "4. SMOTE", "f1_macro": 0.4057, "f1_weighted": 0.4704, "min_f1": 0.2353, "kappa": 0.0954, "balanced_acc": 0.4311},
    {"strategy": "5. ADASYN", "f1_macro": 0.3877, "f1_weighted": 0.4561, "min_f1": 0.2570, "kappa": 0.0822, "balanced_acc": 0.4195},
    {"strategy": "6. Borderline-SMOTE", "f1_macro": 0.4139, "f1_weighted": 0.4811, "min_f1": 0.2859, "kappa": 0.1241, "balanced_acc": 0.4409},
    {"strategy": "7. SMOTE+Tomek", "f1_macro": 0.4334, "f1_weighted": 0.5063, "min_f1": 0.2771, "kappa": 0.1623, "balanced_acc": 0.4657},
    {"strategy": "8. SMOTE+ENN", "f1_macro": 0.3370, "f1_weighted": 0.3135, "min_f1": 0.2093, "kappa": 0.1301, "balanced_acc": 0.4863},
    {"strategy": "9. Alpha (balanced)", "f1_macro": 0.3920, "f1_weighted": 0.4554, "min_f1": 0.2634, "kappa": 0.0888, "balanced_acc": 0.4270},
    {"strategy": "10. Alpha (sqrt)", "f1_macro": 0.4055, "f1_weighted": 0.5324, "min_f1": 0.1660, "kappa": 0.1317, "balanced_acc": 0.4101},
    {"strategy": "11. Focal Loss", "f1_macro": 0.3614, "f1_weighted": 0.4233, "min_f1": 0.2127, "kappa": 0.0622, "balanced_acc": 0.3986},
    {"strategy": "12. Oversample+Alpha", "f1_macro": 0.4156, "f1_weighted": 0.4585, "min_f1": 0.2741, "kappa": 0.1750, "balanced_acc": 0.4810},
    {"strategy": "13. SMOTE+Alpha", "f1_macro": 0.3584, "f1_weighted": 0.3551, "min_f1": 0.2415, "kappa": 0.1288, "balanced_acc": 0.4810},
    {"strategy": "14. Oversample+Focal", "f1_macro": 0.3902, "f1_weighted": 0.4148, "min_f1": 0.2588, "kappa": 0.1592, "balanced_acc": 0.4859},
    {"strategy": "15. SMOTE+Focal", "f1_macro": 0.3861, "f1_weighted": 0.3863, "min_f1": 0.2799, "kappa": 0.1615, "balanced_acc": 0.5222},
    {"strategy": "16. Oversample+Alpha(sqrt)", "f1_macro": 0.4044, "f1_weighted": 0.4482, "min_f1": 0.2763, "kappa": 0.1547, "balanced_acc": 0.4719},
    {"strategy": "17. SMOTE+Alpha(sqrt)", "f1_macro": 0.4048, "f1_weighted": 0.4493, "min_f1": 0.3022, "kappa": 0.1240, "balanced_acc": 0.4623},
    {"strategy": "18. SMOTE+Tomek+Alpha", "f1_macro": 0.4191, "f1_weighted": 0.4459, "min_f1": 0.2816, "kappa": 0.1811, "balanced_acc": 0.5057},
]

baseline = results[0]

print("="*100)
print("PERCENTAGE IMPROVEMENT FROM BASELINE")
print("="*100)
print(f"\nBaseline values:")
print(f"  F1 Macro: {baseline['f1_macro']:.4f}")
print(f"  F1 Weighted: {baseline['f1_weighted']:.4f}")
print(f"  Min F1: {baseline['min_f1']:.4f}")
print(f"  Kappa: {baseline['kappa']:.4f}")
print(f"  Balanced Acc: {baseline['balanced_acc']:.4f}")

# Calculate improvements
improvements = []
for r in results[1:]:  # Skip baseline
    imp = {
        'strategy': r['strategy'],
        'f1_macro_pct': ((r['f1_macro'] - baseline['f1_macro']) / baseline['f1_macro']) * 100,
        'f1_weighted_pct': ((r['f1_weighted'] - baseline['f1_weighted']) / baseline['f1_weighted']) * 100,
        'min_f1_pct': ((r['min_f1'] - baseline['min_f1']) / baseline['min_f1']) * 100,
        'kappa_pct': ((r['kappa'] - baseline['kappa']) / baseline['kappa']) * 100,
        'balanced_acc_pct': ((r['balanced_acc'] - baseline['balanced_acc']) / baseline['balanced_acc']) * 100,
    }
    # Average improvement across all metrics
    imp['avg_improvement'] = (imp['f1_macro_pct'] + imp['f1_weighted_pct'] + imp['min_f1_pct'] +
                              imp['kappa_pct'] + imp['balanced_acc_pct']) / 5
    improvements.append(imp)

# Sort by average improvement
improvements.sort(key=lambda x: x['avg_improvement'], reverse=True)

print("\n" + "="*100)
print("IMPROVEMENTS BY STRATEGY (sorted by average improvement)")
print("="*100)
print(f"\n{'Strategy':<25} {'F1 Macro':<12} {'F1 Wght':<12} {'Min F1':<12} {'Kappa':<12} {'Bal Acc':<12} {'AVG':<10}")
print("-"*100)

for imp in improvements:
    print(f"{imp['strategy']:<25} {imp['f1_macro_pct']:>+8.1f}%   {imp['f1_weighted_pct']:>+8.1f}%   "
          f"{imp['min_f1_pct']:>+8.1f}%   {imp['kappa_pct']:>+8.1f}%   {imp['balanced_acc_pct']:>+8.1f}%   "
          f"{imp['avg_improvement']:>+7.1f}%")

print("\n" + "="*100)
print("TOP 3 STRATEGIES BY AVERAGE IMPROVEMENT")
print("="*100)

for rank, imp in enumerate(improvements[:3], 1):
    print(f"\n🏆 #{rank}: {imp['strategy']}")
    print(f"   Average improvement: {imp['avg_improvement']:+.1f}%")
    print(f"   ├─ F1 Macro:      {imp['f1_macro_pct']:+.1f}%")
    print(f"   ├─ F1 Weighted:   {imp['f1_weighted_pct']:+.1f}%")
    print(f"   ├─ Min F1:        {imp['min_f1_pct']:+.1f}%")
    print(f"   ├─ Kappa:         {imp['kappa_pct']:+.1f}%")
    print(f"   └─ Balanced Acc:  {imp['balanced_acc_pct']:+.1f}%")

# Also show best by each metric
print("\n" + "="*100)
print("BEST IMPROVEMENT BY EACH METRIC")
print("="*100)

metrics = [
    ('F1 Macro', 'f1_macro_pct'),
    ('F1 Weighted', 'f1_weighted_pct'),
    ('Min F1 (minority class)', 'min_f1_pct'),
    ('Cohen\'s Kappa', 'kappa_pct'),
    ('Balanced Accuracy', 'balanced_acc_pct'),
]

for metric_name, metric_key in metrics:
    best = max(improvements, key=lambda x: x[metric_key])
    print(f"\n{metric_name}:")
    print(f"  Best: {best['strategy']} ({best[metric_key]:+.1f}%)")

# Summary recommendation
print("\n" + "="*100)
print("SUMMARY")
print("="*100)
print(f"""
Based on percentage improvement from baseline:

TOP 3 OVERALL:
1. {improvements[0]['strategy']} - {improvements[0]['avg_improvement']:+.1f}% avg improvement
2. {improvements[1]['strategy']} - {improvements[1]['avg_improvement']:+.1f}% avg improvement
3. {improvements[2]['strategy']} - {improvements[2]['avg_improvement']:+.1f}% avg improvement

KEY INSIGHT:
- Min F1 shows the LARGEST improvements (>1000%) because baseline nearly ignores R class
- Kappa improvements are also substantial (>200%) showing better agreement
- F1 Weighted may decrease for some strategies because they sacrifice majority class
  performance to improve minority class - this is often a good tradeoff!
""")
