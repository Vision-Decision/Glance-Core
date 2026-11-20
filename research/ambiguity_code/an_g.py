import json, sys, bisect
d = json.load(open(sys.argv[1])); R = d['rows']
def auroc(pos, neg):
    neg = sorted(neg); s = 0
    for p in pos:
        lo = bisect.bisect_left(neg, p); hi = bisect.bisect_right(neg, p); s += lo + 0.5 * (hi - lo)
    return round(s / (len(pos) * len(neg)), 4)
mean = lambda x: round(sum(x) / len(x), 4) if x else None
med = lambda x: round(sorted(x)[len(x) // 2], 4) if x else None
S = [r for r in R if r['nt'] == 1]; M = [r for r in R if r['nt'] >= 2]; Z = [r for r in R if r['nt'] == 0]
ps = sorted(r['p'] for r in S); tau = ps[int(0.01 * len(ps))]
o = dict(n_single=len(S), n_multi=len(M), n_absent=len(Z), tau_dial_1pct_refusal_refcoco_val=round(tau, 4),
         single_p_lt_0_5=mean([r['p'] < 0.5 for r in S]), multi_p_lt_0_5=mean([r['p'] < 0.5 for r in M]),
         multi_p_lt_tau=mean([r['p'] < tau for r in M]), N_acc_0_5=mean([r['p'] < 0.5 for r in Z]), N_acc_tau=mean([r['p'] < tau for r in Z]),
         single_acc50=mean([r['iou'][0] >= 0.5 for r in S]),
         multi_hit_any=mean([max(r['iou']) >= 0.5 for r in M]), multi_iou_union_ge_0_5=mean([r['iou_union'] >= 0.5 for r in M]),
         multi_hit_largest_given_hit=mean([max(range(len(r['iou'])), key=lambda k: r['iou'][k]) == max(range(len(r['area'])), key=lambda k: r['area'][k]) for r in M if max(r['iou']) >= 0.5]),
         kappa_mean_single=mean([r['k'] for r in S]), kappa_mean_multi=mean([r['k'] for r in M]),
         kappa_vs_hit_any_multi_gap=round(mean([r['k'] for r in M]) - mean([max(r['iou']) >= 0.5 for r in M]), 4),
         kappa_vs_hit_single_gap=round(mean([r['k'] for r in S]) - mean([r['iou'][0] >= 0.5 for r in S]), 4),
         mass_split_multi_frac=mean([sorted(r['mass'], reverse=True)[1] / max(1e-9, max(r['mass'])) >= 0.3 for r in M]),
         n_inst_mass_ge_0_1_multi=mean([sum(x >= 0.1 for x in r['mass']) for r in M]),
         H_med_single=med([r['H'] for r in S]), H_med_multi=med([r['H'] for r in M]), H_med_absent=med([r['H'] for r in Z]),
         mop15_med_single=med([r['mop15'] for r in S]), mop15_med_multi=med([r['mop15'] for r in M]))
P = lambda f: auroc([f(r) for r in M], [f(r) for r in S])
o['auroc_multi_vs_single'] = {'center_entropy_norm': P(lambda r: r['H']), 'center_perplexity': P(lambda r: r['ppl']), 'neg_top1': P(lambda r: -r['top1']),
    'mass_out_pred15': P(lambda r: r['mop15']), 'mass_out_pred': P(lambda r: r['mop']), 'n_peaks': P(lambda r: r['npk']), 'peak2_over_peak1': P(lambda r: r['pk21']),
    'one_minus_kappa': P(lambda r: 1 - r['k']), 'one_minus_p': P(lambda r: 1 - r['p'])}
# size-matched: entropy vs single with similar total GT area
import random
o['auroc_absent_vs_present_p'] = auroc([1 - r['p'] for r in Z], [1 - r['p'] for r in S + M])
o['auroc_absent_vs_present_entropy'] = auroc([r['H'] for r in Z], [r['H'] for r in S + M])
# multi-target expressions where the gate says "present" but box hits no instance
o['multi_present_box_miss'] = mean([r['p'] >= 0.5 and max(r['iou']) < 0.5 and r['iou_union'] < 0.5 for r in M])
print(json.dumps(o, indent=1)); json.dump(o, open(sys.argv[2], 'w'), indent=1)
