"""Metrics, post-hoc calibrators, prior correction, official GREC scoring, risk control and bootstrap for the
absence/confidence sprint. numpy + scipy only (CPU)."""
import gzip, json, math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata, beta as beta_dist, binom


# ----------------------------------------------------------------------------------------------- io / basics
def load_rows(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


# ----------------------------------------------------------------------------------------------- metrics
def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(p, y, n_bins=15, scheme="width"):
    """width: equal-width bins (eval.py's ECE, 15 bins); mass: equal-mass (adaptive) bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return float("nan")
    if scheme == "width":
        b = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    else:
        order = np.argsort(p, kind="stable")
        b = np.empty(len(p), int)
        b[order] = np.minimum((np.arange(len(p)) * n_bins) // len(p), n_bins - 1)
    tot = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def nll(p, y, eps=1e-6):
    p, y = np.clip(np.asarray(p, float), eps, 1 - eps), np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calib_metrics(p, y):
    return dict(ece15=ece(p, y), ece15_mass=ece(p, y, 15, "mass"), ece10=ece(p, y, 10), brier=brier(p, y), nll=nll(p, y),
                mean_p=float(np.mean(p)), rate=float(np.mean(y)), n=int(len(p)))


def aurc(conf, err):
    """area under the risk-coverage curve (mean selective risk over all coverages), ties broken pessimistically."""
    conf, err = np.asarray(conf, float), np.asarray(err, float)
    o = np.lexsort((-err, -conf))
    r = np.cumsum(err[o]) / np.arange(1, len(err) + 1)
    return float(r.mean())


# ----------------------------------------------------------------------------------------------- calibrators
class Logistic:
    """L2-light logistic regression on a feature matrix (intercept added unless fit_intercept=False)."""

    def __init__(self, fit_intercept=True, l2=1e-4):
        self.fi, self.l2 = fit_intercept, l2

    def _X(self, X):
        X = np.asarray(X, float)
        X = X[:, None] if X.ndim == 1 else X
        return np.hstack([X, np.ones((len(X), 1))]) if self.fi else X

    def fit(self, X, y, w=None):
        X, y = self._X(X), np.asarray(y, float)
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        w = w / w.mean()

        def f(b):
            z = X @ b
            l = np.logaddexp(0, z) - y * z
            g = X.T @ (w * (sig(z) - y)) / len(y) + self.l2 * b
            return float(np.mean(w * l) + 0.5 * self.l2 * b @ b), g
        b0 = np.zeros(X.shape[1])
        if not self.fi:
            b0[0] = 1.0
        self.b = minimize(f, b0, jac=True, method="L-BFGS-B").x
        return self

    def logit(self, X):
        return self._X(X) @ self.b

    def __call__(self, X):
        return sig(self.logit(X))


class Temperature:
    def fit(self, z, y):
        self.m = Logistic(fit_intercept=False, l2=0).fit(np.asarray(z)[:, None], y)
        self.T = 1.0 / self.m.b[0]
        return self

    def __call__(self, z):
        return sig(np.asarray(z) / self.T)


class Platt:
    def fit(self, z, y, w=None):
        self.m = Logistic(l2=0).fit(np.asarray(z)[:, None], y, w)
        return self

    def __call__(self, z):
        return self.m(np.asarray(z)[:, None])


class BetaCal:
    """Kull et al. 2017: logistic on [ln p, -ln(1-p)]."""
    def feats(self, z):
        p = np.clip(sig(z), 1e-7, 1 - 1e-7)
        return np.stack([np.log(p), -np.log(1 - p)], 1)

    def fit(self, z, y, w=None):
        self.m = Logistic(l2=0).fit(self.feats(z), y, w)
        return self

    def __call__(self, z):
        return self.m(self.feats(z))


class Isotonic:
    """PAV on the raw score, linear interpolation between block centers, clipped to [eps, 1-eps]."""
    def fit(self, z, y, w=None):
        z, y = np.asarray(z, float), np.asarray(y, float)
        o = np.argsort(z, kind="stable"); z, y = z[o], y[o]
        w = np.ones(len(y)) if w is None else np.asarray(w, float)[o]
        vals, wts, xs = [], [], []
        for zi, yi, wi in zip(z, y, w):
            vals.append(yi); wts.append(wi); xs.append([zi, zi])
            while len(vals) > 1 and vals[-2] >= vals[-1]:
                v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
                ww = wts[-2] + wts[-1]; x = [xs[-2][0], xs[-1][1]]
                vals[-2:] = [v]; wts[-2:] = [ww]; xs[-2:] = [x]
        self.x = np.array([(a + b) / 2 for a, b in xs]); self.v = np.array(vals)
        return self

    def __call__(self, z):
        return np.clip(np.interp(np.asarray(z, float), self.x, self.v), 1e-4, 1 - 1e-4)


class MultiPlatt:
    """logistic on several observable features (column order fixed by the caller)."""
    def fit(self, X, y, w=None):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.m = Logistic(l2=1e-3).fit((X - self.mu) / self.sd, y, w)
        return self

    def __call__(self, X):
        return self.m((X - self.mu) / self.sd)


def make_cal(name):
    return dict(temp=Temperature, platt=Platt, beta=BetaCal, iso=Isotonic)[name]()


# ----------------------------------------------------------------------------------------------- prior shift
def prior_shift(p, pi_src, pi_tgt):
    """Bayes label-shift correction of calibrated posteriors from source prior pi_src to target prior pi_tgt."""
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    a = pi_tgt / pi_src * p
    b = (1 - pi_tgt) / (1 - pi_src) * (1 - p)
    return a / (a + b)


def em_prior(p, pi_src, iters=200, tol=1e-7, clip=(1e-3, 1 - 1e-3)):
    """Saerens et al. 2002: unsupervised estimate of the target prior from unlabeled calibrated posteriors."""
    pi = pi_src
    for _ in range(iters):
        q = prior_shift(p, pi_src, pi)
        new = float(np.clip(q.mean(), *clip))
        if abs(new - pi) < tol:
            pi = new; break
        pi = new
    return pi, prior_shift(p, pi_src, pi)


# ----------------------------------------------------------------------------------------------- GREC (official)
def giou_mat(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda t: np.clip(t[:, 2] - t[:, 0], 0, None) * np.clip(t[:, 3] - t[:, 1], 0, None)
    union = ar(a)[:, None] + ar(b)[None, :] - inter
    iou = inter / np.maximum(union, 1e-9)
    cx1 = np.minimum(a[:, None, 0], b[None, :, 0]); cy1 = np.minimum(a[:, None, 1], b[None, :, 1])
    cx2 = np.maximum(a[:, None, 2], b[None, :, 2]); cy2 = np.maximum(a[:, None, 3], b[None, :, 3])
    c = np.maximum((cx2 - cx1) * (cy2 - cy1), 1e-9)
    return iou, iou - (c - union) / c


def grec_sample(pred_boxes, pred_scores, gt_boxes, thresh_score=0.7, thresh_iou=0.5, use_giou=True):
    """Port of henghuiding/gRefCOCO mdetr/datasets/refexp.py RefExpEvaluator.summarize for one sample.
    Returns (F1==1, no_target, predicted_nonempty)."""
    keep = [b for b, s in sorted(zip(pred_boxes, pred_scores), key=lambda t: -t[1]) if s >= thresh_score]
    no_t = len(gt_boxes) == 0
    if no_t:
        return (len(keep) == 0), True, len(keep) > 0
    TP = 0
    if keep:
        iou, giou = giou_mat(keep, gt_boxes)
        m = (giou if use_giou else iou).copy()
        for _ in range(min(len(keep), len(gt_boxes))):
            k = int(np.argmax(m)); v = m.flat[k]
            if v < thresh_iou:
                break
            i, j = divmod(k, len(gt_boxes))
            TP += 1; m[i, :] = 0.0; m[:, j] = 0.0
    FP, FN = len(keep) - TP, len(gt_boxes) - TP
    f1 = 2 * TP / (2 * TP + FP + FN)
    return f1 >= 1.0, False, len(keep) > 0


def grec_metrics(samples, **kw):
    """samples: list of (pred_boxes, pred_scores, gt_boxes). Official Pr@(F1=1, IoU>=0.5), N-acc, T-acc (+ per-kind)."""
    res = [grec_sample(*s, **kw) for s in samples]
    ok = np.array([r[0] for r in res]); nt = np.array([r[1] for r in res]); ne = np.array([r[2] for r in res])
    ng = np.array([len(s[2]) for s in samples])
    return dict(pr_f1=float(ok.mean()), N_acc=float((~ne[nt]).mean()) if nt.any() else float("nan"),
                T_acc=float(ne[~nt].mean()), pr_single=float(ok[ng == 1].mean()) if (ng == 1).any() else float("nan"),
                pr_multi=float(ok[ng > 1].mean()) if (ng > 1).any() else float("nan"),
                n=len(samples), n_no_target=int(nt.sum()), n_single=int((ng == 1).sum()), n_multi=int((ng > 1).sum()))


# ----------------------------------------------------------------------------------------------- risk control
def cp_upper(k, n, delta):
    """Clopper-Pearson one-sided (1-delta) upper bound for a binomial proportion."""
    if n == 0:
        return 1.0
    return 1.0 if k >= n else float(beta_dist.ppf(1 - delta, k + 1, n - k))


def binom_pval(k, n, alpha):
    """p-value of H0: risk > alpha given k losses in n (exact binomial, one-sided)."""
    if n == 0:
        return 1.0
    return float(binom.cdf(k, n, alpha))


def rcps_threshold(score, loss, alpha, delta=0.1, mode="joint", grid=None):
    """Pick the most permissive threshold t (accept iff score >= t) whose risk is certified <= alpha w.p. >= 1-delta.
    mode="joint":     risk = P(accept and loss)            (monotone in t; UCB = Clopper-Pearson; RCPS)
    mode="selective": risk = P(loss | accept)              (Learn-then-Test, fixed-sequence binomial tests from strict to lax)
    Candidate thresholds = a FIXED grid (data-independent, as LTT requires) for scores in [0,1]:
    1 - logspace(-4, 0, 400), or `grid` (e.g. quantiles of the score on an independent training-distribution split, so
    the sequence starts at a point with enough accepted items), tested strict -> lax; stop at the first non-certified
    one. Returns t (inf = abstain)."""
    s, l = np.asarray(score, float), np.asarray(loss, float)
    n = len(s)
    best = np.inf
    grid = (1 - np.logspace(-4, 0, 400)) if grid is None else np.sort(np.asarray(grid, float))[::-1]
    for t in grid:
        acc = s >= t
        k, m = int(l[acc].sum()), int(acc.sum())
        if mode == "joint":
            ok = cp_upper(k, n, delta) <= alpha
        else:
            if m == 0:
                continue  # nothing accepted yet: no hypothesis tested at this point
            ok = binom_pval(k, m, alpha) <= delta
        if not ok:
            break
        best = t
    return best


def sgr_threshold(score, loss, alpha, delta=0.1):
    """Selection with Guaranteed Risk (Geifman & El-Yaniv, NeurIPS 2017): binary search over the number of accepted
    calibration items with a Clopper-Pearson bound at delta / ceil(log2 m); guarantees P(selective risk > alpha) <= delta.
    Accept iff score >= returned threshold (inf = abstain on everything)."""
    s, l = np.asarray(score, float), np.asarray(loss, float)
    o = np.argsort(-s, kind="stable"); s, l = s[o], l[o]
    m = len(s)
    k = max(1, int(math.ceil(math.log2(m))))
    cum = np.cumsum(l)
    lo, hi, best = 0, m, None
    for _ in range(k):
        j = int(math.ceil((lo + hi) / 2))
        if j <= 0:
            break
        b = cp_upper(int(cum[j - 1]), j, delta / k)
        if b <= alpha:
            best = j; lo = j
        else:
            hi = j
        if hi - lo <= 1:
            break
    if best is None:
        return np.inf
    return s[best - 1]


def conformal_absent_threshold(score_absent, alpha):
    """Split-conformal: accept iff score > q gives P(accept | absent) <= alpha (marginal over calibration draws)."""
    s = np.sort(np.asarray(score_absent, float))
    n = len(s)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else s[k - 1]


# ----------------------------------------------------------------------------------------------- bootstrap
def cluster_boot(groups, fn, B=1000, seed=0, paired_fns=None):
    """Cluster (image-level) bootstrap. groups: array of group ids per row; fn(idx)->float.
    Returns point estimate, 2.5/97.5 percentiles."""
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    uniq, inv = np.unique(g, return_inverse=True)
    members = [[] for _ in range(len(uniq))]
    for i, k in enumerate(inv):
        members[k].append(i)
    members = [np.array(m) for m in members]
    full = np.arange(len(g))
    est = fn(full)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        vals.append(fn(idx))
    vals = np.array(vals, float)
    return dict(est=float(est), lo=float(np.nanpercentile(vals, 2.5)), hi=float(np.nanpercentile(vals, 97.5)),
                p_le0=float(np.mean(vals <= 0)), p_ge0=float(np.mean(vals >= 0)))
