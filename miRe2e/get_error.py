import sklearn.metrics
import numpy as np


def get_error(ref, out):
    auc = sklearn.metrics.roc_auc_score(ref, out)
    pre, rec, th = sklearn.metrics.precision_recall_curve(ref, out)
    f1max = 0
    premax = 0
    recmax = 0
    for p, r in zip(pre, rec):
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > f1max:
            f1max = f1
            premax = p
            recmax = r
    return auc, f1max, premax, recmax


def count_rep(res):
    aux = 0
    total_aux = 0
    for j in range(len(res) - 1):
        if res[j]:
            aux += 1
        else:
            aux = 0
        if aux >= 2 and not res[j + 1]: total_aux += (aux - 1)
    return total_aux


def recpre2(rlabs, clabs, repp):
    """

    Parameters
    ----------
    rlabs: real labels 0/1
    clabs: classifier labels 0/1
    repp: repeated positives

    Returns
    -------

    """

    tp = np.sum(rlabs & clabs)
    fp = np.sum(np.logical_not(rlabs) & clabs) - repp
    fn = np.sum(rlabs & np.logical_not(clabs))

    tpr = tp / float(tp + fn)
    pre = tp / float(tp + fp)

    return (tpr, pre)


def del_rep2d2(ref, res, scores):

    aux, id_aux = 0, []
    del_pos = []
    for j in range(len(res) - 1):

        if res[j]:
            aux += 1
            if not ref[j] == 1:
                id_aux.append(j)
        else:
            aux = 0
            id_aux = []
        if aux >= 2 and not res[j + 1]:
            if sum(ref[j - aux + 1:j + 1]):
                del_pos.extend(id_aux)
            else:
                del_pos.extend(id_aux[1:])

    if len(del_pos) > 0:
        y_ref = np.delete(ref, del_pos)
        y_pred = np.delete(res, del_pos)
        y_scores = np.delete(scores, del_pos)
    else:
        y_ref = ref
        y_pred = res
        y_scores = scores

    return y_ref, y_pred, y_scores


def get_error2d2(ref, out, th, testtime=False):
    pre = np.zeros_like(th)
    rec = np.zeros_like(th)

    for ind in range(len(th)):
        res = out > th[ind]
        fprep = count_rep(res)
        rec[ind], pre[ind] = recpre2(ref.astype(bool), res, fprep)

    f1 = np.nan_to_num(2 * pre * rec / (pre + rec))
    imax = np.argmax(f1)

    if testtime:
        auroc, aucpr = -1.0, -1.0
    else:
        res1 = out > th[imax]
        ref1, res1, out1 = del_rep2d2(ref.astype(bool), res1, out)

        auroc = sklearn.metrics.roc_auc_score(ref1, out1)
        pr, re, ths = sklearn.metrics.precision_recall_curve(ref1, out1)
        aucpr = sklearn.metrics.auc(re, pr)

    return auroc, aucpr, f1[imax], pre[imax], rec[imax], th[imax], pr, re
