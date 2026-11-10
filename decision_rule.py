"""Shared action criterion: weighted majority, optional tie band, legacy caution."""
import math

def evaluate(probability, converged, rule='majority', alpha=.1, tie_margin=0.):
    if rule not in ('majority','cautious'):raise ValueError('unknown decision rule')
    if not 0<alpha<1 or not 0<=tie_margin<.5:raise ValueError('invalid decision settings')
    high=1-alpha if rule=='cautious' else .5+tie_margin
    low=high if rule=='cautious' else .5-tie_margin
    result=dict(decision_rule=rule,decision_push_threshold=high,decision_avoid_threshold=low,
                decision_tie_margin=tie_margin,decision_push=False,decision_label='unavailable',
                decision_reason='probability_unavailable')
    if probability is None:return result
    if not math.isfinite(probability) or not 0<=probability<=1:raise ValueError('invalid probability')
    if not converged:
        result.update(decision_label='uncertain',decision_reason='integration_not_converged');return result
    if rule=='cautious':
        push=probability>=high
        result.update(decision_push=push,decision_label='continue' if push else 'bypass',
                      decision_reason='cautious_threshold_met' if push else 'cautious_threshold_not_met')
    elif probability>high:
        result.update(decision_push=True,decision_label='continue',decision_reason='push_weight_majority')
    elif probability<low:
        result.update(decision_label='bypass',decision_reason='bypass_weight_majority')
    else:
        result.update(decision_label='uncertain',decision_reason='weight_tie_or_tie_band')
    return result

def from_snapshot(snap, alpha=.1):
    # Older recordings genuinely ran the 90% criterion: never relabel their route.
    return evaluate(snap.get('push_probability'),snap.get('integration_converged',False),
                    snap.get('decision_rule','cautious'),snap.get('alpha',alpha),
                    snap.get('decision_tie_margin',0.))

def description(snap, alpha=.1):
    r=from_snapshot(snap,alpha)
    if snap.get('push_limit') is None and snap.get('bounds') is not None:
        return '当前几何比较不可用','#bccbd4'
    if r['decision_label']=='unavailable':return '信息不足：不能判定续推','#bccbd4'
    if r['decision_reason']=='integration_not_converged':return '积分未收敛：未定并回退','#f4a261'
    if r['decision_label']=='uncertain':return '两侧权重未分胜负：未定并回退','#bccbd4'
    if r['decision_push']:return '多数权重：继续交互' if r['decision_rule']=='majority' else '保守门限满足：继续交互','#48d5aa'
    return '多数权重：退回绕行' if r['decision_rule']=='majority' else '保守门限未满足：绕行','#f4a261'
