# -*- coding: utf-8 -*-
"""合成情境測試：不需要板子就能驗證狀態機。"""
import sys
from elevator_input import ElevatorInput, ElevatorConfig, B_CLASS

FPS = 4.0
DT = 1.0 / FPS


def run(script, cfg=None, verbose=False):
    """script: [(秒數, cls 或 None, conf), ...] 依序播放。"""
    sm = ElevatorInput(cfg or ElevatorConfig())
    t = 0.0
    out = []
    for dur, cls, conf in script:
        n = int(round(dur / DT))
        for _ in range(n):
            dets = [] if cls is None else [[0, 0, 50, 50, conf, cls]]
            for e in sm.update(t, dets):
                if not e['event'].endswith('progress'):
                    out.append(e)
                    if verbose:
                        print(f"   {e['t']:6.2f}  {e['event']:<18} "
                              f"{ {k: v for k, v in e.items() if k not in ('t','event','state')} }")
            t += DT
    return out


def dispatched(evs):
    return [e['floor'] for e in evs if e['event'] == 'floor_dispatched']


def kinds(evs):
    return [e['event'] for e in evs]


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case('比 1 再比 2 → 12F')
def _():
    evs = run([(3.5, 1, 80), (1.2, None, 0), (3.5, 2, 80),
               (1.2, None, 0), (3.5, None, 0)])
    assert dispatched(evs) == ['12F'], dispatched(evs)


@case('比 B 再比 1 → B1')
def _():
    evs = run([(3.5, B_CLASS, 75), (1.2, None, 0), (3.5, 1, 80),
               (1.2, None, 0), (3.5, None, 0)])
    assert dispatched(evs) == ['B1'], dispatched(evs)


@case('比 0 再比 5 → 5F')
def _():
    evs = run([(3.5, 0, 85), (1.2, None, 0), (3.5, 5, 70),
               (1.2, None, 0), (3.5, None, 0)])
    assert dispatched(evs) == ['5F'], dispatched(evs)


@case('00 不合法，不送出')
def _():
    evs = run([(3.5, 0, 85), (1.2, None, 0), (3.5, 0, 85),
               (1.2, None, 0), (3.5, None, 0)])
    assert dispatched(evs) == [], dispatched(evs)
    assert 'floor_invalid' in kinds(evs), kinds(evs)


@case('B0 不合法')
def _():
    evs = run([(3.5, B_CLASS, 80), (1.2, None, 0), (3.5, 0, 80),
               (1.2, None, 0), (3.5, None, 0)])
    assert dispatched(evs) == []
    assert 'floor_invalid' in kinds(evs)


@case('第二位比 B → 不合法')
def _():
    evs = run([(3.5, 1, 80), (1.2, None, 0), (3.5, B_CLASS, 80),
               (1.2, None, 0), (3.5, None, 0)])
    assert dispatched(evs) == []
    assert 'floor_invalid' in kinds(evs)


@case('類別一直跳 → 被拒絕，不會亂鎖')
def _():
    # 1 和 7 交替，誰都拿不到 70%
    script = []
    for i in range(32):
        script.append((DT, 1 if i % 2 == 0 else 7, 60))
    evs = run(script)
    assert 'digit_locked' not in kinds(evs), kinds(evs)
    assert 'digit_rejected' in kinds(evs), kinds(evs)


@case('偶爾閃一幀雜訊 → 仍然鎖得住')
def _():
    script = []
    for i in range(16):
        script.append((DT, 7 if i == 5 else 1, 70))
    script += [(1.2, None, 0), (3.5, 2, 80), (1.2, None, 0), (3.5, None, 0)]
    evs = run(script)
    assert dispatched(evs) == ['12F'], dispatched(evs)


@case('信心度太低 → 拒絕（模擬目前的 cls9）')
def _():
    evs = run([(4.0, 9, 25)])
    assert 'digit_locked' not in kinds(evs)
    r = [e for e in evs if e['event'] == 'digit_rejected']
    assert r, kinds(evs)
    # conf_floor=20 擋不住，應該是 confidence_low 或 frames_too_few
    assert r[0]['reason'] in ('confidence_low', 'frames_too_few'), r[0]


@case('確認期間舉手 → 取消，不送出')
def _():
    evs = run([(3.5, 1, 80), (1.2, None, 0), (3.5, 2, 80),
               (1.2, None, 0), (1.0, None, 0), (0.75, 5, 80), (2.0, None, 0)])
    assert dispatched(evs) == [], dispatched(evs)
    assert 'cancelled' in kinds(evs), kinds(evs)


@case('沒有分隔就換手勢 → 不會被當成兩位')
def _():
    # 連續 7 秒不放手，中間從 1 變成 2，中途沒有 gap
    evs = run([(3.5, 1, 80), (3.5, 2, 80), (1.2, None, 0), (3.5, None, 0)])
    # 第一位鎖定後進入 gapwait，手還在 → 不會進到第二位
    assert dispatched(evs) == [], dispatched(evs)


@case('兩位都是同一個數字 11F')
def _():
    evs = run([(3.5, 1, 80), (1.2, None, 0), (3.5, 1, 80),
               (1.2, None, 0), (3.5, None, 0)])
    assert dispatched(evs) == ['11F'], dispatched(evs)


@case('中途手離開太久 → 整個重來')
def _():
    evs = run([(2.0, 1, 80), (2.0, None, 0), (3.5, 2, 80),
               (1.2, None, 0), (3.5, None, 0)])
    # 第一次停留沒滿 3 秒就斷了，2 變成第一位 → 只有一位，不會送出
    assert dispatched(evs) == [], dispatched(evs)


@case('99F 上限內、100F 不可能（只有兩位）')
def _():
    evs = run([(3.5, 9, 80), (1.2, None, 0), (3.5, 9, 80),
               (1.2, None, 0), (3.5, None, 0)])
    assert dispatched(evs) == ['99F'], dispatched(evs)


@case('2 fps 剛好及格（中間視窗 5 幀 = 下限）')
def _():
    global DT
    old = DT
    DT = 0.5
    try:
        evs = run([(4.0, 1, 80)])
        assert 'digit_locked' in kinds(evs), kinds(evs)
    finally:
        DT = old


@case('掉到 1.7 fps → 幀數不足，拒絕而不是亂猜')
def _():
    global DT
    old = DT
    DT = 0.6          # 中間 2 秒只有 4 幀 < min_frames 5
    try:
        evs = run([(4.0, 1, 80)])
        assert 'digit_locked' not in kinds(evs), kinds(evs)
        r = [e for e in evs if e['event'] == 'digit_rejected']
        assert r and r[0]['reason'] == 'frames_too_few', r
    finally:
        DT = old


if __name__ == '__main__':
    ok = fail = 0
    for name, fn in CASES:
        try:
            fn()
            print(f'  [通過] {name}')
            ok += 1
        except AssertionError as e:
            print(f'  [失敗] {name}\n         {e}')
            fail += 1
    print(f'\n{ok} 通過 / {fail} 失敗')
    sys.exit(1 if fail else 0)
