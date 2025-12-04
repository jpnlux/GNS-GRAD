import numpy as np
import random
import math

# =============================================================
# ★ パラメータ設定
# =============================================================

N_GS = 10                    # 地上局数
d_min = 2000e3               # GS間最小距離 2000 km
d_max = 3000e3               # GS間最大距離 3000 km

Z_GS = 0.0                   # 地上局高さ

# 衛星
y_orbit_A = +80e3            # 軌道Aのy位置 +80 km
y_orbit_B = -100e3           # 軌道Bのy位置 -100 km（非対称 → HO分散）
z_sat = 1000e3               # 衛星高度 1000 km
dx_sat = 5e5                 # 衛星間隔 500 km

# 可視最低仰角
elev_min = 30.0              # 30度

# 衛星速度（両方 x軸正方向に流す）
v_LEO = 7600.0               # 7.6 km/s
vx_A = v_LEO
vx_B = v_LEO * 0.985         # Bを少し遅くFしてHOをより分散

# 乱数シード（再現性が欲しければ固定）
random.seed(0)
np.random.seed(0)


# =============================================================
# ★ 地上局の生成
# =============================================================
def generate_ground_stations(N, d_min, d_max):
    xs = [0.0]
    for i in range(1, N):
        dx = random.uniform(d_min, d_max)
        xs.append(xs[-1] + dx)

    # y方向を ±3 km だけランダムに揺らす
    gs_list = [(x, random.uniform(-3000, 3000), Z_GS) for x in xs]
    return gs_list


# =============================================================
# ★ 衛星の等間隔配置
# =============================================================
def generate_satellites_by_spacing(x_min, x_max, dx, y, z):
    span = x_max - x_min
    N = int(span // dx) + 1
    xs = [x_min + i * dx for i in range(N)]
    sats = [(float(x), float(y), float(z)) for x in xs]
    return sats, N


# =============================================================
# ★ 仰角計算
# =============================================================
def elevation(gs, sat):
    dx = sat[0] - gs[0]
    dy = sat[1] - gs[1]
    dz = sat[2] - gs[2]
    horizontal = math.sqrt(dx*dx + dy*dy)
    return math.degrees(math.atan2(dz, horizontal))


def visible(gs, sat, elev_min):
    return elevation(gs, sat) >= elev_min


# =============================================================
# ★ 衛星の位置（時刻 t）
# =============================================================
def satellite_pos(sat0, vx, t):
    return (sat0[0] + vx * t, sat0[1], sat0[2])


# =============================================================
# ★ 可視範囲から抜ける時刻の推定
# =============================================================
def time_to_leave(gs, sat0, vx, elev_min, t_now):
    """
    t_now 以降で、仰角が elev_min を下回る最初の時刻を探す。
    粗探索 + しきい値チェックで求める簡易版。
    """
    def diff(t):
        sat = satellite_pos(sat0, vx, t)
        return elevation(gs, sat) - elev_min

    # そもそも現在時刻で可視でなければ離脱なし
    if diff(t_now) < 0:
        return None

    dt = 0.2  # ★ 粗探索ステップ（小さめにして見逃し防止）
    t = t_now
    prev = diff(t)

    # 最大 200000 ステップ → 4万秒（十分長い）
    for _ in range(200000):
        t += dt
        now = diff(t)
        if prev >= 0 and now < 0:
            # ここで「visible→not visible」に切り替わった
            return t
        prev = now

    return None  # 探索範囲内で可視のまま


# =============================================================
# ★ 初期接続：可視衛星の中からランダムに1つ（重複禁止）
# =============================================================
def assign_initial_links(gs_list, sats_init, vx, elev_min, t0):
    M = len(gs_list)
    used = set()
    assignment = {}

    for gi, gs in enumerate(gs_list):
        # 可視衛星リスト
        visible_list = []
        for idx, sat0 in enumerate(sats_init):
            sat = satellite_pos(sat0, vx, t0)
            if visible(gs, sat, elev_min):
                visible_list.append(idx)

        if not visible_list:
            assignment[gi] = None
            continue

        # シャッフルしてランダム順に
        random.shuffle(visible_list)

        # 未使用衛星を優先して割り当てる
        chosen = None
        for sat_idx in visible_list:
            if sat_idx not in used:
                chosen = sat_idx
                used.add(sat_idx)
                break

        # fallback（全て使用済みの場合）
        if chosen is None:
            chosen = visible_list[0]
            used.add(chosen)

        assignment[gi] = chosen

    return assignment


# =============================================================
# ★ HO 先の衛星選択：可視な衛星の中で x が最小のもの
# =============================================================
def find_next_sat(gs, sats_init, vx, t_event, elev_min):
    """
    HO発生時刻 t_event において，
      ・可視範囲にある衛星
      ・その中で x 座標が最小の衛星
    を選び，その衛星 index を返す。
    該当衛星がなければ None を返す。
    """
    best_si = None
    best_x = float('inf')

    for si, sat0 in enumerate(sats_init):
        sat = satellite_pos(sat0, vx, t_event)
        if visible(gs, sat, elev_min):
            if sat[0] < best_x:
                best_x = sat[0]
                best_si = si

    return best_si


# =============================================================
# ★ HO スナップショット作成
# =============================================================
def make_snapshot(t, topology, sats_A_init, sats_B_init, vx_A, vx_B, hoinfo):
    # 全衛星の x 座標
    sat_pos_A = {i: satellite_pos(sats_A_init[i], vx_A, t)[0]
                 for i in range(len(sats_A_init))}
    sat_pos_B = {i: satellite_pos(sats_B_init[i], vx_B, t)[0]
                 for i in range(len(sats_B_init))}

    # 地上局ごとの接続衛星の x 座標も記録
    connections_with_x = {}
    for gs, con in topology.items():
        idxA = con["A"]
        idxB = con["B"]
        connections_with_x[gs] = {
            "A": {"idx": idxA, "x": sat_pos_A[idxA]},
            "B": {"idx": idxB, "x": sat_pos_B[idxB]}
        }

    # HO前後の衛星位置
    if hoinfo is not None:
        old_idx = hoinfo["old_sat"]
        new_idx = hoinfo["new_sat"]

        if hoinfo["orbit"] == "A":
            old_x = sat_pos_A[old_idx]
            new_x = sat_pos_A[new_idx]
        else:
            old_x = sat_pos_B[old_idx]
            new_x = sat_pos_B[new_idx]

        ho_positions = {
            "old_idx": old_idx, "old_x": old_x,
            "new_idx": new_idx, "new_x": new_x
        }
    else:
        ho_positions = None

    return {
        "time": t,
        "connections": topology.copy(),
        "connections_x": connections_with_x,
        "sat_positions_x": {"A": sat_pos_A, "B": sat_pos_B},
        "ho_event": hoinfo,
        "ho_positions_x": ho_positions
    }


# =============================================================
# ★ 公開関数：HOシミュレーション本体
# =============================================================
def run_ho_simulation():
    """
    HOシミュレーションを実行し，
      snapshots: HOイベントごとのスナップショット（リスト）
      gs_list  : 地上局位置のリスト
    を返す。
    """
    # 1. 地上局
    gs_list = generate_ground_stations(N_GS, d_min, d_max)

    # 2. 衛星配置範囲
    x_min = gs_list[0][0] - 15e8
    x_max = gs_list[-1][0] + 15e8

    # 軌道A
    sats_A_init, NA = generate_satellites_by_spacing(
        x_min, x_max, dx_sat, y_orbit_A, z_sat
    )

    # 軌道B（位相ずらし）
    sats_B_init, NB = generate_satellites_by_spacing(
        x_min + dx_sat/2.0, x_max + dx_sat/2.0, dx_sat, y_orbit_B, z_sat
    )

    # 3. 初期接続
    initial_A = assign_initial_links(gs_list, sats_A_init, vx_A, elev_min, t0=0.0)
    initial_B = assign_initial_links(gs_list, sats_B_init, vx_B, elev_min, t0=0.0)

    topology = {gi: {"A": initial_A[gi], "B": initial_B[gi]} for gi in range(N_GS)}

    # HO回数
    ho_count_A = [0]*N_GS
    ho_count_B = [0]*N_GS

    snapshots = []
    t_now = 0.0

    # =========================================================
    # ★ イベントループ（HO発生を順に処理）
    # =========================================================
    while True:
        # 全GSで A/B が2回ずつ HO 完了 → 終了
        if all(ho_count_A[i] >= 6 and ho_count_B[i] >= 6 for i in range(N_GS)):
            break

        events = []

        for gi, gs in enumerate(gs_list):
            # A軌道のHO候補
            if ho_count_A[gi] < 6:
                idxA = topology[gi]["A"]
                if idxA is not None:
                    tA = time_to_leave(gs, sats_A_init[idxA], vx_A, elev_min, t_now)
                    if tA is not None:
                        events.append((tA, gi, "A"))

            # B軌道のHO候補
            if ho_count_B[gi] < 6:
                idxB = topology[gi]["B"]
                if idxB is not None:
                    tB = time_to_leave(gs, sats_B_init[idxB], vx_B, elev_min, t_now)
                    if tB is not None:
                        events.append((tB, gi, "B"))

        if len(events) == 0:
            # これ以上離脱イベントなし
            break

        # 最も早いイベントのみ処理
        events.sort()
        t_event, gi_event, orbit_event = events[0]

        old_sat = topology[gi_event][orbit_event]
        gs_ev = gs_list[gi_event]

        # 新しい衛星を割当（可視な衛星のうち x が最小）
        if orbit_event == "A":
            new_sat = find_next_sat(gs_ev, sats_A_init, vx_A, t_event, elev_min)
            topology[gi_event]["A"] = new_sat
            ho_count_A[gi_event] += 1
        else:
            new_sat = find_next_sat(gs_ev, sats_B_init, vx_B, t_event, elev_min)
            topology[gi_event]["B"] = new_sat
            ho_count_B[gi_event] += 1

        # スナップショット作成
        hoinfo = {
            "gs": gi_event,
            "orbit": orbit_event,
            "old_sat": old_sat,
            "new_sat": new_sat
        }
        snapshots.append(
            make_snapshot(t_event, topology, sats_A_init, sats_B_init, vx_A, vx_B, hoinfo)
        )

        t_now = t_event

    return snapshots, gs_list


# =============================================================
# ★ スクリプトとして実行されたときのテスト出力
# =============================================================
if __name__ == "__main__":
    snaps, gs_list = run_ho_simulation()
    print("全地上局で A/B の HO を2回ずつ完了したか？")
    print("地上局数:", len(gs_list))
    print("取得スナップショット数:", len(snaps))  # 理論上 10 * 2 * 2 = 40 になるはず

    # 先頭3つだけ HO イベントログを表示
    print("\n===== HO EVENT LOG (first 3 only) =====")
    for i, snap in enumerate(snaps[:3]):
        t = snap["time"]
        ho = snap["ho_event"]
        ho_pos = snap["ho_positions_x"]

        gs = ho["gs"]
        orbit = ho["orbit"]
        old_sat = ho["old_sat"]
        new_sat = ho["new_sat"]

        old_x = ho_pos["old_x"]
        new_x = ho_pos["new_x"]

        print(f"\n--- HO #{i} ---")
        print(f"Time: {t:.2f} s")
        print(f"GS: {gs}, Orbit: {orbit}")
        print(f"HO: sat {old_sat} (x={old_x:.1f})  →  sat {new_sat} (x={new_x:.1f})")
