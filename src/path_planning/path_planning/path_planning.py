import numpy as np
import matplotlib.pyplot as plt
import yaml
import cv2
import os

def load_map(yaml_path, pgm_path):
    yaml_path = os.path.expanduser(yaml_path)  # パスを展開
    pgm_path = os.path.expanduser(pgm_path)  # パスを展開
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    resolution = config['resolution']
    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    grid = (img < 127).astype(int)

    # 座標変換用の情報を計算
    map_height, map_width = grid.shape
    origin = (map_height // 2, map_width // 2)  # 画像中央を原点に設定

    return grid, resolution, origin

def generate_cost_map(grid, wall_cost=10, near_wall_cost=5):
    """
    OpenCVのdilationを使ってコストマップを生成。
    壁は侵入禁止エリアとし、壁付近のコストを増加させる。
    """
    # 壁（侵入禁止エリア）のマスク
    wall_mask = (grid == 1).astype(np.uint8)

    # 円形のカーネルを作成
    kernel_size = 10  # カーネルの直径
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 壁周囲の領域を膨張
    dilated_wall = cv2.dilate(wall_mask, kernel, iterations=1)

    # コストマップを初期化
    cost_map = np.zeros_like(grid, dtype=float)

    # 壁そのもの（侵入禁止エリア）
    cost_map[wall_mask == 1] = wall_cost

    # 壁周辺（膨張した領域）にコストを設定
    near_wall_mask = (dilated_wall == 1) & (wall_mask == 0)
    cost_map[near_wall_mask] = near_wall_cost

    return cost_map


# A*アルゴリズム
def a_star(grid, cost_map, start, goal):
    open_list = []
    closed_list = set()
    came_from = {}

    # 初期化
    open_list.append((heuristic(start, goal), 0, start))
    g_score = {start: 0}

    while open_list:
        _, current_cost, current = min(open_list, key=lambda x: x[0])
        open_list = [x for x in open_list if x[2] != current]

        if current == goal:
            return reconstruct_path(came_from, current)

        closed_list.add(current)

        for neighbor in get_neighbors(grid, current):
            if neighbor in closed_list:
                continue

            # コストマップを加味したg_scoreの計算
            move_cost = heuristic(current, neighbor)  # 移動コスト（距離に基づく）
            tentative_g_score = g_score[current] + move_cost + cost_map[neighbor]
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                open_list.append((f_score, tentative_g_score, neighbor))
                came_from[neighbor] = current

    return None

def heuristic(a, b):
    """ユークリッド距離を計算"""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def get_neighbors(grid, pos):
    """上下左右＋斜め移動を考慮"""
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # 上下左右
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 斜め移動
    for d in directions:
        neighbor = (pos[0] + d[0], pos[1] + d[1])
        if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
            if grid[neighbor] == 0:  # 障害物でない場合
                neighbors.append(neighbor)
    return neighbors


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# 可視化
def plot_path(grid, cost_map, path, start, goal):
    plt.figure(figsize=(12, 6))

    # コストマップ表示
    plt.subplot(1, 2, 1)
    plt.imshow(cost_map, cmap='viridis')
    plt.colorbar(label="Cost")
    plt.title("Cost Map")

    # マップ表示
    plt.subplot(1, 2, 2)
    plt.imshow(grid, cmap='gray')
    path = np.array(path)
    plt.plot(path[:, 1], path[:, 0], color='red')
    plt.scatter(start[1], start[0], color='blue', label='Start')
    plt.scatter(goal[1], goal[0], color='green', label='Goal')
    plt.legend()
    plt.title("Path on Grid Map")

    plt.show()

if __name__ == "__main__":
    # マップ読み取り
    grid, resolution, origin = load_map('~/map.yaml', '~/map.pgm')

    # コストマップ生成
    cost_map = generate_cost_map(grid, wall_cost=10, near_wall_cost=5)

    # スタートとゴール位置（マップ座標系）
    start = (56, 10)  # 原点からの相対位置で設定
    goal = (20, 90)  # 原点からの相対位置で設定

    # A*アルゴリズムで経路生成
    path = a_star(grid, cost_map, start, goal)
    if path:
        plot_path(grid, cost_map, path, start, goal)
    else:
        print("Path not found!")
