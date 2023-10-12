import random
from PIL import Image, ImageDraw, ImageFont
import math
import time
import numpy as np
import os


def get_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


def calculate_accuracy(optimal_path, algorithm_path):
    optimal_set = set(optimal_path)
    algorithm_set = set(algorithm_path)
    common_nodes = optimal_set.intersection(algorithm_set)
    return len(common_nodes) / len(optimal_set) * 100


class Space:
    def __init__(self, height, width, num_hospitals):
        self.height = height
        self.width = width
        self.num_hospitals = num_hospitals
        self.houses = set()
        self.hospitals = set()
        self.best_tsp_path = None
        self.best_tsp_cost = None

    def add_house(self, row, col):
        self.houses.add((row, col))

    def available_space(self):
        candidates = set(
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
        )
        for house in self.houses:
            candidates.remove(house)
        for hospital in self.hospitals:
            candidates.remove(hospital)
        return candidates

    def get_cost(self, hospitals):
        cost = 0
        for house in self.houses:
            cost += min(
                abs(house[0] - hospital[0]) + abs(house[1] - hospital[1])
                for hospital in hospitals
            )
        return cost

    def get_neighbors(self, row, col):
        candidates = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1)
        ]
        neighbors = []
        for r, c in candidates:
            if (r, c) in self.houses or (r, c) in self.hospitals:
                continue
            if 0 <= r < self.height and 0 <= c < self.width:
                neighbors.append((r, c))
        return neighbors

    def output_image(self, filename):
        cell_size = 100
        cell_border = 2
        cost_size = 40
        padding = 10

        img = Image.new("RGBA",
                        (self.width * cell_size, self.height * cell_size + cost_size + padding * 2),
                        "white"
                        )
        house = Image.open("assets/images/House.png").resize((cell_size, cell_size))
        hospital = Image.open("assets/images/Hospital.png").resize((cell_size, cell_size))
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 30)
        draw = ImageDraw.Draw(img)

        for i in range(self.height):
            for j in range(self.width):
                rect = (
                    (j * cell_size + cell_border, i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)
                )
                draw.rectangle(rect, fill="black")

                if (i, j) in self.houses:
                    img.paste(house, tuple(rect[0]), house)
                if (i, j) in self.hospitals:
                    img.paste(hospital, tuple(rect[0]), hospital)

        draw.rectangle(
            (0, self.height * cell_size, self.width * cell_size,
             self.height * cell_size + cost_size + padding * 2),
            "black"
        )

        draw.text(
            (padding, self.height * cell_size + padding),
            f"Cost: {self.get_cost(self.hospitals)}",
            fill="white",
            font=font
        )

        img.save(filename)

    def hill_climb(self, maximum=None, image_prefix=None, log=False):
        count = 0
        self.hospitals = set()
        for i in range(self.num_hospitals):
            self.hospitals.add(random.choice(list(self.available_space())))
        if log:
            print("Initial state: Cost", self.get_cost(self.hospitals))
        if image_prefix:
            self.output_image(f"{image_prefix}{str(count).zfill(9)}.png")

        local_minimum = False
        local_maximum = False
        global_minimum = False
        global_maximum = False

        while maximum is None or count < maximum:
            count += 1
            best_neighbors = []
            best_neighbor_cost = None

            for hospital in self.hospitals:
                for replacement in self.get_neighbors(*hospital):
                    neighbor = self.hospitals.copy()
                    neighbor.remove(hospital)
                    neighbor.add(replacement)

                    cost = self.get_cost(neighbor)
                    if best_neighbor_cost is None or cost < best_neighbor_cost:
                        best_neighbor_cost = cost
                        best_neighbors = [neighbor]
                    elif best_neighbor_cost == cost:
                        best_neighbors.append(neighbor)

            if best_neighbor_cost >= self.get_cost(self.hospitals):
                if local_maximum:
                    global_maximum = True
                    break
                local_maximum = True
            elif best_neighbor_cost <= self.get_cost(self.hospitals):
                if local_minimum:
                    global_minimum = True
                local_minimum = True

            if global_minimum and global_maximum:
                break

            if log:
                print(f"Found Better Neighbor: cost {best_neighbor_cost}")
            self.hospitals = random.choice(best_neighbors)

            if image_prefix:
                self.output_image(f"{image_prefix}{str(count).zfill(9)}.png")

        return self.hospitals

    def simple_hill_climbing(self, image_prefix=None, log=False):
        count = 0
        self.hospitals = set()
        for i in range(self.num_hospitals):
            self.hospitals.add(random.choice(list(self.available_space())))
        if log:
            print("Initial state: Cost", self.get_cost(self.hospitals))
        if image_prefix:
            self.output_image(f"{image_prefix}{str(count).zfill(9)}.png")

        local_minimum = False
        local_maximum = False
        global_minimum = False
        global_maximum = False

        while True:
            count += 1
            best_neighbors = []
            best_neighbor_cost = None

            for hospital in self.hospitals:
                for replacement in self.get_neighbors(*hospital):
                    neighbor = self.hospitals.copy()
                    neighbor.remove(hospital)
                    neighbor.add(replacement)

                    cost = self.get_cost(neighbor)
                    if best_neighbor_cost is None or cost < best_neighbor_cost:
                        best_neighbor_cost = cost
                        best_neighbors = [neighbor]
                    elif best_neighbor_cost == cost:
                        best_neighbors.append(neighbor)

            if best_neighbor_cost >= self.get_cost(self.hospitals):
                if local_maximum:
                    global_maximum = True
                    break
                local_maximum = True
            elif best_neighbor_cost <= self.get_cost(self.hospitals):
                if local_minimum:
                    global_minimum = True
                local_minimum = True

            if global_minimum and global_maximum:
                break

            if log:
                print(f"Found Better Neighbor: cost {best_neighbor_cost}")
            self.hospitals = random.choice(best_neighbors)

            if image_prefix:
                self.output_image(f"{image_prefix}{str(count).zfill(9)}.png")

        return self.hospitals

    def tsp_brute_force(self):
        def calculate_path_cost(path):
            total_cost = 0
            for i in range(len(path) - 1):
                total_cost += get_distance(path[i], path[i + 1])
            return total_cost

        def permute(path, start, end):
            if start == end:
                current_cost = calculate_path_cost(path)
                if current_cost < self.best_tsp_cost:
                    self.best_tsp_cost = current_cost
                    self.best_tsp_path = path.copy()
            else:
                for i in range(start, end + 1):
                    path[start], path[i] = path[i], path[start]
                    permute(path, start + 1, end)
                    path[start], path[i] = path[i], path[start]

        houses_list = list(self.houses)
        num_houses = len(houses_list)
        self.best_tsp_path = []
        self.best_tsp_cost = math.inf
        permute(houses_list, 0, num_houses - 1)
        return self.best_tsp_path

    def steepest_ascent_hill_climbing(self, image_prefix=None, log=False):
        count = 0
        self.hospitals = set()
        for i in range(self.num_hospitals):
            self.hospitals.add(random.choice(list(self.available_space())))
        if log:
            print("Initial state: Cost", self.get_cost(self.hospitals))
        if image_prefix:
            self.output_image(f"{image_prefix}{str(count).zfill(9)}.png")

        while True:
            count += 1
            best_neighbors = []
            best_neighbor_cost = None

            for hospital in self.hospitals:
                for replacement in self.get_neighbors(*hospital):
                    neighbor = self.hospitals.copy()
                    neighbor.remove(hospital)
                    neighbor.add(replacement)

                    cost = self.get_cost(neighbor)
                    if best_neighbor_cost is None or cost < best_neighbor_cost:
                        best_neighbor_cost = cost
                        best_neighbors = [neighbor]
                    elif best_neighbor_cost == cost:
                        best_neighbors.append(neighbor)

            if best_neighbor_cost >= self.get_cost(self.hospitals):
                if log:
                    print("Reached local maximum.")
                break

            if log:
                print(f"Found Better Neighbor: cost {best_neighbor_cost}")
            self.hospitals = random.choice(best_neighbors)

            if image_prefix:
                self.output_image(f"{image_prefix}{str(count).zfill(9)}.png")

        return self.hospitals

    def stochastic_hill_climbing(self, image_prefix=None, log=False):
        count = 0
        self.hospitals = set()
        for i in range(self.num_hospitals):
            self.hospitals.add(random.choice(list(self.available_space())))
        if log:
            print("Initial state: Cost", self.get_cost(self.hospitals))
        if image_prefix:
            self.output_image(f"{image_prefix}{str(count).zfill(9)}.png")

        local_minimum = False
        local_maximum = False
        global_minimum = False
        global_maximum = False

        while True:
            count += 1
            current_cost = self.get_cost(self.hospitals)

            best_neighbors = []
            best_neighbor_cost = None

            for hospital in self.hospitals:
                for replacement in self.get_neighbors(*hospital):
                    neighbor = self.hospitals.copy()
                    neighbor.remove(hospital)
                    neighbor.add(replacement)

                    cost = self.get_cost(neighbor)
                    if best_neighbor_cost is None or cost < best_neighbor_cost:
                        best_neighbor_cost = cost
                        best_neighbors = [neighbor]
                    elif best_neighbor_cost == cost:
                        best_neighbors.append(neighbor)

            if best_neighbor_cost >= current_cost:
                if local_maximum:
                    global_maximum = True
                    break
                local_maximum = True
            elif best_neighbor_cost <= current_cost:
                if local_minimum:
                    global_minimum = True
                local_minimum = True

            if global_minimum and global_maximum:
                break

            if log:
                print(f"Found Better Neighbor: cost {best_neighbor_cost}")
            self.hospitals = random.choice(best_neighbors)

            if image_prefix:
                self.output_image(f"{image_prefix}{str(count).zfill(9)}.png")

        return self.hospitals

    def a_star_search(self, start):
        def calculate_heuristic_cost(point1, point2):
            return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

        open_set = [(start, 0, calculate_heuristic_cost(start, random.choice(list(self.houses))))]
        came_from = {}
        g_score = {house: math.inf for house in self.houses}
        g_score[start] = 0
        f_score = {house: math.inf for house in self.houses}
        f_score[start] = calculate_heuristic_cost(start, random.choice(list(self.houses)))

        while open_set:
            current, g, f = min(open_set, key=lambda x: x[2])
            open_set.remove((current, g, f))

            if current in self.houses:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path, g

            for neighbor in self.get_neighbors(*current):
                tentative_g = g_score[current] + 1

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + calculate_heuristic_cost(neighbor,
                                                                               random.choice(list(self.houses)))

                    if neighbor not in [item[0] for item in open_set]:
                        open_set.append((neighbor, tentative_g, f_score[neighbor]))

        return None, math.inf

    def dfs(self, start):
        def dls_helper(node, visited):
            visited.add(node)
            path.append(node)

            for neighbor in self.get_neighbors(*node):
                if neighbor not in visited:
                    dls_helper(neighbor, visited)

        visited = set()
        path = []
        dls_helper(start, visited)

        return path

    def bfs(self, start):
        visited = set()
        path = []
        queue = [start]

        while queue:
            current_node = queue.pop(0)
            visited.add(current_node)
            path.append(current_node)

            for neighbor in self.get_neighbors(*current_node):
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)

        return path

    def depth_limited_search(self, start, depth_limit):
        def dls_helper(node, depth, visited):
            visited.add(node)
            path.append(node)

            if depth == depth_limit:
                return

            for neighbor in self.get_neighbors(*node):
                if neighbor not in visited:
                    dls_helper(neighbor, depth + 1, visited)

        visited = set()
        path = []
        dls_helper(start, 0, visited)

        return path

    def greedy_search(self, start):
        def calculate_heuristic_cost(point1, point2):
            return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

        unvisited = self.houses.copy()
        unvisited.remove(start)
        path = [start]
        total_cost = 0

        while unvisited:
            current_point = path[-1]
            best_next_point = None
            best_next_cost = math.inf

            for next_point in unvisited:
                heuristic_cost = calculate_heuristic_cost(current_point, next_point)
                if heuristic_cost < best_next_cost:
                    best_next_cost = heuristic_cost
                    best_next_point = next_point

            total_cost += best_next_cost
            path.append(best_next_point)
            unvisited.remove(best_next_point)

        total_cost += calculate_heuristic_cost(path[-1], start)
        path.append(start)

        return path, total_cost

    def uniform_cost_search(self, start):
        def calculate_actual_cost(path):
            total_cost = 0
            for i in range(len(path) - 1):
                total_cost += abs(path[i][0] - path[i + 1][0]) + abs(path[i][1] - path[i + 1][1])
            return total_cost

        unvisited = self.houses.copy()
        unvisited.remove(start)
        path = [start]
        total_cost = 0

        while unvisited:
            current_point = path[-1]
            best_next_point = None
            best_next_cost = math.inf

            for next_point in unvisited:
                actual_cost = calculate_actual_cost(path + [next_point])
                if actual_cost < best_next_cost:
                    best_next_cost = actual_cost
                    best_next_point = next_point

            total_cost += best_next_cost
            path.append(best_next_point)
            unvisited.remove(best_next_point)

        total_cost += calculate_actual_cost(path + [start])
        path.append(start)

        return path, total_cost

    def bidirectional_search(self, start, goal):
        def calculate_actual_cost(path):
            total_cost = 0
            for i in range(len(path) - 1):
                total_cost += abs(path[i][0] - path[i + 1][0]) + abs(path[i][1] - path[i + 1][1])
            return total_cost

        forward_queue = [start]
        backward_queue = [goal]
        forward_visited = set([start])
        backward_visited = set([goal])
        intersect = None

        while forward_queue and backward_queue:
            current_forward_node = forward_queue.pop(0)

            for neighbor in self.get_neighbors(*current_forward_node):
                if neighbor not in forward_visited:
                    forward_queue.append(neighbor)
                    forward_visited.add(neighbor)
            current_backward_node = backward_queue.pop(0)

            for neighbor in self.get_neighbors(*current_backward_node):
                if neighbor not in backward_visited:
                    backward_queue.append(neighbor)
                    backward_visited.add(neighbor)

            intersect = forward_visited.intersection(backward_visited)
            if intersect:
                break

        path = list(intersect)
        forward_path = forward_queue + path
        backward_path = list(reversed(backward_queue + path))
        total_cost = calculate_actual_cost(forward_path + backward_path)
        return forward_path + backward_path, total_cost

    def greedy_best_first_search(self, start):
        def calculate_heuristic_cost(point1, point2):
            return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

        unvisited = self.houses.copy()
        unvisited.remove(start)
        path = [start]
        total_cost = 0

        while unvisited:
            current_point = path[-1]
            best_next_point = None
            best_next_cost = math.inf

            for next_point in unvisited:
                heuristic_cost = calculate_heuristic_cost(current_point, next_point)
                if heuristic_cost < best_next_cost:
                    best_next_cost = heuristic_cost
                    best_next_point = next_point

            total_cost += best_next_cost
            path.append(best_next_point)
            unvisited.remove(best_next_point)

        total_cost += calculate_heuristic_cost(path[-1], start)
        path.append(start)
        return path, total_cost

    def simulated_annealing(self, start_temperature=1000, cooling_rate=0.03):
        def calculate_heuristic_cost(point1, point2):
            return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

        current_state = self.hospitals
        current_cost = self.get_cost(current_state)
        best_state = current_state.copy()
        best_cost = current_cost

        temperature = start_temperature

        while temperature > 1:
            neighbor = set()
            while not neighbor:
                for hospital in current_state:
                    if random.random() < 0.5:
                        neighbor.add(hospital)
                    else:
                        neighbor = neighbor.union(set(self.get_neighbors(*hospital)))

            new_state = current_state.union(neighbor)
            new_cost = self.get_cost(new_state)

            if new_cost < current_cost:
                current_state = new_state
                current_cost = new_cost
                if new_cost < best_cost:
                    best_state = new_state
                    best_cost = new_cost
            else:
                probability = math.exp((current_cost - new_cost) / temperature)
                if random.random() < probability:
                    current_state = new_state
                    current_cost = new_cost

            temperature *= 1 - cooling_rate

        return best_state, best_cost


def run_algorithm_randomly_once():
    s = Space(height=6, width=12, num_hospitals=3)
    for i in range(9):
        s.add_house(random.randrange(s.height), random.randrange(s.width))

    # Hill Climbing
    start_time = time.time()
    hospitals_hc = s.hill_climb(image_prefix="hospitals", log=True)
    end_time = time.time()
    time_hc = end_time - start_time

    # Simple Hill Climbing
    start_time = time.time()
    hospitals_shc = s.simple_hill_climbing(image_prefix="hospitals_shc", log=True)
    end_time = time.time()
    time_shc = end_time - start_time

    # Steepest-Ascent Hill-Climbing
    start_time = time.time()
    hospitals_sahc = s.steepest_ascent_hill_climbing(image_prefix="hospitals_sahc", log=True)
    end_time = time.time()
    time_sahc = end_time - start_time

    # Stochastic Hill Climbing
    start_time = time.time()
    hospitals_shc = s.stochastic_hill_climbing(image_prefix="hospitals_shc", log=True)
    end_time = time.time()
    time_shc = end_time - start_time

    # TSP - Brute Force
    start_time = time.time()
    optimal_path = s.tsp_brute_force()
    end_time = time.time()
    time_tsp = end_time - start_time

    # A* Search
    start_time = time.time()
    start_astar = random.choice(list(s.houses))
    astar_path, astar_cost = s.a_star_search(start_astar)
    end_time = time.time()
    time_astar = end_time - start_time

    # DFS
    start_time = time.time()
    start_dfs = random.choice(list(s.houses))
    dfs_path = s.dfs(start_dfs)
    end_time = time.time()
    time_dfs = end_time - start_time

    # BFS
    start_time = time.time()
    start_bfs = random.choice(list(s.houses))
    bfs_path = s.bfs(start_bfs)
    end_time = time.time()
    time_bfs = end_time - start_time

    # DLS
    start_time = time.time()
    start_dls = random.choice(list(s.houses))
    depth_limit = 5
    dls_path = s.depth_limited_search(start_dls, depth_limit)
    end_time = time.time()
    time_dls = end_time - start_time

    # Bidirectional Search
    start_time = time.time()
    start_bd = random.choice(list(s.houses))
    goal_bd = random.choice(list(s.houses))
    bd_path, bd_cost = s.bidirectional_search(start_bd, goal_bd)
    end_time = time.time()
    time_bd = end_time - start_time

    # Greedy Search
    start_time = time.time()
    start_greedy = random.choice(list(s.houses))
    greedy_path, greedy_cost = s.greedy_search(start_greedy)
    end_time = time.time()
    time_greedy = end_time - start_time

    # Greedy Best First Search
    start_time = time.time()
    start_gbfs = random.choice(list(s.houses))
    gbfs_path, gbfs_cost = s.greedy_best_first_search(start_gbfs)
    end_time = time.time()
    time_gbfs = end_time - start_time

    # Uniform Cost Search
    start_time = time.time()
    start_uniform = random.choice(list(s.houses))
    uniform_path, uniform_cost = s.uniform_cost_search(start_uniform)
    end_time = time.time()
    time_uniform = end_time - start_time

    # Local Temp. Simulated Annealing Seach
    start_time = time.time()
    sa_hospitals, sa_cost = s.simulated_annealing()
    sa_path = s.dfs(list(sa_hospitals)[0])
    end_time = time.time()
    time_sa = end_time - start_time

    # Accuracy Scores
    accuracy_hc = calculate_accuracy(optimal_path, hospitals_hc)
    accuracy_shc = calculate_accuracy(optimal_path, hospitals_shc)
    accuracy_sahc = calculate_accuracy(optimal_path, hospitals_sahc)
    accuracy_shc = calculate_accuracy(optimal_path, hospitals_shc)
    accuracy_astar = calculate_accuracy(optimal_path, astar_path)
    accuracy_dfs = calculate_accuracy(optimal_path, dfs_path)
    accuracy_bfs = calculate_accuracy(optimal_path, bfs_path)
    accuracy_dls = calculate_accuracy(optimal_path, dls_path)
    accuracy_bd = calculate_accuracy(optimal_path, bd_path)
    accuracy_greedy = calculate_accuracy(optimal_path, greedy_path)
    accuracy_gbfs = calculate_accuracy(optimal_path, gbfs_path)
    accuracy_uniform = calculate_accuracy(optimal_path, uniform_path)
    accuracy_sa = calculate_accuracy(optimal_path, sa_path)

    # Accuracy Scores for Avg
    accuracy_scores = {
        'Hill Climbing': accuracy_hc,
        'Simple Hill Climbing': accuracy_shc,
        'Steepest-Ascent Hill-Climbing': accuracy_sahc,
        'Stochastic Hill Climbing': accuracy_shc,
        'A* Search': accuracy_astar,
        'DFS': accuracy_dfs,
        'BFS': accuracy_bfs,
        'Depth-Limited Search': accuracy_dls,
        'Bidirectional Search': accuracy_bd,
        'Greedy Search': accuracy_greedy,
        'Greedy Best First Search': accuracy_gbfs,
        'Uniform Cost Search': accuracy_uniform,
        'Simulated Annealing': accuracy_sa,
    }

    # Result time
    result_times = {
        'Hill Climbing': time_hc,
        'Simple Hill Climbing': time_shc,
        'Steepest-Ascent Hill-Climbing': time_sahc,
        'Stochastic Hill Climbing': time_shc,
        'A* Search': time_astar,
        'DFS': time_dfs,
        'BFS': time_bfs,
        'Depth-Limited Search': time_dls,
        'Bidirectional Search': time_bd,
        'Greedy Search': time_greedy,
        'Greedy Best First Search': time_gbfs,
        'Uniform Cost Search': time_uniform,
        'Simulated Annealing': time_sa,
    }

    print("All Hill-Climbing Algorithms:")
    print("Hill Climbing:", f"{accuracy_hc:.2f}%", " Time:", time_hc)
    print("Simple Hill Climbing:", f"{accuracy_shc:.2f}%", " Time:", time_shc)
    print("Steepest-Ascent Hill-Climbing:", f"{accuracy_sahc:.2f}%", " Time:", time_sahc)
    print("Stochastic Hill Climbing:", f"{accuracy_shc:.2f}%", " Time:", time_shc)
    print("==================================================================================")
    print("Accuracy Scores for All Hill-Climbing Algorithms:")
    print("A* Search:", f"{accuracy_astar:.2f}%", " Time:", time_astar)
    print("DFS:", f"{accuracy_dfs:.2f}%", " Time:", time_dfs)
    print("BFS:", f"{accuracy_bfs:.2f}%", " Time:", time_bfs)
    print("Depth-Limited Search:", f"{accuracy_dls:.2f}%", " Time:", time_dls)
    print("Bidirectional Search:", f"{accuracy_bd:.2f}%", " Time:", time_bd)
    print("Greedy Search:", f"{accuracy_greedy:.2f}%", " Time:", time_greedy)
    print("Greedy Best First Search:", f"{accuracy_gbfs:.2f}%", " Time:", time_gbfs)
    print("Uniform Cost Search:", f"{accuracy_uniform:.2f}%", " Time:", time_uniform)
    print("Simulated Annealing:", f"{accuracy_sa:.2f}%", "Simulated Annealing Cost:", sa_cost, " Time:", time_sa)
    print("==================================================================================")
    average_accuracy = sum(accuracy_scores.values()) / len(accuracy_scores)
    print("Average Accuracy Score for All Algorithms:", f"{average_accuracy:.2f}%")
    print("==================================================================================")
    best_accuracy = max(accuracy_scores.values())
    best_algorithms = [name for name, accuracy in accuracy_scores.items() if accuracy == best_accuracy]
    print("Best Accuracy Score(s):")
    for algorithm in best_algorithms:
        print("Algorithm:", algorithm, " Accuracy Score: ", f"{best_accuracy:.2f}%")
    print("==================================================================================")

    # Identify local minimum, global minimum, or skip for local maximum, global maximum
    print("Search Results:")
    if accuracy_hc == 0:
        print("Hill Climbing reached a local minimum.")
    elif accuracy_hc == 1:
        print("Hill Climbing reached a global minimum.")
    if accuracy_shc == 0:
        print("Simple Hill Climbing reached a local maximum.")
    elif accuracy_shc == 1:
        print("Simple Hill Climbing reached a global minimum.")
    if accuracy_sahc == 0:
        print("Steepest-Ascent Hill-Climbing reached a local maximum.")
    elif accuracy_sahc == 1:
        print("Steepest-Ascent Hill-Climbing reached a global minimum.")
    if accuracy_shc == 0:
        print("Stochastic Hill Climbing reached a local maximum.")
    elif accuracy_shc == 1:
        print("Stochastic Hill Climbing reached a global minimum.")
    if accuracy_astar == 0:
        print("A* Search reached a local minimum.")
    elif accuracy_astar == 1:
        print("A* Search reached a global minimum.")
    if accuracy_dfs == 0:
        print("DFS reached a local minimum.")
    elif accuracy_dfs == 1:
        print("DFS reached a global minimum.")
    if accuracy_bfs == 0:
        print("BFS reached a local minimum.")
    elif accuracy_bfs == 1:
        print("BFS reached a global minimum.")
    if accuracy_dls == 0:
        print("Depth-Limited Search reached a local maximum.")
    elif accuracy_dls == 1:
        print("Depth-Limited Search reached a global maximum.")
    if accuracy_bd == 0:
        print("Bidirectional Search reached a local minimum.")
    elif accuracy_bd == 1:
        print("Bidirectional Search reached a global minimum.")
    if accuracy_greedy == 0:
        print("Greedy Search reached a local minimum.")
    elif accuracy_greedy == 1:
        print("Greedy Search reached a global minimum.")
    if accuracy_gbfs == 0:
        print("Greedy Best First Search reached a local minimum.")
    elif accuracy_gbfs == 1:
        print("Greedy Best First Search reached a global minimum.")
    if accuracy_uniform == 0:
        print("Uniform Cost Search reached a local minimum.")
    elif accuracy_uniform == 1:
        print("Uniform Cost Search reached a global minimum.")
    if accuracy_sa == 0:
        print("Simulated Annealing reached a local minimum.")
    elif accuracy_sa == 1:
        print("Simulated Annealing reached a global minimum.")


def run_algorithm_randomly_all():
    s = Space(height=6, width=12, num_hospitals=3)
    for i in range(9):
        s.add_house(random.randrange(s.height), random.randrange(s.width))

        # Hill Climbing
        start_time = time.time()
        hospitals_hc = s.hill_climb(image_prefix="hospitals", log=True)
        end_time = time.time()
        time_hc = end_time - start_time

        # Simple Hill Climbing
        start_time = time.time()
        hospitals_shc = s.simple_hill_climbing(image_prefix="hospitals_shc", log=True)
        end_time = time.time()
        time_shc = end_time - start_time

        # Steepest-Ascent Hill-Climbing
        start_time = time.time()
        hospitals_sahc = s.steepest_ascent_hill_climbing(image_prefix="hospitals_sahc", log=True)
        end_time = time.time()
        time_sahc = end_time - start_time

        # Stochastic Hill Climbing
        start_time = time.time()
        hospitals_shc = s.stochastic_hill_climbing(image_prefix="hospitals_shc", log=True)
        end_time = time.time()
        time_shc = end_time - start_time

        # TSP - Brute Force
        start_time = time.time()
        optimal_path = s.tsp_brute_force()
        end_time = time.time()
        time_tsp = end_time - start_time

        # A* Search
        start_time = time.time()
        start_astar = random.choice(list(s.houses))
        astar_path, astar_cost = s.a_star_search(start_astar)
        end_time = time.time()
        time_astar = end_time - start_time

        # DFS
        start_time = time.time()
        start_dfs = random.choice(list(s.houses))
        dfs_path = s.dfs(start_dfs)
        end_time = time.time()
        time_dfs = end_time - start_time

        # BFS
        start_time = time.time()
        start_bfs = random.choice(list(s.houses))
        bfs_path = s.bfs(start_bfs)
        end_time = time.time()
        time_bfs = end_time - start_time

        # DLS
        start_time = time.time()
        start_dls = random.choice(list(s.houses))
        depth_limit = 5
        dls_path = s.depth_limited_search(start_dls, depth_limit)
        end_time = time.time()
        time_dls = end_time - start_time

        # Bidirectional Search
        start_time = time.time()
        start_bd = random.choice(list(s.houses))
        goal_bd = random.choice(list(s.houses))
        bd_path, bd_cost = s.bidirectional_search(start_bd, goal_bd)
        end_time = time.time()
        time_bd = end_time - start_time

        # Greedy Search
        start_time = time.time()
        start_greedy = random.choice(list(s.houses))
        greedy_path, greedy_cost = s.greedy_search(start_greedy)
        end_time = time.time()
        time_greedy = end_time - start_time

        # Greedy Best First Search
        start_time = time.time()
        start_gbfs = random.choice(list(s.houses))
        gbfs_path, gbfs_cost = s.greedy_best_first_search(start_gbfs)
        end_time = time.time()
        time_gbfs = end_time - start_time

        # Uniform Cost Search
        start_time = time.time()
        start_uniform = random.choice(list(s.houses))
        uniform_path, uniform_cost = s.uniform_cost_search(start_uniform)
        end_time = time.time()
        time_uniform = end_time - start_time

        # Local Temp. Simulated Annealing Seach
        start_time = time.time()
        sa_hospitals, sa_cost = s.simulated_annealing()
        sa_path = s.dfs(list(sa_hospitals)[0])
        end_time = time.time()
        time_sa = end_time - start_time

        # Accuracy Scores
        accuracy_hc = calculate_accuracy(optimal_path, hospitals_hc)
        accuracy_shc = calculate_accuracy(optimal_path, hospitals_shc)
        accuracy_sahc = calculate_accuracy(optimal_path, hospitals_sahc)
        accuracy_shc = calculate_accuracy(optimal_path, hospitals_shc)
        accuracy_astar = calculate_accuracy(optimal_path, astar_path)
        accuracy_dfs = calculate_accuracy(optimal_path, dfs_path)
        accuracy_bfs = calculate_accuracy(optimal_path, bfs_path)
        accuracy_dls = calculate_accuracy(optimal_path, dls_path)
        accuracy_bd = calculate_accuracy(optimal_path, bd_path)
        accuracy_greedy = calculate_accuracy(optimal_path, greedy_path)
        accuracy_gbfs = calculate_accuracy(optimal_path, gbfs_path)
        accuracy_uniform = calculate_accuracy(optimal_path, uniform_path)
        accuracy_sa = calculate_accuracy(optimal_path, sa_path)

        # Accuracy Scores for Avg
        accuracy_scores = {
            'Hill Climbing': accuracy_hc,
            'Simple Hill Climbing': accuracy_shc,
            'Steepest-Ascent Hill-Climbing': accuracy_sahc,
            'Stochastic Hill Climbing': accuracy_shc,
            'A* Search': accuracy_astar,
            'DFS': accuracy_dfs,
            'BFS': accuracy_bfs,
            'Depth-Limited Search': accuracy_dls,
            'Bidirectional Search': accuracy_bd,
            'Greedy Search': accuracy_greedy,
            'Greedy Best First Search': accuracy_gbfs,
            'Uniform Cost Search': accuracy_uniform,
            'Simulated Annealing': accuracy_sa,
        }

        # Result time
        result_times = {
            'Hill Climbing': time_hc,
            'Simple Hill Climbing': time_shc,
            'Steepest-Ascent Hill-Climbing': time_sahc,
            'Stochastic Hill Climbing': time_shc,
            'A* Search': time_astar,
            'DFS': time_dfs,
            'BFS': time_bfs,
            'Depth-Limited Search': time_dls,
            'Bidirectional Search': time_bd,
            'Greedy Search': time_greedy,
            'Greedy Best First Search': time_gbfs,
            'Uniform Cost Search': time_uniform,
            'Simulated Annealing': time_sa,
        }

        print("All Hill-Climbing Algorithms:")
        print("Hill Climbing:", f"{accuracy_hc:.2f}%", " Time:", time_hc)
        print("Simple Hill Climbing:", f"{accuracy_shc:.2f}%", " Time:", time_shc)
        print("Steepest-Ascent Hill-Climbing:", f"{accuracy_sahc:.2f}%", " Time:", time_sahc)
        print("Stochastic Hill Climbing:", f"{accuracy_shc:.2f}%", " Time:", time_shc)
        print("==================================================================================")
        print("Accuracy Scores for All Hill-Climbing Algorithms:")
        print("A* Search:", f"{accuracy_astar:.2f}%", " Time:", time_astar)
        print("DFS:", f"{accuracy_dfs:.2f}%", " Time:", time_dfs)
        print("BFS:", f"{accuracy_bfs:.2f}%", " Time:", time_bfs)
        print("Depth-Limited Search:", f"{accuracy_dls:.2f}%", " Time:", time_dls)
        print("Bidirectional Search:", f"{accuracy_bd:.2f}%", " Time:", time_bd)
        print("Greedy Search:", f"{accuracy_greedy:.2f}%", " Time:", time_greedy)
        print("Greedy Best First Search:", f"{accuracy_gbfs:.2f}%", " Time:", time_gbfs)
        print("Uniform Cost Search:", f"{accuracy_uniform:.2f}%", " Time:", time_uniform)
        print("Simulated Annealing:", f"{accuracy_sa:.2f}%", "Simulated Annealing Cost:", sa_cost, " Time:", time_sa)
        print("==================================================================================")
        average_accuracy = sum(accuracy_scores.values()) / len(accuracy_scores)
        print("Average Accuracy Score for All Algorithms:", f"{average_accuracy:.2f}%")
        print("==================================================================================")
        best_accuracy = max(accuracy_scores.values())
        best_algorithms = [name for name, accuracy in accuracy_scores.items() if accuracy == best_accuracy]
        print("Best Accuracy Score(s):")
        for algorithm in best_algorithms:
            print("Algorithm:", algorithm, " Accuracy Score: ", f"{best_accuracy:.2f}%")
        print("==================================================================================")

        # Identify local minimum, global minimum, or skip for local maximum, global maximum
        print("Search Results:")
        if accuracy_hc == 0:
            print("Hill Climbing reached a local minimum.")
        elif accuracy_hc == 1:
            print("Hill Climbing reached a global minimum.")
        if accuracy_shc == 0:
            print("Simple Hill Climbing reached a local maximum.")
        elif accuracy_shc == 1:
            print("Simple Hill Climbing reached a global minimum.")
        if accuracy_sahc == 0:
            print("Steepest-Ascent Hill-Climbing reached a local maximum.")
        elif accuracy_sahc == 1:
            print("Steepest-Ascent Hill-Climbing reached a global minimum.")
        if accuracy_shc == 0:
            print("Stochastic Hill Climbing reached a local maximum.")
        elif accuracy_shc == 1:
            print("Stochastic Hill Climbing reached a global minimum.")
        if accuracy_astar == 0:
            print("A* Search reached a local minimum.")
        elif accuracy_astar == 1:
            print("A* Search reached a global minimum.")
        if accuracy_dfs == 0:
            print("DFS reached a local minimum.")
        elif accuracy_dfs == 1:
            print("DFS reached a global minimum.")
        if accuracy_bfs == 0:
            print("BFS reached a local minimum.")
        elif accuracy_bfs == 1:
            print("BFS reached a global minimum.")
        if accuracy_dls == 0:
            print("Depth-Limited Search reached a local maximum.")
        elif accuracy_dls == 1:
            print("Depth-Limited Search reached a global maximum.")
        if accuracy_bd == 0:
            print("Bidirectional Search reached a local minimum.")
        elif accuracy_bd == 1:
            print("Bidirectional Search reached a global minimum.")
        if accuracy_greedy == 0:
            print("Greedy Search reached a local minimum.")
        elif accuracy_greedy == 1:
            print("Greedy Search reached a global minimum.")
        if accuracy_gbfs == 0:
            print("Greedy Best First Search reached a local minimum.")
        elif accuracy_gbfs == 1:
            print("Greedy Best First Search reached a global minimum.")
        if accuracy_uniform == 0:
            print("Uniform Cost Search reached a local minimum.")
        elif accuracy_uniform == 1:
            print("Uniform Cost Search reached a global minimum.")
        if accuracy_sa == 0:
            print("Simulated Annealing reached a local minimum.")
        elif accuracy_sa == 1:
            print("Simulated Annealing reached a global minimum.")


def reset_pngs():
    for filename in os.listdir("."):
        if filename.endswith(".png"):
            os.remove(filename)


def ask_user_reset_or_continue():
    while True:
        print("**********************************************************************************")
        user_input = input("Enter 'RESET' to clear generated images, or 'CONTINUE' to keep them: ")
        print("**********************************************************************************")
        if user_input.lower() == 'reset':
            reset_pngs()
            return True
        elif user_input.lower() == 'continue':
            return False
        else:
            print("********************************************************************************")
            print("Invalid input. Please enter 'RESET' or 'CONTINUE'.")
            print("*********************************************************************************")


def ask_user_to_quit_or_play():
    while True:
        print("*****************************************************************")
        user_input = input("Enter 'QUIT' to end the run, or 'PLAY' to continue: ")
        print("*****************************************************************")
        if user_input.lower() == 'quit':
            return False
        elif user_input.lower() == 'play':
            return True
        else:
            print("**************************************************************")
            print("Invalid input. Please enter 'QUIT' or 'PLAY'.")
            print("**************************************************************")


def run_algorithm_randomly():
    while True:
        print("***************************************************************************")
        user_input = input("Enter 'STEP' to run step by step or 'ALL' to see all results: ")
        print("***************************************************************************")
        if user_input.lower() == 'step':
            run_algorithm_randomly_once()
            ask_user_reset_or_continue()
        elif user_input.lower() == 'all':
            run_algorithm_randomly_all()
            ask_user_reset_or_continue()
        else:
            print("************************************************************************")
            print("Invalid input. Please enter 'STEP' or 'ALL'.")
            print("************************************************************************")
            continue

        if not ask_user_to_quit_or_play():
            break


if __name__ == "__main__":
    run_algorithm_randomly()
