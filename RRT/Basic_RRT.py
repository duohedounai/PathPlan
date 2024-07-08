import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, x, y):
        self.parent_id = None
        self.id = None
        self.x = x
        self.y = y


class Rectangle_obstacle:
    def __init__(self, x1, y1, x2, y2):
        # Coordinates of the lower-left point
        self.left_top_node = Node(x1, y1)
        # Coordinates of the upper-right point
        self.right_bottom_node = Node(x2, y2)
        # Length of the rectangle
        self.length = abs(x2 - x1)
        # Height of the rectangle
        self.height = abs(y1 - y2)


def distance(node1, node2):
    return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)


# Find the closest point in the tree
def find_nearest_neighbor(nodes_deque, rnd_node):
    return min(nodes_deque, key=lambda node: distance(node, rnd_node))


# Expand new node
def steer(node, rnd_node, step_size):
    dx = rnd_node.x - node.x
    dy = rnd_node.y - node.y
    dist = np.sqrt(dx ** 2 + dy ** 2)
    # If the distance between the randomly sampled point and the current node is less than the step size, the random point is returned directly
    if dist < step_size:
        return Node(rnd_node.x, rnd_node.y)
    # The distance of the specified step size is offset along the direction of the random point to generate a new node
    else:
        return Node(node.x + step_size * dx / dist, node.y + step_size * dy / dist)


def collision_check(node, obstacles, ax):
    for curr_obstacle in obstacles:
        curr_rec_point_list = []
        p1 = Node(curr_obstacle.left_top_node.x, curr_obstacle.left_top_node.y)
        p2 = Node(curr_obstacle.left_top_node.x + curr_obstacle.length, curr_obstacle.left_top_node.y)
        p3 = Node(curr_obstacle.left_top_node.x + curr_obstacle.length,
                  curr_obstacle.left_top_node.y - curr_obstacle.height)
        p4 = Node(curr_obstacle.left_top_node.x, curr_obstacle.left_top_node.y - curr_obstacle.height)
        curr_rec_point_list.append(p1)
        curr_rec_point_list.append(p2)
        curr_rec_point_list.append(p3)
        curr_rec_point_list.append(p4)

        # Detects if the new node is within the obstacle area
        if nodeInAreaCheck(node, curr_rec_point_list):
            # ax.plot(node.x, node.y, marker='o', color='yellow')
            return True

    return False


# Determine whether a point is within a given area
def nodeInAreaCheck(new_point, area_point_list):
    in_area_flag = True

    cross_times = 0

    # Allowable accuracy error
    precision = 1e-3

    p = new_point

    for node_index in range(len(area_point_list)):
        p1 = area_point_list[node_index]

        next_point_index = -1
        if node_index + 1 >= len(area_point_list):
            next_point_index = 0
        else:
            next_point_index = node_index + 1

        p2 = area_point_list[next_point_index]

        if (p1.x == p.x and p1.y == p.y) or (p2.x == p.x and p2.y == p.y):
            return in_area_flag

        if p.y <= max(p1.y, p2.y) and p.y >= min(p1.y, p2.y):
            if p.x <= max(p1.x, p2.x) and p.x >= min(p1.x, p2.x):
                if p1.y == p2.y:
                    if p.y == p1.y:
                        return in_area_flag

                if p1.x == p2.x:
                    if p.x == p1.x:
                        return in_area_flag

                on_line_Y = p1.y + (p.x - p1.x) * (p2.y - p1.y) / (p2.x - p1.x)
                if abs(on_line_Y - p.y) < precision:
                    return in_area_flag

            if p.x <= min(p1.x, p2.x):
                continue

            if p1.y != p2.y and p.y != min(p1.y, p2.y):
                cross_times += 1

    if cross_times % 2 == 0:
        # The number of times a ray emitted horizontally to the left from the point passes through the area is even,
        # indicating that the point is outside the area
        return False

    return in_area_flag


# Global path backtracking function
def get_final_node_path(new_node, nodes_map):
    final_path = []
    final_path.append(goal_node)

    former_node = new_node
    final_path.insert(0, former_node)
    while former_node.parent_id != None:
        former_node = nodes_map[former_node.parent_id]
        final_path.insert(0, former_node)

    final_path.insert(0, start_node)
    return final_path


# The main process of the RRT algorithm
def rrt(start_node, goal_node, obstacles, step_size, distance_threshold, max_nodes, ax, max_x, max_y):
    nodes_map = {}
    nodes_list = []
    nodes_list.append(Node(start_node.x, start_node.y))

    while len(nodes_list) < max_nodes:
        # Random sampling to get a node
        rnd_node = Node(np.random.uniform(-max_x, max_x), np.random.uniform(-max_y, max_y))
        # Find the node in the tree that is closest to the new node
        nearest_node_in_tree = find_nearest_neighbor(nodes_list, rnd_node)

        if not collision_check(nearest_node_in_tree, obstacles, ax):
            # Get a new node in the direction of the random node
            new_node = steer(nearest_node_in_tree, rnd_node, step_size)

            # Save the information of the new node
            new_node.id = len(nodes_list) + 1
            new_node.parent_id = nearest_node_in_tree.id
            nodes_map[new_node.id] = new_node

            #  If the distance between the new node and the target point is less than the threshold,
            #  the target point is considered to have been reached
            if distance(new_node, Node(goal_node.x, goal_node.y)) <= distance_threshold:
                nodes_list.append(new_node)

                # Backtrack to get the global path
                final_node_path = get_final_node_path(new_node, nodes_map)
                ax.plot([node.x for node in final_node_path], [node.y for node in final_node_path], linestyle='-',
                        color='lime')
                break
            else:
                nodes_list.append(new_node)

            # Draw the new branch
            ax.plot([nearest_node_in_tree.x, new_node.x], [nearest_node_in_tree.y, new_node.y], linestyle='-',
                    color='darkorange')
            # plt.pause(0.1)


if __name__ == '__main__':
    # The coordinates of the rectangular obstacle
    rectangle_obstacles = [Rectangle_obstacle(-7, 7, -5, -8),
                           Rectangle_obstacle(-4, 4, 0, 2.5),
                           Rectangle_obstacle(-1.5, 1.5, 1, -3.5),
                           Rectangle_obstacle(2.5, 10, 3.5, -4),
                           Rectangle_obstacle(-2, -6, 8, -8)]

    # The coordinates of start and goal points
    start_node = Node(-8, -4)
    goal_node = Node(4.5, 0)

    # The step size of the exploration
    step_size = 0.5
    # The threshold of the distance at which the end point is considered to have been reached
    distance_threshold = 0.5
    # The maximum number of nodes allowed to be explored, beyond which the algorithm is stopped
    max_nodes = 10000

    # Draw the map
    fig, ax = plt.subplots()
    max_x = 10
    max_y = 10
    ax.set_xlim(-max_x, max_x)
    ax.set_ylim(-max_y, max_y)
    ax.plot(start_node.x, start_node.y, color='red', marker='o', markersize=10, label='Start')
    ax.plot(goal_node.x, goal_node.y, color='dodgerblue', marker='o', markersize=10, label='Goal')
    ax.legend()

    for obstacle in rectangle_obstacles:
        # Note that the coordinates of the bottom-left point of the rectangle need to be given here
        ax.add_patch(plt.Rectangle((obstacle.left_top_node.x, obstacle.left_top_node.y - obstacle.height),
                                   obstacle.length, obstacle.height, color='k', alpha=0.5))

    # Execute the RRT algorithm
    rrt(start_node, goal_node, rectangle_obstacles, step_size, distance_threshold, max_nodes,
        ax, max_x, max_y)

    plt.show()
