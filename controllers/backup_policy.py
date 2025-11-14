import numpy as np
from shapely.geometry import Polygon, LineString, MultiLineString, Point
from shapely.ops import unary_union
from scipy.spatial import Voronoi
import networkx as nx


def sample_polygon_edges(poly, num_points_per_edge=5):
    points = []
    coords = list(poly.exterior.coords)
    for i in range(len(coords)-1):
        p0, p1 = np.array(coords[i]), np.array(coords[i+1])
        for t in np.linspace(0,1,num_points_per_edge,endpoint=False):
            points.append(p0*(1-t)+p1*t)
    return np.array(points)


def voronoi_corridor_paths(obstacle_arrays, boundary_array, agent_radius=0.5, sample_points_per_edge=5):

    obstacles = [Polygon(obs).buffer(agent_radius) for obs in obstacle_arrays]
    free_space = Polygon(boundary_array).difference(unary_union(obstacles))
    
    points = np.vstack([sample_polygon_edges(obs, sample_points_per_edge) for obs in obstacles])
    
    vor = Voronoi(points)
    
    paths = []
    for (p1, p2) in vor.ridge_vertices:
        if p1 >= 0 and p2 >= 0:
            pt1 = vor.vertices[p1]
            pt2 = vor.vertices[p2]
            line = LineString([pt1, pt2])
            if free_space.contains(line):
                paths.append([tuple(pt1), tuple(pt2)])
    
    return paths


def attract_point_to_voronoi_paths(point, voronoi_paths, advance_distance = 0.5):
    """
    point_xy: tuple (x,y)
    voronoi_paths: liste de segments [[(x1,y1),(x2,y2)], [(x3,y3),(x4,y4)], ...]
    Retourne: point projeté (x_proj, y_proj), index du segment
    """
    pt = Point(point)
    min_dist = float('inf')
    best_segment = None
    best_idx = -1
    
    for i, seg in enumerate(voronoi_paths):
        line = LineString(seg)
        dist = pt.distance(line)
        if dist < min_dist:
            min_dist = dist
            best_segment = line
            best_idx = i
    
    proj_distance = best_segment.project(pt)
    new_distance = proj_distance + advance_distance
    new_distance = min(new_distance, best_segment.length)
    
    new_point_shapely = best_segment.interpolate(new_distance)
    new_point = np.array([new_point_shapely.x, new_point_shapely.y])

    x0, y0 = best_segment.coords[0]
    x1, y1 = best_segment.coords[1]
    vec = np.array([x1 - x0, y1 - y0])
    tangent = vec / np.linalg.norm(vec)
    angle = np.arctan2(tangent[1], tangent[0])

    return np.array([new_point[0], new_point[1], angle])


class Backup_Policy :


    def __init__(self, 
                obstacles, 
                bounds, 
                K) :

        self.paths = voronoi_corridor_paths(obstacles, bounds, agent_radius=2.0, sample_points_per_edge=10)

        self._K = K



    def compute_agent_vect(self, init_pose, target_pose) :

        pose_x, pose_y, pose_yaw = init_pose[0], init_pose[1], init_pose[2]
        target_x, target_y, target_yaw = target_pose[0], target_pose[1], target_pose[2]

        dx = target_x - pose_x
        dy = target_y - pose_y

        forward_error =  np.cos(pose_yaw) * dx + np.sin(pose_yaw) * dy

        angle_error = np.arctan2(
            -np.sin(pose_yaw) * dx + np.cos(pose_yaw) * dy,
            np.cos(pose_yaw) * dx + np.sin(pose_yaw) * dy
        )

        return forward_error, angle_error



    def act(self, state) :

        pos = state[0:3]
        target_pose = attract_point_to_voronoi_paths(pos, self.paths)

        forward_error, angle_error = self.compute_agent_vect(pos, target_pose)

        u_cmd = self._K @ np.array([forward_error, angle_error]).T

        return u_cmd


