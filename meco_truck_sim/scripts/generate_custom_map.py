import numpy as np
from PIL import Image
import yaml
import os

def generate_free_space_map(
    filename_prefix="free_space",
    map_width_m=50.0,
    map_height_m=50.0,
    resolution=0.05
):
    """
    Generates a free space occupancy grid map (PNG) and configuration (YAML).
    
    Args:
        filename_prefix: Name for output files (e.g., 'map' -> 'map.png', 'map.yaml')
        map_width_m: Physical width of the map in meters
        map_height_m: Physical height of the map in meters
        resolution: Map resolution in meters per pixel
    """
    
    # 1. Calculate dimensions in pixels
    width_px = int(np.ceil(map_width_m / resolution))
    height_px = int(np.ceil(map_height_m / resolution))
    
    print(f"Generating map: {width_px}x{height_px} pixels ({map_width_m}x{map_height_m}m)")

    # 2. Initialize Grid (0 = Black/Occupied, 255 = White/Free)
    # in ROS map_server: 
    #   Pixel 0 (Black)   => Occupancy 100 (Occupied)
    #   Pixel 255 (White) => Occupancy 0   (Free)
    #   Pixel 205 (Gray)  => Unknown
    grid = np.full((height_px, width_px), 255, dtype=np.uint8)

    return grid

def generate_occupied_map(
    filename_prefix="occupied_space",
    map_width_m=50.0,
    map_height_m=50.0,
    resolution=0.05
):
    """
    Generates a fully occupied occupancy grid map (PNG) and configuration (YAML).
    
    Args:
        filename_prefix: Name for output files (e.g., 'map' -> 'map.png', 'map.yaml')
        map_width_m: Physical width of the map in meters
        map_height_m: Physical height of the map in meters
        resolution: Map resolution in meters per pixel
    """
    
    # 1. Calculate dimensions in pixels
    width_px = int(np.ceil(map_width_m / resolution))
    height_px = int(np.ceil(map_height_m / resolution))
    
    print(f"Generating map: {width_px}x{height_px} pixels ({map_width_m}x{map_height_m}m)")

    # 2. Initialize Grid (0 = Black/Occupied, 255 = White/Free)
    # in ROS map_server: 
    #   Pixel 0 (Black)   => Occupancy 100 (Occupied)
    #   Pixel 255 (White) => Occupancy 0   (Free)
    #   Pixel 205 (Gray)  => Unknown
    grid = np.full((height_px, width_px), 0, dtype=np.uint8)

    return grid

def add_rectangular_obstacle(
    grid,
    bottom_left_m,
    top_right_m,
    resolution
):
    """
    Adds a rectangular obstacle to the occupancy grid map.
    
    Args:
        grid: 2D numpy array representing the occupancy grid
        bottom_left_m: (x, y) coordinates of the bottom-left corner in meters
        top_right_m: (x, y) coordinates of the top-right corner in meters
        resolution: Map resolution in meters per pixel
    """
    height_px, width_px = grid.shape

    # Convert meters to pixels (round to nearest pixel)
    bl_x = int(np.round(bottom_left_m[0] / resolution))
    bl_y = int(np.round(bottom_left_m[1] / resolution))
    tr_x = int(np.round(top_right_m[0] / resolution))
    tr_y = int(np.round(top_right_m[1] / resolution))

    # Clip to image bounds
    bl_x_px = int(np.clip(bl_x, 0, width_px - 1))
    tr_x_px = int(np.clip(tr_x, 0, width_px - 1))

    # Invert Y axis: user coords origin is bottom-left (meters), numpy array origin is top-left (rows)
    bl_y_px = int(np.clip(height_px - 1 - bl_y, 0, height_px - 1))
    tr_y_px = int(np.clip(height_px - 1 - tr_y, 0, height_px - 1))

    # Ensure start < end for slicing
    y_start = min(bl_y_px, tr_y_px)
    y_end = max(bl_y_px, tr_y_px) + 1
    x_start = min(bl_x_px, tr_x_px)
    x_end = max(bl_x_px, tr_x_px) + 1

    # Set obstacle area to occupied (0/Black)
    grid[y_start:y_end, x_start:x_end] = 0

    return grid

def add_circular_obstacle(
    grid,
    center_m,
    radius_m,
    resolution
):
    """
    Adds a circular obstacle to the occupancy grid map.
    
    Args:
        grid: 2D numpy array representing the occupancy grid
        center_m: (x, y) coordinates of the circle center in meters
        radius_m: Radius of the circle in meters
        resolution: Map resolution in meters per pixel
    """
    height_px, width_px = grid.shape

    # Convert center and radius to pixels (round to nearest pixel)
    center_x = int(np.round(center_m[0] / resolution))
    center_y = int(np.round(center_m[1] / resolution))
    radius_px = int(np.round(radius_m / resolution))

    # Clip and map Y from meters-origin (bottom-left) to numpy rows (top-left)
    center_x_px = int(np.clip(center_x, 0, width_px - 1))
    center_y_px = int(np.clip(height_px - 1 - center_y, 0, height_px - 1))

    # Create a meshgrid for distance calculation
    y_indices, x_indices = np.ogrid[:height_px, :width_px]
    distance_from_center = np.sqrt((x_indices - center_x_px) ** 2 + (y_indices - center_y_px) ** 2)

    # Set pixels within the radius to occupied (0/Black)
    grid[distance_from_center <= radius_px] = 0

    return grid

def add_rectangular_road(
    grid,
    bottom_left_m,
    top_right_m,
    resolution
):  
    """
    Adds a rectangular road (free space) to the occupancy grid map.
    
    Args:
        grid: 2D numpy array representing the occupancy grid
        bottom_left_m: (x, y) coordinates of the bottom-left corner in meters
        top_right_m: (x, y) coordinates of the top-right corner in meters
        resolution: Map resolution in meters per pixel
    """
    height_px, width_px = grid.shape

    # Convert meters to pixels (round to nearest pixel)
    bl_x = int(np.round(bottom_left_m[0] / resolution))
    bl_y = int(np.round(bottom_left_m[1] / resolution))
    tr_x = int(np.round(top_right_m[0] / resolution))
    tr_y = int(np.round(top_right_m[1] / resolution))

    # Clip to image bounds
    bl_x_px = int(np.clip(bl_x, 0, width_px - 1))
    tr_x_px = int(np.clip(tr_x, 0, width_px - 1))

    # Invert Y axis: user coords origin is bottom-left (meters), numpy array origin is top-left (rows)
    bl_y_px = int(np.clip(height_px - 1 - bl_y, 0, height_px - 1))
    tr_y_px = int(np.clip(height_px - 1 - tr_y, 0, height_px - 1))

    # Ensure start < end for slicing
    y_start = min(bl_y_px, tr_y_px)
    y_end = max(bl_y_px, tr_y_px) + 1
    x_start = min(bl_x_px, tr_x_px)
    x_end = max(bl_x_px, tr_x_px) + 1

    # Set road area to free space (255/White)
    grid[y_start:y_end, x_start:x_end] = 255

    return grid

def add_circular_free_space(
    grid,
    center_m,
    radius_m,
    resolution
):  
    """
    Adds a circular free space area to the occupancy grid map.
    
    Args:
        grid: 2D numpy array representing the occupancy grid
        center_m: (x, y) coordinates of the circle center in meters
        radius_m: Radius of the circle in meters
        resolution: Map resolution in meters per pixel
    """
    height_px, width_px = grid.shape

    # Convert center and radius to pixels (round to nearest pixel)
    center_x = int(np.round(center_m[0] / resolution))
    center_y = int(np.round(center_m[1] / resolution))
    radius_px = int(np.round(radius_m / resolution))

    # Clip and map Y from meters-origin (bottom-left) to numpy rows (top-left)
    center_x_px = int(np.clip(center_x, 0, width_px - 1))
    center_y_px = int(np.clip(height_px - 1 - center_y, 0, height_px - 1))

    # Create a meshgrid for distance calculation
    y_indices, x_indices = np.ogrid[:height_px, :width_px]
    distance_from_center = np.sqrt((x_indices - center_x_px) ** 2 + (y_indices - center_y_px) ** 2)

    # Set pixels within the radius to free space (255/White)
    grid[distance_from_center <= radius_px] = 255

    return grid


def save_map_and_yaml(
    grid,
    filename_prefix,
    resolution,
    origin=(-1.0, -1.0, 0.0)
):
    """
    Saves the occupancy grid map as a PNG file and its configuration as a YAML file.
    
    Args:
        grid: 2D numpy array representing the occupancy grid
        filename_prefix: Name for output files (e.g., 'map' -> 'map.png', 'map.yaml')
        resolution: Map resolution in meters per pixel
        origin: (x, y, theta) origin of the map in meters and radians
    """
    # Save PNG
    png_filename = f"{filename_prefix}.png"
    img = Image.fromarray(grid, mode='L')
    img.save(png_filename)
    print(f"Saved map image to {png_filename}")

    # Save YAML
    yaml_filename = f"{filename_prefix}.yaml"
    map_info = {
        'image': os.path.basename(png_filename),
        'resolution': resolution,
        'origin': list(origin),
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196
    }
    with open(yaml_filename, 'w') as yaml_file:
        yaml.dump(map_info, yaml_file, default_flow_style=False)
    print(f"Saved map configuration to {yaml_filename}")


def parking_one(parking_width_m=0.7, parking_center=4.75, save=True, resolution=0.05):
    """Generates a free space with parking spot."""
    grid = generate_free_space_map(
        filename_prefix="custom_map",
        map_width_m=3.0,
        map_height_m=6.0,
        resolution=resolution
    )

    parking_start = parking_center - (parking_width_m / 2)
    parking_end = parking_center + (parking_width_m / 2)

    # add long boundary line
    add_rectangular_obstacle(
        grid,
        bottom_left_m=(2.8, 0.1),
        top_right_m=(2.9, 5.9),
        resolution=resolution
    )

    # add short boundary lines
    add_rectangular_obstacle(
        grid,
        bottom_left_m=(1.1, 5.8),
        top_right_m=(2.8, 5.9),
        resolution=resolution
    )
    add_rectangular_obstacle(
        grid,
        bottom_left_m=(1.1, 0.1),
        top_right_m=(2.8, 0.2),
        resolution=resolution
    )

    # add lines to build a narrow passage
    add_rectangular_obstacle(
        grid,
        bottom_left_m=(1.1, 6.0 - parking_start),
        top_right_m=(1.2, 5.9),
        resolution=resolution
    )
    add_rectangular_obstacle(
        grid,
        bottom_left_m=(1.1, 0.1),
        top_right_m=(1.2, 6.0 - parking_end),
        resolution=resolution
    )

    # add boundaries of the narrow passage
    add_rectangular_obstacle(
        grid,
        bottom_left_m=(0.1, 6.0 - parking_start),
        top_right_m=(1.1, 6.1 - parking_start),
        resolution=resolution
    )
    add_rectangular_obstacle(
        grid,
        bottom_left_m=(0.1, 5.9 - parking_end),
        top_right_m=(1.1, 6.0 - parking_end),
        resolution=resolution
    )
    add_rectangular_obstacle(
        grid,
        bottom_left_m=(0.1, 5.9 - parking_end),
        top_right_m=(0.2, 6.1 - parking_start),
        resolution=resolution
    )

    # Save the map and YAML configuration
    if save:
        save_map_and_yaml(grid, "parking_one", resolution)

    else:
        return grid


def parking_two(parking_width_m=0.7, parking_center=4.75, roundabout_radius_m=0.2, save=True, resolution=0.05):
    '''Generates a parking scenario with two circular obstacles and a line between them (as in two roundabouts).'''
    grid = parking_one(parking_width_m=parking_width_m, parking_center=parking_center, save=False, resolution=resolution)

    # add circular obstacles
    add_circular_obstacle(
        grid,
        center_m=(2.0, 4.45),
        radius_m=roundabout_radius_m,
        resolution=resolution
    )
    add_circular_obstacle(
        grid,
        center_m=(2.0, 1.55),
        radius_m=roundabout_radius_m,
        resolution=resolution
    )

    # add line between circular obstacles
    add_rectangular_obstacle(
        grid,
        bottom_left_m=(1.95, 1.55),
        top_right_m=(2.05, 4.45),
        resolution=resolution
    )   

    # Save the map and YAML configuration
    if save:
        save_map_and_yaml(grid, "parking_two", resolution)
    else:
        return grid

def intersection(road_width_m=0.5, save=True, resolution=0.05):
    '''Generates a roundabout intersection scenario.'''
    grid = generate_occupied_map(
        filename_prefix="custom_map",
        map_width_m=3.0,
        map_height_m=6.0,
        resolution=resolution
    )

    map_center = (1.5, 3.0)

    # add roads leading to/from the center
    add_rectangular_road(
        grid,
        bottom_left_m=(0.0, map_center[1] - road_width_m / 2),
        top_right_m=(map_center[0], map_center[1] + road_width_m / 2),
        resolution=resolution
    )
    add_rectangular_road(
        grid,
        bottom_left_m=(map_center[0], map_center[1] - road_width_m / 2),
        top_right_m=(3.0, map_center[1] + road_width_m / 2),
        resolution=resolution
    )
    add_rectangular_road(
        grid,
        bottom_left_m=(map_center[0] - road_width_m / 2, map_center[1]),
        top_right_m=(map_center[0] + road_width_m / 2, 6.0),
        resolution=resolution
    )
    add_rectangular_road(
        grid,
        bottom_left_m=(map_center[0] - road_width_m / 2, 0.0),
        top_right_m=(map_center[0] + road_width_m / 2, map_center[1]),
        resolution=resolution
    )


    # Save the map and YAML configuration
    if save:
        save_map_and_yaml(grid, "intersection", resolution)
    else:
        return grid

def intersection_roundabout(roundabout_outer_radius_m=1.0, roundabout_inner_radius_m=0.2, road_width_m=0.5, resolution=0.05):
    '''Generates a roundabout intersection scenario.'''
    grid = intersection(road_width_m=road_width_m, save=False, resolution=resolution)

    map_center = (1.5, 3.0)

    # add circular free space (roundabout)
    add_circular_free_space(
        grid,
        center_m=map_center,
        radius_m=roundabout_outer_radius_m,
        resolution=resolution
    )

    # add circular obstacle to the roundabout center
    add_circular_obstacle(
        grid,
        center_m=map_center,
        radius_m=roundabout_inner_radius_m,
        resolution=resolution
    )


    # Save the map and YAML configuration
    save_map_and_yaml(grid, "intersection_roundabout", resolution)

if __name__ == "__main__":
    resolution = 0.05
    parking_one(parking_width_m=0.7, resolution=resolution)
    parking_two(parking_width_m=0.7, roundabout_radius_m=0.2, resolution=resolution)
    intersection(road_width_m=0.8, resolution=resolution)
    intersection_roundabout(roundabout_outer_radius_m=1.0, roundabout_inner_radius_m=0.2, road_width_m=0.6, resolution=resolution)