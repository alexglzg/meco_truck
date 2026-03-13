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


def generate_racetrack(track_width_m=0.8, resolution=0.05):
    """
    Generates an oval racetrack layout.
    The track fills most of the 3x6m space.
    """
    # Start with a fully occupied map
    grid = generate_occupied_map(
        filename_prefix="racetrack",
        map_width_m=3.0,
        map_height_m=6.0,
        resolution=resolution
    )

    # Dimensions for the outer boundary (white area)
    outer_radius = 1.3  # meters
    inner_radius = outer_radius - track_width_m
    center_top = (1.5, 4.5)
    center_bottom = (1.5, 1.5)

    # 1. Create the outer "pill" shape (Free Space)
    add_circular_free_space(grid, center_top, outer_radius, resolution)
    add_circular_free_space(grid, center_bottom, outer_radius, resolution)
    add_rectangular_road(grid, (1.5 - outer_radius, 1.5), (1.5 + outer_radius, 4.5), resolution)

    # 2. Create the inner "infield" (Occupied Space)
    add_circular_obstacle(grid, center_top, inner_radius, resolution)
    add_circular_obstacle(grid, center_bottom, inner_radius, resolution)
    add_rectangular_obstacle(grid, (1.5 - inner_radius, 1.5), (1.5 + inner_radius, 4.5), resolution)

    save_map_and_yaml(grid, "racetrack", resolution)

def generate_slalom(lane_width_m=1.0, resolution=0.05):
    """
    Generates a zig-zag slalom path.
    """
    grid = generate_occupied_map(
        filename_prefix="slalom",
        map_width_m=3.0,
        map_height_m=6.0,
        resolution=resolution
    )

    # Define path points for a 'Z' or 'S' shape
    # Bottom horizontal
    add_rectangular_road(grid, (0.5, 0.5), (2.5, 0.5 + lane_width_m), resolution)
    # Diagonal/Vertical connector
    add_rectangular_road(grid, (2.5 - lane_width_m, 0.5), (2.5, 5.5), resolution)
    # Top horizontal
    add_rectangular_road(grid, (0.5, 5.5 - lane_width_m), (2.5, 5.5), resolution)

    # Add a few circular "pylons" in the middle of the lanes
    add_circular_obstacle(grid, (1.5, 3.0), 0.2, resolution)

    save_map_and_yaml(grid, "slalom", resolution)

def generate_figure_8(track_width_m=0.8, resolution=0.05):
    """
    Generates a Figure-8 layout.
    Two loops merge in the center to create a dynamic intersection.
    """
    grid = generate_occupied_map(
        filename_prefix="figure_8",
        map_width_m=3.0,
        map_height_m=6.0,
        resolution=resolution
    )

    outer_radius = 1.3
    inner_radius = outer_radius - track_width_m
    
    # Centers for the top and bottom loops
    center_top = (1.5, 4.2)
    center_bottom = (1.5, 1.8)

    # 1. Draw outer boundary (Free Space)
    # The overlap in the middle creates the crossover junction
    add_circular_free_space(grid, center_top, outer_radius, resolution)
    add_circular_free_space(grid, center_bottom, outer_radius, resolution)

    # 2. Draw inner islands (Occupied Space)
    add_circular_obstacle(grid, center_top, inner_radius, resolution)
    add_circular_obstacle(grid, center_bottom, inner_radius, resolution)

    save_map_and_yaml(grid, "figure_8", resolution)

def generate_grid_world(lane_width_m=0.7, resolution=0.05):
    """
    Generates a Manhattan-style grid with 2 vertical and 3 horizontal lanes.
    Great for testing global pathfinding algorithms.
    """
    grid = generate_occupied_map(
        filename_prefix="grid_world",
        map_width_m=3.0,
        map_height_m=6.0,
        resolution=resolution
    )

    # Two vertical lanes
    add_rectangular_road(grid, (0.5, 0.0), (0.5 + lane_width_m, 6.0), resolution)
    add_rectangular_road(grid, (3.0 - 0.5 - lane_width_m, 0.0), (2.5, 6.0), resolution)

    # Three horizontal crossing lanes
    # Bottom cross
    add_rectangular_road(grid, (0.0, 1.0), (3.0, 1.0 + lane_width_m), resolution)
    # Middle cross
    add_rectangular_road(grid, (0.0, 3.0 - (lane_width_m / 2)), (3.0, 3.0 + (lane_width_m / 2)), resolution)
    # Top cross
    add_rectangular_road(grid, (0.0, 5.0 - lane_width_m), (3.0, 5.0), resolution)

    save_map_and_yaml(grid, "grid_world", resolution)

def generate_serpentine(lane_width_m=0.9, resolution=0.05):
    """
    Generates a snaking S-curve track with sharp hairpin turns.
    """
    grid = generate_occupied_map(
        filename_prefix="serpentine",
        map_width_m=3.0,
        map_height_m=6.0,
        resolution=resolution
    )

    # Segment 1: Enter bottom-left, go up
    add_rectangular_road(grid, (0.2, 0.0), (0.2 + lane_width_m, 2.0), resolution)
    
    # Segment 2: Cross to the right
    add_rectangular_road(grid, (0.2, 2.0 - lane_width_m), (2.8, 2.0), resolution)
    
    # Segment 3: Go up on the right side
    add_rectangular_road(grid, (2.8 - lane_width_m, 2.0 - lane_width_m), (2.8, 4.0), resolution)
    
    # Segment 4: Cross back to the left
    add_rectangular_road(grid, (0.2, 4.0 - lane_width_m), (2.8, 4.0), resolution)
    
    # Segment 5: Exit top-left
    add_rectangular_road(grid, (0.2, 4.0 - lane_width_m), (0.2 + lane_width_m, 6.0), resolution)

    save_map_and_yaml(grid, "serpentine", resolution)

def add_path_free_space(grid, waypoints_m, width_m, resolution):
    """
    Draws a continuous path connecting waypoints.
    Uses point-to-line-segment distance to create perfectly rounded corners.
    """
    height_px, width_px = grid.shape
    
    # Create coordinate grids
    y_indices, x_indices = np.ogrid[:height_px, :width_px]
    x_m = x_indices * resolution
    # Map Y from bottom-left (user space) to top-left (image space)
    y_m = (height_px - 1 - y_indices) * resolution
    
    # Initialize with infinite distance
    min_dist = np.full((height_px, width_px), np.inf)
    
    # Calculate shortest distance to any line segment in the path
    for i in range(len(waypoints_m) - 1):
        x0, y0 = waypoints_m[i]
        x1, y1 = waypoints_m[i+1]
        
        dx = x1 - x0
        dy = y1 - y0
        l2 = dx**2 + dy**2
        
        if l2 == 0:
            dist = np.sqrt((x_m - x0)**2 + (y_m - y0)**2)
        else:
            t = ((x_m - x0) * dx + (y_m - y0) * dy) / l2
            t = np.clip(t, 0.0, 1.0)
            proj_x = x0 + t * dx
            proj_y = y0 + t * dy
            dist = np.sqrt((x_m - proj_x)**2 + (y_m - proj_y)**2)
            
        min_dist = np.minimum(min_dist, dist)
        
    # Set pixels within the track width to free space (255/White)
    grid[min_dist <= (width_m / 2.0)] = 255
    return grid


def generate_mini_spa(track_width_m=0.9, resolution=0.05):
    """
    Generates a 3x6m approximation of the Spa-Francorchamps circuit.
    Designed with sweeping curves for Ackermann steering robots.
    """
    grid = generate_occupied_map(
        filename_prefix="mini_spa",
        map_width_m=3.0,
        map_height_m=6.0,
        resolution=resolution
    )

    # Waypoints mapping the iconic corners (X, Y in meters)
    spa_waypoints = [
        (1.5, 1.0),  # Start/Finish Straight
        (2.2, 0.6),  # Turn 1: La Source (tight hairpin right)
        (2.6, 1.0),  # Exiting La Source
        (2.0, 2.2),  # Plunge down to Eau Rouge
        (1.4, 2.6),  # Eau Rouge (left flick)
        (1.0, 3.0),  # Raidillon (right flick and uphill)
        (0.6, 4.5),  # Kemmel Straight (flat out!)
        (0.8, 5.4),  # Les Combes entry
        (1.4, 5.6),  # Les Combes / Malmedy chicane
        (2.2, 5.4),  # Bruxelles / Pouhon (sweeping double apex)
        (2.6, 4.8),  # Pouhon exit
        (2.6, 3.0),  # Blanchimont (fast return straight)
        (2.2, 1.8),  # Bus Stop chicane entry (hard braking)
        (1.5, 1.4),  # Bus Stop chicane exit
        (1.5, 1.0)   # Close the loop at Start/Finish
    ]

    add_path_free_space(grid, spa_waypoints, track_width_m, resolution)
    save_map_and_yaml(grid, "mini_spa", resolution)

def generate_irregular_figure_8(track_width_m=0.85, resolution=0.05):
    """
    Generates a smooth, asymmetrical figure-8 track.
    Perfect for Ackermann steering as it features sweeping, non-uniform curves.
    """
    grid = generate_occupied_map(
        filename_prefix="irregular_8",
        map_width_m=3.0,
        map_height_m=6.0,
        resolution=resolution
    )

    # A series of waypoints creating two asymmetrical lobes.
    # The vectors through the center (1.5, 3.0) are aligned for a smooth crossover.
    waypoints = [
        (1.5, 3.0),  # Center crossover (start)
        
        # --- Top Lobe (Taller, leans slightly left at the peak) ---
        (2.1, 3.6),  # Exit center going up-right
        (2.5, 4.5),  # Sweep wide right
        (2.0, 5.5),  # Tighten toward the top
        (1.0, 5.5),  # Flat top section
        (0.5, 4.5),  # Drop down the left side
        (0.9, 3.6),  # Approach center going down-right
        
        (1.5, 3.0),  # Center crossover
        
        # --- Bottom Lobe (Shorter, wider, tighter bottom corners) ---
        (2.1, 2.4),  # Exit center going down-right
        (2.6, 1.5),  # Sweep wide right
        (2.2, 0.5),  # Tight bottom right corner
        (1.2, 0.4),  # Flat bottom section
        (0.4, 1.2),  # Tight bottom left corner
        (0.9, 2.4),  # Approach center going up-right
        
        (1.5, 3.0)   # Close the loop at center
    ]

    # Requires the add_path_free_space function from the previous step!
    add_path_free_space(grid, waypoints, track_width_m, resolution)
    save_map_and_yaml(grid, "irregular_8", resolution)

if __name__ == "__main__":
    resolution = 0.05
    # parking_one(parking_width_m=0.7, resolution=resolution)
    # parking_two(parking_width_m=0.7, roundabout_radius_m=0.2, resolution=resolution)
    # intersection(road_width_m=0.8, resolution=resolution)
    # intersection_roundabout(roundabout_outer_radius_m=1.0, roundabout_inner_radius_m=0.2, road_width_m=0.6, resolution=resolution)
    # generate_racetrack(track_width_m=0.8, resolution=resolution)
    # generate_slalom(lane_width_m=1.0, resolution=resolution)
    # generate_figure_8(track_width_m=0.8, resolution=resolution)
    # generate_grid_world(lane_width_m=0.7, resolution=resolution)
    # generate_serpentine(lane_width_m=0.9, resolution=resolution)
    # generate_mini_spa(track_width_m=0.9, resolution=resolution)
    generate_irregular_figure_8(track_width_m=0.85, resolution=resolution)