import os
import glob
from tqdm import tqdm
import cv2
import numpy as np
from equilib import Equi2Pers

def ensure_even_dimensions(width, height):
    width = int(width)
    height = int(height)
    return width - (width % 2), height - (height % 2)

def calculate_dimensions(width, aspect_ratio):
    height = int(width / aspect_ratio)
    return ensure_even_dimensions(width, height)

def load_image(equi_img_path):
    equi_img = cv2.imread(equi_img_path)
    equi_img = cv2.cvtColor(equi_img, cv2.COLOR_BGR2RGB)
    return np.transpose(equi_img, (2, 0, 1))

def generate_perspective_view(equi_img, rots, height, width, fov_x):
    equi2pers = Equi2Pers(height=height, width=width, fov_x=fov_x, mode="nearest")
    return equi2pers(equi=equi_img, rots=rots)

def create_video_commands(output_dir, fps, width, height, gif_width, gif_height):
    input_pattern = os.path.join(output_dir, 'frame_%03d.png')
    mp4_output = os.path.join(output_dir, 'animation.mp4')
    gif_output = os.path.join(output_dir, 'animation.gif')
    palette_path = os.path.join(output_dir, 'palette.png')

    commands = {
        'mp4': (
            f'ffmpeg -y -framerate {fps} -i "{input_pattern}" '
            f'-c:v libx264 -pix_fmt yuv420p -crf 23 '
            f'-vf "scale={width}:{height}" "{mp4_output}"'
        ),
        'palette': (
            f'ffmpeg -y -i "{mp4_output}" -vf '
            f'"fps={fps},scale={gif_width}:{gif_height}:flags=lanczos,'
            f'palettegen=max_colors=128:stats_mode=single" '
            f'"{palette_path}"'
        ),
        'gif': (
            f'ffmpeg -y -i "{mp4_output}" -i "{palette_path}" -lavfi '
            f'"fps={fps},scale={gif_width}:{gif_height}:flags=lanczos[x];'
            f'[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" '
            f'-loop 0 "{gif_output}"'
        )
    }
    return commands, palette_path

def cleanup_frames(output_dir):
    """Remove all frame files from the output directory."""
    frame_files = glob.glob(os.path.join(output_dir, "frame_*.png"))
    for file in frame_files:
        os.remove(file)
    print(f"Cleaned up {len(frame_files)} frame files.")

def create_animation(input_path, output_dir='output', fps=30, width=640, height=None, 
         aspect_ratio=16/9, fov=90.0, num_frames=72, gif_size=None, cleanup=False):
    """
    Main function to create animated rotation from equirectangular image.
    
    Args:
        input_path (str): Path to input equirectangular image
        output_dir (str): Directory for output files
        fps (int): Frames per second for animation
        width (int): Width of output frames
        height (int): Height of output frames (optional)
        aspect_ratio (float): Aspect ratio for output
        fov (float): Field of view in degrees
        num_frames (int): Number of frames in animation
        gif_size (int): Height of output GIF in pixels
        cleanup (bool): Whether to remove frame files after creating animation
    """
    
    # Create and clear output directory
    os.makedirs(output_dir, exist_ok=True)
    for file in glob.glob(os.path.join(output_dir, "frame_*.png")):
        os.remove(file)

    # Calculate dimensions
    if height is None:
        width, height = calculate_dimensions(width, aspect_ratio)
    else:
        width = int(height * aspect_ratio)
        width, height = ensure_even_dimensions(width, height)

    # Calculate GIF dimensions
    if gif_size is not None:
        gif_height = gif_size
        gif_width = int(gif_height * aspect_ratio)
        gif_width, gif_height = ensure_even_dimensions(gif_width, gif_height)
    else:
        gif_width, gif_height = width, height

    print(f"Output dimensions: {width}x{height} (aspect ratio: {width/height:.2f})")

    # Load image and generate frames
    equi_img = load_image(input_path)
    rotation_steps = np.linspace(0, 360, num_frames, endpoint=False)

    print("Generating frames...")
    for i, angle in enumerate(tqdm(rotation_steps)):
        rots = {'roll': 0.0, 'pitch': 0.0, 'yaw': np.radians(angle)}
        pers_img = generate_perspective_view(equi_img, rots, height, width, fov)
        
        frame = np.transpose(pers_img, (1, 2, 0))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(os.path.join(output_dir, f"frame_{i:03d}.png"), frame)

    # Create video and GIF
    commands, palette_path = create_video_commands(
        output_dir, fps, width, height, gif_width, gif_height
    )

    print("\nCreating MP4...")
    os.system(commands['mp4'])
    
    print("Generating color palette...")
    os.system(commands['palette'])
    
    print("Creating optimized GIF...")
    os.system(commands['gif'])

    # Cleanup
    if os.path.exists(palette_path):
        os.remove(palette_path)

    # Clean up frame files if requested
    if cleanup:
        cleanup_frames(output_dir)

    print("\nAnimation creation completed!")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate animated rotation of an equirectangular image')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input equirectangular image')
    parser.add_argument('--output_dir', type=str, default='images',
                        help='Directory to store frames (default: images)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for animation (default: 20)')
    parser.add_argument('--width', type=int, default=720,
                        help='Width of the output frames (default: 720)')
    parser.add_argument('--height', type=int, default=None,
                        help='Height of the output frames (default: calculated from aspect ratio)')
    parser.add_argument('--aspect_ratio', type=float, default=9/16,
                        help='Aspect ratio (width/height) for the output (default: 19.5/9)')
    parser.add_argument('--fov', type=float, default=70.0,
                        help='Field of view in degrees (default: 70.0)')
    parser.add_argument('--num_frames', type=int, default=240,
                        help='Number of frames in the animation (default: 180)')
    parser.add_argument('--gif_size', type=int, default=200,
                        help='Height of the GIF in pixels (default: 200)')
    parser.add_argument('--cleanup', action='store_true',
                        help='Remove frame files after creating animation')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.input_path):
        print(f"Error: Input file '{args.input_path}' does not exist.")
        exit(1)
    if args.width <= 0 or (args.height is not None and args.height <= 0):
        print("Error: Width and height must be positive numbers.")
        exit(1)
    if args.aspect_ratio <= 0:
        print("Error: Aspect ratio must be positive.")
        exit(1)
    if args.fov <= 0 or args.fov >= 180:
        print("Error: FOV must be between 0 and 180 degrees.")
        exit(1)
    if args.num_frames < 1:
        print("Error: Number of frames must be at least 1.")
        exit(1)

    # Run main function with parsed arguments
    create_animation(
        input_path=args.input_path,
        output_dir=args.output_dir,
        fps=args.fps,
        width=args.width,
        height=args.height,
        aspect_ratio=args.aspect_ratio,
        fov=args.fov,
        num_frames=args.num_frames,
        gif_size=args.gif_size,
        cleanup=args.cleanup
    )