# TODO add variable for angle to start from
import os
import glob
import argparse
from tqdm import tqdm

import cv2
import numpy as np
from equilib import Equi2Pers

def load_image(equi_img_path):
    equi_img = cv2.imread(equi_img_path)
    equi_img = cv2.cvtColor(equi_img, cv2.COLOR_BGR2RGB)
    equi_img = np.transpose(equi_img, (2, 0, 1))
    return equi_img

def calculate_dimensions(width, aspect_ratio):
    return int(width / aspect_ratio)

def generate_perspective_view(equi_img, rots, height, width, fov_x):
    equi2pers = Equi2Pers(height=height, width=width, fov_x=fov_x, mode="bilinear")
    pers_img = equi2pers(equi=equi_img, rots=rots)
    return pers_img

def create_animation(input_path, output_dir, fps=30, width=640, height=None, 
                    aspect_ratio=16/9, fov=90.0, num_frames=72, gif_size=None):

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear existing files in output directory
    for file in glob.glob(os.path.join(output_dir, "frame_*.png")):
        os.remove(file)

    # Calculate dimensions if height is not specified
    if height is None:
        height = calculate_dimensions(width, aspect_ratio)
    else:
        # If height is specified, recalculate width to maintain aspect ratio
        width = int(height * aspect_ratio)

    print(f"Output dimensions: {width}x{height} (aspect ratio: {aspect_ratio:.2f})")

    # Calculate GIF dimensions if specified
    if gif_size is not None:
        gif_height = gif_size
        gif_width = int(gif_height * aspect_ratio)
    else:
        gif_width = width
        gif_height = height

    # Parameters
    rotation_steps = np.linspace(0, 360, num_frames, endpoint=False)

    # Load equirectangular image
    equi_img = load_image(input_path)

    # Generate frames
    print("Generating frames...")
    for i, angle in enumerate(tqdm(rotation_steps)):
        rots = {
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': np.radians(angle)
        }

        # Generate perspective view
        pers_img = generate_perspective_view(equi_img, rots, height, width, fov)
        
        # Convert and save frame
        frame = np.transpose(pers_img, (1, 2, 0))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frame_path = os.path.join(output_dir, f"frame_{i:03d}.png")
        cv2.imwrite(frame_path, frame)

    # Create video using ffmpeg
    create_video(output_dir, fps, width, height, gif_width, gif_height)

def create_video(output_dir, fps, width, height, gif_width, gif_height):
    input_pattern = os.path.join(output_dir, 'frame_%03d.png')
    mp4_output = os.path.join(output_dir, 'animation.mp4')
    gif_output = os.path.join(output_dir, 'animation.gif')
    palette_path = os.path.join(output_dir, 'palette.png')

    # Create MP4
    mp4_cmd = (
        f'ffmpeg -y -framerate {fps} -i "{input_pattern}" '
        f'-c:v libx264 -pix_fmt yuv420p -crf 23 '
        f'-vf "scale={width}:{height}" "{mp4_output}"'
    )
    print("\nCreating MP4...")
    os.system(mp4_cmd)

    # Generate optimized palette
    palette_cmd = (
        f'ffmpeg -y -i "{mp4_output}" -vf '
        f'"fps={fps},scale={gif_width}:{gif_height}:flags=lanczos,'
        f'palettegen=max_colors=128:stats_mode=single" '
        f'"{palette_path}"'
    )
    print("\nGenerating color palette...")
    os.system(palette_cmd)

    # Create optimized GIF
    gif_cmd = (
        f'ffmpeg -y -i "{mp4_output}" -i "{palette_path}" -lavfi '
        f'"fps={fps},scale={gif_width}:{gif_height}:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" '
        f'-loop 0 "{gif_output}"'
    )
    print("\nCreating optimized GIF...")
    os.system(gif_cmd)

    # Clean up palette file
    if os.path.exists(palette_path):
        os.remove(palette_path)

def main():
    parser = argparse.ArgumentParser(description='Generate animated rotation of an equirectangular image')
    parser.add_argument('input_path', type=str, 
                        help='Path to the input equirectangular image')
    parser.add_argument('--output_dir', type=str, default='animation_frames',
                        help='Directory to store frames (default: animation_frames)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for animation (default: 10)')
    parser.add_argument('--width', type=int, default=320,
                        help='Width of the output frames (default: 320)')
    parser.add_argument('--height', type=int, default=None,
                        help='Height of the output frames (default: calculated from aspect ratio)')
    parser.add_argument('--aspect_ratio', type=float, default=1,
                        help='Aspect ratio (width/height) for the output (default: 1)')
    parser.add_argument('--fov', type=float, default=80.0,
                        help='Field of view in degrees (default: 90.0)')
    parser.add_argument('--num_frames', type=int, default=90,
                        help='Number of frames in the animation (default: 90)')
    parser.add_argument('--gif_size', type=int, default=200,
                        help='Height of the GIF in pixels (default: 200, width will be calculated from aspect ratio)')

    args = parser.parse_args()

    # Validate input file
    if not os.path.isfile(args.input_path):
        print(f"Error: Input file '{args.input_path}' does not exist.")
        return

    # Validate numeric arguments
    if args.width <= 0 or (args.height is not None and args.height <= 0):
        print("Error: Width and height must be positive numbers.")
        return

    if args.aspect_ratio <= 0:
        print("Error: Aspect ratio must be positive.")
        return

    if args.fov <= 0 or args.fov >= 180:
        print("Error: FOV must be between 0 and 180 degrees.")
        return

    if args.num_frames < 1:
        print("Error: Number of frames must be at least 1.")
        return

    create_animation(
        args.input_path,
        args.output_dir,
        args.fps,
        args.width,
        args.height,
        args.aspect_ratio,
        args.fov,
        args.num_frames,
        args.gif_size
    )
    print("\nAnimation creation completed!")

if __name__ == "__main__":
    main()