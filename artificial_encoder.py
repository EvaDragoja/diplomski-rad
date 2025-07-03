import rawpy
import numpy as np
import glob
import os
import subprocess
import re
import argparse
import cv2


def demosaick(bayer):
    dim_x, dim_y = bayer.shape
    parity = 0
    r_dem = np.zeros((dim_x, dim_y))
    g_dem = np.zeros((dim_x, dim_y))
    b_dem = np.zeros((dim_x, dim_y))
    for x in range(0, dim_x, 2):
        for y in range(0, dim_y, 2):
            r_dem[x][y] = bayer[x][y]
            r_dem[x + 1][y] = bayer[x][y]
            r_dem[x][y + 1] = bayer[x][y]
            r_dem[x + 1][y + 1] = bayer[x][y]
    for x in range(0, dim_x, 2):
        for y in range(0, dim_y):
            parity = 1 - parity
            g_dem[x][y] = bayer[x + parity][y]
            g_dem[x + 1][y] = bayer[x + parity][y]
    for x in range(0, dim_x, 2):
        for y in range(0, dim_y, 2):
            b_dem[x][y] = bayer[x + 1][y + 1]
            b_dem[x + 1][y] = bayer[x + 1][y + 1]
            b_dem[x][y + 1] = bayer[x + 1][y + 1]
            b_dem[x + 1][y + 1] = bayer[x + 1][y + 1]
    return r_dem, g_dem, b_dem


def generate_raw_for_artificial(input_folder, output_file):
    # Initialize YUV file
    yuv_file = open(output_file, 'wb')

    # Loop through DNG files in the folder
    for filename in glob.glob(os.path.join(input_folder, '*.DNG')):
        with rawpy.imread(filename) as raw:

            bayer = raw.raw_image

            # Dimensions
            dim_x, dim_y = bayer.shape

            bayer = bayer.astype(int)

            img_r, img_g, img_b = demosaick(bayer)

            # White balance
            img_g = img_g / (img_g.mean() / img_r.mean())
            img_b = img_b / (img_b.mean() / img_r.mean())

            # Clip outliers
            min_max = min(max(map(max, img_r)), max(map(max, img_g)), max(map(max, img_b)))

            img_r = img_r.clip(max=min_max)
            img_g = img_g.clip(max=min_max)
            img_b = img_b.clip(max=min_max)

            # Scale to 0.0 - 1.0 range
            img_r = img_r.astype(float) / 4095
            img_g = img_g.astype(float) / 4095
            img_b = img_b.astype(float) / 4095

            # Gamma correction
            img_r = img_r ** (1 / 2.2)
            img_g = img_g ** (1 / 2.2)
            img_b = img_b ** (1 / 2.2)

            img_g = (img_g * 255).astype(int)
            img_b = (img_b * 255).astype(int)
            img_r = (img_r * 255).astype(int)

            # Y (g) component
            for i in range(0, dim_x):
                for j in range(0, dim_y):
                    yuv_file.write(int(img_r[i][j]).to_bytes(1, byteorder='big'))
                    yuv_file.write(int(img_g[i][j]).to_bytes(1, byteorder='big'))
                    yuv_file.write(int(img_b[i][j]).to_bytes(1, byteorder='big'))

    # Close YUV file
    yuv_file.close()


def extract_channels(bayer):
    # Assuming bayer is a 2D numpy array with the RGGB pattern
    # Red channel: Every other pixel in every other row starting from [0, 0]
    r = bayer[0::2, 0::2]
    # Green channel: A mix of every other pixel starting from [0, 1] and [1, 0]
    g1 = bayer[0::2, 1::2]  # Even rows, odd columns
    g2 = bayer[1::2, 0::2]  # Odd rows, even columns
    g = np.zeros((bayer.shape[0] // 2, bayer.shape[1]))  # Preparing green channel
    g[:, 0::2] = g2  # Filling in from g2
    g[:, 1::2] = g1  # Filling in from g1
    # Blue channel: Every other pixel in every other row starting from [1, 1]
    b = bayer[1::2, 1::2]

    return r, g, b

def convert_dng_to_artificial_yuv(input_folder, output_file):
    yuv_file = open(output_file, 'wb')

    for filename in glob.glob(os.path.join(input_folder, '*.DNG')):
        with rawpy.imread(filename) as raw:

            bayer = raw.raw_image

            # Dimensions
            dim_x, dim_y = bayer.shape

            bayer = bayer.astype(int)

            # Extract G, R, and B channels directly from RGGB
            img_r, img_g, img_b = extract_channels(bayer)

            # White balance
            img_g = img_g / (img_g.mean() / img_r.mean())
            img_b = img_b / (img_b.mean() / img_r.mean())

            # Clip outliers
            min_max = min(max(map(max, img_r)), max(map(max, img_g)), max(map(max, img_b)))

            img_r = img_r.clip(max=min_max)
            img_g = img_g.clip(max=min_max)
            img_b = img_b.clip(max=min_max)

            # Scale to 0.0 - 1.0 range
            img_r = img_r.astype(float) / 4095
            img_g = img_g.astype(float) / 4095
            img_b = img_b.astype(float) / 4095

            # Gamma correction
            img_r = img_r ** (1 / 2.2)
            img_g = img_g ** (1 / 2.2)
            img_b = img_b ** (1 / 2.2)

            img_g = (img_g * 255).astype(int)
            img_b = (img_b * 255).astype(int)
            img_r = (img_r * 255).astype(int)

            # Write pixels in YUV422P format (planar)
            # Y (g) component
            for i in range(0, dim_x // 2):
                for j in range(0, dim_y):
                    yuv_file.write(int(img_g[i][j]).to_bytes(1, byteorder='big'))

            # U (b) component
            for i in range(0, dim_x // 2):
                for j in range(0, dim_y // 2):
                    yuv_file.write(int(img_b[i][j]).to_bytes(1, byteorder='big'))

            # V (r) component
            for i in range(0, dim_x // 2):
                for j in range(0, dim_y // 2):
                    yuv_file.write(int(img_r[i][j]).to_bytes(1, byteorder='big'))

    yuv_file.close()


def encode_yuv_with_ffmpeg(input_file, output_file, width, height, log_file, yuv_format, crf):

    expected_frame_size = int(width) * int(height) * 3  # for yuv444p (1B per channel)
    file_size = os.path.getsize(input_file)
    total_frames = file_size // expected_frame_size
    print(f"YUV file size: {file_size} bytes, Frame size: {expected_frame_size}, Total frames: {total_frames}")

    command = [
        "ffmpeg",
        "-s", f"{width}x{height}",
        "-pixel_format", f"{yuv_format}",
        "-i", input_file,
        "-c:v", "libx265",
        "-crf", crf,
        "-y",
        "-x265-params", "psnr=1",
        output_file
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    # Capture PSNR
    psnr_pattern = r"Global PSNR: ([\d\.]+)"
    psnr_match = re.search(psnr_pattern, result.stderr)
    psnr = psnr_match.group(1) if psnr_match else "N/A"


    # Capture encoding time and average QP
    time_pattern = r"encoded \d+ frames in ([\d\.]+)s \(([\d\.]+) fps\), ([\d\.]+) kb/s, Avg QP:([\d\.]+)"
    time_match = re.search(time_pattern, result.stderr)
    encode_time = time_match.group(1) if time_match else "N/A"
    avg_qp = time_match.group(4) if time_match else "N/A"

    # Compute compression ratio
    raw_size = os.path.getsize(input_file)
    encoded_size = os.path.getsize(output_file)
    compression_ratio = raw_size / encoded_size

    # Write details to log file
    with open(log_file, 'w') as f:
        f.write(f"PSNR: {psnr}\n")
        f.write(f"Encoding Time (s): {encode_time}\n")
        f.write(f"Average QP: {avg_qp}\n")
        f.write(f"Compression Ratio: {compression_ratio:.2f}\n")

    print(f"Metrics written to {log_file}")


def play_video(video_path, display_width=960):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error opening video file.")
        return

    # Original size of the video
    original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale_factor = display_width / original_width
    display_height = int(original_height * scale_factor)

    while True:
        ret, frame = video.read()

        if ret:
            # Change frame size
            frame_resized = cv2.resize(frame, (display_width, display_height))

            cv2.imshow('Final Video', frame_resized)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            print("Video ended. Close the window to exit.")
            while True:
                if cv2.getWindowProperty('Final Video', cv2.WND_PROP_VISIBLE) < 1:
                    break
                cv2.waitKey(100)
            break

    video.release()
    cv2.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Video decoder script.")
    parser.add_argument('input_file', type=str, help='Input video file path')
    parser.add_argument('output_file', type=str, help='Output file path')
    parser.add_argument('-r', '--resolution', type=str, help='Resolution in WIDTHxHEIGHT format', required=True)

    args = parser.parse_args()

    # Argument reading
    input_file = args.input_file
    output_file = args.output_file
    resolution = args.resolution

    # Parsing the resolution
    try:
        width, height = map(int, resolution.lower().split('x'))
    except ValueError:
        print("Resolution format is incorrect. Please use WIDTHxHEIGHT format, e.g., 1920x1080.")
        return

    return input_file, output_file, width, height



if __name__ == "__main__":
    input_folder, encoded_output_artificial, width, height = parse_arguments()
    crf = "45"
    
    yuv_output = 'raw.yuv'
    artificial_yuv_output = "artificial.yuv"
    log_filename_artificial = "artificial_encoding_metrics.txt"

    generate_raw_for_artificial(input_folder, yuv_output)
    convert_dng_to_artificial_yuv(input_folder, artificial_yuv_output)
    encode_yuv_with_ffmpeg(artificial_yuv_output, encoded_output_artificial, width, int(height) // 2, log_filename_artificial, "yuv422p", crf)
    # Show artificial_encoded_video.mp4
    play_video(encoded_output_artificial)
