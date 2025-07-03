import rawpy
import numpy as np
import glob
import os
import subprocess
import re
import argparse
import cv2



def rgb_yuv(img_r, img_g, img_b):
    # Assuming img_r, img_g, and img_b are NumPy arrays of the same shape.
    Y = 0.299 * img_r + 0.587 * img_g + 0.114 * img_b
    U = -0.147 * img_r - 0.289 * img_g + 0.436 * img_b
    V = 0.615 * img_r - 0.515 * img_g - 0.100 * img_b

    # Normalize and convert to integer values as before, if necessary.
    Y = (Y * 255).astype(int)
    U = ((U + 0.5) * 255).astype(int)
    V = ((V + 0.5) * 255).astype(int)

    return Y, U, V

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


def generate_raw_for_normal(input_folder, output_file):
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

            y, u, v = rgb_yuv(img_r, img_g, img_b)


            # Y (g) component
            for i in range(0, dim_x):
                for j in range(0, dim_y):
                    yuv_file.write(int(y[i][j]).to_bytes(1, byteorder='big'))
            # U (b) component
            for i in range(0, dim_x):
                for j in range(0, dim_y):
                    yuv_file.write(int(u[i][j]).to_bytes(1, byteorder='big'))

            # V (r) component
            for i in range(0, dim_x):
                for j in range(0, dim_y):
                    yuv_file.write(int(v[i][j]).to_bytes(1, byteorder='big'))

    # Close YUV file
    yuv_file.close()

def encode_yuv_with_ffmpeg(input_file, output_file, width, height, log_file, yuv_format, crf):
    #if roi_x != "":
    #    command = [
    #        "ffmpeg -f "{width}x{height} -pixel_format  yuv_format -i gdfsfdgfdinput_file, #final_output
    #        "-vf", f"addroi=x={roi_x}:y={roi_y}:w={roi_x_dim}:h={roi_y_dim}:qoffset={roi_crf}",
     #       "-c:v", "libx265",
      #      "-crf", crf,
      #      "-y",  # overwrite output file if it exists
      #      "-x265-params", "psnr=1",
      #      output_file
      #  ]
    #else:
    command = [
        "ffmpeg",
        "-s", f"{width}x{height}",
        "-pixel_format", f"{yuv_format}",
        "-i", input_file,
        "-c:v", "libx265",
        "-crf", crf,
        "-y",  # overwrite output file if it exists
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

def convert_normal_mp4_to_yuv(encoded_output_normal, encoded_yuv_output_normal, yuv_format):
    subprocess.run([
        "ffmpeg",
        "-i", f"{encoded_output_normal}",
        "-c:v", "rawvideo",
        "-pix_fmt", f"{yuv_format}",
        "-y",
        f"{encoded_yuv_output_normal}"
    ])    

def decode_normal_yuv_to_rgb(decoded_yuv_file, final_rgb_output_file, input_folder, width, height):

    # Odredi broj frameova na osnovu broja .DNG fajlova
    filenames = [f for f in glob.glob(os.path.join(input_folder, '*.DNG'))]
    total_frames = len(filenames)

    with open(decoded_yuv_file, 'rb') as yuv_file, open(final_rgb_output_file, 'wb') as rgb_file:
        for frame in range(total_frames):
            print(f"Decoding frame {frame + 1}/{total_frames}...")

            # Pročitaj YUV444 frame
            y = np.frombuffer(yuv_file.read(width * height), dtype=np.uint8).reshape((height, width))
            u = np.frombuffer(yuv_file.read(width * height), dtype=np.uint8).reshape((height, width))
            v = np.frombuffer(yuv_file.read(width * height), dtype=np.uint8).reshape((height, width))

            # YUV -> RGB konverzija
            y = y.astype(float)
            u = u.astype(float) - 128
            v = v.astype(float) - 128

            r = y + 1.402 * v
            g = y - 0.344136 * u - 0.714136 * v
            b = y + 1.772 * u

            r = np.clip(r, 0, 255).astype(np.uint8)
            g = np.clip(g, 0, 255).astype(np.uint8)
            b = np.clip(b, 0, 255).astype(np.uint8)

            rgb_frame = np.stack((r, g, b), axis=2)
            rgb_file.write(rgb_frame.tobytes())

def yuv_to_mp4(input_yuv, output_mp4, width, height):
    command = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}',
        '-i', input_yuv,
        '-c:v', 'libx264',
        '-y',
        output_mp4
    ]
    subprocess.run(command, check=True)

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
    input_folder, final_normal_rgb_output, width, height = parse_arguments()
    crf = "45"

    yuv_output_normal = 'raw_n.yuv'
    encoded_output = 'normal_encoded_video.mp4'
    log_filename = "normal_encoding_metrics.txt"
    encoded_yuv_normal = "normal_encoded_video.yuv"
    final_normal_yuv_output = "final_normal_output.yuv"


    generate_raw_for_normal(input_folder, yuv_output_normal)
    encode_yuv_with_ffmpeg(yuv_output_normal, encoded_output, width, height, log_filename, "yuv444p", crf)
    convert_normal_mp4_to_yuv(encoded_output, encoded_yuv_normal, "yuv444p")
    decode_normal_yuv_to_rgb(encoded_yuv_normal, final_normal_yuv_output, input_folder, width, height)
    yuv_to_mp4(final_normal_yuv_output, final_normal_rgb_output, width, height)
