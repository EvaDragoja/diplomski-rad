import numpy as np
import os
import subprocess
import argparse
import cv2


def convert_artificial_mp4_to_yuv(encoded_output_artificial, encoded_yuv_output_artificial, yuv_format):
    subprocess.run([
        "ffmpeg",
        "-i", f"{encoded_output_artificial}",
        "-c:v", "rawvideo",
        "-pix_fmt", f"{yuv_format}",
        "-y",
        f"{encoded_yuv_output_artificial}"
    ])


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


def calculate_frames(encoded_yuv_artificial, width, height):    
    frame_size = int(width * height * 2)
    total_size = os.path.getsize(encoded_yuv_artificial)
    frame_count = total_size // frame_size

    return frame_count


def upsample(img):
    dim_x, dim_y = img.shape
    img_n = np.zeros((dim_x, dim_y * 2))
    for i in range(dim_x):
        for j in range(0, dim_y * 2, 2):
            img_n[i][j] = img[i][j // 2]
            img_n[i][j + 1] = img[i][j // 2]
    return img_n


def demosaick_wb_gamma(encoded_yuv_artificial, final_artificial_yuv_output, frames, height, width, gamma=2.2):
    yuv_file = open(final_artificial_yuv_output, 'wb')
    video = open(encoded_yuv_artificial, 'rb')

    bayer_rev = np.zeros((height, width))

    for frame in range(frames):
        print("Encoding frame " + str(frame) + "...")

        g = np.zeros((height // 2, width))
        b = np.zeros((height // 2, width // 2))
        r = np.zeros((height // 2, width // 2))

        # G channel
        for i in range(height // 2):
            for j in range(width):
                byte = video.read(1)
                g[i, j] = int.from_bytes(byte, 'big')

        # B channel
        for i in range(height // 2):
            for j in range(width // 2):
                byte = video.read(1)
                b[i, j] = int.from_bytes(byte, 'big')

        # R channel
        for i in range(height // 2):
            for j in range(width // 2):
                byte = video.read(1)
                r[i, j] = int.from_bytes(byte, 'big')

        # Upsample B and R channels
        b_upsampled = upsample(b)
        r_upsampled = upsample(r)

        for x in range(0, height, 2):
            for y in range(0, width, 2):
                bayer_rev[x, y] = r_upsampled[x // 2, y]                # R
                bayer_rev[x, y + 1] = g[x // 2, y + 1]                  # G
                bayer_rev[x + 1, y] = g[x // 2, y]                      # G
                bayer_rev[x + 1, y + 1] = b_upsampled[x // 2, y + 1]    # B

        r_dem, g_dem, b_dem = demosaick(bayer_rev)

        # Normalization
        max_val = max(r_dem.max(), g_dem.max(), b_dem.max())
        if max_val > 255:
            scale = 255.0 / max_val
            r_dem *= scale
            g_dem *= scale
            b_dem *= scale

        # White balance
        r_dem = np.clip(r_dem * 1.1, 0, 255) 
        g_dem = np.clip(g_dem * 1.1, 0, 255)
        b_dem = np.clip(b_dem * 1.3, 0, 255) 

        image_rec = np.zeros((height, width, 3))
        image_rec[:, :, 0] = r_dem
        image_rec[:, :, 1] = g_dem
        image_rec[:, :, 2] = b_dem

        # Gamma correction
        gamma = 1 / 0.6
        image_rec = np.clip(((image_rec / 255.0) ** gamma) * 255, 0, 255)

        image_rec_uint8 = image_rec.astype(np.uint8)
        yuv_file.write(image_rec_uint8.tobytes())

    yuv_file.close()
    video.close()




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



if __name__ == "__main__":
    input_mp4_file, final_artificial_yuv_output, width, height = parse_arguments()
    crf = "45"

    encoded_yuv_artificial = "artificial_encoded_video.yuv"
    final_artificial_mp4_output = "final_artificial_output.mp4"


    #encode_yuv_with_dec265(artificial_yuv_output, encoded_output_artificial)
    convert_artificial_mp4_to_yuv(input_mp4_file, encoded_yuv_artificial, "yuv422p")
    # For calculating number of frames since we don't have input folder with all the .DNG files
    frames = calculate_frames(encoded_yuv_artificial, width, height)
    demosaick_wb_gamma(encoded_yuv_artificial, final_artificial_yuv_output, frames, height, width)
    # Format it into .mp4 file so that it can be played
    yuv_to_mp4(final_artificial_yuv_output, final_artificial_mp4_output, width, height)
    play_video(final_artificial_mp4_output)
