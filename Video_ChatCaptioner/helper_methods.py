import os
import csv
import math
import subprocess
from PIL import Image, ImageDraw, ImageFont



def convert_frames_to_mp4(folder_name):
    for folder in os.listdir(folder_name + "/frames"):
        cmd = (
            "ffmpeg -framerate 30 -pattern_type glob -i '{0}/frames/{1}/*.jpg' "
            + "-c:v libx264 -pix_fmt yuv420p {0}/videos/{1}.mp4"
        ).format(folder_name, folder)
        os.system(cmd)


# tken from https://stackoverflow.com/questions/3844430/how-to-get-the-duration-of-a-video-in-python
def get_length(filename):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            filename,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return math.ceil(float(result.stdout))


# taken from https://www.pythontutorial.net/python-basics/python-write-csv-file/
def wirte_stc_csv_file(folder_name, csv_file_name):
    header = ["videoid", "contentUrl", "duration", "page_dir", "name"]
    data = []
    video_folder_name = folder_name + "/videos"
    for folder in os.listdir(video_folder_name):
        full_path = video_folder_name + "/" + folder
        video_length = get_length(full_path)
        data.append(
            [os.path.splitext(folder)[0], full_path, "PT00H00M" + str(video_length) + "S", full_path, "Test"]
        )
    with open(folder_name + '/' + csv_file_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)

def save_image_with_title(pil_image, title_text, output_path, title_font_size=40, title_position=(10, 10)):
    # Create an ImageDraw object
    draw = ImageDraw.Draw(pil_image)

    # Load a font for the title
    try:
        font = ImageFont.truetype("arial.ttf", title_font_size)
    except IOError:
        font = ImageFont.load_default()
        print("Warning: Arial font not found. Using the default font instead.")

    # Define the position where the title will be placed
    x, y = title_position

    # Add the title text to the image
    draw.text((x, y), title_text, fill="white", font=font)

    # Save the image with the title
    pil_image.save(output_path)
    print('saved + ' + output_path)


if __name__ == "__main__":
    # convert_frames_to_mp4("Video_ChatCaptioner/stc/shanghai_tech_dataset/testing")
    wirte_stc_csv_file(folder_name="Video_ChatCaptioner/stc/shanghai_tech_dataset/testing", csv_file_name="videos.csv")
    print("help")
