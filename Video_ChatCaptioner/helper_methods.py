import os
import csv
import math
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from PIL import Image, ImageDraw, ImageFont
from chatcaptioner.video_reader import (
    read_video_sampling,
)
from skimage.io import imread
import shutil



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


def get_testing_numpy_frames(frames_path):
    """returns a list of frames in frames_path

    Args:
        frames_path (str): the path of the list of frames

    Returns:
        PIL.Image list: a list of frames in frames_path 
    """    
    frames = []
    frame_names = os.listdir(frames_path)
    frame_names.sort()
    # we sort the names since the order is important - later on we enumarate over the frames.
    for frame_name in frame_names:
        try:
            frame = imread(frames_path + '/' + frame_name)
            # Convert to the PIL.Image format 
            frames.append(Image.fromarray(np.asarray(frame)))
        except Exception as e:
            print('Failed to get image ' + str(e))
    return frames
            

def move_captions():
    path = 'output/Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/captions'
    for folder in os.listdir(path):
        folder_path = path + '/' + folder
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                source = folder_path + '/' + file
                destination = path + '/' + file
                shutil.move(source, destination)
                if len(os.listdir(folder_path)) == 0:
                    os.rmdir(folder_path)
            

def remove_last_empty_line(file_path):
    with open(file_path, 'r+') as file:
        lines = file.readlines()
        if lines and lines[-1].strip() == '':
            # If the last line is empty, truncate the file to remove it
            file.seek(0, os.SEEK_END)
            file.seek(file.tell() - len(lines[-1]), os.SEEK_SET)
            file.truncate()
    

def append_npys(path):
    npys_concatinated = []
    for npy_file in os.listdir(path):
        npy = np.load(path + '/' + npy_file)
        npys_concatinated = npys_concatinated + npy.tolist()
    return np.array(npys_concatinated)


def calc_auc(predicted_path, gt_path):
    predicted_npys_concatinated = append_npys(predicted_path)
    gt_npys_concatinated = append_npys(gt_path)
    fpr, tpr, _ = metrics.roc_curve(gt_npys_concatinated,  predicted_npys_concatinated)
    auc = metrics.roc_auc_score(gt_npys_concatinated, predicted_npys_concatinated)

    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()
    plt.savefig("dummy_name.png")
    return metrics.roc_auc_score(gt_npys_concatinated, predicted_npys_concatinated)


def assert_no_files_end_with_empty_line(path):
    for file in os.listdir(path):
        with open(path + '/' + file, "r") as f:
            lines = f.readlines()
            if not lines:
                continue
            if lines[-1].strip() == '':
                return False
    return True 
    

if __name__ == "__main__":
    # convert_frames_to_mp4("Video_ChatCaptioner/stc/shanghai_tech_dataset/testing")
    # wirte_stc_csv_file(folder_name="Video_ChatCaptioner/stc/shanghai_tech_dataset/testing", csv_file_name="videos.csv")
    # frames = get_testing_numpy_frames("Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/frames/12_0175")
    # sampled_frames = read_video_sampling(
    #             "Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/videos/12_0175.mp4",
    #             num_frames=100,
    #         )
    # move_captions()
    # folder = 'output/Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/captions'
    # for file in os.listdir(folder):
    #     remove_last_empty_line(folder + '/' + file)
    
    # print(assert_no_files_end_with_empty_line(folder))
    predicted_npys = 'output/Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/npys'
    test_frame_mask = 'Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/test_frame_mask'
    a = calc_auc(predicted_npys, test_frame_mask)
    print("help")
