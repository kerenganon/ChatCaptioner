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
import replicate
from pathlib import Path


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
            [
                os.path.splitext(folder)[0],
                full_path,
                "PT00H00M" + str(video_length) + "S",
                full_path,
                "Test",
            ]
        )
    with open(folder_name + "/" + csv_file_name, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(data)


def save_image_with_title(
    pil_image, title_text, output_path, title_font_size=40, title_position=(10, 10)
):
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
    print("saved + " + output_path)


def get_testing_numpy_frames(frames_path):
    """returns a list of frames in frames_path

    Args:
        frames_path (str): the path of the list of frames

    Returns:
        PIL.Image list: a list of frames in frames_path
    """
    frames = []
    frame_names = os.listdir(frames_path)
    # we sort the names since the order is important - later on we enumarate over the frames.
    frame_names.sort()

    for frame_name in frame_names:
        try:
            frame = imread(frames_path + "/" + frame_name)
            # Convert to the PIL.Image format
            frames.append(Image.fromarray(np.asarray(frame)))
        except Exception as e:
            print("Failed to get image " + str(e))
    return frames


def move_captions(path):
    for folder in os.listdir(path):
        folder_path = path + "/" + folder
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                source = folder_path + "/" + file
                destination = path + "/" + file
                shutil.move(source, destination)
                if len(os.listdir(folder_path)) == 0:
                    os.rmdir(folder_path)


def remove_last_empty_line(path):
    with open(path, "r+", encoding="utf-8") as file:
        lines = file.readlines()
        if lines and lines[-1].endswith("\n"):
            # Move the pointer (similar to a cursor in a text editor) to the end of the file
            file.seek(0, os.SEEK_END)

            # This code means the following code skips the very last character in the file -
            # i.e. in the case the last line is null we delete the last line
            # and the penultimate one
            pos = file.tell() - 1

            # Read each character in the file one at a time from the penultimate
            # character going backwards, searching for a newline character
            # If we find a new line, exit the search
            while pos > 0 and file.read(1) != "\n":
                pos -= 1
                file.seek(pos, os.SEEK_SET)

            # So long as we're not at the start of the file, delete all the characters ahead
            # of this position
            if pos > 0:
                file.seek(pos, os.SEEK_SET)
                file.truncate()


# 1. flattens the captions dir
# 2. removes the last line in a text file if it's empty
def fix_txt_files(path):
    move_captions(path)
    for file in os.listdir(path):
        remove_last_empty_line(path + "/" + file)


def append_npys(path):
    npys_concatinated = []
    for npy_file in os.listdir(path):
        npy = np.load(path + "/" + npy_file)
        npys_concatinated = npys_concatinated + npy.tolist()
    return np.array(npys_concatinated)


def calc_auc(predicted_path, gt_path, normalize_predictions=False):
    predicted_npys_concatinated = append_npys(predicted_path)
    gt_npys_concatinated = append_npys(gt_path)
    if normalize_predictions:
        max_pred = predicted_npys_concatinated.max()
        min_pred = predicted_npys_concatinated.min()
        predicted_npys_concatinated = (predicted_npys_concatinated - min_pred) / (
            max_pred - min_pred
        )
    fpr, tpr, _ = metrics.roc_curve(gt_npys_concatinated, predicted_npys_concatinated)
    auc = metrics.roc_auc_score(gt_npys_concatinated, predicted_npys_concatinated)

    # create ROC curve
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc=4)
    plt.show()
    plt.savefig("dummy_name.png")
    return metrics.roc_auc_score(gt_npys_concatinated, predicted_npys_concatinated)


def assert_no_files_end_with_empty_line(path):
    for file in os.listdir(path):
        with open(path + "/" + file, "r") as f:
            lines = f.readlines()
            if not lines:
                continue
            if lines[-1].strip() == "":
                return False
    return True


def check_captions(captions_path, frames_path):
    wrong_captions = []
    for frames in os.listdir(frames_path):
        with open(captions_path + "/" + frames + ".txt", "r") as f:
            lines = f.readlines()
            if len(lines) != len(os.listdir(frames_path + "/" + frames)):
                wrong_captions.append(frames)

    return wrong_captions


def average_and_med_npys(npys_path):
    npys_flettened = append_npys(npys_path)
    return np.mean(npys_flettened), np.median(npys_flettened)


# taken from https://replicate.com/joehoover/instructblip-vicuna13b/versions/c4c54e3c8c97cd50c2d2fec9be3b6065563ccf7d43787fb99f84151b867178fe/api#output-schema
def call_instruct_blip(frame_path):
    prompt = """
        If there are people in the frame, describe what they are doing in a short single sentence 
    """
    # Give a short caption to the frame in a single sentance, if there are people focus on what they are doing.

    path = Path(frame_path)
    output = replicate.run(
        "gfodor/instructblip:ca869b56b2a3b1cdf591c353deb3fa1a94b9c35fde477ef6ca1d248af56f9c84",
        input={
            "prompt": prompt,
            "image_path": path,
            # "len_penalty": 1,
            "max_length": 10,
        },
    )
    # The joehoover/instructblip-vicuna13b model can stream output as it's running.
    # The predict method returns an iterator, and you can iterate over that output.
    str_output = ""
    for item in output:
        str_output += item
    return str_output


# Taken from https://huggingface.co/Salesforce/instructblip-flan-t5-xl, smaller model
def call_instruct_blip_2(frame_path):
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
    import torch
    from PIL import Image
    import requests

    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl"
    )
    processor = InstructBlipProcessor.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    image = Image.open(frame_path).convert("RGB")
    prompt = """
        If there are people in the frame, describe what they are doing in a short sentence.
        If there aren't people, say that there aren't any people.
    """
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=35,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=0,
        temperature=1,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[
        0
    ].strip()
    print(generated_text)


if __name__ == "__main__":
    # convert_frames_to_mp4("Video_ChatCaptioner/stc/shanghai_tech_dataset/testing")
    # wirte_stc_csv_file(folder_name="Video_ChatCaptioner/stc/shanghai_tech_dataset/testing", csv_file_name="videos.csv")
    # frames = get_testing_numpy_frames("Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/frames/12_0175")
    # sampled_frames = read_video_sampling(
    #             "Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/videos/12_0175.mp4",
    #             num_frames=100,
    #         )
    # move_captions(path = "output/Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/captions_without_prompt")
    folder = "output/Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/captions_without_prompt"
    # for file in os.listdir(folder):
    #     remove_last_empty_line(folder + '/' + file)

    # print(assert_no_files_end_with_empty_line(folder))
    # check_captions('output/Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/captions', 'Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/frames')
    # fix_txt_files(folder)

    gt_mean, gt_median = average_and_med_npys(
        "Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/test_frame_mask"
    )
    pred_mean, pred_median = average_and_med_npys(
        "output/Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/npys_without_prompt_without_ppl_detection"
    )
    predicted_npys = "output/Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/npys_without_prompt_without_ppl_detection"
    test_frame_mask = (
        "Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/test_frame_mask"
    )
    a = calc_auc(predicted_npys, test_frame_mask)
    # call_instruct_blip_2(
    #     "Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/frames/01_0015/015.jpg"
    # )
    print("end")
