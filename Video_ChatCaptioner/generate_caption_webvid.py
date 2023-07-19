import cv2
import numpy as np
import json
import sys
import os
import yaml
import torch
from PIL import Image
import csv
from helper_methods import save_image_with_title, get_testing_numpy_frames


from chatcaptioner.video_chat import set_openai_key, caption_for_video
from chatcaptioner.blip2 import Blip2
from chatcaptioner.utils import RandomSampledDataset, plot_img, print_info
from chatcaptioner.video_reader import (
    read_video_with_timestamp,
    read_video_sampling,
    read_video_with_timestamp_key_frame,
)
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


FRAME_FOLDER = sys.argv[1]
CAPTION_FILE = sys.argv[2]
OUTPUT_FOLDER = sys.argv[3]
VIDEO_LIMIT = int(sys.argv[4])
SAVE_FRAME_WITH_CAPPTION = False
# VIDEO_FOLDER = "Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/videos"
# CAPTION_FILE = "Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/videos.csv"
# OUTPUT_FOLDER = "Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/output/"
# VIDEO_LIMIT = 1


blip2s = {"FlanT5 XXL": Blip2("FlanT5 XXL", device_id=0, bit8=True)}


data_file = {}
with open(CAPTION_FILE, "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header row
    for row in csv_reader:
        video_id = row[0]
        duration = row[2]
        folder = row[3]
        caption = row[-1]

        data_file[video_id] = {
            "duration": duration,
            "folder": folder,
            "caption": caption,
        }


# find all the frames paths
frames_files = []
for frame_folder in os.listdir(FRAME_FOLDER):
    full_path = os.path.join(FRAME_FOLDER, frame_folder)
    frames_files.append(full_path)
# for root, dirs, files in os.walk(FRAME_FOLDER):
#     for filename in files:
#         full_path = os.path.join(root, filename)
#         frames_files.append(full_path)
# print(full_path)

# extract the video frames with uniform sampling
frames_list = []
for frames_path in frames_files[:VIDEO_LIMIT]:
    # video_id = frames_path.split("/")[-1].replace(".mp4", "")
    video_id = frames_path.split("/")[-1]
    # print(data_file.keys(), video_id)
    if video_id in data_file.keys():
        new_json_file = {}
        new_json_file["video_id"] = video_id
        new_json_file["frames_path"] = frames_path
        new_json_file["annotation"] = data_file[video_id]

        try:
            # sampled_frames = read_video_sampling(
            #     "Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/videos/"
            #     + video_id,
            #     num_frames=100,
            # )
            frames = get_testing_numpy_frames(frames_path)
            new_json_file["features"] = frames
            frames_list.append(new_json_file)
            print("Added " + str(len(frames)) + " frames for video_id: " + video_id)
        except Exception as e:
            print("Error Extracting Features: " + str(e))


# print(video_list)

for sample in frames_list:
    video_id = sample["video_id"]
    features = sample["features"]
    output = sample["annotation"]["folder"]
    print("captioning video_id: " + video_id)
    if not os.path.exists(OUTPUT_FOLDER + output):
        os.makedirs(OUTPUT_FOLDER + output)
    with open(OUTPUT_FOLDER + output + "/" + video_id + ".txt", "w") as f:
        for i, feat in enumerate(features):
            try:
                sub_summaries = caption_for_video(
                    blip2s["FlanT5 XXL"],
                    [feat],
                    print_mode="no",
                    n_rounds=1,
                    model="gpt-3.5-turbo",
                )
            except Exception as e:
                print("exception thrown: " + str(e))
            f.write(
                "Frame: "
                + str(i)
                + ": "
                + sub_summaries["BLIP2+OurPrompt"]["caption"]
                + "\n"
            )

            if SAVE_FRAME_WITH_CAPPTION:
                save_image_with_title(
                    pil_image=feat,
                    title_text=sub_summaries["BLIP2+OurPrompt"]["caption"],
                    output_path=OUTPUT_FOLDER
                    + output
                    + "/"
                    + video_id
                    + "_frame_"
                    + str(i)
                    + ".png",
                )
        print("captioned video_id: " + video_id + ". Captions are saved in: " + f.name)
        # print("Frame: " + str(i) + ": " + sub_summaries["BLIP2+OurPrompt"]["caption"])
        # sub_summaries = caption_for_video(blip2s['FlanT5 XXL'], features, print_mode="chat",n_rounds=30, model='gpt-3.5-turbo')
        # caption =  sample["annotation"]["caption"]
        # f.write("ground truth: "+ caption+"\n\n\n")
        # f.write("chatCaptioner: " +sub_summaries["ChatCaptioner"]["caption"]+"\n\n\n")
        # f.write("chat log:\n")
        # for element in sub_summaries["ChatCaptioner"]["chat"]:
        #     f.write(element["content"]+"\n")
