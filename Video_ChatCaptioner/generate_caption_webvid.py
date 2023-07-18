import cv2
import numpy as np
import json
import sys
import os
import yaml
import torch
from PIL import Image
import csv
from helper_methods import save_image_with_title


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


VIDEO_FOLDER = sys.argv[1]
CAPTION_FILE = sys.argv[2]
OUTPUT_FOLDER = sys.argv[3]
VIDEO_LIMIT = int(sys.argv[4])
SAVE_FRAME_WITH_CAPPTION = False
# VIDEO_FOLDER = "Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/videos"
# CAPTION_FILE = "Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/videos.csv"
# OUTPUT_FOLDER = "Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/output/"
# VIDEO_LIMIT = 3


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


# find all the video paths
video_files = []
for root, dirs, files in os.walk(VIDEO_FOLDER):
    for filename in files:
        full_path = os.path.join(root, filename)
        video_files.append(full_path)
        print(full_path)

# extract the video frames with uniform sampling
video_list = []
for video_path in video_files[:VIDEO_LIMIT]:
    video_id = video_path.split("/")[-1].replace(".mp4", "")
    print(data_file.keys(), video_id)
    if video_id in data_file.keys():
        new_json_file = {}
        new_json_file["video_id"] = video_id
        new_json_file["video_path"] = video_path
        new_json_file["annotation"] = data_file[video_id]

        try:
            sampled_frames = read_video_sampling(video_path, num_frames=100)
            new_json_file["features"] = sampled_frames
            video_list.append(new_json_file)
        except:
            print("Error Extracting Features")


# print(video_list)

for sample in video_list:
    video_id = sample["video_id"]
    features = sample["features"]
    output = sample["annotation"]["folder"]
    print(video_id)
    if not os.path.exists(OUTPUT_FOLDER + output):
        os.makedirs(OUTPUT_FOLDER + output)
    with open(OUTPUT_FOLDER + output + "/" + video_id + ".txt", "w") as f:
        for i, feat in enumerate(features):
            sub_summaries = caption_for_video(
                blip2s["FlanT5 XXL"],
                [feat],
                print_mode="no",
                n_rounds=1,
                model="gpt-3.5-turbo",
            )
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

            # print("Frame: " + str(i) + ": " + sub_summaries["BLIP2+OurPrompt"]["caption"])
        # sub_summaries = caption_for_video(blip2s['FlanT5 XXL'], features, print_mode="chat",n_rounds=30, model='gpt-3.5-turbo')
        # caption =  sample["annotation"]["caption"]
        # f.write("ground truth: "+ caption+"\n\n\n")
        # f.write("chatCaptioner: " +sub_summaries["ChatCaptioner"]["caption"]+"\n\n\n")
        # f.write("chat log:\n")
        # for element in sub_summaries["ChatCaptioner"]["chat"]:
        #     f.write(element["content"]+"\n")
