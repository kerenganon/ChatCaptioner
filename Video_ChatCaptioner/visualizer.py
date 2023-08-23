import os
from random import sample
import numpy as np

FRAMES_PATH = "Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/frames"
CAPTIONS_TXT_PATH = "output/Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/captions_with_instruct_blip"
PREDICTED_NPY_PATH = "output/Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/npys_with_instruct_blip"
GT_NPY_PATH = "Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/test_frame_mask"
MAX_WIDTH_IN_PX = 500
VIDEO_CAPTIONER_PROMPT = "None"
INSTRUCT_BLIP_PROMPT = """
    Write a caption of the image focusing on what the people in it are doing in a short sentence. \n
    If there are no people, say that there aren't any people.
"""
LLM_PROMPT = "None"
AUC = 0.59


def print_frame_caption(caption, html_out):
    print(f"<br><b> Caption: {caption}</b><br>", file=html_out)


def print_abnormal_scores(frame_dir, frame_number, html_out):
    gt = np.load(GT_NPY_PATH + "/" + frame_dir + ".npy")
    pred = np.load(PREDICTED_NPY_PATH + "/" + frame_dir + ".npy")
    print(f"<br><b> Abnormal scoring: </b><br>", file=html_out)
    print(
        f"<table><tr> <th>Predicted</th> <th>GT</th> <th>Delta</th></tr>", file=html_out
    )
    pred_score = pred[frame_number]
    gt_score = gt[frame_number]
    delta = abs(gt_score - pred_score)
    delta_colors = [
        "#69B34C",  # green
        "#ACB334",  # light green
        "#FAB733",  # yellow
        "#FF8E15",  # light orange
        "#FF4E11",  # orange
        "#FF0D0D",  # red
    ]
    delta_color = delta_colors[int(delta * 6)]
    print(
        f'<tr><td>{pred_score}</td><td>{gt_score}</td><td><p style="background-color:{delta_color};">{delta}</p></td></tr></table>',
        file=html_out,
    )


def print_img(image_path, html_out):
    img_tag = '<img src="{0}" style="max-width: {1}px; height: auto;">'.format(
        image_path, MAX_WIDTH_IN_PX
    )
    print(img_tag, file=html_out)


def sample_frames(clip_id, sampled_frames=5):
    sampled_frames = sample(os.listdir(FRAMES_PATH + "/" + clip_id), sampled_frames)
    sampled_frames.sort()
    return sampled_frames


def sample_frame_paths(sampled_dirs=5, sampled_frames_per_dir=10):
    frame_paths_dict = {}
    for sampled_frame_dir in sample(os.listdir(FRAMES_PATH), sampled_dirs):
        frame_paths_dict[sampled_frame_dir] = []
        sampled_frames = sample_frames(sampled_frame_dir, sampled_frames_per_dir)
        # sampled_frames = sample(
        #     os.listdir(FRAMES_PATH + "/" + sampled_frame_dir), sampled_frames_per_dir
        # )
        sampled_frames.sort()
        for sampled_frame in sampled_frames:
            frame_paths_dict[sampled_frame_dir].append(sampled_frame)
    return frame_paths_dict


def get_top_and_bottom_clips(num_of_clips=5):
    """
    Returns the clips that have the smalless and largest average delta from GT

    Parameters:
        num_of_clips (int): the number of top and bottom clips we want

    Returns:
        list, list: The top and bottom num_of_clips ids
    """
    clips_dict = {}
    for clip in os.listdir(FRAMES_PATH):
        gt = np.load(GT_NPY_PATH + "/" + clip + ".npy")
        pred = np.load(PREDICTED_NPY_PATH + "/" + clip + ".npy")
        delta = np.average(abs(gt - pred))
        clips_dict[clip] = delta
    sorted_clips_tuples = sorted(clips_dict.items(), key=lambda item: item[1])
    sorted_clips = [item[0] for item in sorted_clips_tuples]
    return sorted_clips[:num_of_clips], sorted_clips[:-num_of_clips]


def get_frame_number(frame_caption):
    # Each frame caption has the following format: "Frame: <frame_number>: <frame_caption>"
    return int(frame_caption.split(":")[1])


def remove_frame_prefix(frame_caption):
    # Each frame caption has the following format: "Frame: <frame_number>: <frame_caption>"
    # return only the actual <frame_caption>
    return frame_caption.split(":")[-1]


def get_frame_caption(frame_dir, frame_number):
    with open(CAPTIONS_TXT_PATH + "/" + frame_dir + ".txt") as f:
        lines = f.readlines()
        caption = next(
            filter(
                lambda line: get_frame_number(line) == frame_number,
                lines,
            )
        )
        return remove_frame_prefix(caption)


def create_samples_html():
    html_out = open("vizualizer.html", "w")
    print('<head><meta charset="UTF-8"></head>', file=html_out)
    print("<h1>Results</h1>", file=html_out)
    print("<h2>Instruuct blip prompt:</h2>", file=html_out)
    print("{0}<br>".format(INSTRUCT_BLIP_PROMPT), file=html_out)
    print("<h2>LLM prompt:</h2>", file=html_out)
    print("{0}<br>".format(LLM_PROMPT), file=html_out)
    print("<h2>AUC score:</h2>", file=html_out)
    print("{0}<br>".format(AUC), file=html_out)
    print("<h2>Frame samples:</h2>", file=html_out)
    print("<style>table, th, td {border:1px solid black;}</style>", file=html_out)
    image_paths = sample_frame_paths()

    for dir in image_paths.keys():
        for frame in image_paths[dir]:
            path = FRAMES_PATH + "/" + dir + "/" + frame
            print(f"<br><b>{dir + '/' + frame}</b><br>", file=html_out)
            frame_number = int(frame.removesuffix(".jpg"))
            print_frame_caption(get_frame_caption(dir, frame_number), html_out)
            print_abnormal_scores(dir, frame_number, html_out)
            print_img(path, html_out)
            print(f"<br><b></b><br>", file=html_out)
            print(f"<br><b></b><br>", file=html_out)

    print("<hr>", file=html_out)
    html_out.close()


def print_clips(clip_to_frame_dict, html_out):
    for key, value in clip_to_frame_dict.items():
        print("<h3>Clip {0}:</h3>".format(key), file=html_out)
        for frame in value:
            print("<h4>Frame {0}:</h4>".format(frame), file=html_out)
            frame_number = int(frame.removesuffix(".jpg"))
            print_frame_caption(get_frame_caption(key, frame_number), html_out)
            print_abnormal_scores(key, frame_number, html_out)
            print_img(FRAMES_PATH + "/" + key + "/" + frame, html_out)
            print(f"<br><b></b><br>", file=html_out)


def create_best_and_worst_clips_html():
    html_out = open("best_and_worst_clips.html", "w")
    print('<head><meta charset="UTF-8"></head>', file=html_out)
    print("<h2>Instruct blip prompt:</h2>", file=html_out)
    print("{0}<br>".format(INSTRUCT_BLIP_PROMPT), file=html_out)
    print("<h2>LLM prompt:</h2>", file=html_out)
    print("{0}<br>".format(LLM_PROMPT), file=html_out)
    print("<h2>AUC score:</h2>", file=html_out)
    print("{0}<br>".format(AUC), file=html_out)
    print("<style>table, th, td {border:1px solid black;}</style>", file=html_out)

    best_clips, worst_clips = get_top_and_bottom_clips()
    best_clips_sampled_frames = {clip: sample_frames(clip) for clip in best_clips}
    worst_clips_sampled_frames = {clip: sample_frames(clip) for clip in worst_clips}

    print("<h1>Best clips:</h1>", file=html_out)
    print_clips(best_clips_sampled_frames, html_out)
    print("<h1>Worst clips:</h1>", file=html_out)
    print_clips(worst_clips_sampled_frames, html_out)
    print("<hr>", file=html_out)
    html_out.close()


if __name__ == "__main__":
    a = np.load("Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/test_frame_mask/10_0038.npy")
    # create_samples_html()
    create_best_and_worst_clips_html()
