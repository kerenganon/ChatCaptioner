from transformers import pipeline
import torch
import argparse
import os
import numpy as np

INITIAL_PROMT = """
    Normal behaviour is considered walking, and any behaviour which is normal in a university campus.
    Is the following frame description defining normal behaviour?
    (answer yes or no).\n"""


def get_captions_per_frame(txt_caption_filename):
    with open(txt_caption_filename, "r") as f:
        return f.read().split("\n")


def get_llm_anomaly_score(frame_caption, llm_pipe, yes_idx, no_idx):
    """Prompts the LLM model with the frame_caption, asking whether or not it is normal.

    Args:
        frame_caption (str): a caption of a frame of a video
        llm_pipe (Pipeline): an LLM pipeline that can be promted with a question

    Returns:
        float: the prediction of the llm model on how normal the frame caption is.
    """
    prompt = INITIAL_PROMT + frame_caption
    # answer = llm_pipe(prompt, length_penalty=0, num_beams=3, num_return_sequences=1)
    # return answer
    with torch.no_grad():
        inp = llm_pipe.tokenizer(prompt, text_target="", return_tensors="pt").to("cuda")
        out = llm_pipe.model(**inp)
        probs = out.logits[0, 0].sigmoid()
        y_score, n_score = probs[yes_idx].item(), probs[no_idx].item()
        total = y_score + n_score
        return n_score / total


def get_frame_number(frame_caption):
    # Each frame caption has the following format: "Frame: <frame_number>: <frame_caption>"
    return int(frame_caption.split(":")[1])


def remove_frame_prefix(frame_caption):
    # Each frame caption has the following format: "Frame: <frame_number>: <frame_caption>"
    # return only the actual <frame_caption>
    return frame_caption.split(":")[-1]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--caption_folder_name",
        help="contains txt files with frame captions",
    )
    parser.add_argument(
        "-o",
        "--npys_output_folder_name",
        help="the path to where we want to save the npy files",
    )
    args = parser.parse_args()
    return args.caption_folder_name, args.npys_output_folder_name


def main():
    llm_pipe = pipeline("text2text-generation", model="google/flan-t5-xl", device=0)
    yes_idx = llm_pipe.tokenizer("yes").input_ids[0]
    no_idx = llm_pipe.tokenizer("no").input_ids[0]
    caption_folder_name, npys_output_folder_name = get_args()
    for file in os.listdir(caption_folder_name):
        # run only on text files
        if file.endswith(".txt"):
            frame_captions = get_captions_per_frame(caption_folder_name + "/" + file)
            frame_mask = np.zeros(len(frame_captions))
            for frame_caption in frame_captions:
                abnormal_score = get_llm_anomaly_score(
                    remove_frame_prefix(frame_caption), llm_pipe, yes_idx, no_idx
                )
                # answer = get_llm_anomaly_prediction(frame_caption, llm_pipe)
                print(frame_caption)
                print("abnormal_score: " + str(abnormal_score))
                frame_mask[get_frame_number(frame_caption)] = abnormal_score
            np.save(
                npys_output_folder_name + "/" + file.removesuffix(".txt") + ".npy",
                frame_mask,
            )


if __name__ == "__main__":
    data = np.load(
        "Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/test_frame_mask/01_0014.npy"
    )
    main()
