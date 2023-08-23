
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

FRAMES_PATH = 'Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/frames'
FRAMES_GT_NPYS_PATH = 'Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/test_frame_mask'
CAPTIONS_PATH = 'output/Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/captions_with_instruct_blip'
NPYS_PATH = 'output/Video_ChatCaptioner/stc/shanghai_tech_dataset/testing/npys_with_instruct_blip'
PLOTS_OUTPUT_PATH = 'output/plots'

def get_gt_abnormal_ranges(gt_npy):
    indices = np.where(gt_npy == 1)[0]
    index_ranges = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
    return [(rng[0], rng[-1]) for rng in index_ranges]
    

def plot_gt_vs_pred_for_clip(clip_id):
    gt_npy = np.load(FRAMES_GT_NPYS_PATH + "/" + clip_id + '.npy')
    pred_npy = np.load(NPYS_PATH + "/" + clip_id + '.npy')
    smoothed_pred = gaussian_filter(pred_npy, sigma=3)
    plt.figure().set_figwidth(20)
    plt.plot(pred_npy, linewidth=0.6, label='predicted')
    plt.plot(smoothed_pred, label='smoothed predicted')
    # plt.fill(gt_npy, facecolor='red', alpha=0.25, label='GT')
    gt_abnormal_ranges = get_gt_abnormal_ranges(gt_npy)
    for range in gt_abnormal_ranges:
        plt.axvspan(xmin=range[0], xmax=range[1], color='red', alpha=0.25)
    plt.legend(loc='best')
    output_path = PLOTS_OUTPUT_PATH + '/gt_vs_pred_' + clip_id + '.png'
    plt.savefig(output_path)
    return output_path


if __name__ == "__main__":
    plot_gt_vs_pred_for_clip('10_0038')