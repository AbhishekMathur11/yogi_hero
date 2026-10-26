import argparse
import os
import re
from visualize import vis_utils
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='stick figure mp4 file to be rendered.')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    params = parser.parse_args()

    assert params.input_path.endswith('.mp4'), "input_path must end with .mp4"

    # sample00_rep00.mp4, or the combined file samples_00_to_00.mp4
    basename = os.path.splitext(os.path.basename(params.input_path))[0]
    pair = re.fullmatch(r"sample(\d+)_rep(\d+)", basename)
    combined = re.fullmatch(r"samples_(\d+)_to_(\d+)", basename)
    if pair:
        sample_i, rep_i = int(pair.group(1)), int(pair.group(2))
    elif combined:
        sample_i, rep_i = int(combined.group(1)), 0
    else:
        raise ValueError(
            "expected sample00_rep00.mp4 or samples_00_to_00.mp4, got " + basename
        )

    sample_dir = os.path.dirname(params.input_path)
    npy_path = os.path.join(sample_dir, 'results.npy')
    out_dir = os.path.join(sample_dir, "output_mesh")
    os.makedirs(out_dir, exist_ok=True)
    out_npy_path = os.path.join(out_dir, "smpl_params.npy")
    results_dir = os.path.join(out_dir, "obj")

    # Remove and recreate the results directory to ensure it's empty
    if os.path.exists(results_dir) and os.path.isdir(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    npy2obj = vis_utils.npy2obj(npy_path, sample_i, rep_i,
                                device=params.device, cuda=params.cuda)

    print('Saving obj files to [{}]'.format(os.path.abspath(results_dir)))
    for frame_i in tqdm(range(npy2obj.real_num_frames)):
        npy2obj.save_obj(os.path.join(results_dir, 'frame{:03d}.obj'.format(frame_i)), frame_i)

    print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))
    npy2obj.save_npy(out_npy_path)
