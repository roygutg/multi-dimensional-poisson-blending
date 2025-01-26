import numpy as np
from skimage.io import imread
import os
import requests
import nibabel as nib
from PoissonBlending import Poisson2DBlender, Colored, Poisson3DBlender


def get_2d_inputs(example_dir: str, as_gray: bool = True):
    source = imread(os.path.join(example_dir, "source.png"), as_gray=as_gray)
    target = imread(os.path.join(example_dir, "target.png"), as_gray=as_gray)
    mask = imread(os.path.join(example_dir, "mask.png"), as_gray=True)
    return source, target, mask


def get_3d_inputs():
    target = read_3d_spine()
    source = read_3d_brain()

    mask = np.zeros_like(source)
    depth, height, width = source.shape
    mask[depth // 6:-depth // 6, width // 8:-width // 2, width // 6:-width // 6] = 1

    return source, target, mask


def read_3d_brain():
    brain_nii_path = r"examples/3d/sub-01122021301_task-arousal_bold.nii.gz"
    if not os.path.isfile(brain_nii_path):
        brain_nii_url = r"https://s3.amazonaws.com/openneuro.org/ds005530/sub-01122021301/func/sub-01122021301_task-arousal_bold.nii.gz?versionId=JKDbAVf6TxwKVEc3OzUZRYMXL_bRR2wN"
        download_binary_file(brain_nii_url, brain_nii_path)

    full_brain_timeline = nib.load(brain_nii_path).get_fdata()
    return full_brain_timeline[:, :, :, 0]


def read_3d_spine():
    spine_nii_path = r"examples/3d/sub-01_ses-01_T2w.nii.gz"
    if not os.path.isfile(spine_nii_path):
        spine_nii_url = r"https://s3.amazonaws.com/openneuro.org/ds004926/sub-01/ses-01/anat/sub-01_ses-01_T2w.nii.gz?versionId=NvJcfK42b8Xck6QmyG.8MEy.2B_GRn7x"
        download_binary_file(spine_nii_url, spine_nii_path)

    full_spine_image = nib.load(spine_nii_path).get_fdata()
    return full_spine_image[:, 128:200, 128:]


def download_binary_file(url, output_path):
    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, "wb") as file:
        file.write(response.content)


def demo_2d():
    all_2d_examples_dir = r"examples/2d/"
    example_numbers = (1, 2, 3)
    example_dir_gen = (os.path.join(all_2d_examples_dir, str(example_i)) for example_i in example_numbers)
    for example_dir in example_dir_gen:
        for as_gray in (True, False):
            source, target, mask = get_2d_inputs(example_dir, as_gray)
            blender = Poisson2DBlender(source, target, mask)
            if not as_gray:
                blender = Colored(blender)
            blender.blend()
            fig = blender.show_results()

            color_label = "gray" if as_gray else "rgb"
            fig.savefig(os.path.join(example_dir, f"result_{color_label}.png"))


def demo_3d():
    source, target, mask = get_3d_inputs()
    blender = Poisson3DBlender(source, target, mask, mix_gradients=False)
    blender.blend()
    blender.show_results()


if __name__ == '__main__':
    demo_2d()
    demo_3d()
