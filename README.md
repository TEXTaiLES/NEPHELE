# Nephele
<p align="center">
  <img src="readme_images/nephele_logo_2.png" alt="Pipeline Image" width="400"/>
</p>


"Nephele" is a system that processes images to generate 3D meshes without backgrounds. By combining **SAM2** (Surface-Aligned Mesh) and **SUGAR** (Surface-Aligned Gaussian Splatting), it aligns surfaces and creates efficient 3D models, perfect for applications requiring clean, background-free reconstructions.
#### Pipeline:


<p align="center">
  <img src="readme_images/pipeline_SAMplify_SuGaR.png" alt="Pipeline Image" width="600"/>
</p>

## Quick Start

To get started with this project, follow these simple steps:

### 1. Clone the repository

Run the following command to clone the repository to your local machine:

```bash
git clone --recurse-submodules https://github.com/TEXTaiLES/SAMplify_SuGaR
# If you forgot --recurse-submodules:
git submodule update --init --recursive
```

### 2. Navigate to the project directory

After cloning, go to the project folder, rename folder to "NEPHELE" and then:

```bash
cd NEPHELE
```


### 3. Prepare dataset

You need to prepare the dataset by creating a **data** folder inside the **SAM2-Docker** directory. Inside this **data** folder, you will store the images with a `.jpg` extension. The images should be placed in a subfolder named after your dataset.

Follow these steps:

1. Navigate to the **SAM2-Docker** directory:

   ```bash
   cd SAM2
   ```


2. Create a folder inside **data** to store your images. Name the folder after your dataset, for example:

   ```bash
   mkdir -p data/input/your_dataset_name
   ```

3. Place your `.jpg` images into the **your\_dataset\_name** folder.


Now, your **SAM2** directory should have the following structure:

```
SAM2/
└── data/
    └── input/
        └── <DATASET_NAME>/
            ├── image1.jpg
            ├── image2.jpg
            └── ...

```

These images will now be ready to be processed by the SAM2-Docker system.

---


### 4. Set environment variables for the paths
Before running the pipeline, you need to set up the environment variables for the paths of the **SAM2-Docker** and **SuGaR-Docker** repositories.

Set the paths for **sam_fit_sugar**  by adding this line to your terminal:

```bash
export nephele_PATH="/path/to/nephele_PAT"
```

Make sure to replace `"/path/to/nephele_PAT"` with the correct path where the repositorie is located on your system.

### 5. Run the pipeline

Once the paths are set, you can run the pipeline with the following command:

```bash
cd ..
bash run_pipeline.sh your_dataset_name
```

This will execute the **SAM2** and **SuGaR** pipelines for the specified dataset.

---



#UI Usage

## Overview

This project uses the SAM2 and SuGaR frameworks for 3D reconstruction of images, generating high-quality models with background removal. By clicking on significant points in an image, the SAM2 model generates a mask that isolates the object of interest.

## Workflow

## Upload Data
### 1. Upload Options

In this project you can upload either images from a local folder or either upload a video from which frames will be extracted. There is a third option, which is to get a scan dataset that it is already uploaded to "HESTIA", a database of TexTaiLES project.
<p align="center">
  <img src="readme_images/upload_data.png" alt="Example Image (GUI points)" width="640">
</p>
## Mask Generation
### 1. Image Loading and Point Annotation

The process starts by loading an image. The user can click on the image to add points of interest:

- **Left-click** to add **positive points** (marked in **green**).
- **Right-click** to add **negative points** (marked in **red**).


The tool allows the user to select specific regions of the image, which will later be used to create a mask.

<p align="center">
  <img src="readme_images/dress_gui.png" alt="Example Image (GUI points)" width="640">
</p>

#### Example Image (before mask creation):
<p align="center">
  <img src="readme_images/dress_original_1760625683.jpg" alt="Example Image (original)" width="640">
</p>

### 2. Generating the Mask with SAM2

Once enough points are added, the SAM2 model generates a **mask** that isolates the object of interest. The mask is a binary image that shows the identified area.

#### Mask Output:
<p align="center">
  <img src="readme_images/dress_mask_1760625683.jpg" alt="Mask Output" width="640">
</p>

### 3. Overlay of Mask on Image

After the mask is created, it is applied to the original image, showing the object against a transparent or black background, as shown below:

<p align="center">
  <img src="readme_images/dress_masked_1760625683.jpg" alt="Image overlay" width="640">
</p>


### 4. Final Output

The final result is an image where the object is isolated from the background, ready for further processing or 3D reconstruction.

##  Structure-from-Motion with COLMAP

The masked images are passed to **COLMAP** to estimate **camera poses** and generate a **sparse/dense point cloud**.

<p align="center">
  <img src="readme_images/colmap_dress.png" alt="Image overlay" width="640">
</p>

## Surface-Aligned Gaussian Splatting (SuGaR)

The COLMAP reconstruction initializes a SuGaR model, producing a **3D Gaussian field**.

<p align="center">
  <img src="readme_images/gs3d.png" alt="Gaussian Splatting" width="650">
</p>

We can visualize both the **splat points** and the **ellipsoids** that represent their covariance:

<p align="center">
  <img src="readme_images/ellipsoids_3d_gs.png" alt="Ellipsoids Visualization" width="650">
</p>

### Mesh Extraction and Texturing

Finally, the SuGaR output is converted to a **Poisson mesh** and textured.

**Textured Mesh Output:**

<p align="center">
  <img src="readme_images/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.png" alt="Textured Mesh" width="640">
</p>



## Steps to Use

1. **Clone the repository** and set up the environment.
2. **Load your images** into the appropriate directory.
3. **Run the pipeline** using the `run_pipeline.sh` script.
4. **Annotate the points** by clicking on the important areas of the image.
5. **Generate the mask**, which will be saved for further use.

For more details, check the documentation in the repository or the command line help.


# NEPHELE
NEPHELE is a system that processes images to generate 3D meshes without backgrounds. By combining SAM2 and SUGAR (Surface-Aligned Gaussian Splatting), it aligns surfaces and creates efficient 3D models, perfect for applications requiring clean, background-free reconstructions.



---

## Citation
If you use this software, please cite it using the following BibTeX entry:

```bibtex
@software{Nephele_TEXTaiLES_2026,
  author  = {{Athena Research Center}},
  title   = {{Nephele: SAM2 + COLMAP + SuGaR pipeline for background-free 3D mesh reconstruction}},
  url     = {https://github.com/TEXTaiLES/SAMplify_SuGaR},
  version = {0.1.0},
  year    = {2025},
  license = {MIT}
}
```

## Third-party citations

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}

@inproceedings{Schonberger2016SfM,
  title     = {Structure-from-Motion Revisited},
  author    = {Sch{\"o}nberger, Johannes L. and Frahm, Jan-Michael},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2016}
}

@article{guedon2023sugar,
  title   = {SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering},
  author  = {Gu{\'e}don, Antoine and Lepetit, Vincent},
  journal = {CVPR},
  year    = {2024}
}

@article{kerbl2023gaussiansplatting,
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal = {ACM Transactions on Graphics},
  year    = {2023}
}
```

## License
This project is licensed under the MIT License. 

