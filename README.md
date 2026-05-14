# MorphGAN
Official PyTorch implementation of "MorphGAN: An intrinsically optimized generative framework for high-fidelity white blood cell synthesis".
## 🛠️ Dependencies

The code is implemented in Python and requires the following packages. We recommend using a virtual environment (e.g., conda or venv) for installation.

* PyTorch == 2.0.0
* Torchvision == 0.15.1
* NumPy == 1.23.5
* OpenCV == 4.5.4.60
* pandas == 1.5.3

**Installation:**

Clone this repository and install the required dependencies via `pip`:

```bash
git clone [https://github.com/pushisnz/MorphGAN.git](https://github.com/pushisnz/MorphGAN.git)
cd MorphGAN
pip install -r requirements.txt
## 📂 Data Preparation

The training script expects the dataset to be structured as a main folder containing one or more subfolders with images. For example, prepare your White Blood Cell (WBC) dataset as follows:

```text
./dataset_path/
├── class_1/
│   ├── img1.png
│   ├── img2.png
│   └── ...
├── class_2/
│   ├── img3.png
│   └── ...

### ⚙️ Training

To train the model, simply run the following command:

```bash
python train.py --path ./data --output_path ./results --name exp_01 --im_size 256 --batch_size 16 --iter 100000 --save_interval 5000
