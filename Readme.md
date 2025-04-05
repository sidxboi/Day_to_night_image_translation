
# 🌗 CycleGAN: Day to Night Image Translation

This project implements a **CycleGAN** to perform **unpaired image-to-image translation** — transforming daytime urban scenes into realistic night-time counterparts using deep learning. Built using TensorFlow and trained on a small custom dataset.

![Day to Night Sample](generated_images/epoch_50.png)

---

## 🧠 What is CycleGAN?

CycleGAN is a type of Generative Adversarial Network that learns **bidirectional translation** between two visual domains **without paired training examples**. In this case:
- Domain A: Daytime cityscapes  
- Domain B: Nighttime cityscapes  

---

## ✨ Features

- ✅ Unpaired image-to-image translation (no need for matching day-night images)
- ✅ Fully custom training loop in TensorFlow 2.x
- ✅ Data augmentation: resize, crop, flip, rotate, color jitter
- ✅ Regular checkpoint saving and image logging
- ✅ Lightweight and optimized for **Google Colab (T4 GPU)**

---

## 📁 Dataset

Place your images in this structure:
```
dataset/
├── day/      # 522 images (jpg or png)
└── night/    # 227 images
```

You can use your own dataset, just maintain this folder structure.

---

## ⚙️ Training

### 💻 Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy, Matplotlib, Pillow

Install dependencies:

```bash
pip install -r requirements.txt
```

### 🚀 Start Training (Colab recommended)

```bash
python train.py
```

Training settings:
- **Image size**: 256x256  
- **Batch size**: 2  
- **Epochs**: 50  
- **Cycle loss weight**: 15.0  
- **Identity loss**: 7.5  

Checkpoints and output images will be saved to:
- `checkpoints/`
- `generated_images/`

---

## 🖼️ Inference

To generate night-style image from a test input:

```python
from PIL import Image
import tensorflow as tf

# Load model
generator = tf.keras.models.load_model('path_to_G_day_to_night.h5')

# Load and preprocess input
input_img = preprocess_image('your_image.jpg')[None, ...]
night_img = generator(input_img, training=False)[0]

# Save or display result
Image.fromarray((night_img.numpy() * 127.5 + 127.5).astype('uint8')).save('night_result.jpg')
```

---

## 📊 Results

Generated image after 50 epochs:

![Example Output](generated_images/epoch_50.png)

---

## 🧾 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [CycleGAN paper](https://arxiv.org/abs/1703.10593) by Zhu et al.  
- TensorFlow 2.x  
- Google Colab (T4 GPU)

---

> Created by Sidhant Ranjan Medhi(https://github.com/sidxboi)
```

---

Let me know if you’d like me to generate the `train.py` or any other file too!
