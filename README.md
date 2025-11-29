# 2D Computer Graphics with OpenCV and Python

A streamlined curriculum for learning 2D computer graphics using OpenCV with Python.

---

## Quick Start

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv opencv_env
opencv_env\Scripts\activate  # Windows
# source opencv_env/bin/activate  # Linux/Mac

# Install core packages
pip install opencv-python opencv-contrib-python
pip install numpy matplotlib jupyter
```

### Your First OpenCV Program
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and display image
img = cv2.imread('image.jpg')
cv2.imshow('My Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## Prerequisites

**Python Skills**: Intermediate level (functions, classes, NumPy basics)
**Math**: Linear algebra, basic calculus, trigonometry
**Time Commitment**: 12-16 weeks (10-15 hours/week)

---

## Course Modules

> **📓 Each module has an interactive Jupyter notebook. Click the badges to open in Google Colab!**

### Module 0: Prerequisites (Before Week 1)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/computer-graphics/blob/main/notebooks/00_Prerequisites.ipynb)
- Python fundamentals (functions, classes, comprehensions)
- NumPy essentials (arrays, slicing, operations)
- Mathematics review (linear algebra, trigonometry, calculus basics)
- Practice exercises

**Key Concepts**: Python programming, NumPy arrays, vectors, matrices, transformations

---

### Module 1: Foundations (Weeks 1-2)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/computer-graphics/blob/main/notebooks/01_Foundations.ipynb)
- Image representation and file I/O
- NumPy arrays as images
- Pixel manipulation and ROI
- Basic visualization

**Key Functions**: `cv2.imread()`, `cv2.imwrite()`, `cv2.imshow()`, array slicing

---

### Module 2: Drawing & Color (Weeks 3-4)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/computer-graphics/blob/main/notebooks/02_Drawing_and_Color.ipynb)
- Drawing primitives (lines, circles, rectangles, text)
- Color spaces (RGB, BGR, HSV, LAB)
- Color transformations
- Thresholding techniques

**Key Functions**: `cv2.line()`, `cv2.circle()`, `cv2.putText()`, `cv2.cvtColor()`, `cv2.threshold()`

---

### Module 3: Filtering & Enhancement (Weeks 5-6)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/computer-graphics/blob/main/notebooks/03_Filtering_Enhancement.ipynb)
- Convolution fundamentals
- Blurring and smoothing
- Sharpening and edge enhancement
- Noise reduction

**Key Functions**: `cv2.filter2D()`, `cv2.GaussianBlur()`, `cv2.medianBlur()`, `cv2.bilateralFilter()`

---

### Module 4: Geometric Transformations (Weeks 7-8)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/computer-graphics/blob/main/notebooks/04_Module.ipynb)
- Translation, rotation, scaling
- Affine transformations
- Perspective transformations
- Image warping

**Key Functions**: `cv2.warpAffine()`, `cv2.warpPerspective()`, `cv2.getRotationMatrix2D()`, `cv2.resize()`

---

### Module 5: Edge Detection (Week 9)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/computer-graphics/blob/main/notebooks/05_Module.ipynb)
- Gradient operators (Sobel, Scharr)
- Canny edge detection
- Laplacian edge detection

**Key Functions**: `cv2.Sobel()`, `cv2.Canny()`, `cv2.Laplacian()`

---

### Module 6: Morphology (Week 10)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/computer-graphics/blob/main/notebooks/06_Module.ipynb)
- Erosion and dilation
- Opening and closing
- Morphological gradient
- Advanced operations

**Key Functions**: `cv2.erode()`, `cv2.dilate()`, `cv2.morphologyEx()`, `cv2.getStructuringElement()`

---

### Module 7: Contours & Shapes (Weeks 11-12)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/computer-graphics/blob/main/notebooks/07_Module.ipynb)
- Contour detection
- Shape properties and analysis
- Bounding boxes and fitting
- Hough transforms

**Key Functions**: `cv2.findContours()`, `cv2.drawContours()`, `cv2.contourArea()`, `cv2.HoughLines()`, `cv2.HoughCircles()`

---

### Module 8: Histograms (Week 13)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/computer-graphics/blob/main/notebooks/08_Module.ipynb)
- Histogram calculation
- Histogram equalization
- Adaptive equalization (CLAHE)
- Histogram-based segmentation

**Key Functions**: `cv2.calcHist()`, `cv2.equalizeHist()`, `cv2.createCLAHE()`

---

### Module 9: Segmentation (Week 14)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/computer-graphics/blob/main/notebooks/09_Module.ipynb)
- Thresholding methods
- Watershed algorithm
- GrabCut algorithm
- K-means clustering

**Key Functions**: `cv2.watershed()`, `cv2.grabCut()`, `cv2.kmeans()`

---

### Module 10: Feature Detection (Week 15)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/computer-graphics/blob/main/notebooks/10_Module.ipynb)
- Corner detection (Harris, Shi-Tomasi)
- Blob detection
- SIFT, ORB descriptors
- Feature matching

**Key Functions**: `cv2.cornerHarris()`, `cv2.goodFeaturesToTrack()`, `cv2.ORB_create()`, `cv2.BFMatcher()`

---

### Module 11: Advanced Topics (Week 16)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/computer-graphics/blob/main/notebooks/11_Module.ipynb)
- Image pyramids and blending
- Panorama stitching
- Template matching
- Real-time video processing

**Key Functions**: `cv2.pyrDown()`, `cv2.pyrUp()`, `cv2.matchTemplate()`, `cv2.VideoCapture()`

---

## Capstone Projects

### 1. Document Scanner (Intermediate)
- Edge detection → Perspective correction → Enhancement → Export

### 2. Photo Filter App (Intermediate)
- Load images → Apply filters → Adjust colors → Save results

### 3. Object Counter (Intermediate)
- Segmentation → Morphology → Contour detection → Count & measure

### 4. Panorama Stitcher (Advanced)
- Feature detection → Matching → Homography → Blending

### 5. Real-time Face Effects (Advanced)
- Video capture → Face detection → Apply effects → Display

---

## Essential Resources

### Books
1. **Mastering OpenCV 4 with Python** - Alberto Fernández Villán ⭐ Primary
2. **Practical Python and OpenCV** - Adrian Rosebrock
3. **Digital Image Processing** - Gonzalez & Woods (Theory)

### Online
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) - Official docs
- [PyImageSearch](https://pyimagesearch.com) - Practical tutorials
- [Learn OpenCV](https://learnopencv.com) - Advanced techniques

### Practice
- Kaggle computer vision competitions
- OpenCV GitHub examples
- Create a portfolio GitHub repository

---

## Code Snippets Reference

### Basic Template
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('input.jpg')

# Process (example: convert to grayscale)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display with matplotlib (converts BGR to RGB)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.show()

# Save result
cv2.imwrite('output.jpg', gray)
```

### Common Patterns
```python
# ROI extraction
roi = img[y:y+h, x:x+w]

# Color channel access
b, g, r = cv2.split(img)
blue_channel = img[:, :, 0]

# Create blank image
blank = np.zeros((height, width, 3), dtype=np.uint8)

# Copy image (avoid reference)
img_copy = img.copy()

# Normalize to 0-1 range
normalized = img.astype(np.float32) / 255.0
```

---

## Study Tips

1. **Use Jupyter Notebooks** for interactive exploration
2. **Visualize everything** - display intermediate results
3. **Start small** - test on small images first
4. **Read error messages** - check shape, dtype, value ranges
5. **Remember BGR** - OpenCV uses BGR, not RGB
6. **Profile code** - use `%%timeit` in Jupyter
7. **Build incrementally** - add one feature at a time

---

## Progress Tracking

- [ ] Week 1-2: Environment setup, basic I/O, pixel operations
- [ ] Week 3-4: Drawing, color spaces, thresholding
- [ ] Week 5-6: Filtering and enhancement
- [ ] Week 7-8: Geometric transformations
- [ ] Week 9: Edge detection
- [ ] Week 10: Morphological operations
- [ ] Week 11-12: Contours and shapes
- [ ] Week 13: Histograms
- [ ] Week 14: Segmentation
- [ ] Week 15: Feature detection
- [ ] Week 16: Advanced topics + Capstone

---

**License**: Educational use
**Last Updated**: November 2025
**Version**: 2.0 - Python Edition

---

## 📚 Directory Structure

```
computer-graphics/
├── README.md                          # This file - complete syllabus
├── .gitignore                         # Git ignore configuration
│
└── notebooks/                         # Jupyter notebooks for each module
    ├── 00_Prerequisites.ipynb         # ✅ Python, NumPy, Math review
    ├── 01_Foundations.ipynb           # ✅ Fully developed
    ├── 02_Drawing_and_Color.ipynb     # ✅ Fully developed
    ├── 03_Filtering_Enhancement.ipynb # 📝 Template (expand with content)
    └── 04-11_Module.ipynb             # 📝 Additional module templates
```

## 🚀 Getting Started

### Option 1: Use Google Colab (Recommended - No Installation!)

1. **Fork or upload this repository to your GitHub account**
2. **Replace `YOUR_USERNAME`** in all notebook links:
   - In README.md: Find `YOUR_USERNAME` → Replace with your GitHub username
   - In all `.ipynb` files in `notebooks/` folder
   - Use find & replace in your editor or run this command:
     ```bash
     # Linux/Mac
     find . -name "*.ipynb" -o -name "*.md" | xargs sed -i 's/YOUR_USERNAME/your-github-username/g'
     ```
3. **Click any "Open in Colab" badge** to start learning immediately
4. **Start with Module 0 (Prerequisites)** to assess your readiness

### Option 2: Local Installation

```bash
# Create virtual environment
python -m venv opencv_env
opencv_env\Scripts\activate  # Windows
# source opencv_env/bin/activate  # Linux/Mac

# Install packages
pip install opencv-python opencv-contrib-python numpy matplotlib jupyter
```

### Learning Path

1. 📚 **Module 0**: [Prerequisites](notebooks/00_Prerequisites.ipynb) - Python, NumPy, Math review
2. 🎨 **Module 1-2**: Foundations & Drawing (fully developed)
3. 🔧 **Module 3-11**: Additional topics (templates to expand)
4. 🚀 **Capstone Projects**: Build real applications

## 📝 Note

- Modules 1-2 have full content and examples
- Modules 3-11 are templates ready for you to expand
- Use the official OpenCV documentation to fill in details
- Contribute improvements via pull requests!

---

*Start coding, visualize results, and build projects. Computer graphics is learned by doing!*
