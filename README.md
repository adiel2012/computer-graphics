# 2D Computer Graphics with OpenCV and Python

A comprehensive, production-ready curriculum for learning 2D computer graphics using OpenCV with Python. All modules include complete theory, mathematical foundations, and hands-on code examples.

> ✅ **Production Ready**: All 12 modules fully developed with comprehensive content
>
> ✅ **Interactive Learning**: Every module has a Google Colab-ready Jupyter notebook
>
> ✅ **Complete Coverage**: From basics to advanced topics with 150+ hours of material

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended - No Installation!)

1. **Click any module badge below** to open in Google Colab
2. **No setup required** - runs in your browser
3. **Free GPU access** for intensive operations

### Option 2: Local Installation

```bash
# Create and activate virtual environment
python -m venv opencv_env
source opencv_env/bin/activate  # Linux/Mac
# opencv_env\Scripts\activate  # Windows

# Install required packages
pip install opencv-python opencv-contrib-python
pip install numpy matplotlib jupyter scipy
```

### Your First OpenCV Program

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and display image
img = cv2.imread('image.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('My First Image')
plt.axis('off')
plt.show()
```

---

## 📚 Course Modules

> **All modules are production-ready with complete theory, code, and visualizations**

### Module 0: Prerequisites ✅
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-graphics/blob/main/notebooks/00_Prerequisites.ipynb)

**Before Week 1** - Essential foundations
- Python fundamentals (functions, classes, comprehensions)
- NumPy essentials (arrays, broadcasting, vectorization)
- Mathematics review (linear algebra, trigonometry, calculus)
- 20+ practice exercises with solutions

**Key Concepts**: Python programming, NumPy arrays, vectors, matrices, transformations

---

### Module 1: Foundations ✅
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-graphics/blob/main/notebooks/01_Foundations.ipynb)

**Weeks 1-2** - Image fundamentals
- Digital image representation and coordinate systems
- File I/O and image formats (JPEG, PNG, BMP)
- Pixel manipulation and Region of Interest (ROI)
- Channels, color depth, and image properties

**Key Functions**: `cv2.imread()`, `cv2.imwrite()`, `cv2.imshow()`, array indexing

---

### Module 2: Drawing & Color ✅
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-graphics/blob/main/notebooks/02_Drawing_and_Color.ipynb)

**Weeks 3-4** - Graphics and color theory
- Drawing primitives (lines, circles, rectangles, polygons, text)
- Color spaces (RGB, BGR, HSV, LAB, YCrCb)
- Color transformations and conversions
- Thresholding techniques (binary, adaptive, Otsu's)

**Key Functions**: `cv2.line()`, `cv2.circle()`, `cv2.rectangle()`, `cv2.putText()`, `cv2.cvtColor()`, `cv2.threshold()`

---

### Module 3: Filtering & Enhancement ✅
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-graphics/blob/main/notebooks/03_Filtering_Enhancement.ipynb)

**Weeks 5-6** - Image processing techniques
- Convolution theory and kernel operations
- Blurring (Gaussian, median, bilateral)
- Sharpening and unsharp masking
- Noise types and reduction strategies

**Key Functions**: `cv2.filter2D()`, `cv2.GaussianBlur()`, `cv2.medianBlur()`, `cv2.bilateralFilter()`

---

### Module 4: Geometric Transformations ✅
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-graphics/blob/main/notebooks/04_Module.ipynb)

**Weeks 7-8** - Spatial transformations
- Translation, rotation, and scaling
- Affine transformations and homogeneous coordinates
- Perspective transformations and homography
- Image warping and interpolation methods

**Key Functions**: `cv2.warpAffine()`, `cv2.warpPerspective()`, `cv2.getRotationMatrix2D()`, `cv2.resize()`, `cv2.getPerspectiveTransform()`

---

### Module 5: Edge Detection ✅
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-graphics/blob/main/notebooks/05_Module.ipynb)

**Week 9-10** - Gradient-based methods
- Image gradients and derivative operators
- Sobel, Prewitt, Scharr operators
- Laplacian and Laplacian of Gaussian (LoG)
- Canny edge detection algorithm (5 steps)

**Key Functions**: `cv2.Sobel()`, `cv2.Scharr()`, `cv2.Laplacian()`, `cv2.Canny()`

---

### Module 6: Feature Detection ✅
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-graphics/blob/main/notebooks/06_Module.ipynb)

**Week 11** - Keypoints and descriptors
- Harris corner detection (structure tensor)
- Shi-Tomasi Good Features to Track
- SIFT (Scale-Invariant Feature Transform)
- ORB (Oriented FAST and Rotated BRIEF)
- Feature matching (Brute-Force, FLANN)

**Key Functions**: `cv2.cornerHarris()`, `cv2.goodFeaturesToTrack()`, `cv2.SIFT_create()`, `cv2.ORB_create()`, `cv2.BFMatcher()`

---

### Module 7: Contours & Shapes ✅
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-graphics/blob/main/notebooks/07_Module.ipynb)

**Week 12** - Shape analysis
- Contour detection and hierarchy
- Shape properties (area, perimeter, moments)
- Bounding boxes and convex hulls
- Hough transforms (lines, circles)
- Shape approximation and matching

**Key Functions**: `cv2.findContours()`, `cv2.drawContours()`, `cv2.contourArea()`, `cv2.arcLength()`, `cv2.HoughLines()`, `cv2.HoughCircles()`

---

### Module 8: Histograms ✅
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-graphics/blob/main/notebooks/08_Module.ipynb)

**Week 13** - Statistical analysis
- Histogram theory and calculation
- Histogram equalization (global contrast enhancement)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Color image equalization (HSV, YCrCb, LAB)

**Key Functions**: `cv2.calcHist()`, `cv2.equalizeHist()`, `cv2.createCLAHE()`

---

### Module 9: Segmentation ✅
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-graphics/blob/main/notebooks/09_Module.ipynb)

**Week 14** - Region extraction
- Thresholding methods (Otsu, adaptive)
- Watershed algorithm for segmentation
- GrabCut interactive segmentation
- K-means clustering for color quantization

**Key Functions**: `cv2.threshold()`, `cv2.adaptiveThreshold()`, `cv2.watershed()`, `cv2.grabCut()`, `cv2.kmeans()`

---

### Module 10: Morphological Operations ✅
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-graphics/blob/main/notebooks/10_Module.ipynb)

**Week 15** - Mathematical morphology
- Erosion and dilation (basic operations)
- Opening and closing (noise removal)
- Morphological gradient (boundary extraction)
- Top-hat and black-hat transforms
- Structuring elements (shapes and sizes)

**Key Functions**: `cv2.erode()`, `cv2.dilate()`, `cv2.morphologyEx()`, `cv2.getStructuringElement()`

---

### Module 11: Advanced Topics ✅
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/computer-graphics/blob/main/notebooks/11_Module.ipynb)

**Week 16** - Real-world applications
- Image pyramids (Gaussian, Laplacian)
- Template matching and object detection
- Panorama stitching with homography
- Video processing fundamentals

**Key Functions**: `cv2.pyrDown()`, `cv2.pyrUp()`, `cv2.matchTemplate()`, `cv2.VideoCapture()`, `cv2.findHomography()`

---

## 🎯 Capstone Projects

Apply your skills with these comprehensive projects:

### 1. Document Scanner App (Intermediate)
**Skills**: Edge detection, perspective transformation, image enhancement
- Detect document edges using Canny
- Apply perspective correction
- Enhance with adaptive thresholding
- Export as PDF

### 2. Photo Filter Application (Intermediate)
**Skills**: Color spaces, filtering, artistic effects
- Instagram-style filters
- Vintage, sepia, vignette effects
- Batch processing
- GUI with tkinter

### 3. Intelligent Object Counter (Intermediate-Advanced)
**Skills**: Segmentation, morphology, contour analysis
- Automatic thresholding
- Morphological operations for cleanup
- Contour detection and filtering
- Measure and classify objects

### 4. Panorama Stitcher (Advanced)
**Skills**: Feature detection, matching, homography
- Detect and match SIFT/ORB features
- Estimate homography with RANSAC
- Warp and blend images
- Multi-image stitching

### 5. Real-time Face Effects (Advanced)
**Skills**: Video processing, face detection, live filtering
- Webcam capture and processing
- Haar Cascade face detection
- Apply real-time effects
- Performance optimization

---

## 📖 Essential Resources

### Books
1. **Mastering OpenCV 4 with Python** - Alberto Fernández Villán ⭐ Primary resource
2. **Learning OpenCV 4 Computer Vision with Python 3** - Joseph Howse
3. **Digital Image Processing** - Gonzalez & Woods (Theory reference)
4. **Computer Vision: Algorithms and Applications** - Richard Szeliski

### Online Documentation
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) - Official documentation
- [OpenCV API Reference](https://docs.opencv.org/4.x/) - Complete function reference
- [NumPy Documentation](https://numpy.org/doc/) - Array operations
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html) - Visualization examples

### Tutorial Sites
- [PyImageSearch](https://pyimagesearch.com) - Practical tutorials and projects
- [Learn OpenCV](https://learnopencv.com) - Advanced techniques and deep learning
- [Real Python - Image Processing](https://realpython.com/image-processing-with-the-python-pillow-library/) - Python fundamentals

### Practice Platforms
- [Kaggle Computer Vision](https://www.kaggle.com/competitions?searchQuery=computer+vision) - Competitions
- [OpenCV GitHub Examples](https://github.com/opencv/opencv/tree/master/samples/python) - Official samples
- [Papers With Code](https://paperswithcode.com/area/computer-vision) - Latest research

---

## 💡 Code Patterns & Best Practices

### Standard Workflow Template
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image
img = cv2.imread('input.jpg')
if img is None:
    raise FileNotFoundError("Image not found")

# 2. Convert color space if needed (OpenCV uses BGR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Process image
processed = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(processed, 100, 200)

# 4. Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_rgb)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(gray, cmap='gray')
axes[1].set_title('Grayscale')
axes[1].axis('off')

axes[2].imshow(edges, cmap='gray')
axes[2].set_title('Edges')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# 5. Save result
cv2.imwrite('output.jpg', edges)
```

### Common Patterns

```python
# ROI (Region of Interest) extraction
roi = img[y:y+h, x:x+w]

# Channel splitting and access
b, g, r = cv2.split(img)
blue_channel = img[:, :, 0]  # Faster than split

# Create blank images
blank_gray = np.zeros((height, width), dtype=np.uint8)
blank_color = np.zeros((height, width, 3), dtype=np.uint8)
white_img = np.ones((height, width, 3), dtype=np.uint8) * 255

# Safe image copying (avoid references)
img_copy = img.copy()

# Normalize to [0, 1] range
normalized = img.astype(np.float32) / 255.0

# Resize maintaining aspect ratio
def resize_with_aspect_ratio(img, width=None, height=None):
    h, w = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        ratio = height / h
        return cv2.resize(img, (int(w * ratio), height))
    ratio = width / w
    return cv2.resize(img, (width, int(h * ratio)))
```

### Error Handling Best Practices

```python
# Check image loaded successfully
img = cv2.imread('image.jpg')
if img is None:
    print("Error: Could not load image")
    exit()

# Check image dimensions
if len(img.shape) == 2:
    print("Grayscale image")
elif len(img.shape) == 3:
    print(f"Color image with {img.shape[2]} channels")

# Validate kernel size (must be odd)
ksize = 5 if ksize % 2 == 1 else ksize + 1

# Ensure proper data type
img_float = img.astype(np.float32)
img_uint8 = np.clip(result, 0, 255).astype(np.uint8)
```

---

## 🎓 Learning Path & Study Tips

### Recommended Schedule (16 Weeks)

**Weeks 1-2**: Foundations
- Set up environment
- Complete Module 0 (Prerequisites)
- Master Module 1 (Image basics)
- Practice pixel manipulation

**Weeks 3-4**: Graphics & Color
- Module 2 (Drawing & Color)
- Build a drawing app
- Experiment with color spaces

**Weeks 5-6**: Filtering
- Module 3 (Filtering & Enhancement)
- Implement custom kernels
- Noise reduction project

**Weeks 7-8**: Transformations
- Module 4 (Geometric Transformations)
- Build document scanner
- Perspective correction

**Week 9-10**: Edges
- Module 5 (Edge Detection)
- Compare edge detectors
- Optimize Canny parameters

**Week 11**: Features
- Module 6 (Feature Detection)
- Implement image matching
- SIFT vs ORB comparison

**Week 12**: Shapes
- Module 7 (Contours & Shapes)
- Object counting project
- Shape recognition

**Week 13**: Histograms
- Module 8 (Histograms)
- Contrast enhancement
- Color correction

**Week 14**: Segmentation
- Module 9 (Segmentation)
- Background removal
- GrabCut project

**Week 15**: Morphology
- Module 10 (Morphological Operations)
- Noise cleanup
- Feature extraction

**Week 16**: Advanced & Capstone
- Module 11 (Advanced Topics)
- Choose capstone project
- Build portfolio piece

### Study Tips

1. **Interactive Learning**: Use Jupyter notebooks for experimentation
2. **Visualize Everything**: Display intermediate results at each step
3. **Start Small**: Test algorithms on small images first (faster iteration)
4. **Read Error Messages**: Check image shape, dtype, and value ranges
5. **BGR vs RGB**: Remember OpenCV uses BGR by default
6. **Profile Performance**: Use `%%timeit` in Jupyter for optimization
7. **Incremental Development**: Add one feature at a time
8. **Read the Docs**: OpenCV documentation has great examples
9. **Build Projects**: Apply concepts to real problems
10. **Share Your Work**: Create a GitHub portfolio

### Common Pitfalls to Avoid

❌ Forgetting to convert BGR to RGB for matplotlib
❌ Not checking if image loaded (None check)
❌ Using even kernel sizes (must be odd)
❌ Modifying original image (use .copy())
❌ Incorrect data types (uint8 vs float32)
❌ Not normalizing when needed
❌ Ignoring aspect ratio when resizing

---

## 📊 Progress Tracking

Track your learning journey:

- [ ] **Week 0**: Environment setup and prerequisites
- [ ] **Week 1-2**: Foundations - Image I/O and pixel operations
- [ ] **Week 3-4**: Drawing and color spaces
- [ ] **Week 5-6**: Filtering and enhancement
- [ ] **Week 7-8**: Geometric transformations
- [ ] **Week 9-10**: Edge detection
- [ ] **Week 11**: Feature detection and matching
- [ ] **Week 12**: Contours and shape analysis
- [ ] **Week 13**: Histograms and equalization
- [ ] **Week 14**: Image segmentation
- [ ] **Week 15**: Morphological operations
- [ ] **Week 16**: Advanced topics
- [ ] **Final**: Complete capstone project

---

## 📂 Repository Structure

```
computer-graphics/
├── README.md                           # This file - complete course guide
├── .gitignore                          # Git configuration
│
└── notebooks/                          # All course modules
    ├── 00_Prerequisites.ipynb          # ✅ Python, NumPy, Math (Complete)
    ├── 01_Foundations.ipynb            # ✅ Image basics (Complete)
    ├── 02_Drawing_and_Color.ipynb      # ✅ Graphics & color (Complete)
    ├── 03_Filtering_Enhancement.ipynb  # ✅ Filters (Complete)
    ├── 04_Module.ipynb                 # ✅ Transformations (Complete)
    ├── 05_Module.ipynb                 # ✅ Edge detection (Complete)
    ├── 06_Module.ipynb                 # ✅ Features (Complete)
    ├── 07_Module.ipynb                 # ✅ Contours (Complete)
    ├── 08_Module.ipynb                 # ✅ Histograms (Complete)
    ├── 09_Module.ipynb                 # ✅ Segmentation (Complete)
    ├── 10_Module.ipynb                 # ✅ Morphology (Complete)
    └── 11_Module.ipynb                 # ✅ Advanced topics (Complete)
```

---

## 🤝 Contributing

This is an educational resource. Contributions welcome:
- Report issues or bugs
- Suggest improvements
- Add practice exercises
- Share project ideas

---

## 📜 License

This curriculum is provided for **educational purposes**.

**OpenCV License**: Apache 2.0
**Course Materials**: Educational use encouraged

---

## 🌟 Acknowledgments

- **OpenCV Community** for the incredible library
- **NumPy & Matplotlib** teams for essential tools
- **PyImageSearch** and **Learn OpenCV** for inspiration
- All contributors to computer vision education

---

## 📧 Support

**Questions or Issues?**
- Check the [OpenCV documentation](https://docs.opencv.org/4.x/)
- Review notebook comments and markdown cells
- Search for error messages in [Stack Overflow](https://stackoverflow.com/questions/tagged/opencv)
- Open an issue in this repository

---

**Version**: 3.0 (December 2024)
**Status**: Production Ready - All 12 modules complete
**Total Content**: 150+ hours of learning material

---

*Start your computer vision journey today. Master OpenCV, build amazing projects, and transform images into insights!* 🚀

