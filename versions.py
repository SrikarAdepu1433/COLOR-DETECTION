import streamlit as st
import cv2
import matplotlib
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw

# Print library versions
print(f"streamlit=={st.__version__}")
print(f"opencv-python=={cv2.__version__}")
print(f"matplotlib=={matplotlib.__version__}")
print(f"numpy=={np.__version__}")
print(f"scikit-learn=={np.__version__}")
print(f"Pillow=={Image.__version__}")