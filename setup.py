"""
VisionGuard-AD — Setup Script
Unsupervised Visual Anomaly Detection for Industrial Manufacturing QC.
"""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8") if (this_directory / "README.md").exists() else ""

setup(
    name="visionguard-ad",
    version="1.0.0",
    author="VisionGuard Team",
    description="Unsupervised Visual Anomaly Detection for Industrial Manufacturing Quality Control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/visionguard/VisionGuard-AD",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "seaborn>=0.12.0",
        "streamlit>=1.28.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "Pillow>=10.0.0",
        "pandas>=2.0.0",
        "faiss-cpu>=1.7.4",
        "einops>=0.7.0",
        "timm>=0.9.0",
        "FrEIA>=0.2",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
    entry_points={
        "console_scripts": [
            "vg-train=train:main",
            "vg-evaluate=evaluate:main",
            "vg-inference=inference:main",
            "vg-benchmark=benchmark:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    license="MIT",
)
