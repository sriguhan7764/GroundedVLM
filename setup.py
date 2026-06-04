from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="groundedvlm",
    version="1.0.0",
    description="Open-Vocabulary Visual Grounding with Vision-Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your@email.com",
    url="https://github.com/yourusername/GroundedVLM",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "scripts*"]),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "timm>=0.9.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pycocotools>=2.0.6",
        "opencv-python>=4.8.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "einops>=0.6.0",
        "open_clip_torch>=2.20.0",
        "accelerate>=0.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
        "viz": [
            "plotly>=5.0",
            "wandb>=0.15.0",
            "ipython>=8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "groundedvlm-run=scripts.run_grounding:main",
            "groundedvlm-eval=scripts.eval_coco:main",
            "groundedvlm-bench=scripts.benchmark_encoders:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
