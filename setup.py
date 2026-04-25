from setuptools import find_packages, setup

setup(
    name="smart-vision",
    version="0.2.0",
    description="Real-time scene segmentation with a live camera feed.",
    packages=find_packages(include=["project", "project.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "opencv-python",
        "pillow",
        "torch",
        "torchvision",
        "transformers",
    ],
    entry_points={
        "console_scripts": [
            "smart-vision=project.main:main",
        ]
    },
)
