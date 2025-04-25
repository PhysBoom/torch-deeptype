from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="torch-deeptype",
    version="0.1.0",
    description="PyTorch implementation of DeepType with clustering and sparsity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Matthew Chak",
    author_email="mchak@calpoly.edu",
    url="https://github.com/physboom/torch-deeptype",
    packages=find_packages(exclude=["test", "tests"]),
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "scikit-learn",
        "numpy",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
