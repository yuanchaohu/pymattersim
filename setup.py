from setuptools import find_packages, setup

setup(
    name="PyMatterSim",  # Replace with your project name
    version="0.0.4",
    author="Yuan-Chao Hu",
    author_email="ychu0213@gmail.com",
    description="A python data analysis library for computer simulations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yuanchaohu/pymattersim",
    packages=find_packages(),  # Automatically discover packages
    install_requires=[i.strip("\n") for i in open("requirements.txt").readlines()],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
