from setuptools import setup, find_packages

setup(
    name="cascade2p",
    version="1.0",
    description="Calibrated inference of spiking from calcium ΔF/F data using deep networks",
    author="Peter Rupprecht",
    author_email="",
    packages=find_packages(),
    python_requires=">=3.7, <3.8",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "tensorflow==2.3",  # pip install CPU and GPU tensorflow
        "keras==2.3.1",
        "h5py",
        "seaborn",
        "ruamel.yaml",
        "spyder",
    ],
)
