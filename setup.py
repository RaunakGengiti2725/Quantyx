from setuptools import setup, find_packages

setup(
    name="quantumproject",
    version="0.1.0",
    description="Quantum Geometry Project - Mapping entanglement entropy to discrete bulk geometry",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "networkx",
        "matplotlib",
        "pennylane",
        "scipy",
        "dgl",
        "tensorboard",
        "scikit-learn",
    ],
    python_requires=">=3.8",
) 