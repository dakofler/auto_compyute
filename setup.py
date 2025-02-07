"""Setup file for auto compyute."""

import pathlib

from setuptools import find_packages, setup

setup(
    name="compyute_autograd",
    version=pathlib.Path("auto_compyute/VERSION").read_text(encoding="utf-8"),
    description="AutoCompyute: Lightweight Deep Learning with Pure Python.",
    long_description=pathlib.Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/dakofler/auto_compyute/",
    author="Daniel Kofler",
    author_email="dkofler@outlook.com",
    license="MIT",
    project_urls={
        "Source Code": "https://github.com/dakofler/auto_compyute",
        "Issues": "https://github.com/dakofler/auto_compyute/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.12",
    install_requires=[
        "cupy_cuda12x>=13.0.0",
        "numpy>=1.26.4",
        "opt_einsum>=3.4.0",
        "mermaid-python>=0.1",
    ],
    extras_require={
        "dev": [
            "mypy>=1.11.2",
            "pytest>=8.2.0",
            "pytest-cov>=5.0.0",
            "torch>=2.5.0",
            "twine>=5.1.1",
            "wheel>=0.43.0",
            "Sphinx>=7.4.7",
            "pydata_sphinx_theme>=0.15.4",
        ]
    },
    packages=find_packages(exclude=["tests", ".github", ".venv", "docs"]),
    include_package_data=True,
)
