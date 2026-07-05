from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


setup(
    name="bgm-iv",
    version="0.1.0",
    author="Guyue Luo, Qiao Liu",
    description=(
        "Latent Bayesian generative modeling for nonlinear instrumental-variable "
        "analysis with high-dimensional covariates."
    ),
    long_description=_read_text(ROOT / "README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/liuq-lab/BGM-IV",
    project_urls={
        "Source": "https://github.com/liuq-lab/BGM-IV",
        "Issues": "https://github.com/liuq-lab/BGM-IV/issues",
        "README": "https://github.com/liuq-lab/BGM-IV#readme",
        "Paper": "https://arxiv.org/abs/2605.07029",
    },
    packages=find_packages(include=["bgm_iv", "bgm_iv.*"]),
    include_package_data=False,
    install_requires=[
        "numpy==1.24.2",
        "tensorflow==2.10.0",
        "tensorflow-probability==0.18.0",
        "pyyaml",
        "tqdm",
        "python-dateutil",
    ],
    license="MIT",
    license_files=["LICENSE"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9,<3.11",
)
