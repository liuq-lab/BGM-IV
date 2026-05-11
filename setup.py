import setuptools


setuptools.setup(
    name="bgm-iv",
    version="0.1.0",
    author="Qiao Liu",
    author_email="qiao.liu@yale.edu",
    description="Minimal BGM-IV package for demand-design IV experiments.",
    long_description=(
        "A clean research package containing the BGM-IV implementation "
        "and the three demand-design IV experiment configurations used for "
        "the NeurIPS 2026 submission."
    ),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy==1.24.2",
        "tensorflow==2.10.0",
        "tensorflow-probability==0.18.0",
        "pyyaml",
        "tqdm",
        "python-dateutil",
    ],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7, <3.11",
)
