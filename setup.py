from setuptools import setup

from ezancestry import __project__

setup(
    name="ezancestry",
    version="0.1",
    description="Genetic ancestry predictions",
    url="http://github.com/arvkevi/ezancestry",
    author="Kevin Arvai",
    author_email="arkvevi@gmail.com",
    license="MIT",
    packages=["ezancestry"],
    include_package_data=True,
    keywords="genetics ancestry dimensionality-reduction machine-learning",
    zip_safe=False,
    entry_points={
        "console_scripts": [
            f"{__project__} = {__project__}.__main__:main",
        ]
    },
    install_requires=[
        "fire",
        "loguru",
        "numpy",
        "pandas",
        "plotly",
        "scikit-learn",
        "scipy",
        "snps",
        "streamlit",
        "umap-learn",
    ],
    extras_require={"dev": ["pytest", "pytest-cov", "isort"]},
)
