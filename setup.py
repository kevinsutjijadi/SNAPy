from setuptools import find_packages, setup

with open("README.md", 'r') as f:
    long_description = f.read

setup(
    name="SNAPy",
    version="0.01",
    description="""Spatial Network Analysis Python Module
 A package of urban network analysis tools based on Geopandas dataframe and networkx pathfinding""",
    package_dir={"": "SNAPy"},
    packages=find_packages(where="SNAPy"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kevinsutjijadi/SNAPy",
    author="kevinsutjijadi",
    author_email="kevinsutjijadi@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    install_requires=["pandas >= 1.5.2",
                      "geopandas >= 0.9.0",
                      "networkx >= 2.7.1",
                      "scipy >= 1.10.0",
                      "numpy >= 1.24.1",
                      "shapely >= 2.0.0"
                      ],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.11",
)