from setuptools import find_packages, setup
import os

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name="snapy-toolbox",
    version="0.2.14",
    description="""Spatial Network Analysis Python Module
 A package of urban network analysis tools based on Geopandas dataframe and networkx pathfinding""",
    # package_dir={"": "SNAPy"},
    packages=['', 'SNAPy', 'SNAPy.SGACy'],
    package_data={'SNAPy.SGACy': ['*.pyx', '*.cpp', '*.pyd', '*.so'],},
    include_package_data=True,
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
    install_requires=["pandas >= 1.5.3",
                      "geopandas >= 1.0.1",
                      "scipy >= 1.10.0",
                      "numpy >= 1.23.5",
                      "shapely >= 2.0.0",
                      "pydeck >= 0.8.0"
                      ],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires="~=3.11",
)

# python setup.py sdist bdist_wheel
# twine upload dist/*