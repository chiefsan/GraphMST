import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="graphmst",
    version="0.0.1",
    author="Nitisha Bharathi",
    author_email="nitishasam@gmail.com",
    description="A small graph package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chiefsan/graphmst",
    license="MIT",
    tests_require=["pytest"],
    py_modules="graph_mst",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
