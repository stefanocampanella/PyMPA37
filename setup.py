import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pympa",
    version="0.0.1.dev0",
    author="Alessandro Vuan, Stefano Campanella",
    author_email="avuan@inogs.it, scampanella@inogs.it",
    description="A software package for phase match filtering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stefanocampanella/PyMPA37",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires='>=3.8',
)
