import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eqfit",
    version="0.0.3",
    author="Sam Ingram",
    author_email="sp_ingram12@yahoo.co.uk",
    description="Equation fitting automation made simple with python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SamPIngram/eqfit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
   'numpy>=1.19.2',
   'pandas>=1.1.2',
   'matplotlib>=3.3.1',
   'scikit-learn>=0.23.2'
    ],
    python_requires='>=3.5',
)
