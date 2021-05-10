import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="miRe2e",
    version="0.17",
    author="Jonathan Raad",
    author_email="jraad@sinc.unl.edu.ar",
    description="An end-to-end deep neural network based on Transformers for "
                "pre-miRNA prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sinc-lab/miRe2e",
    project_urls={
        "Webdemo": "https://sinc.unl.edu.ar/web-demo/miRe2e/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=['torch>=1.7',
                      'biopython>=1.78',
                      'scikit-learn>=0.23',
                      'tqdm']
)