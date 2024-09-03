from setuptools import setup, find_packages

setup(
    name="beyondfid",
    version="0.1.0",
    packages=find_packages(),  # Automatically find and include your package
    entry_points={
        "console_scripts": [
            "beyondfid=beyondfid.run:main",
        ],
    },
    install_requires=[
        # List your dependencies here, e.g.:
        # "numpy>=1.18.0",
    ],
    author="Mischa Dombrowski",
    author_email="mischa.dombrowski@gmail.com",
    description="Python package to efficiently compute common metrics used for unconditional image generation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    #url="https://your_project_url.com",
    #classifiers=[
    #    "Programming Language :: Python :: 3",
    #    "License :: OSI Approved :: MIT License",
    #    "Operating System :: OS Independent",
    #],
    python_requires='>=3.6',  # Adjust this as necessary
)