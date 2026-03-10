from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = []
    for line in fh:
        line = line.strip()
        if line and not line.startswith("#"):
            if "#" in line:
                line = line.split("#")[0].strip()
            if line:
                requirements.append(line)

setup(
    name="unicbench-evaluation",
    version="1.0.0",
    author="Unified Counting Benchmark Team",
    description="Official evaluation toolkit for Unified Counting Benchmark for MLLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rongchenggang/UNICBench",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
)