from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cerebra-ai",
    version="2.0.0",
    author="AI Developer",
    author_email="ai@example.com",
    description="Cerebra AI - Advanced Intelligent Text System with GPT-like Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cerebra-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "lxml>=4.9.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
    ],
    python_requires=">=3.8",
    keywords="ai, artificial-intelligence, gpt, transformer, nlp, text-generation",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/cerebra-ai/issues",
        "Source": "https://github.com/yourusername/cerebra-ai",
        "Documentation": "https://github.com/yourusername/cerebra-ai/blob/main/DOCUMENTATION.md",
    },
)