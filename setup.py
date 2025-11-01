from setuptools import setup, find_packages

setup(
    name="cerebra-ai",
    version="1.0.0",
    description="Cerebra AI - Интеллектуальная текстовая система",
    author="Ваше имя",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
    ],
    python_requires=">=3.8",
)