from setuptools import setup, find_packages


def read_readme() -> str:
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "fin_pet: A robust Python library for derivative pricing, greeks, implied volatility, and risk metrics."


setup(
    name="fin_pet",
    version="0.1.0",
    description="Derivative pricing, greeks, implied volatility, and risk metrics",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Georgiy Dzakhoev",
    url="https://github.com/gdzakhoev/fin_pet",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.8",
        "pandas>=1.3",
        "scikit-learn>=1.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU General Public License v3.0",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Financial and Insurance Industry",
    ],
    include_package_data=True,
)


