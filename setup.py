from setuptools import setup, find_packages

setup(
    name="FinanceDatabase",        # Name of your package
    version="0.1",                 # Version number
    packages=find_packages(),      # Automatically find subpackages
    install_requires=[],           # Add dependencies if needed
    author="Chidi & Zino",            # Your name
    description="A financial database for Python",
    #long_description=open("README.md").read(),  # Optional: Use your README as a description
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
