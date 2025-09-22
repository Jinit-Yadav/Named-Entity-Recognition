import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

    __version__ = "0.0.0"

    REPO_NAME = "Text-Summarization-Project"
    AUTHOR_USER_NAME = "JINIT YADAV"
    SRC_REPO = "textSummarizer"
    AUTHOR_EMAIL = "yadavjinit8@gmail.com"
    
    setuptools.setup(
        name=REPO_NAME,
        version=__version__,
        author=AUTHOR_USER_NAME,
        author_email=AUTHOR_EMAIL,
        description="A small python package for text summarization",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/JINITYADAV/" + REPO_NAME,
        packages=setuptools.find_packages(where="src"),
        package_dir={"": "src"},
    )