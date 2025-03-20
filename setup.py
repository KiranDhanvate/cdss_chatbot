from setuptools import setup, find_packages

setup(
    name="cdss_chatbot",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "Django>=4.2.7",
        "google-generativeai>=0.3.1",
        "python-dotenv>=1.0.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "matplotlib>=3.7.2",
        "nltk>=3.8.1",
        "scikit-learn>=1.3.0",
        "transformers>=4.30.2",
        "tqdm>=4.65.0",
        "whitenoise>=6.6.0",
        "django-cors-headers>=4.3.1",
        "gunicorn>=21.2.0",
    ],
    python_requires=">=3.9",
) 