# Easy-Bot-Service Prototype
## Configured for development with PyCharm

## Setup

Make sure to install the dependencies from requirements.txt. Additional requirement is llama-cpp-python package.

```bash
# install llama-cpp-python
pip install llama-cpp-python
```

If you encounter any problems while installing llama-cpp-python package, please refer to official documentation for troubleshooting at https://llama-cpp-python.readthedocs.io/en/latest/.

## Development Server

Start the server on `http://localhost:4000`:

```bash
python app.py
```

## Models

All models should be in GGUF format and be put inside ./models directory. Be sure to name them as they are named in sentence_similarity.py and text_generation.py files, or rename them in corresponding lines.

## Additional information

Easy-Bot-Service is built with Python 3.11. Please, before proceed be sure to check official documentation on corresponding technology.

# Copyright

EasyOneWeb LLC 2020 - 2024. All rights reserved. See LICENSE.md for licensing and usage information.