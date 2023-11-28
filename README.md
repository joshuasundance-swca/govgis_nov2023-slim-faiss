---
title: govgis_nov2023-slim-faiss
emoji: 🌎
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.28.2
app_file: app.py
pinned: true
license: mit
---

# govgis_nov2023-slim-faiss

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

[![Push to HuggingFace Space](https://github.com/joshuasundance-swca/govgis_nov2023-slim-faiss/actions/workflows/hf-space.yml/badge.svg)](https://github.com/joshuasundance-swca/govgis_nov2023-slim-faiss/actions/workflows/hf-space.yml)
[![Open HuggingFace Space](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/joshuasundance/govgis_nov2023-slim-faiss)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)


# govgis_nov2023-slim-faiss

🤖 This README was written by GPT-4. 🤖

## Features

- **Semantic Search on GIS Metadata**: Leverages the `govgis_nov2023` dataset to provide detailed insights into numerous GIS servers and layers.
- **Natural Language Query Processing**: Uses Claude-Instant and Claude-2.1 models to interpret and rephrase user queries (optional).
- **Advanced Document Retrieval**: Integrates FAISS vector store for efficient and relevant document retrieval based on query semantics.
- **Customizable User Experience**: Sidebar controls to adjust search parameters and input fields for queries.

## Dataset Overview

- **Content**: The app is built around the `govgis_nov2023` dataset, which documents metadata from 1684 government ArcGIS servers, detailing almost a million individual layers.
- **Unique Snapshot**: Provides a unique snapshot of these servers, with metadata including field information for feature layers and cell size for raster layers.

## User Interface Guide

- Adjust search settings like result limits and response generation parameters in the sidebar.
- Securely enter your Anthropic API key for model access.
- Submit natural language queries related to GIS data.

## Contributions

We welcome contributions. Please follow the standard fork and pull request process.

## Support and Contact

For support, please raise an issue on GitHub or in the HuggingFace space.

## License

This project is under the [MIT License](LICENSE.md).

## Acknowledgments

Thanks to the Huggingface and Streamlit communities, and special acknowledgment to Joseph Elfelt and the creators of the `restgdf` library for their contributions to the GIS field.


## TODO
- [ ] Add an open source model like `HuggingFaceH4/zephyr-7b-beta`
- [ ] Hybrid search w/ bm25 or similar
- [ ] Find a lightweight way to incorporate geospatial filtering
- [ ] Add more parameters
