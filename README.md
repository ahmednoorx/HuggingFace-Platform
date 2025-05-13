# HuggingFace-Platform

Experiments and demos using the Hugging Face platform and ecosystem.

---

## Overview

This repository contains a collection of Jupyter notebooks demonstrating a variety of Natural Language Processing (NLP), Computer Vision, Speech, and Multimodal tasks using the Hugging Face Transformers, Datasets, and Diffusers libraries. The projects are designed for learning, experimentation, and showcasing the capabilities of state-of-the-art models.

---

## Contents

- **HuggingFace_demo.ipynb**  
  A comprehensive demo notebook covering:
  - Sentiment analysis
  - Text classification
  - Token classification (NER)
  - Question answering
  - Text generation
  - Summarization
  - Translation
  - Tokenization and encoding
  - Fine-tuning on IMDB dataset
  - Using Hugging Face pipelines for vision, speech, and multimodal tasks
  - ArXiv paper summarization

- **Text_Summarizer_project.ipynb**  
  Fine-tuning and evaluating a text summarization model (Pegasus) on the SAMSum dataset, including:
  - Data loading and preprocessing
  - Model training and evaluation (ROUGE metrics)
  - Saving and loading models/tokenizers
  - Inference and summary generation

- **Text_to_speech_generation_with_LLM_with_hugging_face.ipynb**  
  Text-to-speech generation using Hugging Face pipelines (e.g., Suno Bark), including:
  - Generating speech audio from text
  - Playing audio output in the notebook

- **Copy_of_Text_to_Image_generation_with_LLM_with_hugging_face.ipynb**  
  Text-to-image generation using Stable Diffusion and Diffusers, including:
  - Generating images from text prompts
  - Customizing generation parameters (steps, size, negative prompts)
  - Displaying generated images

---

## Getting Started

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)
- [Jupyter Notebook](https://jupyter.org/)
- (Optional) NVIDIA GPU for faster model inference

### Installation

Install the required libraries (run in your terminal or notebook):

```bash
pip install transformers diffusers datasets accelerate matplotlib torch arxiv sacrebleu rouge_score py7zr
```

Some notebooks may require additional packages (see the first cells of each notebook).

---

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ahmednoorx/HuggingFace-Platform.git
   cd HuggingFace-Platform
   ```

2. **Open any notebook in Jupyter:**
   ```bash
   jupyter notebook
   ```
   Or open directly in VS Code.

3. **Run the cells step by step.**  
   Follow the markdown instructions in each notebook.

---

## Notes

- **No secrets or API keys** are included in this repository.
- If you use your own Hugging Face API keys or tokens, store them in a `.env` file and add `.env` to `.gitignore`.
- Some models may require significant RAM/VRAM. Use a GPU-enabled environment for best performance.

---

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/index)
- [Hugging Face Diffusers Documentation](https://huggingface.co/docs/diffusers/index)

---

## License

This repository is for educational and research purposes.  
Check individual model licenses on the Hugging Face Hub before commercial use.

---

## Author

[ahmednoorx](https://github.com/ahmednoorx)