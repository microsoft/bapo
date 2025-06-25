# Overview

This repository contains all scripts for re-producing the results of the our paper 
[Lost in Transmission: When and Why LLMs Fail to Reason Globally](https://arxiv.org/abs/2505.08140).

**Reference:**
```bibtex
@misc{schnabel2025bapo,
      title={Lost in Transmission: When and Why LLMs Fail to Reason Globally}, 
      author={Tobias Schnabel and Kiran Tomlinson and Adith Swaminathan and Jennifer Neville},
      year={2025},
      eprint={2505.08140},
      url={https://arxiv.org/abs/2505.08140}, 
}
```

## How To Run

### Install requirements
We recommend using a new environment for the requirements. You can do this using `venv` or `conda`.

For `conda`:
```bash
conda env create -f environment.yml
conda activate runbapo
```

For `venv`:
```bash  
python -m venv runbapo
source runbapo/bin/activate  # On Windows use `runbapo\Scripts\activate`
pip install -r requirements.txt
```

### Set up API keys
Set the `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GOOGLE_API_KEY` environment variables, e.g.,

```bash
 export OPENAI_API_KEY=<your_openai_api_key>
 export ANTHROPIC_API_KEY=<your_anthropic_api_key>
 export GOOGLE_API_KEY=<your_google_api_key>
```

### Pre-process the Space Digest dataset
1. Download the raw Space digest dataset from [this link](https://github.com/stangelid/qt) as well as the subset from the [ZeroScrolls benchmark](https://huggingface.co/datasets/tau/zero_scrolls/resolve/main/space_digest.zip) benchmark.
2. Place the files in a new directory called `processed_data`.
3. Run the preprocessing script:
   ```bash
   python preprocess_space_digest.py
   ```

### Run Experiments
```bash
python bapo_experiments.py
```
Code structure:
- Each experiment is implemented as a subclass inheriting from the `Experiment` base class.
- `generate_data()` provides the main functionality for generating the data used in the experiment

### Generate Plots
```bash
python plot_results.py
```

## Questions and Issues
If you have any questions or issues regarding this code, please open an issue on the GitHub repository.
For questions related to the paper, please contact the authors via email.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.