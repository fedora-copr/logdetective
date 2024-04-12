Log Detective
=============

A Python tool to analyze logs using a Language Model (LLM) and Drain template miner.

Installation
------------

    # optionaly when you prefer system packages
    dnf install python3-jsonpickle python3-tiktoken
    # install all remaining packages
    pip install .

Usage
-----

To analyze a log file, run the script with the following command line arguments:
- `url` (required): The URL of the log file to be analyzed.
- `--model` (optional, default: "Mistral-7B-Instruct-v0.2-GGUF"): The path or URL of the language model for analysis.
- `--summarizer` (optional, default: "drain"): Choose between LLM and Drain template miner as the log summarizer. You can also provide the path to an existing language model file instead of using a URL.
- `--n_lines` (optional, default: 5): The number of lines per chunk for LLM analysis. This only makes sense when you are summarizing with LLM.

Example usage:

    logdetective https://example.com/logs.txt --model https://mymodel.co/model_file --summarizer drain --n_lines 10


Contributing
------------

Contributions are welcome! Please submit a pull request if you have any improvements or new features to add. Make sure your changes pass all existing tests before submitting.

To develop logdetective, you should fork this repository, clone your fork, and install dependencies using pip:

    git clone https://github.com/yourusername/logdetective.git
    cd logdetective
    pip install .

Make changes to the code as needed and run pre-commit.

License
-------

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.
