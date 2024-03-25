Log Buddy
=========

A Python tool to analyze logs using Language Model (LLM) and Drain template miner.

Installation
------------

    pip install .

Usage
-----

To analyze a log file, run the script with the following command line arguments:
- url (required): The URL of the log file to be analyzed.
- --model: The path or URL of the language model for analysis. If not provided, it will use a predefined one (Mistral-7B-Instruct-v0.2-GGUF).
- --summarizer (optional, default: "drain"): Choose between LLM and Drain template miner as the log summarizer. You can also provide the path to an existing language model file instead of using a URL.
- --n_lines (optional, default: 5): The number of lines per chunk for LLM analysis.

Example usage:

    logbuddy https://example.com/logs.txt --model=https://mymodel.co/model_file --summarizer=drain --nlines=10


Contributing
------------

Contributions are welcome! Please submit a pull request if you have any improvements or new features to add. Make sure your changes pass all existing tests before submitting.

To develop logbuddy, you should fork this repository, clone your fork, and install dependencies using pip:

    git clone https://github.com/yourusername/logbuddy.git
    cd logbuddy
    pip install -r requirements.txt .

Make changes to the code as needed and run pre-commit.

License
-------

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.
