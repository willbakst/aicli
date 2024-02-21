# aicli
A copy-cat version of Samuel Colvin's aicli implemented with Mirascope and Typer

## Installation

```shell
pip install wb-aicli
```

## Usage

### Single-Query Chat

This will make a single call to the LLM:

```shell
wb chat "what's the best developer tool for python data validation?"
Pydantic is considered one of the best developer tools for Python data validation due to its simplicity and robustness.
```

### Multi-Query Chat

This will maintain a history of the chat and inject it into the prompt with each query:

```shell
wb chat
```

```shell
wb: aicli ➤ what's the best developer tool for python data validation?
```

```
The best developer tool for Python data validation is the Pydantic library.
It provides data validation and settings management using Python type annotations.
```

```shell
wb: aicli ➤ why?
```

```
Pydantic is considered one of the best developer tools for Python data validation due to its simplicity and robustness.
It leverages Python type annotations for data validation, making the code easier to read and maintain.
Additionally, it provides powerful features for data parsing, validation, and settings management,
which are essential for maintaining data integrity and consistency in Python applications.
```

### File Extraction

This will extract structured information from a file based on your query:

```shell
wb extract pyproject.toml "package name and version"
```

```
Extracted Info: package_name='wb-aicli' package_version='0.1.1'
```
