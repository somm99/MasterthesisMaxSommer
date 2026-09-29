# MasterthesisMaxSommer
This is the prototype developed during the masterthesis with the title "Implementation, prototyping, and evaluation of visualization methods for tangles in datasets". The thesis was written at the FernUniversität Hagen. It prototypes 8 different visualizations for tangles.

The project heavily relies on the tangles package by Reinhard Diestel - Copyright (c) 2019-2023, Tangle Software Contributors The package can be found at: https://github.com/tangle-software/tangles.git

To install the protoype and run it locally on your system, you need to follow the installation instructions


## Requirements

- Python 3.11 (the application was built and tested on this version; other versions are not guaranteed to work)
- Git

## Installation

1. **Clone the repository and open it in a terminal**

```
   git clone https://github.com/somm99/MasterthesisMaxSommer.git
   cd MasterthesisMaxSommer
```

   If you use an IDE such as PyCharm, you can also open the cloned folder there and use its built-in terminal.

2. **Create a virtual environment**

```
   python -m venv .venv
```

3. **Activate the virtual environment**

   Windows:
```
   .venv\Scripts\activate
```

   macOS / Linux:
```
   source .venv/bin/activate
```

   The terminal prompt should now start with `(.venv)`.

4. **Install the dependencies**

```
   pip install -r requirements.txt
```

5. **Install the `tangles` package**

```
   cd tangles-main
   pip install -e .
```

6. **(Optional) Verify the installation**

```
   pytest -n auto
```

   Then return to the project root:

```
   cd ..
```

## Starting the application

1. Open a terminal in the project root folder (`MasterthesisMaxSommer`).
2. Activate the virtual environment (skip this if `(.venv)` is already shown in the prompt), see step 3 above.
3. Start the app:

```
   python run_app.py
```
The app opens in your browser