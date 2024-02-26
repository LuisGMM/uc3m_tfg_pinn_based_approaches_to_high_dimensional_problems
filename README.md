# README

## Installing

### In Windows

```bash
.\install.bat
```

```bash
.\venv_win\Scripts\Activate
```

```bash
python .\src\tfg\nn\cos.py
```

### In Linux

Run the following command:

```bash
make install
```

Make sure to have installed some basic packages like `make`. Otherwise, install
the requirements from `requirements/required.txt` and go on.

To make and see the documentation:

```bash
make docs && google-chrome $(pwd)/docs/build/html/index.html
```
