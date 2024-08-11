# Curvetopia: A Journey into the World of Curves

## Overview

Curvetopia is an interactive Python application designed to analyze, manipulate, and complete 2D curves. It provides tools for curve regularization, shape classification, symmetry detection, and curve completion. This project is ideal for those working with geometric data, computer graphics, or anyone interested in curve analysis and manipulation.

## Features

1. **Curve Regularization**: Identify and group similar curves using clustering techniques.
2. **Shape Classification**: Automatically classify curves into categories such as lines, circles, ellipses, and rectangles.
3. **Symmetry Detection**: Detect and visualize symmetry in curves using machine learning techniques.
4. **Curve Completion**: Complete partial curves based on their identified shape and characteristics.

## Output Examples

### Curve Completion
![Curve Completion 1](https://raw.githubusercontent.com/2105789/Shapeo/main/output/Curve%20Completion%201.png)
![Curve Completion 2](https://raw.githubusercontent.com/2105789/Shapeo/main/output/Curve%20Completion%202.png)

### Curve Regularization
![Regularize Curves](https://raw.githubusercontent.com/2105789/Shapeo/main/output/Regularize%20Curves.png)

### Shape Classification
![Shape Classification 1](https://raw.githubusercontent.com/2105789/Shapeo/main/output/Shape%20Classification%201.png)
![Shape Classification 2](https://raw.githubusercontent.com/2105789/Shapeo/main/output/Shape%20Classification%202.png)
![Shape Classification 3](https://raw.githubusercontent.com/2105789/Shapeo/main/output/Shape%20Classification%203.png)

### Symmetry Detection
![Symmetry Detection](https://raw.githubusercontent.com/2105789/Shapeo/main/output/Symmetry%20Detection.png)

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Steps

1. Clone the repository:
   ```
   git clone https://github.com/2105789/Shapeo.git
   cd curvetopia
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the address shown in the terminal (usually `http://localhost:8501`).

3. Upload a CSV file containing your curve data.

4. Use the sidebar to select the desired operation:
   - Regularize Curves
   - Shape Classification
   - Symmetry Detection
   - Curve Completion

5. Adjust parameters as needed and view the results.

## Input Format

The input should be a CSV file where each row represents a point on a curve. The format should be:

```
curve_id,segment_id,x,y
```

Where:
- `curve_id`: Identifier for each curve
- `segment_id`: Identifier for each segment within a curve
- `x`: X-coordinate of the point
- `y`: Y-coordinate of the point

## Project Structure

- `app.py`: Main Streamlit application file
- `modules.py`: Contains core functions for curve analysis and manipulation
- `requirements.txt`: List of Python dependencies

## Contributing

Contributions to Curvetopia are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


