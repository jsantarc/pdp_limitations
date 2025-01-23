# pdp_limitations

# Model Interpretation Comparison Package

This package demonstrates the limitations of Partial Dependence Plots (PDPs) and M-Plots while highlighting the advantages of Accumulated Local Effects (ALE) Plots in Python.

## Features
- Generate synthetic data to demonstrate interpretation challenges
- Visualize and compare PDP, M-Plot, and ALE interpretations
- Show how ALE plots overcome common limitations of PDPs see Examples
- Demonstrate real-world examples with UCI datasets

## Installation
```bash
pip install -r requirements.txt
```

## Required Dependencies
```python
tqdm
numpy 
pandas 
matplotlib 
seaborn 
scikit-learn
alibi
ucimlrepo
```

## Usage
See `example.ipynb` for detailed examples and comparisons of different interpretation methods.

## References
1. **Interpreting Black-Box Supervised Learning Models Via Accumulated Local Effects**
   - Christoph Molnar
   - [Link to the Book](https://christophm.github.io/interpretable-ml-book/ale.html)

2. **YouTube Video on Interpreting Models**
   - [Watch the Video](https://www.youtube.com/watch?v=06knUxoig9Y&t=248s)

3. **Alibi Explainers API Documentation**
   - [Alibi Explainers API](https://docs.seldon.io/projects/alibi/en/latest/api/alibi.explainers.html)

4. **IML (Interpretable Machine Learning)**
   - [IML Documentation](https://slds-lmu.github.io/iml/)

## Contributing
Feel free to open issues or submit pull requests for improvements.

## License
MIT License
