# Exoplanet Detection Dataset Builder

A comprehensive Python toolkit for building machine learning datasets from Kepler exoplanet data, including real confirmed exoplanets, negative samples, and synthetic transit injections.

## Features

- **Real Exoplanet Data**: Downloads and processes confirmed exoplanets from NASA's Kepler dataset
- **Negative Samples**: Generates samples from stars without confirmed exoplanets
- **Synthetic Transits**: Creates artificial exoplanet transits using realistic transit modeling
- **Progress Tracking**: Visual progress bars and comprehensive logging
- **Data Visualization**: Built-in tools for inspecting and analyzing generated datasets
- **Modular Design**: Clean object-oriented architecture with reusable components

## Project Structure

```
├── dataset_building/          # Core dataset generation
│   ├── main.py               # Main execution script
│   ├── StarDataset.py        # Base dataset class
│   ├── Positives.py          # Confirmed exoplanet dataset
│   ├── Negatives.py          # Negative samples dataset
│   ├── Synthetics.py         # Synthetic transit dataset
├── model/                    # Machine learning models
│   ├── encoder.py           # Lightcurve encoder
│   ├── model.py             # Main model architecture
│   └── processor.py         # Parameter estimation
├── data/                     # Generated datasets
│   ├── samples/             # Lightcurve arrays (.npz files)
│   │   ├── positive/        # Confirmed exoplanets
│   │   ├── negative/        # Negative samples
│   │   └── synthetic/       # Synthetic transits
│   └── *.csv                # Dataset metadata
├── resources/               # Reference data and documentation
├── config.json              # Configuration settings
└── requirements.txt         # Python dependencies
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd exoplanet-dataset-builder

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `config.json` to set your parameters:

```json
{
    "username": "your_kaggle_username",
    "key": "your_kaggle_api_key",
    "dataset_name": "nasa/kepler-exoplanet-search-results",
    "star_catalog_files": "resources/estrellas_cleaned.txt",
    "positive_sample_count": 500,
    "negative_sample_count": 1000,
    "synthetic_samples_count": 250,
    "planets_per_synthetic_star": 4,
    "period_filter": 40,
    "valid_lightcurve_lenght_threshold": 1000,
    "distribution_params": {
        "period": {"min": 1, "max": 40, "bins": 100},
        "duration": {"min": 1, "max": 20, "bins": 100}
    },
    "lightcurve_length": 1500,
    "min_lightcurve_length": 1000,
    "positive": "data/samples/positive",
    "negative": "data/samples/negative",
    "synthetic": "data/samples/synthetic"
}
```

### Generate Dataset

```bash
# Run the main dataset generation
python dataset_building/main.py
```

## Dataset Types

### Positives Dataset
- **Source**: NASA Kepler confirmed exoplanets
- **Content**: Lightcurves with confirmed planetary transits
- **Features**: Period, duration, and error distributions
- **Usage**: Training data for exoplanet detection

### Negatives Dataset
- **Source**: Stars without confirmed exoplanets
- **Content**: Clean stellar lightcurves
- **Features**: Zero-weight distributions (no transits)
- **Usage**: Negative training examples

### Synthetics Dataset
- **Source**: Artificial transit injection
- **Content**: Real stellar lightcurves with injected transits
- **Features**: Realistic transit parameters and errors
- **Usage**: Data augmentation and validation

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `positive_sample_count` | Number of confirmed exoplanets to process | 500 |
| `negative_sample_count` | Number of negative samples to process | 1000 |
| `synthetic_samples_count` | Number of synthetic transits to generate | 250 |
| `planets_per_synthetic_star` | Max planets per synthetic star | 4 |
| `period_filter` | Maximum period in days | 40 |
| `lightcurve_length` | Target lightcurve length in points | 1500 |
| `min_lightcurve_length` | Minimum valid lightcurve length | 1000 |

## Data Format

Each generated sample is saved as a `.npz` file containing:

- **`arr_0`**: Lightcurve flux array (normalized)
- **`arr_1`**: Period distribution weights

## Advanced Usage

### Custom Dataset Generation

```python
from dataset_building.Positives import Positives
from dataset_building.Negatives import Negatives
from dataset_building.Synthetics import Synthetics

# Load configuration
import json
with open('config.json') as f:
    config = json.load(f)

# Initialize datasets
positives = Positives(config)
negatives = Negatives(config)
synthetics = Synthetics(config)

# Generate data
positives.download_data()
negatives.download_data()
synthetics.download_data()
```

## TODO

This project currently focuses on dataset generation. The next major step is to implement and train the machine learning model:

- **Model Training**: Implement training pipeline for the neural network models in the `model/` directory
- **Hyperparameter Tuning**: Optimize model parameters for exoplanet detection
- **Evaluation Metrics**: Add comprehensive evaluation and validation
- **Model Deployment**: Create inference pipeline for new lightcurve data
- **Performance Optimization**: Optimize training speed and memory usage

The dataset generation is complete and ready for model training.

## Troubleshooting

### Common Issues

**Permission Errors on Windows:**
- The cache cleanup handles Windows file locking automatically
- If issues persist, manually clear `~/.lightkurve/cache/`

**Memory Issues:**
- Reduce `sample_count` parameters in config.json
- Process datasets individually instead of all at once

**Download Failures:**
- Check Kaggle API credentials in config.json
- Ensure internet connection is stable
- Verify lightkurve cache permissions

### Performance Tips

- **Parallel Processing**: Run different dataset types simultaneously
- **Incremental Generation**: Resume interrupted downloads automatically
- **Cache Management**: Automatic cleanup every 20 downloads
- **Progress Tracking**: Visual progress bars for long operations

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **lightkurve**: Kepler data access
- **batman-package**: Transit modeling
- **kagglehub**: Dataset downloads
- **tqdm**: Progress bars
- **torch**: Deep learning (for model training)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **NASA Kepler Mission**: For the exoplanet discovery data
- **Lightkurve Team**: For the excellent Kepler data access tools
- **Batman Transit Modeling**: For realistic transit simulations
- **Kaggle**: For hosting the Kepler dataset

---

This project provides a solid foundation for exoplanet detection research. The dataset generation pipeline is complete and ready for machine learning model development.

