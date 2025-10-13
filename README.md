# Exoplanet Detection Model

The purpose of this model is to detect exoplanets in lightcurves. This is done by estimating the period of transits based on the provided lightcurve data. 

The model is to be trained using the datasets to estimate the most likely periods of exoplanet transits. This model can then be used in conjunction with other machine learning models to detect exoplanets in lightcurves, or using the predictions to phase fold the lightcurve data and verify the presence of exoplanets on that period.

The model output is a $d$-dimensional vector where each element corresponds to a different period. The period represented for each entry depends on the configuration parameters used to generate the dataset.

## Features

- **Real Exoplanet Data**: Downloads and processes confirmed exoplanets from NASA's Kepler dataset
- **Negative Samples**: Generates samples from stars without confirmed exoplanets
- **Synthetic Transits**: Creates artificial exoplanet transits using realistic transit modeling
- **Modular Design**: Clean object-oriented architecture with reusable components for data generation and training.

## Project Structure

```
├── dataset_building/          # Core dataset generation
│   ├── main.py               # Main execution script
│   ├── StarDataset.py        # Base dataset class
│   ├── Positives.py          # Confirmed exoplanet dataset
│   ├── Negatives.py          # Negative samples dataset
│   ├── Synthetics.py         # Synthetic transit dataset
├── model/                    # Machine learning models
│   ├── model.py             # Main model architecture
├── data/                     # Generated datasets (Not available due to size)
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
        "period": {"min": 1, "max": 40, "bins": 100}
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
| `lightcurve_length` | Target lightcurve length in points | 3000 |
| `min_lightcurve_length` | Minimum valid lightcurve length | 1500 |

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

The dataset generation is complete and ready for model training (I think, lol).

## Troubleshooting

### Common Issues

**Download Time:**

The current download script is *slow*. The root cause is the lightkurve API. They are going to send me a SWAT team if I keep hitting their API.

## License

This project is licensed under the "I don't care" license. Just send me a message if you want to work on this.

## Acknowledgments

- **NASA Kepler Mission**: For the exoplanet discovery data
- **Lightkurve Team**: For the excellent Kepler data access tools
- **Batman Transit Modeling**: For realistic transit simulations
- **Kaggle**: For hosting the Kepler dataset



