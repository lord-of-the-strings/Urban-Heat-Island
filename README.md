# Urban Heat Island Analysis on Ahmedabad, Gujarat

## Project Structure

```text
uhi/
├── data/                   # Local georeferenced raster warehouse
│   ├── ahmedabad_B4.tif    # Landsat 8 - Red Band (Surface Reflectance)
│   ├── ahmedabad_B5.tif    # Landsat 8 - Near-Infrared Band (Surface Reflectance)
│   ├── ahmedabad_B10.tif   # Landsat 8 - Thermal Infrared Band (Surface Temperature)
│   ├── ahmedabad_NDVI.tif  # Processed Normalised Difference Vegetation Index Grid
│   └── ahmedabad_LST.tif   # Processed Land Surface Temperature Matrix (°C)
├── download.py             # Ingestion engine utilizing Google Earth Engine (GEE) API
├── process.py              # Raster array processing, calibration, and edge-case handling
├── plot.py                 # Core statistical analysis, sampling, and regression diagnostics for a naive linear regression approach

```

---

## 🛰️ Phase 1: Data Engineering & Calibration Pipeline

### 1. High-Fidelity Cloud Masking (`download.py`)

Atmospheric interference (such as thin cirrus clouds, haze, or cloud edge shadows) artificially deflates thermal band readings and skews vegetation indicators. To guarantee data integrity, the ingestion pipeline cleans the Landsat 8 scene by performing bitwise operations on the 16-bit Quality Assessment band (`QA_PIXEL`).

The engine builds a strict binary clear-sky mask by asserting that the cloud, dilated cloud, and cloud shadow bits are strictly evaluated to zero:

* **Bit 3:** Cloud Shadow
* **Bit 4:** Dilated Cloud
* **Bit 5:** Cloud

```python
is_clear_shadow = qa.bitwiseAnd(1 << 3).eq(0)
is_clear_dilated = qa.bitwiseAnd(1 << 4).eq(0)
is_clear_cloud = qa.bitwiseAnd(1 << 5).eq(0)
clean_mask = is_clear_shadow & is_clear_dilated & is_clear_cloud
return image.updateMask(clean_mask)

```

Any contaminated pixels are automatically masked out as `NaN` values before local GeoTIFF serialization.

### 2. Scientific Matrix Formulations (`process.py`)

#### Normalised Difference Vegetation Index (NDVI)

Quantifies vegetation canopy density by calculating the normalized ratio between Near-Infrared (highly reflected by healthy plant cell walls) and Red light (absorbed by chlorophyll):

$$\text{NDVI} = \frac{\text{Band 5} - \text{Band 4}}{\text{Band 5} + \text{Band 4}}$$

* *Numerical Stability:* Wrapped within a NumPy error-state environment (`np.errstate(divide='ignore', invalid='ignore')`) to gracefully intercept divide-by-zero anomalies occurring over deep shadow zones or empty tile boundaries, mapping those instances to a steady `0.0`.

#### Land Surface Temperature (LST)

Calculated by transforming the raw digital numbers of Thermal Band 10 into Top-of-Atmosphere (TOA) spectral radiance, adjusting for split-window land surface emissivity, and converting the absolute thermodynamic scale from Kelvin to Celsius:

$$T_{\text{Celsius}} = T_{\text{Kelvin}} - 273.15$$

---

## 📈 Baseline Findings & Statistical Diagnostics

To test the direct statistical relationship between surface greenness and physical surface temperature, we extracted the corresponding pixel vectors from `ahmedabad_NDVI.tif` and `ahmedabad_LST.tif`, filtered out water bodies ($\text{NDVI} \le 0.0$), and computed a first-order linear regression profile (`plot.py`):

* **Slope ($m$):** Negative ($m < 0$)
* *Diagnostic:* Thermodynamically consistent. Confirms that as vegetative volume increases, localized surface skin temperatures drop due to active transpirational cooling.


* **Coefficient of Determination ($R^2$):** `0.1352`
* *Diagnostic:* NDVI variations explain exactly **13.52%** of the observed Land Surface Temperature variance across Ahmedabad.



### 🚨 Core Scientific Problem Statement

The high data scatter and low $R^2$ score prove that **standard empirical, statistical, or traditional machine learning models are fundamentally incapable of accurately modeling urban heat dynamics**.

A basic linear regression or "black-box" neural network misses the remaining **86.48%** of the microclimate profile because it ignores the unobserved physics governing the landscape:

1. **Material Heterogeneity & Thermal Inertia:** Tightly packed commercial concrete, asphalt transit networks, industrial metal roofs, and barren dirt patches all yield identical low NDVI signatures (around $0.0$ to $0.1$), yet they exhibit vastly different heat capacities and re-radiation rates.
2. **Microclimate Atmospheric Advection:** Heat is fluid. Micro-scale regional wind vectors physically carry heat across pixel boundaries (advection). A standard isolated pixel-by-pixel statistical framework cannot account for this horizontal thermal transport.

---

## 🧠 Phase 2 Core Architecture: The Physics-Informed Neural Network (PINN)

To bridge the 86.48% descriptive gap exposed by our baseline analytics, Phase 2 implements a **Physics-Informed Neural Network (PINN)**. Instead of acting as a blind statistical predictor, the neural network's internal optimization parameters are directly constrained by the fundamental laws of thermodynamics.

### The Multi-Component Loss Function

The PINN minimizes a unified loss function that forces predictions to simultaneously match real-world observations and satisfy the **2D Heat Advection-Diffusion Partial Differential Equation (PDE)**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \mathcal{L}_{\text{physics}} + \mathcal{L}_{\text{boundary}}$$

#### 1. Data Fidelity Loss ($\mathcal{L}_{\text{data}}$)

Ensures the network's predicted temperatures ($T_{\text{pred}}$) match the high-confidence, cloud-masked Landsat satellite pixels ($T_{\text{observed}}$):

$$\mathcal{L}_{\text{data}} = \frac{1}{N} \sum_{i=1}^{N} \left| T_{\text{pred}}(x_i, y_i, t_i) - T_{\text{observed}}(x_i, y_i, t_i) \right|^2$$

#### 2. Thermodynamic Regularization Loss ($\mathcal{L}_{\text{physics}}$)

Utilizes **Automatic Differentiation (Autograd)** to compute exact spatial and temporal partial derivatives of the neural network with respect to its input coordinates $(x, y, t)$, penalizing the model when it violates the 2D Heat Equation:

$$\frac{\partial T}{\partial t} = \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right) - \mathbf{u} \cdot \nabla T + S(\text{NDVI})$$

* **Thermal Diffusivity ($\alpha$):** Parameterizes spatial heat dispersion rates based on localized surface material classifications.
* **Advection Term ($-\mathbf{u} \cdot \nabla T$):** Links dynamic, horizontal atmospheric vector fluxes (Wind components $u, v$) extracted directly from localized hourly **ERA5-Land** reanalysis datasets.
* **Source/Sink Term ($S$):** Explicitly driven by the spatial configurations of the `ahmedabad_NDVI.tif` layer to simulate localized cooling capacities.

---

## 🚀 Execution & Operational Workflows

### 1. Ingest Raw Spatial Layers

Trigger the cloud-masked Earth Engine ingestion pipeline to download raw bands into `data/`:

```bash
python download.py

```

The data has already been downloaded and cleaned for you, so no need to run this script.

### 2. Compute Calibration Matrices

Compile raw arrays into physical georeferenced NDVI and LST layers:

```bash
python process.py

```

This script is a dependency of plot.py so just running plot.py works.

### 3. Run Statistical Regression & Analysis

Execute downsampled stochastic sampling to verify data ranges, compute coefficients, and output trend diagnostic visuals:

```bash
python plot.py

```
### Geophysical Notes

#### Normalised Difference Vegetation Index (NDVI)

Quantifies vegetation canopy density by calculating the normalized ratio between Near-Infrared (highly reflected by healthy plant cell walls) and Red light (absorbed by chlorophyll):

$$\text{NDVI} = \frac{\text{Band 5} - \text{Band 4}}{\text{Band 5} + \text{Band 4}}$$

* *Numerical Stability:* Wrapped within a NumPy error-state environment (`np.errstate(divide='ignore', invalid='ignore')`) to gracefully intercept divide-by-zero anomalies occurring over deep shadow zones or empty tile boundaries, mapping those instances to a steady `0.0`.

#### Land Surface Temperature (LST)

Calculated by transforming the raw digital numbers of Thermal Band 10 into Top-of-Atmosphere (TOA) spectral radiance, adjusting for split-window land surface emissivity, and converting the absolute thermodynamic scale from Kelvin to Celsius:

$$T_{\text{Celsius}} = T_{\text{Kelvin}} - 273.15$$

---

## 📈 Baseline Findings & Statistical Diagnostics

To test the direct statistical relationship between surface greenness and physical surface temperature, we extracted the corresponding pixel vectors from `ahmedabad_NDVI.tif` and `ahmedabad_LST.tif`, filtered out water bodies ($\text{NDVI} \le 0.0$), and computed a first-order linear regression profile (`plot.py`):

* **Slope ($m$):** Negative ($m < 0$)
* *Diagnostic:* Thermodynamically consistent. Confirms that as vegetative volume increases, localized surface skin temperatures drop due to active transpirational cooling.


* **Coefficient of Determination ($R^2$):** `0.1352`
* *Diagnostic:* NDVI variations explain exactly **13.52%** of the observed Land Surface Temperature variance across Ahmedabad.

