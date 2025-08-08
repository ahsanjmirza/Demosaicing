# Demosaicing Algorithm

This repository provides an edge-aware, variance‐guided demosaicing implementation for Bayer‐pattern raw images, based on Stephen H. Westin’s ISO 12233 test chart methodology. It converts single‐channel CFA (Color Filter Array) data into full-resolution RGB images.

---

## Contents

* **CFAFilt.py**
  Core class implementing adaptive green interpolation and red/blue residual reconstruction.
* **run.py**
  High-level script: reads configuration, loads raw data, runs the demosaicer, and writes PNG outputs.
* **config.json**
  User‐editable parameters for input/output paths, CFA pattern, thresholds, and image dimensions.
* **/inputs/**
  Folder for your raw `.raw` or packed `.png/.jpg` CFA images.
* **/outputs/**
  Destination for demosaiced `.png` files.

---

## Features

* **Adaptive Green Interpolation**
  Chooses horizontal, vertical, or weighted interpolation based on local gradients and variance.
* **Red/Blue Reconstruction**
  Computes high-frequency residuals along 45°/135° diagonals for R/B channels.
* **Configurable Thresholds**
  Fine-tune `g_grad_th`, `rb_grad_th`, and variance threshold (`var_th`) for your sensor/noise characteristics.
* **Supports All Major Bayer Patterns**
  RGGB, BGGR, GRBG, GBRG.

---

## Installation

1. **Clone this repo**

   ```bash
   git clone https://github.com/yourusername/demosaic-algorithm.git
   cd demosaic-algorithm
   ```

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install numpy scipy matplotlib tqdm imageio
   ```

---

## Configuration

Edit `config.json` to suit your data and preferences:

```json
{
  "image_path": "./inputs",
  "output_path": "./outputs",
  "rows": 1200,
  "cols": 1920,
  "bitdepth": 8,
  "cfa": "RGGB",
  "g_grad_th": 0.0,
  "rb_grad_th": 0.7,
  "var_th": 10.0
}
```

| Key            | Description                                                        |
| -------------- | ------------------------------------------------------------------ |
| `image_path`   | Directory containing raw CFA files or packed CFA images            |
| `output_path`  | Directory for saving demosaiced PNGs                               |
| `rows`, `cols` | Raw image dimensions (used if reading `.raw` binary data directly) |
| `bitdepth`     | Bits per sample in raw data (e.g. 8, 10, 12, 14, 16)               |
| `cfa`          | Bayer pattern: `"RGGB"`, `"BGGR"`, `"GRBG"`, or `"GBRG"`           |
| `g_grad_th`    | Green‐channel gradient threshold for switching interpolation modes |
| `rb_grad_th`   | Red/Blue diagonal gradient threshold for residual selection        |
| `var_th`       | Variance threshold for weighted vs. unweighted green blending      |

---

## Usage

1. **Place your images**

   * **Packed CFA images** (e.g. `.png`/`.jpg`) in `./inputs/`.
   * Or **raw binary** files `.raw` with shape (`rows`, `cols`) and dtype `uint16`.

2. **Run the demosaicer**

   ```bash
   python run.py
   ```

   This will generate demosaiced `.png` images prefixed with `dm_` in `./outputs/`.

---

## Algorithm Overview

1. **CFA Separation**

   * Parses raw Bayer data into three 2D channels (with zeros in missing positions).
2. **Weight & Gradient Computation**

   * Computes first‐ and second‐order horizontal/vertical gradients.
   * Builds exponential weight maps for edge directionality.
   * Computes 5×5 local variance between green and red/blue neighborhoods.
3. **Green Interpolation**

   * Chooses horizontal, vertical, or weighted interpolation based on `g_grad_th`, weights, and variance.
4. **Red/Blue Reconstruction**

   * At R/B pixels: solves for missing R or B by averaging diagonal residuals, guided by `rb_grad_th`.
   * At G pixels: fits local red/blue differences using weighted horizontal/vertical neighbors.
5. **Post‐processing**

   * Clips to valid range, scales to 8-bit, and outputs an RGB image.

---

## Examples

<details>
<summary>Example Input CFA Patch vs. Output RGB</summary>

|              CFA Patch             |             Demosaiced Result            |
| :--------------------------------: | :--------------------------------------: |
| ![raw CFA](examples/cfa_patch.png) | ![rgb result](examples/dm_cfa_patch.png) |

</details>

---

## Notes & Tips

* **Tuning thresholds** is critical:

  * `g_grad_th` > 0.0 biases toward pure horizontal/vertical interpolation.
  * `rb_grad_th` typically in \[0.5, 1.0]; lower values mix more diagonals.
  * `var_th` controls blending between gradient‐weighted vs. pure second‐order weights.
* For **noisy sensors**, consider raising `var_th` to favor robust blending.
* Ensure your input raw data matches the `bitdepth` and `rows`/`cols` settings.

---

## License

This code is released under the MIT License. Feel free to adapt and redistribute!

---

## Acknowledgments

* Based on techniques in ISO 12233 resolution test charts and CFA interpolation research.
* Implementation inspired by Stephen H. Westin’s electronic test charts (Cornell University).
