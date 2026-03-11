# Current Research in Quantum Diamond Microscopy: Data Analysis and Parameter Extraction

*Compiled February 2026 — focus on widefield QDM data analysis, ODMR fitting,
B111/vector field extraction, stress/strain sensing, and ML approaches.*

---

## 1. NV Center Physics Fundamentals

### Electronic Structure and the NV Hamiltonian

The nitrogen-vacancy (NV) center is a point defect in diamond consisting of a substitutional
nitrogen atom adjacent to a lattice vacancy. The negatively charged NV⁻ state (hereafter "NV")
has an S = 1 ground-state spin triplet that can be initialized, manipulated, and read out
optically at room temperature.

The ground-state spin Hamiltonian is:

```
H = D·Sz² + E(Sx² - Sy²) + γ_NV · B·S + H_hf + H_strain + H_T
```

where:

| Term | Symbol | Physical origin |
|------|--------|-----------------|
| Zero-field splitting | D ≈ 2.870 GHz | Crystal field; axial |
| Transverse ZFS | E | Local strain / electric field; lifts m_s = ±1 degeneracy at zero field |
| Zeeman | γ_NV · B·S | γ_NV = 28.024 GHz/T |
| Hyperfine | H_hf | ¹⁴N: A_∥ ≈ −2.14 MHz, A_⊥ ≈ −2.70 MHz; ¹⁵N: A_∥ ≈ +3.03 MHz |
| Strain | H_strain | Couples to d_∥ and d_⊥ susceptibilities |
| Temperature | H_T | dD/dT ≈ −74 kHz/K |

The key result for magnetometry is that an axial magnetic field B₁₁₁ (along the NV axis)
splits the m_s = ±1 resonances symmetrically around D:

```
f± = D ± γ_NV · B₁₁₁
→  B₁₁₁ = (f+ − f−) / (2 · γ_NV)
```

For a widefield diamond sensor with an ensemble of NV centers oriented along all four ⟨111⟩
directions, four pairs of resonances are present in the ODMR spectrum, encoding all three
Cartesian components of the local field.

### Hyperfine Structure

- **¹⁴N** (99.6% natural abundance, nuclear spin I = 1): triplet of lines separated by ~2.16 MHz.
  The ESR14N lineshape is three Lorentzians per resonance branch.
- **¹⁵N** (isotopically enriched, I = 1/2): doublet separated by ~3.03 MHz.
  The ESR15N lineshape is two Lorentzians per branch.

Resolving the hyperfine structure is important for precise center-frequency extraction because
an unresolved triplet biases a single-Lorentzian fit.

### Key Reviews

- **Doherty et al. (2013)** — "The nitrogen-vacancy colour centre in diamond."
  *Physics Reports* 528, 1–45.
  DOI: [10.1016/j.physrep.2013.02.001](https://doi.org/10.1016/j.physrep.2013.02.001)
  — The canonical comprehensive review of NV physics: electronic structure, spin Hamiltonian,
  optical properties, and sensing applications.

- **Rondin et al. (2014)** — "Magnetometry with nitrogen-vacancy defects in diamond."
  *Reports on Progress in Physics* 77, 056503.
  DOI: [10.1088/0034-4885/77/5/056503](https://doi.org/10.1088/0034-4885/77/5/056503)
  — Covers both single-NV and ensemble approaches, experimental protocols (CW-ODMR, pulsed),
  and signal-to-noise analysis for widefield geometries.

---

## 2. ODMR Spectrum Fitting and Parameter Extraction

### Standard Lorentzian Fitting

Conventional CW-ODMR analysis fits the normalized fluorescence dip to a sum of Lorentzians:

```
S(f) = 1 − Σ_i  C_i · (Γ_i/2)² / [(f − f_i)² + (Γ_i/2)²]
```

where f_i is the resonance center, Γ_i the linewidth, and C_i the contrast. For an ESR14N
model, there are 3 resonances per polarity/frange pair; for ESR15N, 2. Fitting proceeds
independently per pixel for widefield data, making GPU acceleration essential for practical
throughput on ~4 Mpixel images.

Extracted parameters and their physical mapping:

| Fit parameter | Physical quantity |
|---------------|------------------|
| f_center (center of triplet) | D + γ_NV B₁₁₁ per NV orientation |
| Γ (linewidth) | Spin coherence T₂*, strain inhomogeneity, power broadening |
| C (contrast) | NV density, collection efficiency, spin-readout fidelity |
| E (transverse ZFS from zero-field splitting of ±1) | Local strain and/or electric field |
| D (from temperature dependence) | Local temperature (dD/dT ≈ −74 kHz/K) |

### Bayesian / Adaptive Methods

- **Dushenko, Ambal & McMichael (2020)** — "Sequential Bayesian experiment design for
  optically detected magnetic resonance of nitrogen-vacancy centers."
  *Physical Review Applied* 14, 054036.
  DOI: [10.1103/PhysRevApplied.14.054036](https://doi.org/10.1103/PhysRevApplied.14.054036)
  — Demonstrates >10× speed-up over fixed frequency-sweep ODMR by adaptively choosing
  which microwave frequencies to probe next based on accumulated posterior. The OptBayesExpt
  framework (NIST open-source) implements this approach.

### Clustering-Based Fitting

- **Stone et al. (2024)** — "Fast characterization of optically detected magnetic resonance
  spectra via data clustering."
  *The Journal of Physical Chemistry C*.
  DOI: [10.1021/acs.jpcc.4c03864](https://doi.org/10.1021/acs.jpcc.4c03864)
  (arXiv: [2405.18648](https://arxiv.org/abs/2405.18648))
  — A data-clustering algorithm (no physical model required) achieves ~1.3× better accuracy,
  ~4.7× higher effective frequency resolution, and uses ~5× fewer data points than standard
  MLE/LSE fitting. Particularly valuable for low-SNR or undersampled spectra.

---

## 3. B111 Magnetic Field Mapping

### Widefield Extraction Protocol

For a single NV orientation, the B₁₁₁ map is constructed as:

```
B₁₁₁ = (f_high − f_low) / (2 · γ_NV)
```

where f_high and f_low are the fitted center frequencies of the upper and lower resonance
branches for one polarity. In practice, two field polarities (positive/negative bias field)
are used to separate remanent and induced components:

```
B_remanent = (B₁₁₁⁺ + B₁₁₁⁻) / 2
B_induced   = (B₁₁₁⁺ − B₁₁₁⁻) / 2
```

Sensitivity limitations come from photon shot noise, spin-projection noise, and collection
efficiency; achieving µT/√Hz per pixel is routine, sub-100 nT/√Hz requires pulsed protocols.

### Sub-second and High Frame-Rate Imaging

- **Parashar et al. (2022)** — "Sub-second temporal magnetic field microscopy using quantum
  defects in diamond."
  *Scientific Reports* 12, 8743.
  DOI: [10.1038/s41598-022-12609-3](https://doi.org/10.1038/s41598-022-12609-3)
  — Lock-in camera detection of frequency-modulated NV fluorescence achieves 50–200 fps
  widefield B-field imaging without sacrificing spatial coverage, enabling sub-second
  imaging of dynamic currents in planar microcoils.

### Neuromorphic / Event-Camera Acquisition

- **Du et al. (2024)** — "Widefield Diamond Quantum Sensing with Neuromorphic Vision Sensors."
  *Advanced Science* 11(2), e2304355.
  DOI: [10.1002/advs.202304355](https://doi.org/10.1002/advs.202304355)
  (arXiv: [2306.14099](https://arxiv.org/abs/2306.14099))
  — Replaces the frame-based camera with an event camera (neuromorphic sensor) that logs
  per-pixel brightness changes asynchronously. Achieves 13× improvement in temporal
  resolution vs. frame-based widefield QDM with comparable ODMR frequency precision;
  opens path to millisecond-scale dynamic field imaging.

---

## 4. Vector Magnetic Field Reconstruction

The full vector B = (Bx, By, Bz) can be recovered from widefield data when at least two NV
orientations are resolvable in the ODMR spectrum (or from a single orientation using
multi-axis bias fields). Approaches range from direct inversion of the projection equations to
Fourier-domain methods.

### Fourier-Domain Reconstruction

- **Guo et al. (2024)** — "Wide-field Fourier magnetic imaging with electron spins in diamond."
  *npj Quantum Information* 10, 24.
  DOI: [10.1038/s41534-024-00818-9](https://doi.org/10.1038/s41534-024-00818-9)
  — Combines widefield NV imaging with Fourier k-space acquisition to push spatial resolution
  20× beyond the optical diffraction limit while maintaining a large field of view. The
  technique reconstructs the full spatial spectrum of the stray field and inverts for the
  source magnetization.

---

## 5. Current Density Reconstruction

Stray magnetic field images are related to the underlying current density J via the
Biot–Savart law, which is invertible in 2D under a single-layer assumption. Three generations
of approaches have emerged:

### Fourier (Wiener filter) Inversion

Classical approach: divide by the transfer function in k-space with a Tikhonov regularizer.
Sensitive to noise at high spatial frequencies and to the choice of standoff distance.

### Bayesian Inference

- **Clement, Sethna & Nowack (2019/2020)** — "Reconstruction of Current Densities from
  Magnetic Images by Bayesian Inference."
  arXiv: [1910.12929](https://arxiv.org/abs/1910.12929)
  — Formulates current reconstruction as MAP inference; arbitrary priors (smoothness,
  device-boundary masks) are easily incorporated. Outperforms Wiener filter for noisy data.

### Optimized Widefield Reconstruction

- **Midha et al. (2024)** — "Optimized Current Density Reconstruction from Widefield Quantum
  Diamond Magnetic Field Maps."
  *Physical Review Applied* 22, 014015.
  DOI: [10.1103/PhysRevApplied.22.014015](https://doi.org/10.1103/PhysRevApplied.22.014015)
  (arXiv: [2402.17781](https://arxiv.org/abs/2402.17781))
  — Systematic comparison of Fourier and Bayesian reconstructions on QDM data; provides
  practical guidelines for standoff distance, regularization, and noise budgeting.

### Deep Learning Reconstruction

- **Reed et al. (2024/2025)** — "Machine Learning for Improved Current Density Reconstruction
  from 2D Vector Magnetic Images."
  *Physical Review Applied* 23, 034035.
  DOI: [10.1103/PhysRevApplied.23.034035](https://doi.org/10.1103/PhysRevApplied.23.034035)
  (arXiv: [2407.14553](https://arxiv.org/abs/2407.14553))
  — A deep convolutional neural network trained on QDM vector-field data significantly
  outperforms analytic Fourier reconstruction at high noise or large standoff distances.
  Demonstrates the advantage of using both in-plane B components as network input.

---

## 6. Stress and Strain Mapping

NV centers are sensitive to local lattice stress through the spin-strain coupling in the
Hamiltonian. The transverse ZFS parameter E and the orbital splitting respond to the
symmetry-breaking strain components ε_xx − ε_yy and ε_xy (shear), while the axial strain
ε_zz shifts D. This makes widefield ODMR a non-contact, all-optical strain microscope.

### Spin-Stress Susceptibilities

The NV spin Hamiltonian strain terms are characterized by coupling constants d_∥ and d_⊥:

```
δD = d_∥ · (ε_xx + ε_yy + ε_zz)  +  ...
δE = d_⊥ · [(ε_xx − ε_yy) ± 2i ε_xy]
```

Typical values: d_∥ ≈ −4.86 PHz/strain, d_⊥ ≈ 15.5 PHz/strain (from Barson et al.).

### Nanomechanical Sensing

- **Barson et al. (2017)** — "Nanomechanical Sensing Using Spins in Diamond."
  *Nano Letters* 17(3), 1496–1503.
  DOI: [10.1021/acs.nanolett.6b04544](https://doi.org/10.1021/acs.nanolett.6b04544)
  — Establishes the quantitative spin-stress coupling framework. Demonstrates NV-based
  sensing of thermally driven nanomechanical motion; provides the key susceptibility
  constants (d_∥, d_⊥) used to convert ODMR frequency shifts to strain tensor components.

### Zero-Field ODMR Strain Extraction

- **Alam et al. (2024)** — "Determining Strain Components in a Diamond Waveguide from
  Zero-Field ODMR Spectra of NV⁻ Center Ensembles."
  *Physical Review Applied* 22, 024055.
  DOI: [10.1103/PhysRevApplied.22.024055](https://doi.org/10.1103/PhysRevApplied.22.024055)
  (arXiv: [2402.06422](https://arxiv.org/abs/2402.06422))
  — Extracts the full relevant strain tensor from continuous-wave zero-field ODMR without
  applying an external bias field. The transverse ZFS E provides the deviatoric strain
  components; the paper demonstrates extraction inside a diamond optical waveguide where
  fabrication-induced stress dominates. A practical fitting pipeline for multi-component
  strain imaging is provided.

---

## 7. Multi-Parameter Sensing

A single ODMR spectrum encodes multiple physical quantities simultaneously. Disentangling
them requires fitting the full Hamiltonian or designing pulse sequences that are sensitive
to one parameter at a time.

| Observable | ODMR signature | Sensitivity |
|------------|---------------|-------------|
| B₁₁₁ | Symmetric frequency shift of ±1 branches | ~10 nT/√Hz (pulsed) |
| Temperature | Overall shift of D (−74 kHz/K) | ~10 mK/√Hz |
| Strain (axial) | Shift of D | Competes with temperature |
| Strain (transverse) | Splitting of ±1 at zero field via E | ~kPa/√Hz |
| Electric field | Shift via d_∥ (axial E-field) or d_⊥ (transverse) | ~(V/cm)/√Hz |

Key challenge: simultaneous B and T sensing requires measurements at multiple bias fields or
use of the full 8-peak ESR14N spectrum to decouple Zeeman from thermal D-shift.

---

## 8. Machine Learning Approaches

### Model-Free B-Field Imaging

- **Tsukamoto et al. (2022)** — "Accurate magnetic field imaging using nanodiamond quantum
  sensors enhanced by machine learning."
  *Scientific Reports* 12, 13942.
  DOI: [10.1038/s41598-022-18115-w](https://doi.org/10.1038/s41598-022-18115-w)
  (arXiv: [2202.00380](https://arxiv.org/abs/2202.00380))
  — Demonstrates B-field imaging at 1.8 µT accuracy using nanodiamond ensembles and ML
  without invoking any physical model. The trained network maps raw fluorescence spectra
  directly to B₁₁₁, bypassing explicit ODMR fitting. Particularly useful when spectra
  are overlapping or partially resolved.

### Edge / Embedded ML Magnetometer

- **Homrighausen et al. (2023)** — "Edge-Machine-Learning-Assisted Robust Magnetometer Based
  on Randomly Oriented NV-Ensembles in Diamond."
  *Sensors* 23(3), 1119.
  DOI: [10.3390/s23031119](https://doi.org/10.3390/s23031119)
  — Trains a neural network on CW-ODMR spectra from randomly oriented NV micro-diamonds,
  then runs inference on an embedded ESP32 microcontroller. Demonstrates a low-cost,
  stand-alone widefield magnetometer that operates without specialized fitting software.

### ML for Current Reconstruction

See §5 (Reed et al. 2024/2025) for ML applied to the post-processing step of converting
B₁₁₁ maps to current density images.

---

## 9. Super-Resolution and Spatial Resolution

Optical diffraction limits the pixel footprint in widefield QDM to ~300–500 nm for visible
light. Two strategies have emerged to improve spatial resolution.

### Structured Illumination

- **Xu et al. (2024)** — "Super-Resolution Enabled Widefield Quantum Diamond Microscopy."
  *ACS Photonics* 11(1), 121–127.
  DOI: [10.1021/acsphotonics.3c01077](https://doi.org/10.1021/acsphotonics.3c01077)
  (arXiv: [2307.14990](https://arxiv.org/abs/2307.14990))
  — A digital micromirror device (DMD) generates rapidly programmable structured illumination
  patterns. Achieves super-resolved ODMR sensing of two nanodiamonds that are unresolvable
  with conventional widefield QDM; also mitigates phototoxicity in biological samples by
  spatially modulating excitation.

### Fourier k-Space Reconstruction

See §4 (Guo et al. 2024, npj Quantum Information) — achieves 20× improvement in spatial
resolution beyond the optical limit via k-space acquisition and inversion.

---

## 10. Implications for QDMpy

Based on the literature, the following analysis capabilities represent high-value targets:

| Capability | Relevant paper(s) | Status in QDMpy |
|------------|-------------------|-----------------|
| ESR14N / ESR15N / single-peak fitting | Standard (Rondin 2014) | Implemented |
| B₁₁₁ remanent + induced extraction | — | Implemented |
| GPU-accelerated per-pixel Lorentzian fit | — | Implemented (pygpufit) |
| Bayesian adaptive frequency selection | Dushenko 2020 | Not implemented |
| Strain (E-parameter) map extraction | Barson 2017, Alam 2024 | Not implemented |
| Temperature map from D-shift | — | Not implemented |
| Current density reconstruction (Fourier/Bayesian) | Midha 2024, Clement 2019 | Not implemented |
| ML-based spectrum → B-field | Tsukamoto 2022 | Not implemented |
| Vector B reconstruction (multi-orientation) | Guo 2024 | Not implemented |
| Clustering-based fast ODMR fitting | Stone 2024 | Not implemented |

---

## References

1. **Doherty et al. (2013)**
   Doherty MW, Manson NB, Delaney P, Jelezko F, Wrachtrup J, Hollenberg LCL.
   "The nitrogen-vacancy colour centre in diamond."
   *Physics Reports* 528, 1–45.
   DOI: [10.1016/j.physrep.2013.02.001](https://doi.org/10.1016/j.physrep.2013.02.001)

2. **Rondin et al. (2014)**
   Rondin L, Tetienne J-P, Hingant T, Roch J-F, Maletinsky P, Jacques V.
   "Magnetometry with nitrogen-vacancy defects in diamond."
   *Reports on Progress in Physics* 77, 056503.
   DOI: [10.1088/0034-4885/77/5/056503](https://doi.org/10.1088/0034-4885/77/5/056503)

3. **Barson et al. (2017)**
   Barson MSJ, Peddibhotla P, Ovartchaiyapong P, Ganesan K, Taylor RL, Gebert M,
   Mielens Z, Koslowski B, Simpson DA, McGuinness LP, McCallum J, Prawer S, Onoda S,
   Ohshima T, Bleszynski Jayich AC, Jelezko F, Manson NB, Doherty MW.
   "Nanomechanical Sensing Using Spins in Diamond."
   *Nano Letters* 17(3), 1496–1503.
   DOI: [10.1021/acs.nanolett.6b04544](https://doi.org/10.1021/acs.nanolett.6b04544)

4. **Dushenko, Ambal & McMichael (2020)**
   Dushenko S, Ambal K, McMichael RD.
   "Sequential Bayesian experiment design for optically detected magnetic resonance of
   nitrogen-vacancy centers."
   *Physical Review Applied* 14, 054036.
   DOI: [10.1103/PhysRevApplied.14.054036](https://doi.org/10.1103/PhysRevApplied.14.054036)

5. **Clement, Sethna & Nowack (2019)**
   Clement CB, Sethna JP, Nowack KC.
   "Reconstruction of Current Densities from Magnetic Images by Bayesian Inference."
   arXiv: [1910.12929](https://arxiv.org/abs/1910.12929)
   Published in *Physical Review Applied*.

6. **Parashar et al. (2022)**
   Parashar M, Bathla A, Shishir D, et al.
   "Sub-second temporal magnetic field microscopy using quantum defects in diamond."
   *Scientific Reports* 12, 8743.
   DOI: [10.1038/s41598-022-12609-3](https://doi.org/10.1038/s41598-022-12609-3)

7. **Tsukamoto et al. (2022)**
   Tsukamoto M, Ito S, Ogawa K, Ashida Y, Sasaki K, Kobayashi K.
   "Accurate magnetic field imaging using nanodiamond quantum sensors enhanced by
   machine learning."
   *Scientific Reports* 12, 13942.
   DOI: [10.1038/s41598-022-18115-w](https://doi.org/10.1038/s41598-022-18115-w)
   arXiv: [2202.00380](https://arxiv.org/abs/2202.00380)

8. **Homrighausen et al. (2023)**
   Homrighausen J, Horsthemke L, Pogorzelski J, Trinschek S, Glösekötter P, Gregor M.
   "Edge-Machine-Learning-Assisted Robust Magnetometer Based on Randomly Oriented
   NV-Ensembles in Diamond."
   *Sensors* 23(3), 1119.
   DOI: [10.3390/s23031119](https://doi.org/10.3390/s23031119)

9. **Alam et al. (2024)**
   Alam MS, Gorrini F, Gawełczyk M, Wigger D, Coccia G, Guo Y, Shahbazi S, Bharadwaj V,
   Kubanek A, Ramponi R, Barclay PE, Bennett AJ, Hadden JP, Bifone A, Eaton SM,
   Machnikowski P.
   "Determining Strain Components in a Diamond Waveguide from Zero-Field ODMR Spectra of
   NV⁻ Center Ensembles."
   *Physical Review Applied* 22, 024055.
   DOI: [10.1103/PhysRevApplied.22.024055](https://doi.org/10.1103/PhysRevApplied.22.024055)
   arXiv: [2402.06422](https://arxiv.org/abs/2402.06422)

10. **Du et al. (2024)**
    Du Z, Gupta M, Xu F, Zhang K, Zhang J, Zhou Y, Liu Y, Wang Z, Wrachtrup J, Wong N,
    Li C, Chu Z.
    "Widefield Diamond Quantum Sensing with Neuromorphic Vision Sensors."
    *Advanced Science* 11(2), e2304355.
    DOI: [10.1002/advs.202304355](https://doi.org/10.1002/advs.202304355)
    arXiv: [2306.14099](https://arxiv.org/abs/2306.14099)

11. **Guo et al. (2024)**
    Guo Z, Huang Y, Cai M, et al.
    "Wide-field Fourier magnetic imaging with electron spins in diamond."
    *npj Quantum Information* 10, 24.
    DOI: [10.1038/s41534-024-00818-9](https://doi.org/10.1038/s41534-024-00818-9)

12. **Midha et al. (2024)**
    Midha S, Parashar M, Bathla A, Broadway DA, Tetienne J-P, Saha K.
    "Optimized Current Density Reconstruction from Widefield Quantum Diamond Magnetic
    Field Maps."
    *Physical Review Applied* 22, 014015.
    DOI: [10.1103/PhysRevApplied.22.014015](https://doi.org/10.1103/PhysRevApplied.22.014015)
    arXiv: [2402.17781](https://arxiv.org/abs/2402.17781)

13. **Stone et al. (2024)**
    Stone DG, et al.
    "Fast Characterization of Optically Detected Magnetic Resonance Spectra via Data
    Clustering."
    *The Journal of Physical Chemistry C*.
    DOI: [10.1021/acs.jpcc.4c03864](https://doi.org/10.1021/acs.jpcc.4c03864)
    arXiv: [2405.18648](https://arxiv.org/abs/2405.18648)

14. **Xu et al. (2024)**
    Xu F, Chen J, Hou Y, Cheng J, Hui TKC, Chen S-C, Chu Z.
    "Super-Resolution Enabled Widefield Quantum Diamond Microscopy."
    *ACS Photonics* 11(1), 121–127.
    DOI: [10.1021/acsphotonics.3c01077](https://doi.org/10.1021/acsphotonics.3c01077)
    arXiv: [2307.14990](https://arxiv.org/abs/2307.14990)

15. **Reed et al. (2024/2025)**
    Reed NR, Bhutto D, Turner MJ, Daly DM, Oliver SM, Tang J, Olsson KS, Langellier N,
    Ku MJH, Rosen MS, Walsworth RL.
    "Machine Learning for Improved Current Density Reconstruction from 2D Vector Magnetic
    Images."
    *Physical Review Applied* 23, 034035.
    DOI: [10.1103/PhysRevApplied.23.034035](https://doi.org/10.1103/PhysRevApplied.23.034035)
    arXiv: [2407.14553](https://arxiv.org/abs/2407.14553)
