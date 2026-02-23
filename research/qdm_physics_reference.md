# Quantum Diamond Microscopy: Physics & Terminology Reference

*A comprehensive reference for the physics, instrumentation, data analysis, and
terminology of quantum diamond microscopy (QDM). Intended as a shared reference for
both human developers and Claude AI sessions working on qdmpy-core.*

*Compiled February 2026.*

---

## Table of Contents

1. [NV Center Physics](#1-nv-center-physics)
2. [QDM Instrument](#2-qdm-instrument)
3. [ODMR Spectroscopy](#3-odmr-spectroscopy)
4. [ODMR Spectrum Structure](#4-odmr-spectrum-structure)
5. [Data Analysis & B111 Extraction](#5-data-analysis--b111-extraction)
6. [Applications](#6-applications)
7. [Physical Constants](#7-physical-constants)
8. [Glossary](#8-glossary)
9. [References](#9-references)

---

## 1. NV Center Physics

### 1.1 Crystal Structure

The nitrogen-vacancy (NV) center is a point defect in the diamond lattice consisting of
a substitutional nitrogen atom adjacent to a vacant lattice site [Doherty2013]. The defect
has C_3v symmetry with the symmetry axis along the nitrogen-vacancy bond direction, which
coincides with one of the four crystallographic <111> directions of the diamond cubic lattice
[Doherty2013].

The NV center exists in two charge states: the neutral NV^0 (zero-phonon line at 575 nm)
and the negatively charged NV^- (zero-phonon line at 637 nm / 1.945 eV) [Doherty2013].
The NV^- state is the magnetically sensitive species used for magnetometry. Throughout this
document, "NV center" refers to NV^- unless stated otherwise.

### 1.2 Electronic Energy Levels

The NV^- center has a spin-1 (S=1) ground state (^3A_2) and a spin-1 excited state (^3E),
connected by an optical transition near 637 nm [Doherty2013]. Between these triplet states
lie two singlet states (^1A_1 and ^1E) that mediate intersystem crossing (ISC) and are
central to the spin-dependent fluorescence mechanism [Goldman2015, Robledo2011].

**Ground state (^3A_2):**

- Spin triplet with sublevels m_s = 0 and m_s = +/-1
- Zero-field splitting D_gs = 2.870 GHz separates m_s = 0 from m_s = +/-1 [Doherty2013]
- The m_s = +/-1 sublevels are degenerate at zero magnetic field
- A transverse zero-field splitting parameter E (typically 0-11 MHz in real diamonds) lifts
  the m_s = +/-1 degeneracy due to local strain [Acosta2010, Doherty2013]

**Excited state (^3E):**

- Spin triplet with zero-field splitting D_es = 1.42 GHz [Doherty2013]
- Radiative lifetime ~12-13 ns in bulk diamond [Goldman2015]
- Phonon sideband emission spans 637-800 nm, peaking near 680 nm
- Debye-Waller factor ~0.04 (only ~4% of emission into the zero-phonon line) [Doherty2013]

**Singlet states:**

- ^1A_1: upper singlet, reached from ^3E via ISC; decays rapidly (~1 ns) to ^1E [Robledo2011]
- ^1E: metastable singlet with lifetime ~250-300 ns at room temperature [Robledo2011]
- ^1E decays preferentially to the m_s = 0 sublevel of ^3A_2, providing optical spin
  polarization [Goldman2015]

### 1.3 Spin Hamiltonian

The ground-state spin Hamiltonian of a single NV center in an external magnetic field
**B** is [Doherty2013]:

```
H = D_gs * S_z^2 + E * (S_x^2 - S_y^2) + gamma_NV * B . S
```

where:

- D_gs = 2.870 GHz is the axial zero-field splitting
- E is the transverse zero-field splitting (strain-dependent, typically 0-11 MHz)
- gamma_NV = g_e * mu_B / h = 28.024 GHz/T is the NV gyromagnetic ratio
- S_z is the spin-1 operator along the NV symmetry axis
- **B** is the external magnetic field vector

For a magnetic field B aligned along the NV axis (B_perp = 0), the eigenvalues are:

```
f(m_s = 0) = 0
f(m_s = +1) = D_gs + gamma_NV * B_parallel
f(m_s = -1) = D_gs - gamma_NV * B_parallel
```

giving two microwave resonance frequencies:

```
f_+ = D_gs + gamma_NV * B_parallel    (high-frequency branch)
f_- = D_gs - gamma_NV * B_parallel    (low-frequency branch)
```

The splitting between these two resonances is 2 * gamma_NV * B_parallel, and their mean is
D_gs (to first order). A non-zero transverse field B_perp causes mixing of spin states and
shifts the resonance frequencies nonlinearly [Doherty2013].

### 1.4 Four NV Orientations

In diamond's cubic lattice, the NV axis can point along any of four crystallographically
equivalent <111> directions [Doherty2013]:

```
[111], [-1-11], [-11-1], [1-1-1]
```

Each orientation family has a different projection of the applied magnetic field onto its
axis. In a QDM, a bias field is aligned along one chosen NV axis (typically [111]) to
spectrally separate the four families [Levine2019]. The on-axis family experiences the
largest Zeeman splitting and produces the most separated pair of ODMR dips; the other three
families have smaller, often overlapping splittings.

### 1.5 Hyperfine Structure

**^14N (99.6% natural abundance):**

The ^14N nucleus has spin I = 1, producing a hyperfine triplet (m_I = -1, 0, +1) on each
m_s = +/-1 resonance [Doherty2013]. The axial hyperfine constant is:

```
A_parallel(14N) = -2.16 MHz
```

Each ODMR resonance line is split into three peaks separated by |A_parallel| = 2.16 MHz,
centered on the resonance frequency. This is the standard case for QDM magnetometry and
gives rise to the "ESR14N" 3-Lorentzian model with equal spacing [Doherty2013].

**^15N (synthetic enrichment):**

The ^15N nucleus has spin I = 1/2, producing a hyperfine doublet on each resonance
[Doherty2013]:

```
A_parallel(15N) = +3.03 MHz
```

Each resonance is split into two peaks separated by |A_parallel| = 3.03 MHz. This gives
the "ESR15N" 2-Lorentzian model.

### 1.6 Temperature Dependence

The ground-state zero-field splitting D_gs has a significant temperature dependence near
room temperature [Acosta2010]:

```
dD/dT = -74.2(7) kHz/K    (near 300 K)
```

This means a 1 K temperature change shifts D by ~74 kHz, corresponding to a spurious
apparent magnetic field of ~74 kHz / gamma_NV ~ 2.6 uT [Acosta2010]. For measurements
targeting ~1 uT sensitivity, temperature must be controlled to better than ~0.4 K. Laser
heating of the diamond is a practical concern in widefield QDM operation [Levine2019].

The physical origin is attributed to local thermal expansion of the diamond lattice
[Acosta2010]. The transverse parameter E also has weak temperature dependence:
dE/(E*dT) = -1.4(3) x 10^-4 K^-1 [Acosta2010].

---

## 2. QDM Instrument

### 2.1 Architecture

The quantum diamond microscope (QDM) is a widefield magnetic imaging instrument that uses
a dense layer of NV centers in a diamond chip as a 2D magnetic field sensor [Levine2019,
Glenn2017]. Unlike scanning-probe NV magnetometry (which uses a single NV on a tip), the
QDM images an entire field of view simultaneously using a camera, trading nanometer
resolution for speed and large area coverage [Levine2019].

Key components [Levine2019]:

- **Diamond sensor chip**: A transparent diamond (typically 2x2x0.5 mm) with a thin
  NV-rich layer (~1-4 um thick) near one surface. Grown by CVD with controlled nitrogen
  doping, then annealed to form NV centers [Ohno2012, Kleinsasser2016]
- **532 nm laser**: Green excitation for NV fluorescence, delivered in widefield
  illumination geometry (either epi or oblique)
- **Microwave antenna**: A microstrip or loop antenna on or near the diamond surface,
  used to drive spin transitions. The MW frequency is swept to record ODMR spectra
  [Levine2019]
- **Bias magnets**: Permanent magnets or Helmholtz coils providing a static magnetic field
  (typically a few mT) aligned along one NV crystallographic axis to separate the four NV
  orientation families spectrally [Levine2019]
- **sCMOS camera**: Widefield detection of NV fluorescence (637-800 nm) with high quantum
  efficiency and low read noise [Levine2019]
- **Optical filters**: Dichroic mirror (reflects 532 nm, transmits >600 nm) and long-pass
  filter to reject laser light and pass NV fluorescence

The sample is placed directly on the diamond surface, minimizing sensor-to-sample standoff
distance (~1-100 um), which is critical for spatial resolution of near-field magnetic
sources [Levine2019].

### 2.2 Diamond Sensor Fabrication

**CVD growth (preferred for QDM)** [Ohno2012, Kleinsasser2016]:

- Electronic-grade diamond substrate grown by chemical vapor deposition (CVD)
- NV-rich layer grown by introducing N2 gas during CVD; typical thickness ~1-4 um
- Nitrogen concentration: ~1-10 ppm N, yielding NV densities ~10^11 to 10^12 NV/cm^3
  (N-to-NV conversion efficiency ~1-10%)
- Post-growth vacuum annealing at 800-1200 C mobilizes vacancies to form NV centers
- Delta-doped ultra-thin layers (~3-5 nm) have been demonstrated with peak N concentration
  ~10^19 cm^-3

**Ion implantation** [Ohno2012]:

- Start with ultrapure CVD diamond; implant N+ ions at controlled energy
- Anneal to form NV centers; better depth control but lower overall NV density
- Preferred for scanning-probe (single-NV) applications rather than widefield QDM

### 2.3 Performance Specifications

Typical QDM performance parameters [Levine2019, Glenn2017]:

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Spatial resolution | 1-5 um | Set by optics and NV layer depth |
| Field of view | 1-4 mm | Glenn2017 reports ~4 mm for geological config |
| Magnetic sensitivity | 1-40 uT/sqrt(Hz) per pixel | Photon-shot-noise limited |
| Volume-normalized sensitivity | ~20 uT*um/sqrt(Hz) | Glenn2017 |
| Effective sensitivity (1 hr) | ~tens of nT | After time integration |
| Vector components | All 3 (Bx, By, Bz) | Simultaneous |
| Operating temperature | Room temperature | Both sensor and sample |
| Bias field | ~few mT | Along one NV axis |

### 2.4 QDM vs SQUID vs Scanning NV

| Property | QDM (widefield) | Scanning SQUID | Scanning NV |
|----------|-----------------|----------------|-------------|
| Spatial resolution | ~1-5 um | ~100-150 um (warm) | ~50 nm |
| Field sensitivity | ~1-40 uT/sqrt(Hz) | <500 fT/sqrt(Hz) | ~10 nT/sqrt(Hz) |
| Field of view | ~1-4 mm (simultaneous) | Scanned, mm-scale | Scanned, <100 um |
| Vector components | 3 simultaneous | Typically 1 (Bz) | 1 per scan |
| Temperature | Room temp | Sensor at 4 K | Room temp |
| Speed | Fast (full FOV at once) | Slow (point scan) | Slow (point scan) |

[Levine2019, Fu2020, Weiss_lab]

The QDM excels in spatial resolution, wide FOV, vector capability, and room-temperature
operation. SQUID excels in raw field sensitivity by orders of magnitude. Scanning NV excels
in nanoscale spatial resolution. For paleomagnetic applications, QDM and SQUID are
complementary; Fu et al. (2020) showed they recover comparable net moments [Fu2020].

---

## 3. ODMR Spectroscopy

### 3.1 CW-ODMR Protocol

Continuous-wave optically detected magnetic resonance (CW-ODMR) is the standard measurement
mode for widefield QDM [Levine2019]:

1. **Continuous green illumination** (532 nm) polarizes NV spins into m_s = 0 and excites
   fluorescence
2. **Microwave frequency sweep**: At each MW frequency, a camera frame is acquired
3. When the MW frequency matches a spin resonance (m_s = 0 -> m_s = +/-1), the NV
   population is driven into m_s = +/-1 states
4. m_s = +/-1 states have higher ISC rate to the dark singlet pathway (~80 MHz vs ~8 MHz
   for m_s = 0) [Goldman2015], reducing fluorescence
5. The result is a **dip in fluorescence** at each resonance frequency

The raw data is a 3D datacube: fluorescence intensity as a function of (x, y, MW_frequency).
At each pixel, extracting the resonance frequencies from the ODMR spectrum yields the local
magnetic field projection along the NV axis [Levine2019].

### 3.2 ODMR Contrast Mechanism

The spin-dependent fluorescence arises from differential intersystem crossing rates
[Goldman2015, Robledo2011]:

- **m_s = 0**: Low ISC rate (~8 MHz from ^3E to ^1A_1). Most cycles are radiative
  (^3E -> ^3A_2), producing bright fluorescence
- **m_s = +/-1**: High ISC rate (~80 MHz from ^3E to ^1A_1). A large fraction of cycles
  go through the dark singlet pathway, reducing fluorescence

The singlet ^1E state decays preferentially back to m_s = 0, providing **optical spin
polarization** to ~80% after a few optical cycles [Goldman2015]. This polarization is what
makes NV magnetometry possible: the system continuously resets to a known spin state.

Typical ensemble ODMR contrast is 1-5% (vs ~30% for single NV), limited by background
fluorescence from NV^0, non-NV defects, and imperfect optical collection [Levine2019].

### 3.3 Sensitivity

The photon-shot-noise-limited CW-ODMR sensitivity is [Levine2019, Eq. 10]:

```
eta_CW = P_F * (h / (g_e * mu_B)) * (Delta_f / (C * sqrt(R)))
```

where:

- P_F ~ 0.77: Lorentzian lineshape factor
- h / (g_e * mu_B) ~ 36 nT/kHz: inverse NV gyromagnetic ratio
- Delta_f: ODMR linewidth (FWHM), typically 5-20 MHz for ensembles
- C: ODMR contrast (fractional dip depth), typically 1-5% for ensembles
- R: photon detection rate (counts/s per pixel)

Sensitivity is degraded by laser intensity noise, microwave power broadening, and diamond
inhomogeneity. Higher NV density increases R but also broadens linewidth via the P1 spin
bath (substitutional nitrogen), so there is an optimal NV concentration [Levine2019].

### 3.4 Pulsed ODMR

Pulsed protocols (Ramsey, Hahn echo, dynamical decoupling) offer improved sensitivity by
decoupling the measurement from laser and MW noise [Levine2019]. The key improvement is
that linewidth is set by T2* (Ramsey) or T2 (echo) rather than by CW power broadening.
However, pulsed protocols require more complex instrumentation (fast MW switching, precise
timing, pulsed laser) and are less commonly used in widefield QDM for geological
applications where CW-ODMR is sufficient [Levine2019].

### 3.5 Green Fluorescence (GF) Artifact

The microwave antenna can produce frequency-dependent parasitic effects on the optical
signal that are not related to magnetic resonance. This "green fluorescence" or "GF"
artifact manifests as a slowly varying baseline modulation across the frequency sweep
and must be corrected during data processing [Levine2019]. Common correction approaches
include polynomial baseline subtraction or reference-pixel normalization.

---

## 4. ODMR Spectrum Structure

### 4.1 Full Spectrum

A typical CW-ODMR spectrum from a QDM pixel with bias field along [111] shows eight groups
of dips [Levine2019]:

- 2 groups from the on-axis NV family (largest splitting, well separated)
- 6 groups from the three off-axis families (smaller, often overlapping splittings)

Each "group" consists of hyperfine-split subpeaks (3 for ^14N, 2 for ^15N).

For magnetometry, typically only the on-axis family is analyzed. The two on-axis resonance
groups are split symmetrically about D_gs = 2.870 GHz by the Zeeman interaction:

```
f_low = D_gs - gamma_NV * B_111    (low-frequency branch, below ZFS)
f_high = D_gs + gamma_NV * B_111   (high-frequency branch, above ZFS)
```

where B_111 is the magnetic field projection along the [111] NV axis.

### 4.2 Spectral Models

**ESR14N (standard, 3 Lorentzians per resonance):**

For ^14N diamond (standard), each resonance consists of three equally-spaced Lorentzian
dips separated by the hyperfine splitting A = 2.16 MHz [Doherty2013]:

```
ODMR(f) = 1 - sum_{k=-1,0,+1} [ c_k / (1 + ((f - f_0 - k*A) / (w/2))^2) ]
```

where f_0 is the center frequency, w is the linewidth (FWHM), and c_k are the contrast
amplitudes of each hyperfine component.

Fit parameters per resonance group: center frequency (f_0), linewidth (w), and up to 3
contrast values (c_{-1}, c_0, c_{+1}), though often a single contrast is used with
fixed relative amplitudes.

**ESR15N (2 Lorentzians per resonance):**

For ^15N-enriched diamond, each resonance is a doublet with splitting A = 3.03 MHz
[Doherty2013].

**Single Lorentzian (simplified):**

For broad linewidths where hyperfine structure is unresolved, a single Lorentzian per
resonance suffices.

### 4.3 Frequency Ranges

In practice, the ODMR spectrum is acquired in two separate frequency sweeps [Levine2019]:

- **Low-frequency range (frange_0)**: Covers the low-frequency resonance branch, typically
  ~2.72-2.87 GHz (below D_gs)
- **High-frequency range (frange_1)**: Covers the high-frequency resonance branch, typically
  ~2.87-3.02 GHz (above D_gs)

Each range is swept with ~50 frequency points. The two ranges are measured sequentially
for each camera frame.

---

## 5. Data Analysis & B111 Extraction

### 5.1 Dual-Polarity Measurement Protocol

To separate remanent (permanent) magnetization from induced (bias-tracking) magnetization,
the QDM measurement is repeated with two opposite bias field polarities [Glenn2017, Fu2014]:

- **Polarity 0 (pol_0)**: Negative applied bias field
- **Polarity 1 (pol_1)**: Positive applied bias field

For each polarity, the full ODMR spectrum (both frequency ranges) is acquired and fitted
to extract resonance frequencies at every pixel.

### 5.2 B111 Extraction

From the fitted resonance center frequencies, the magnetic field along the NV [111] axis
is computed for each polarity [Glenn2017]:

```
dB[pol] = (f_high[pol] - f_low[pol]) / (2 * gamma_NV)
```

where f_high and f_low are the center frequencies of the high and low resonance branches,
and gamma_NV = 28.024 GHz/T. This quantity dB is always positive and represents the
total field magnitude along the NV axis.

### 5.3 Remanent vs Induced Decomposition

The dual-polarity measurement enables separation of remanent and induced contributions
[Glenn2017, Fu2014]:

```
negDiff = -dB[pol_0]    (negative by convention, since pol_0 = negative bias)
posDiff = +dB[pol_1]    (positive by convention)
```

Then:

```
B111_remanent = (negDiff + posDiff) / 2
B111_induced  = (negDiff - posDiff) / 2
```

**Physical interpretation:**

- **B111_remanent** (ferromagnetic/permanent): The component of the sample's magnetization
  that does not change when the bias field flips. This is the paleomagnetically interesting
  signal — it records ancient magnetic fields frozen into the sample [Fu2014, Glenn2017]
- **B111_induced** (paramagnetic/diamagnetic): The component that tracks the applied bias
  field. It flips sign when the bias flips and cancels in the remanent map. Includes
  paramagnetic and diamagnetic susceptibility responses [Glenn2017]

### 5.4 Current Density Reconstruction

For semiconductor failure analysis applications, the measured 2D magnetic field maps can be
inverted to reconstruct the underlying current density distribution using Fourier-space
deconvolution methods [Turner2022]. The QDM's vector measurement capability (Bx, By, Bz)
is particularly valuable for resolving 3D current distributions in multi-layer devices
[Turner2022].

---

## 6. Applications

### 6.1 Paleomagnetism & Geology

The QDM was initially developed for paleomagnetic applications by Roger Fu and Ron Walsworth
at Harvard [Fu2014, Glenn2017]. Key results:

- **Meteorite paleomagnetism**: Fu et al. (2014) used early QDM measurements combined with
  SQUID microscopy to study the Semarkona meteorite, establishing that the solar nebula had
  a magnetic field of ~54 uT at the location where the meteorite's chondrules formed, at
  ~1-3 AU from the Sun [Fu2014]. This was the first direct measurement of the nebular
  magnetic field.
- **ALH 84001 (Mars meteorite)**: Fu et al. (2020) used QDM and SQUID microscopy to study
  this Martian meteorite, finding that Mars had a dynamo field of at least ~20 uT at
  ~3.9 Ga [Fu2020]
- **Allende meteorite**: Fu et al. (2021) used QDM to image individual grain assemblages,
  concluding that remanent magnetization was acquired during parent body alteration
  [Nichols2021]
- **Geological QDM**: Glenn et al. (2017) described a QDM optimized for geological samples
  with 5 um resolution and ~4 mm FOV [Glenn2017]

The QDM enables mapping magnetization at the scale of individual mineral grains (~few um),
which is critical for understanding remanence carriers and avoiding inclusion of
non-remanence-bearing material in bulk measurements [Glenn2017].

### 6.2 Semiconductor Failure Analysis

Turner/Oliver et al. (2022) demonstrated QDM imaging of current distributions in an 8 nm
process node integrated circuit and 3D current distributions in a multi-layer PCB
[Turner2022]. Advantages over existing techniques:

- Vector magnetic imaging (all 3 components simultaneously)
- Room-temperature, non-invasive operation
- Micron-scale resolution with ~1-100 um standoff
- Can image through flip-chip packaging (measuring from the backside)

### 6.3 Biological Imaging

Glenn et al. (2015) demonstrated single-cell magnetic imaging using the QDM, detecting
immunomagnetically labeled mammalian cells with ~1 mm^2 FOV — two orders of magnitude
larger than previous NV imaging [Glenn2015]. Applications include:

- Quantitative profiling of cancer biomarkers via superparamagnetic nanoparticle labels
- Correlated magnetic and fluorescence imaging on the same platform
- Detection of magnetotactic bacteria and biogenic magnetic minerals

### 6.4 Materials Science

QDM is used for imaging domain structures in thin-film magnetic materials, studying
magnetic phase transitions, and characterizing magnetic nanoparticles [Levine2019]. The
combination of wide FOV and micron-scale resolution makes it well suited for studying
mesoscale magnetic phenomena.

---

## 7. Physical Constants

### 7.1 NV Center Constants

| Constant | Symbol | Value | Reference |
|----------|--------|-------|-----------|
| Ground state ZFS | D_gs | 2.870 GHz | [Doherty2013] |
| Excited state ZFS | D_es | 1.42 GHz | [Doherty2013] |
| NV gyromagnetic ratio | gamma_NV | 28.024 GHz/T | [Doherty2013] |
| NV g-factor | g_e | ~2.003 | [Doherty2013] |
| ZFS temperature coefficient | dD/dT | -74.2 kHz/K | [Acosta2010] |
| Zero-phonon line | ZPL | 637 nm (1.945 eV) | [Doherty2013] |
| NV^0 zero-phonon line | ZPL_0 | 575 nm | [Doherty2013] |
| Debye-Waller factor | DW | ~0.04 | [Doherty2013] |
| Excited state lifetime | tau_rad | ~12-13 ns | [Goldman2015] |
| Metastable singlet lifetime | tau_1E | ~250 ns (RT) | [Robledo2011] |

### 7.2 Hyperfine Constants

| Isotope | Nuclear spin | Hyperfine constant A_parallel | Peaks per resonance | Reference |
|---------|-------------|-------------------------------|---------------------|-----------|
| ^14N | I = 1 | -2.16 MHz | 3 (triplet) | [Doherty2013] |
| ^15N | I = 1/2 | +3.03 MHz | 2 (doublet) | [Doherty2013] |
| ^13C | I = 1/2 | ~130 MHz (nearest neighbor) | Typically unresolved | [Doherty2013] |

### 7.3 ISC Rates

| Transition | Rate | Lifetime | Reference |
|------------|------|----------|-----------|
| ^3E (m_s=+/-1) -> ^1A_1 | ~80 MHz | ~12 ns | [Goldman2015] |
| ^3E (m_s=0) -> ^1A_1 | ~8 MHz | ~120 ns | [Goldman2015] |
| ^1A_1 -> ^1E | ~1 GHz | ~1 ns | [Robledo2011] |
| ^1E -> ^3A_2 (m_s=0 pref.) | ~4 MHz | ~250 ns | [Robledo2011] |

### 7.4 Useful Conversions

| Conversion | Value |
|------------|-------|
| 1 GHz in magnetic field | 1 GHz / gamma_NV = 35.68 mT |
| 1 mT in frequency | gamma_NV * 1 mT = 28.024 MHz |
| 1 uT in frequency | 28.024 kHz |
| 1 K temperature shift | dD/dT * 1 K = -74.2 kHz ~ 2.6 uT apparent field |
| Bohr magneton | mu_B = 9.274 x 10^-24 J/T |
| Planck constant | h = 6.626 x 10^-34 J*s |

### 7.5 NV Geometry

The four NV orientations in Cartesian coordinates (unit vectors) [Doherty2013]:

```
nv_1 = [ 1,  1,  1] / sqrt(3)
nv_2 = [-1, -1,  1] / sqrt(3)
nv_3 = [-1,  1, -1] / sqrt(3)
nv_4 = [ 1, -1, -1] / sqrt(3)
```

The tetrahedral angle between any two NV axes is arccos(-1/3) ~ 109.47 degrees.

---

## 8. Glossary

| Term | Definition |
|------|------------|
| **B111** | Magnetic field component along the NV [111] axis, extracted from ODMR splitting |
| **Bias field** | Static external magnetic field applied to separate NV orientation families |
| **Contrast (C)** | Fractional depth of ODMR fluorescence dip; ratio of dip depth to baseline |
| **CW-ODMR** | Continuous-wave ODMR; standard QDM measurement protocol |
| **D (ZFS)** | Zero-field splitting; energy gap between m_s=0 and m_s=+/-1 at zero field |
| **ESR** | Electron spin resonance; sometimes used interchangeably with ODMR for NV centers |
| **frange** | Frequency range; one of two swept MW bands (low or high) in QDM measurement |
| **gamma_NV** | NV gyromagnetic ratio; 28.024 GHz/T |
| **GF artifact** | Green fluorescence artifact from MW antenna coupling; requires baseline correction |
| **Hyperfine** | Splitting of NV spin resonances due to nuclear spin interaction (^14N or ^15N) |
| **Induced magnetization** | Sample magnetization that tracks the applied bias field |
| **ISC** | Intersystem crossing; spin-dependent non-radiative transition between triplet and singlet states |
| **Linewidth (w)** | FWHM of an ODMR resonance dip |
| **NV center** | Nitrogen-vacancy defect in diamond; the quantum sensor |
| **NV^-** | Negatively charged NV center; the magnetically sensitive charge state |
| **NV^0** | Neutral NV center; not magnetically sensitive |
| **ODMR** | Optically detected magnetic resonance |
| **P1 center** | Substitutional nitrogen defect in diamond; spin bath that broadens NV linewidth |
| **Polarity** | Direction of applied bias field; flipped between measurements to separate remanent/induced |
| **QDM** | Quantum diamond microscope |
| **Remanent magnetization** | Sample magnetization that persists regardless of applied field |
| **ZFS** | Zero-field splitting |
| **ZPL** | Zero-phonon line; sharp spectral line of NV emission at 637 nm |

---

## 9. References

- **[Acosta2010]** Acosta, V.M., Bauch, E., Ledbetter, M.P., Waxman, A., Bouber, L.-S.,
  and Budker, D. (2010). "Temperature Dependence of the Nitrogen-Vacancy Magnetic Resonance
  in Diamond." *Phys. Rev. Lett.*, 104, 070801.
  DOI: [10.1103/PhysRevLett.104.070801](https://doi.org/10.1103/PhysRevLett.104.070801)

- **[Doherty2013]** Doherty, M.W., Manson, N.B., Delaney, P., Jelezko, F., Wrachtrup, J.,
  and Hollenberg, L.C.L. (2013). "The nitrogen-vacancy colour centre in diamond."
  *Physics Reports*, 528(1), 1-45.
  DOI: [10.1016/j.physrep.2013.02.001](https://doi.org/10.1016/j.physrep.2013.02.001)

- **[Fu2014]** Fu, R.R., Weiss, B.P., Lima, E.A., Harrison, R.J., Bai, X.-N., Desch, S.J.,
  Ebel, D.S., Suavet, C., Wang, H., Glenn, D., Le Sage, D., Kasama, T., Walsworth, R.L.,
  and Kuan, A.T. (2014). "Solar nebula magnetic fields recorded in the Semarkona meteorite."
  *Science*, 346(6213), 1089-1092.
  DOI: [10.1126/science.1259022](https://doi.org/10.1126/science.1259022)

- **[Fu2020]** Fu, R.R., Kehayias, P., Weiss, B.P., and Walsworth, R.L. (2020).
  "A Paleomagnetic Study of the Allende and Murchison CM Chondrites Using the Quantum
  Diamond Microscope." *Geochem. Geophys. Geosyst.*, 21, e2020GC009147.
  DOI: [10.1029/2020GC009147](https://doi.org/10.1029/2020GC009147)

- **[Glenn2015]** Glenn, D.R., Lee, K., Park, H., Weissleder, R., Yacoby, A., Lukin, M.D.,
  Lee, H., Walsworth, R.L., and Bhatt, S. (2015). "Single-cell magnetic imaging using a
  quantum diamond microscope." *Nature Methods*, 12, 736-738.
  DOI: [10.1038/nmeth.3449](https://doi.org/10.1038/nmeth.3449)

- **[Glenn2017]** Glenn, D.R., Fu, R.R., Kehayias, P., Le Sage, D., Lima, E.A., Weiss, B.P.,
  and Walsworth, R.L. (2017). "Micrometer-scale magnetic imaging of geological samples using
  a quantum diamond microscope." *Geochem. Geophys. Geosyst.*, 18, 3254-3267.
  DOI: [10.1002/2017GC006946](https://doi.org/10.1002/2017GC006946)

- **[Goldman2015]** Goldman, M.L., Sipahigil, A., Doherty, M.W., Yao, N.Y., Bennett, S.D.,
  Markham, M., Twitchen, D.J., Manson, N.B., Kubanek, A., and Lukin, M.D. (2015).
  "Phonon-Induced Population Dynamics and Intersystem Crossing in Nitrogen-Vacancy Centers."
  *Phys. Rev. Lett.*, 114, 145502.
  DOI: [10.1103/PhysRevLett.114.145502](https://doi.org/10.1103/PhysRevLett.114.145502)
  *Note: ISC rates also discussed in Goldman, PRB 91, 165201 (2015).*

- **[Kleinsasser2016]** Kleinsasser, E.E., Stanber, M.M., Barber, B.A., Nelson, G.B.,
  Walsworth, R.L., and Levine, E.V. (2016). "High density nitrogen-vacancy sensing surface
  created via He+ ion implantation of ^12C diamond." *Appl. Phys. Lett.*, 108, 202401.
  DOI: [10.1063/1.4949357](https://doi.org/10.1063/1.4949357)

- **[Levine2019]** Levine, E.V., Turner, M.J., Kehayias, P., Hart, C.A., Langellier, N.,
  Trubko, R., Glenn, D.R., Fu, R.R., and Walsworth, R.L. (2019). "Principles and techniques
  of the quantum diamond microscope." *Nanophotonics*, 8(11), 1945-1973.
  DOI: [10.1515/nanoph-2019-0209](https://doi.org/10.1515/nanoph-2019-0209)

- **[Nichols2021]** Nichols, C.I.O., Bryson, J.F.J., Herrero-Albillos, J., et al. (2021).
  "Meteorite Magnetism: A Decade of Progress and Upcoming Challenges." *AGU Advances*, 2(4),
  e2021AV000511.
  DOI: [10.1029/2021AV000511](https://doi.org/10.1029/2021AV000511)

- **[Ohno2012]** Ohno, K., Joseph Heremans, F., Bassett, L.C., Myers, B.A., Toyli, D.M.,
  Bleszynski Jayich, A.C., Palmstrom, C.J., and Awschalom, D.D. (2012). "Engineering shallow
  spins in diamond with nitrogen delta-doping." *Appl. Phys. Lett.*, 101, 082413.
  DOI: [10.1063/1.4748280](https://doi.org/10.1063/1.4748280)

- **[Robledo2011]** Robledo, L., Bernien, H., van der Sar, T., and Hanson, R. (2011).
  "Spin dynamics in the optical cycle of single nitrogen-vacancy centres in diamond."
  *New J. Phys.*, 13, 025013.
  DOI: [10.1088/1367-2630/13/2/025013](https://doi.org/10.1088/1367-2630/13/2/025013)

- **[Turner2022]** Oliver, S.M., Martynowych, D.J., Turner, M.J., Hopper, D.A.,
  Walsworth, R.L., and Levine, E.V. (2022). "Vector Magnetic Current Imaging of an 8 nm
  Process Node Chip and 3D Current Distributions Using the Quantum Diamond Microscope."
  arXiv: [2202.08135](https://arxiv.org/abs/2202.08135)

---

*This document is a pure physics/research reference. For codebase-specific implementation
details (array shapes, normalization methods, fitting internals), see the qdmpy-core source
documentation and `memory/` files.*
