# Sample and detector geometry

The experiment chamber is **spherical**. The X-ray beam enters from upstream, is defined by slits, and illuminates a **sample plate**. A **goniometer arm** carries both an **Axis Photonique CMOS area detector** and a **GaAs photodiode** for complementary measurements.

## Sample coordinate system

The sample is positioned with four primary motors:


| Axis             | Role                                                                                                                                                                                                                                                                  |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Sample X**     | Translation **along the goniometer axis** (colinear with Sample Theta). Positive typically moves the plate toward the motor side; negative toward the chamber door.                                                                                                   |
| **Sample Y**     | Translation **perpendicular to the beam** when Sample Theta is at 0°. Positive moves the plate **away** from the beam; negative **toward** the beam.                                                                                                                  |
| **Sample Z**     | Translation **normal to the sample plate**. At Sample Theta = 0°, positive is **up** in the lab frame and negative is **down**. Changing Z also changes **sample height relative to the beam** and, except at Sample Theta = 0°, couples to sample–detector distance. |
| **Sample Theta** | **Rotation** of the plate about the axis colinear with Sample X. **90°** places the surface **normal to the incident beam** (grazing-incidence experiments use other angles). 0° and 180° correspond to plate facing up vs down in the lab frame.                     |


Together, Sample X, Y, and Z form a **right-handed** system that **rotates with Sample Theta** for Y and Z, while X stays on the rotation axis.

Additional motors (for example **Sample Azimuthal Rotation**, piezo fine stages, or numbered **SampleRot** axes) provide extra degrees of freedom for specialized holders.

## Detector arm (CCD prefix)

The **same goniometer arm** carries:

- An **Axis Photonique CMOS** **area detector** for **scattering patterns** and imaging (this station **no longer** uses the previous **CCD** camera; frame size and pixel pitch follow the active CMOS mode and ROI).
- A **5 mm × 5 mm GaAs photodiode** for **direct-beam** or **high-flux** monitoring when the **detector** is moved out of the beam.

Arm motors use the **CCD** prefix:


| Axis          | Role                                                                                                                                                                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **CCD Theta** | **Rotation** of the arm in the scattering plane. At 0° the arm can be aligned with the direct beam from the chamber-door perspective; positive rotation follows the same handedness as Sample Theta over roughly **120°** of travel. |
| **CCD Y**     | **Sample–detector distance**. Positive moves the detector **farther** from the sample; negative **closer**. This is the primary lever for **Q** or camera length in scattering.                                                      |
| **CCD X**     | **Lateral** shift of the detector, colinear with the same direction as **Sample X**.                                                                                                                                                 |


**Pollux CCD X/Y** name alternative encoder or staging positions when a Pollux-style detector path is configured.

## Photodiode vs beamstop photodiode

- **Main photodiode** (GaAs): placed in the **direct beam** when the arm is at the **photodiode preset** (typically large CCD Y and small CCD X with CCD Theta near 0°). Used for **alignment**, **normalization**, and **NEXAFS**-style measurements with the **CMOS detector** out of the way.
- **Beamstop photodiode** (small Si diode on the beamstop): used when the **CMOS detector is in** scattering geometry to monitor **transmitted or scattered flux** without saturating the main diode.

## T-2T and reflectivity

**T-2T** couples **Sample Theta** and **CCD Theta** so the detector tracks at twice the sample angle for **specular reflectivity** (θ–2θ geometry).

## Beam stop

The **Beam Stop** motor positions a stop that **blocks the direct beam** from hitting the **CMOS sensor** during scattering, protecting the detector and defining shadowing for certain geometries.

## Alignment order (experimental practice)

1. **Instrument alignment**: put the **photodiode** in the direct beam using the CCD arm; optimize **CCD Theta** (and related positions) so the diode reads the beam cleanly.
2. **Sample alignment**: iterate **Sample Z** (often half-maximum of absorption or scattering) and **Sample Theta** (centroid or symmetry) so the sample intersects the beam as intended.

This repository’s higher-level scan helpers assume sensible shutter and exposure defaults; see [Digital I/O](digital-io.md) and [Analog inputs](analog-inputs.md) for what each signal represents.