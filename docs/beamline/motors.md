# Motors and controlled axes

Motors are grouped by **function**. Each row is a **logical axis** on the beamline: position, angle, energy, aperture, or a derived setpoint. Values are in **millimeters**, **degrees**, **electron volts**, or **dimensionless** as noted in the beamline reference.

## Sample positioning


| Motor                     | Purpose                                                                           |
| ------------------------- | --------------------------------------------------------------------------------- |
| Sample X                  | Along rotation axis; scans the plate through the beam along the goniometer axis   |
| Sample Y                  | Perpendicular to beam at Sample Theta = 0; lateral offset of the illuminated spot |
| Sample Z                  | Normal to plate; height and (when tilted) coupling to sample–detector distance    |
| Sample Theta              | Primary sample tilt; grazing vs normal incidence                                  |
| Sample Azimuthal Rotation | In-plane rotation of the sample about the surface normal                          |
| Sample Y Scaled           | Derived or scaled Y used in specific scan recipes                                 |
| Sample Number             | Logical sample index for bookkeeping (not a mechanical axis)                      |


## Detector arm and reflectivity


| Motor                      | Purpose                                                            |
| -------------------------- | ------------------------------------------------------------------ |
| CCD Theta                  | Detector arm angle in the scattering plane                         |
| CCD X                      | Lateral detector position (same line as Sample X)                  |
| CCD Y                      | Camera length / sample–detector distance                           |
| Pollux CCD X, Pollux CCD Y | Alternate staging for Pollux detector configuration                |
| T-2T                       | Coupled θ–2θ motion for specular reflectivity                      |
| Beam Stop                  | Beamstop position to block direct beam on the area detector (CMOS) |


## Energy and monochromator (M101)


| Motor                                                | Purpose                                          |
| ---------------------------------------------------- | ------------------------------------------------ |
| Beamline Energy                                      | Commanded photon energy at the experiment        |
| Beamline Energy Goal                                 | Target energy for ramped or stepped energy scans |
| Mono Energy                                          | Monochromator internal energy setpoint           |
| Mono 101 Grating                                     | Grating choice for energy range and resolution   |
| Mono 101 Vessel                                      | Monochromator vessel translation                 |
| M101 Feedback                                        | Closed-loop energy stabilization                 |
| M101 Horizontal Deflection, M101 Vertical Deflection | Fine steering at the monochromator               |


## EPU (undulator)


| Motor            | Purpose                                             |
| ---------------- | --------------------------------------------------- |
| EPU Gap          | Undulator gap; sets fundamental and flux            |
| EPU Z            | Longitudinal undulator position                     |
| EPU Polarization | Polarization mode (circular vs linear S vs P, etc.) |


## Mirrors (M103, M121)


| Motor                                  | Purpose                                              |
| -------------------------------------- | ---------------------------------------------------- |
| M103 Yaw, M103 Bend Up, M103 Bend Down | Mirror pointing and curvature for focus and steering |
| M121 Translation                       | Secondary mirror translation                         |


## Entrance and exit slits


| Motor                                        | Purpose                                                 |
| -------------------------------------------- | ------------------------------------------------------- |
| Entrance Slit Width (and variants)           | Horizontal acceptance before dispersion                 |
| Exit Slit Top/Bottom/Left/Right              | Blade positions defining the beam at monochromator exit |
| Horizontal/Vertical Exit Slit Size, Position | Composite horizontal and vertical exit slit controls    |
| Horizontal/Vertical Slit Position, Size      | Additional slit axes where configured                   |


## JJ scatter slits (upstream, middle, in-chamber)

For each of the three stations: **Vert Trans**, **Horz Trans**, **Vert Aperture**, **Horz Aperture** define **centering** and **aperture** in vertical and horizontal directions.

## Higher-order suppressor and diagnostics


| Motor                   | Purpose                                        |
| ----------------------- | ---------------------------------------------- |
| Higher Order Suppressor | Four-bounce harmonic rejection mirror position |
| Diag 106                | Named diagnostic stage (beam diagnostics path) |


## Shutters


| Motor              | Purpose                              |
| ------------------ | ------------------------------------ |
| PiezoShutter Trans | Fast piezo shutter translation       |
| PZT Shutter        | Piezo shutter drive or encoded state |


## Environment and sample environment


| Motor                  | Purpose                                                                             |
| ---------------------- | ----------------------------------------------------------------------------------- |
| Temperature Controller | Hot stage setpoint (often in K with calibration to °C)                              |
| Coolstage              | Cryogenic stage temperature control                                                 |
| Camera Temp Setpoint   | Camera cooling setpoint (legacy name; CMOS stack may differ from old CCD setpoints) |


## Camera and ROI


| Motor                                           | Purpose                                          |
| ----------------------------------------------- | ------------------------------------------------ |
| CCD Camera Shutter Inhibit, CCD Shutter Control | Software or hardware shutter interlock and drive |
| Camera ROI X/Y/Width/Height                     | Region of interest on the sensor                 |
| Camera ROI X Bin, Camera ROI Y Bin              | Binning for speed or SNR                         |


## Fine positioning and outputs


| Motor                       | Purpose                                         |
| --------------------------- | ----------------------------------------------- |
| Piezo Vertical, Piezo Horiz | Piezo fine translation of sample or stage       |
| OSP Adjustment              | Optical sample position trim                    |
| AO 0, AO 1                  | General analog outputs (often auxiliary drives) |


## Multi-channel scaler (MCS)

**MCS_axis0** … **MCS_axis4** are logical axes tied to scaler hardware used in stepped or multi-channel acquisition modes.

## Additional sample rotations

**SampleRot0** … **SampleRot4** are extra rotation axes for multi-axis sample manipulators.

Not every scan uses every axis; experimental plans should list only the motors that move for a given measurement and respect **shutter** and **detector** limits before large moves.