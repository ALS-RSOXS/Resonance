# Area detector (2D CMOS)

The scattering camera is an **Axis Photonique CMOS** large-area detector on the **same goniometer arm** as the main photodiode. It **replaced** the earlier **CCD** camera on this station; control and analog labels may still use **CCD** in names (for example **CCD Theta** motors or **CCD Temperature** readbacks) even though the sensor is CMOS.

## Physical role

- **Position**: set by **CCD Theta**, **CCD X**, and **CCD Y** (see [Sample and detector geometry](sample-and-detector-geometry.md)). Those names are **legacy beamline axes** for the detector arm, not a statement that the sensor is a CCD.
- **Protection**: the **Beam Stop** blocks the **direct beam** from hitting the sensor during scattering; align using the **photodiode** or **beamstop diode** instead of exposing the raw beam on the CMOS.
- **Temperature**: follow vendor and beamline practice for the CMOS head and readout electronics. Legacy analogs **CCD Temperature** and **Camera Temp Setpoint** still report cooling or setpoint for the camera stack; confirm limits against the **Axis Photonique** configuration in use.

## Readout parameters

**Region of interest (ROI)** and **binning** motors (or equivalent software parameters) crop the sensor and reduce readout time at the cost of field of view or spatial resolution. Acquired frame **height and width** come from the instrument payload (they are not fixed to the old 2048 layout). Exposure is gated by **shutter** timing and optional **CCD shutter control** / **inhibit** lines so that motion does not blur images unintentionally.

## Software integration

Application code treats the detector as an **instrument** (default name **Axis Photonique** in this repository) with **acquire**, **status**, and **attribute** access (vendor SDK behind the control layer). Quality metrics such as **exposure time**, **mean intensity**, **max intensity**, and **histogram width** are used to **reject bad frames** (for example shutter edge effects or saturation) before writing files or stacks.

## Relation to other signals

- **AI 6 BeamStop** or **Photodiode** channels provide **scalar flux** concurrent with 2D frames for **normalization** or **absorption** estimates.
- **Digital** lines coordinate **shutter**, **camera trigger**, and **pause** during multi-point scans.

For API-level names used in Python helpers, see the API reference; this page describes **hardware and experiment roles** only.