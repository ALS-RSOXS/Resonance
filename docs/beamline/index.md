# ALS 11.0.1.2 RSoXS beamline

This section describes the **physical beamline** at ALS station 11.0.1.2 (RSoXS): how X-rays reach the sample, how the sample and detector move, and what the analog, digital, and imaging channels represent in an experiment. It is written for experimental planning and alignment, not for low-level control-system API naming.

## What RSoXS does

RSoXS (Resonant Soft X-ray Scattering) uses tunable soft X-rays to probe **samples in vacuum** and **scattering or spectroscopy** signals from **photodiodes**, **electron yield**, or a **large-area 2D detector** (**Axis Photonique CMOS** on this station, replacing the earlier CCD). The control system exposes motors (positions and energies), analog inputs (signals), digital lines (shutters, lights, triggers), and an **area detector** instrument for 2D images.

## How to read these pages

| Page | Contents |
|------|----------|
| [Optical path](optical-path.md) | Source to chamber: undulator, monochromator, mirrors, higher-order suppressor, slits, flux monitoring |
| [Sample and detector geometry](sample-and-detector-geometry.md) | Sample X/Y/Z/Theta, detector arm (CCD-prefixed motors), photodiode vs beamstop, typical alignment roles |
| [Motors](motors.md) | Motor groups: sample, detector arm, energy, EPU, mirrors, slits, shutter, environment, camera ROI |
| [Analog inputs](analog-inputs.md) | Each AI channel: what is measured, typical use, units |
| [Digital I/O](digital-io.md) | Shutter, lights, triggers, inhibits, status bits |
| [Area detector](area-detector.md) | Axis Photonique CMOS 2D detector, exposure quality, relation to the detector arm (CCD-prefixed motors) |

For step-by-step alignment algorithms and checkpointing, see project skills and internal reference material used by beamline scientists; this documentation stays at the component level.
