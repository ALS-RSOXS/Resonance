# Optical path and upstream optics

The beamline splits naturally into **upstream optics** (everything before the spherical experimental chamber) and **in-chamber** sample and detector hardware.

## 1. EPU source

Soft X-rays are produced by an **elliptically polarizing undulator (EPU)** on the storage ring. The undulator gap and polarization setting determine flux, harmonic content, and polarization state (linear vs circular vs intermediate) as a function of energy. Ring **beam current** is read as an analog channel for long-term normalization and drift awareness.

## 2. Horizontal exit slit

After the undulator, a **horizontal exit slit** selects the useful portion of the fan of radiation before dispersion.

## 3. M101 monochromator

**Beamline energy** is set by the **M101 monochromator**, a grating instrument. Different gratings cover different energy ranges (for example lower vs higher photon energies). Monochromator motors include grating selection, vessel position, feedback, and small deflections used for alignment and stability.

## 4. M103 mirror and focusing

Downstream of the monochromator, **M103** (and related mirror optics) **focus and steer** the beam toward the experiment. Mirror motors adjust yaw and bend to shape the focal spot at the sample.

## 5. Higher-order suppressor (HOS)

A **four-bounce mirror assembly** removes **higher harmonic** content that would otherwise contaminate low-energy scattering or spectroscopy. Its incidence angle is adjustable over a limited range, or the assembly can be **withdrawn** from the beam when unfiltered spectrum is required (with the trade-off that harmonics are no longer suppressed).

## 6. Gold mesh and flux monitoring

A **gold mesh** can be inserted upstream of the HOS for photoelectric current-based flux measurement. A second mesh assembly **downstream of the HOS** but **before the first JJ slits** provides **incident flux** suitable for normalization when the upstream spectrum is not appropriate (for example certain low-energy scattering conditions). The corresponding analog channel is commonly used as an **Izero** reference for absorbed intensity.

## 7. JJ scatter slits (three stations)

**JJ scatter slits** define beam size and position at three distances:


| Station           | Approximate role                                                                               |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| **Upstream JJ**   | First collimation ~1.5 m before the sample chamber; removes stray scatter from upstream optics |
| **Middle JJ**     | Further cleaning ~0.6 m before the sample                                                      |
| **In-chamber JJ** | Final beam definition ~0.2 m before the sample                                                 |


Each station has **four motors**: vertical and horizontal **translation** (centering) and **aperture** (size).

## 8. Shutter and chamber entry

Before and inside the chamber, **shutters** (often piezo or air-assisted) gate X-ray exposure. **Digital outputs** open and close the beam; **digital inputs** report dump status and shutter-related interlocks. In-chamber **lighting** is separate from the X-ray shutter and is used for alignment cameras.

## Summary diagram (conceptual)

```text
Ring -> EPU -> exit slit -> M101 (energy) -> mirrors (e.g. M103) -> HOS
  -> [optional mesh] -> Upstream JJ -> Middle JJ -> shutter path
  -> In-chamber JJ -> sample -> detector arm (photodiode / CMOS area detector)
```

Next: [Sample and detector geometry](sample-and-detector-geometry.md) for coordinate axes and alignment roles.