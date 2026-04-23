# Analog inputs (AI)

Analog inputs are **continuous readbacks** used for normalization, alignment, temperature monitoring, and timing diagnostics. Channel indices 0–20 are fixed on this beamline; names below match the **human-readable** labels used in control and acquisition software.


| Index | Name                     | What it represents                                                                  | Typical use                                                   |
| ----- | ------------------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 0     | EPU Polarization         | Undulator polarization mode readback                                                | Confirm circular vs linear settings                           |
| 1     | Coolstage Temp C         | Cryogenic holder temperature                                                        | Low-T experiments                                             |
| 2     | CCD Temperature          | Detector sensor or camera-stack temperature (legacy channel name; hardware is CMOS) | Verify cooling or thermal stability before long exposures     |
| 3     | Beam Current             | Storage ring current                                                                | Long-term drift and ring fill state                           |
| 4     | TEY signal               | Total electron yield from sample                                                    | Surface-sensitive NEXAFS                                      |
| 5     | Izero                    | Upstream intensity (often mesh)                                                     | Normalize absorption or scattering                            |
| 6     | Photodiode               | Main GaAs diode in direct beam                                                      | Alignment, flux, transmission                                 |
| 7     | AI 0                     | Spare analog input                                                                  | User-defined front-end                                        |
| 8     | AI 3 Izero               | Gold mesh downstream of HOS, before first JJ slits                                  | Alternate normalization path                                  |
| 9     | AI 5                     | Spare analog input                                                                  | User-defined                                                  |
| 10    | AI 6 BeamStop            | Small Si diode on beamstop                                                          | Flux while the area detector (CMOS) is in scattering geometry |
| 11    | AI 7                     | Spare analog input                                                                  | User-defined                                                  |
| 12    | Temperature Controller   | Hot stage readback                                                                  | Annealing or high-T chemistry                                 |
| 13    | PZT Shutter              | Piezo shutter position or status analog                                             | Shutter diagnostics                                           |
| 14    | Pause Trigger            | Pause line level                                                                    | Scan pause / interlock visibility                             |
| 15    | LV Memory                | LabVIEW memory usage                                                                | Operator health of control host                               |
| 16    | Deriv Photodiode         | Time derivative of photodiode signal                                                | Detect fast beam flicker                                      |
| 17    | Time Stamp Error         | Clock sync error                                                                    | Diagnose timing between subsystems                            |
| 18    | Time Stamp Transmit Time | Stamp message latency                                                               | Network or driver timing                                      |
| 19    | Time Stamp Server Time   | Server-side time base                                                               | Correlate events to host clock                                |
| 20    | Camera Temp Setpoint     | Camera cooling setpoint readback (legacy name)                                      | Confirm commanded detector temperature                        |


**Sign conventions**: some channels may read **negative** after cabling polarity, amplifier offset, or direction convention. Always **zero or calibrate** before precision absorption work.

**Choosing a normalization channel**: use **Photodiode** when the main diode is in the beam; use **AI 3 Izero** or **Izero** when the mesh path matches your energy and harmonic conditions; use **AI 6 BeamStop** when the **CMOS area detector** is exposed and the direct beam must stay off the main diode.