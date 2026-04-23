# Digital I/O (DIO)

Digital lines are **binary** controls or status bits. **DO** drives an output (shutter, light, trigger); **DI** reads a state (dump, frame loss, shutter feedback).

| Name | Type | Role |
|------|------|------|
| Shutter Rev | DO | Shutter reverse or secondary shutter direction control |
| Lightfiled Frame Loss | DI | Indicates loss of frames from the LightField camera path (when used) |
| Nothing | — | Reserved or unused bit |
| Camera Scan | DO | Pulses or arms camera acquisition during scans |
| Shutter Output | DO | **Main X-ray shutter**; opens or closes the beam |
| Air Shutter Output | DO | Air-actuated shutter segment |
| Light Output | DO | Chamber **visible light** for alignment cameras |
| Beam Dumped | DI | Ring **beam dump** asserted; beam is not delivered |
| PZT Shutter Status | DI | Piezo shutter position or interlock state |
| Camera Shutter In | DI | Camera shutter feedback (when wired) |
| Do Pause Trigger | DO | Drives pause hardware for scans |
| Trigger Pause Trigger | DO | Triggers pause logic from software |
| Shutter Inhibit | DO | **Prevents** shutter from opening when asserted |
| Trigger + Inhibit | DO | Combined trigger and inhibit line for gated acquisition |

## Safety and workflow notes

- Treat **Shutter Output** and **Shutter Inhibit** as **safety-critical**: never open the shutter unless sample, detector, and personnel constraints are satisfied.
- **Beam Dumped** should be checked before interpreting **low counts** as a sample effect.
- **Camera Scan** and pause triggers coordinate **detector readout** with motor or energy motion; misconfiguration can cause **partial frames** or **paused** scans.

Some installations expose additional DIO lines in diagnostics screens; the table lists the **standard visible set** for this beamline configuration.
