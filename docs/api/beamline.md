# Beamline and Connection

`Beamline` is the high-level facade for beamline hardware control. `Connection` holds BCS server settings.

## Connection

::: resonance.api.core.beamline.Connection

## Beamline

::: resonance.api.core.beamline.Beamline

## Example

```python
bl = await Beamline.create()
data = await bl.ai.trigger_and_read(["Photodiode"], acquisition_time=1.0)
await bl.motors.set("Sample X", 10.5)
```
