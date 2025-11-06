# API Connectivity Architecture
## Goal
- [ ] Build safe programatic interface to the beamline computer
- [ ] Recover the ability to run scheduled macro scans from pandas interface
- [ ] Build real time data reduction.
- [ ] Facilitate feedback driven beamline scans.

# Safe programatic interface
We need the ability to safely interface with the beamline computer's control scheme. The BCSz file constains necescary documentation on the control current interface. But the interface allows for too much controll and let's people get into troube by moving a motor with the wrong commands, or by incorrectly reading the data. At this level we want to be abele to do the following things

1. `MOVE MOTOR (SAMPLE X) to 10`
2. `JOG MOTOR (SAMPLE X) by 1`
3. `GET MOTOR (SAMPLE X)`
4. `MOVE MOTOR by 10 in .1 incriments`
5. `GET PHOTODIODE`
6. `GET PHOTODIODE for 10 sec`
7. `ACTUATE (SHUTTER) 10 sec`

These are generally simple interfaces. We can break the interface down into four primary components: AI's (Analog Inputs), Motors, DIO's (Didgital Input Output), Intsruments, and Miscilanious.

## Ai's
These handle sensors that are continuously reading and returing results. When we read out the Ai's, we want to make sure that we are getting an array back, allong with calculating the principle value, and standard deviation for that array value. Though the BCSz API, we can do this using the following commands
```python
response = server.aquire_data(chans=["Photodiode"], time = 1)

```
Both of these results will return back a dictionary. The standard dictionary return obeys the following model.
```python
from pydantic import BaseModel, Field

class AiData(BaseModel):
    chan: str
    period: float
    time: float
    data: list[float]


class AiResponse(BaseModel):
    success: bool
    error_description: str = Field(
        alias="error description"
    )
    log: bool = Field(alias="log?")
    not_found: list[str] = []
    chans: list[AiData]
    API_delta_t: float

    model_config = {"populate_by_name": True}
```

This is what the reponse ooks like, but we want to process the results into something useable. To do this, we want to compute the statistics on the returned data. Most importantly, we want
1. Data Array
2. Array Mean
3. Array Standard error
