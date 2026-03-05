from bcs import BCSz


async def get_instrument_acquired2d_string(conn: BCSz.BCSServer, name: str) -> dict:
    """
    Dependency injection for the get_instrument_acquired2d method.

    Parameters
    ----------
    conn : BCSz.BCSServer
        BCSz.BCSServer object
    name : str
        Name of the instrument

    Returns
    -------
    dict
        Dictionary containing the acquired data
    """
    return await conn.bcs_request('GetInstrumentAcquired2DString', dict(locals()))


class AreaDetector:
    def __init__(self, conn: BCSz.BCSServer):
        self._conn = conn

    ...
