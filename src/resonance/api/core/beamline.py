from bcs import BCSz
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from resonance.api.core.ai import AIAccessor
from resonance.api.core.dio import DIOAccessor
from resonance.api.core.motors import MotorAccessor
from resonance.api.core.scan import ScanExecutor


class Connection(BaseSettings):
    """Configuration for BCS server connection from environment variables."""

    addr: str = Field(default="localhost", alias="BCS_SERVER_ADDRESS")
    port: int = Field(default=5577, alias="BCS_SERVER_PORT")
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Beamline:
    """ """

    def __init__(self, conn: BCSz.BCSServer):
        self._conn = conn
        self.ai = AIAccessor(conn)
        self.motors = MotorAccessor(conn)
        self.dio = DIOAccessor(conn)
        self._executor = ScanExecutor(conn)

    @classmethod
    async def create(cls) -> "Beamline":
        config = Connection()
        server = BCSz.BCSServer()
        await server.connect(**config.model_dump())
        return cls(server)
