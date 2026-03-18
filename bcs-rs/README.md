# bcs-rs: Rust transport and decoding for BCSz

`bcs-rs` is a Rust extension module that accelerates and hardens the client side of the BCSz beamline control system:

- High-performance Z85 decoding for large detector blobs.
- A Rust-backed ZeroMQ + CURVE client for the BCS server.
- A thin PyO3 layer that exposes this functionality as the `bcs_rs._core` Python module.

The design goal is that Python code uses a small, stable API in `bcs_rs._core` while Rust handles all low-level transport, timing, and blob decoding.


## Current Python BCSz behavior (for context)

The existing BCSz client in Python:

- Creates a global ZeroMQ context.
- Uses a REQ socket with CURVE security to talk to the LabVIEW BCS server.
- Fetches the server public key from an unauthenticated endpoint (`public` on `port`).
- Sends JSON requests (`command` + parameters) and receives JSON responses.
- Encodes motor status as a Python `Flag` bitfield.

### Relevant modules and status flags

```python
import sys
if sys.platform[:3] == "win":
    # zmq.asyncio does not support the default (proactor) event loop on Windows.
    # so set the event loop to one zmq supports
    pass  # no asyncio. Do nothing
import zmq
import zmq.utils.z85
import json
import time
from enum import Flag  # for MotorStatus


class MotorStatus(Flag):
    HOME = 1
    FORWARD_LIMIT = 2
    REVERSE_LIMIT = 4
    MOTOR_DIRECTION = 8
    MOTOR_OFF = 16
    MOVE_COMPLETE = 32
    FOLLOWING_ERROR = 64
    NOT_IN_DEAD_BAND = 128
    FORWARD_SW_LIMIT = 256
    REVERSE_SW_LIMIT = 512
    MOTOR_DISABLED = 1024
    RAW_MOTOR_DIRECTION = 2048
    RAW_FORWARD_LIMIT = 4096
    RAW_REVERSE_LIMIT = 8192
    RAW_FORWARD_SW_LIMIT = 16384
    RAW_REVERSE_SW_LIMIT = 32768
    RAW_MOVE_COMPLETE = 65536
    MOVE_LT_THRESHOLD = 131072

    def is_set(self, flag):
        return bool(self._value_ & flag._value_)
```

### BCSz server key fetch

```python
class BCSServer:
    _zmq_socket: zmq.Socket

    @staticmethod
    async def _get_server_public_key(addr, port):
        clear_socket = _zmq_context.socket(zmq.REQ)
        clear_socket.connect(f"tcp://{addr}:{port}")
        await clear_socket.send("public".encode())
        server_public = await clear_socket.recv()
        clear_socket.close()
        return server_public
```

### BCSz connect (ZeroMQ + CURVE)

```python
async def connect(self, addr="127.0.0.1", port=5577):
    global _zmq_context

    if not _zmq_context:
        if "zmq.asyncio" in sys.modules:
            _zmq_context = zmq.asyncio.Context()
        else:
            _zmq_context = zmq.Context()

    self._zmq_socket = _zmq_context.socket(zmq.REQ)

    (client_public_key, client_secret_key) = zmq.curve_keypair()

    server_public_key = await self._get_server_public_key(addr, port)

    print(f"Server Public Key {server_public_key}")

    self._zmq_socket.setsockopt(zmq.CURVE_SERVERKEY, server_public_key)
    self._zmq_socket.setsockopt(zmq.CURVE_PUBLICKEY, client_public_key)
    self._zmq_socket.setsockopt(zmq.CURVE_SECRETKEY, client_secret_key)

    self._zmq_socket.connect(f"tcp://{addr}:{port + 1}")
```

### BCSz request path

```python
async def bcs_request(self, command_name, param_dict, debugging=False):
    if debugging:
        print(f"API command {command_name} BEGIN.")

    api_call_start = time.time()
    param_dict["command"] = command_name
    param_dict["_unused"] = "_unused"
    if "self" in param_dict:
        del param_dict["self"]
    await self._zmq_socket.send(json.dumps(param_dict).encode())
    response_dict = json.loads(await self._zmq_socket.recv())
    response_dict["API_delta_t"] = time.time() - api_call_start

    if debugging:
        print(f"API command {command_name} END {response_dict['API_delta_t']} s.")

    return response_dict
```

All higher-level scan and control methods in BCSz ultimately delegate to `bcs_request`.

---

## Rust implementation: future setup

The Rust implementation replaces the Python transport with a Rust `BcsConnection` while keeping the high-level Python API:

- Rust owns all ZeroMQ + CURVE details and JSON I/O.
- Rust measures and injects `API_delta_t` timing into the response.
- Rust models `MotorStatus` as a typed bitflag set.
- Python keeps the `BCSServer` API and all scan wrappers, but no longer does JSON or timing.

High-level design:

- `bcs-rs` exposes a PyO3 module `bcs_rs._core`.
- `bcs_rs._core.BcsConnection` is the Rust-backed client.
- `BCSServer` in Python is composed of a `BcsConnection` and delegates to it.

Only two Python methods need to change:

- `BCSServer.connect` → construct a Rust `BcsConnection`.
- `BCSServer.bcs_request` → delegate to `BcsConnection.bcs_request` and simply log if `debugging` is enabled.

Everything else in the Python client can remain unchanged.

---

## Crate layout

The `bcs-rs` crate is organized into:

- `z85.rs`: high-performance Z85 decoder (exported as `decode_z85` / `decode_z85_parallel`).
- `transport.rs`: BCS control client implemented in Rust using ZeroMQ + CURVE and JSON.
- `lib.rs`: PyO3 glue exposing both Z85 helpers and the BCS transport API as `bcs_rs._core`.

Only `lib.rs` defines the Python module; `z85` and `transport` are internal Rust modules.

---

## Core Rust transport API

The central Rust type is a connection object that owns the ZeroMQ context and REQ socket:

```rust
pub struct BcsConnection {
    ctx: rzmq::Context,
    req: rzmq::Socket,
}
```

It provides a minimal interface:

```rust
impl BcsConnection {
    pub fn connect(
        addr: &str,
        port: u16,
        recv_timeout: Duration,
        send_timeout: Duration,
    ) -> Result<Self, BcsError>;

    pub fn bcs_request(
        &self,
        command_name: &str,
        params: BTreeMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value, BcsError>;
}
```

- `connect`:
  - Creates a ZeroMQ context.
  - Creates a REQ socket.
  - Generates a client CURVE key pair.
  - Contacts `addr:port` with a plain REQ socket to retrieve the server public key by sending the `"public"` command.
  - Configures `CURVE_SERVERKEY`, `CURVE_PUBLICKEY`, and `CURVE_SECRETKEY`.
  - Sets `RCVTIMEO` and `SNDTIMEO` to the requested timeouts.
  - Connects the secure REQ socket to `tcp://{addr}:{port + 1}`.
- `bcs_request`:
  - Adds `"command"` and `"_unused"` into the parameter map.
  - Serializes the payload to JSON and sends it on the REQ socket.
  - Measures elapsed time with `Instant::now()` / `elapsed()`.
  - Receives the JSON response and deserializes it into `serde_json::Value`.
  - Injects `API_delta_t` (seconds as `f64`) into the response object before returning it.

The error type `BcsError` wraps underlying ZeroMQ and JSON errors and is converted to `PyErr` in `lib.rs`.

---

## Motor status bitflags in Rust

The Python `MotorStatus` enum encodes motor controller state as a bitfield. In Rust this will be modeled as a typed flag set so that callers can inspect motor status without re-implementing bit arithmetic in Python.

Planned Rust representation:

```rust
bitflags::bitflags! {
    pub struct MotorStatus: u32 {
        const HOME                 = 1;
        const FORWARD_LIMIT        = 2;
        const REVERSE_LIMIT        = 4;
        const MOTOR_DIRECTION      = 8;
        const MOTOR_OFF            = 16;
        const MOVE_COMPLETE        = 32;
        const FOLLOWING_ERROR      = 64;
        const NOT_IN_DEAD_BAND     = 128;
        const FORWARD_SW_LIMIT     = 256;
        const REVERSE_SW_LIMIT     = 512;
        const MOTOR_DISABLED       = 1024;
        const RAW_MOTOR_DIRECTION  = 2048;
        const RAW_FORWARD_LIMIT    = 4096;
        const RAW_REVERSE_LIMIT    = 8192;
        const RAW_FORWARD_SW_LIMIT = 16384;
        const RAW_REVERSE_SW_LIMIT = 32768;
        const RAW_MOVE_COMPLETE    = 65536;
        const MOVE_LT_THRESHOLD    = 131072;
    }
}
```

This type will be exposed to Python via PyO3 in one of two forms:

- As a `#[pyclass]` with helper methods such as `is_set(name: str) -> bool`, or
- As integer bitfields on response objects with small helper functions that map raw integers to `MotorStatus` instances.

The long-term intent is that any motor-related responses coming back from the BCS server carry a typed `MotorStatus` alongside raw numeric values so that client code can write clear, intention-revealing checks instead of manual bit masking.

---

## Python-facing PyO3 layer

`lib.rs` exposes the transport API and Z85 helpers to Python through a PyO3 module:

```rust
#[pymodule]
mod _core {
    use super::*;

    #[pyclass]
    pub struct BcsConnection {
        inner: transport::BcsConnection,
    }

    #[pymethods]
    impl BcsConnection {
        #[new]
        fn new(
            addr: String,
            port: u16,
            recv_timeout_ms: Option<u64>,
            send_timeout_ms: Option<u64>,
        ) -> PyResult<Self> {
            let recv = Duration::from_millis(recv_timeout_ms.unwrap_or(5000));
            let send = Duration::from_millis(send_timeout_ms.unwrap_or(5000));
            let inner = transport::BcsConnection::connect(&addr, port, recv, send)?;
            Ok(Self { inner })
        }

        fn bcs_request<'py>(
            &self,
            py: Python<'py>,
            command_name: &str,
            params: &PyAny,
        ) -> PyResult<&'py PyAny> {
            let dict: &PyDict = params.downcast()?;
            let map = python_to_serde_map(dict)?;      // PyDict -> BTreeMap<String, Value>
            let resp_val = self.inner.bcs_request(command_name, map)?;
            let resp_py = serde_to_python(py, &resp_val)?; // Value -> PyObject
            Ok(resp_py.into_ref(py))
        }
    }

    #[pyfunction]
    fn decode_z85(data: &str) -> PyResult<Vec<u8>> { /* existing */ }

    #[pyfunction]
    fn decode_z85_parallel(py: Python<'_>, data: &str) -> PyResult<Vec<u8>> { /* existing */ }
}
```

From Python this appears as:

```python
from bcs_rs._core import BcsConnection, decode_z85, decode_z85_parallel
```

The existing Z85 functions stay unchanged; the new `BcsConnection` class is the transport entry point.

The helper functions `python_to_serde_map` and `serde_to_python` are responsible for converting between Python `dict` / `list` / scalars and `serde_json::Value`. They can be implemented manually or via `pyo3-serde`.

---

## Future Python BCSz integration

The Python `BCSServer` class in BCSz will be composed of a `BcsConnection` and will delegate all transport to it.

```python
from bcs_rs._core import BcsConnection

class BCSServer:
    def __init__(self, addr: str = "127.0.0.1", port: int = 5577) -> None:
        self._addr = addr
        self._port = port
        self._conn: BcsConnection | None = None

    async def connect(self, addr: str | None = None, port: int | None = None) -> None:
        if addr is not None:
            self._addr = addr
        if port is not None:
            self._port = port
        # Blocking Rust constructor → run in a thread
        self._conn = await asyncio.to_thread(
            BcsConnection,
            self._addr,
            self._port,
            5000,  # recv_timeout_ms
            5000,  # send_timeout_ms
        )

    async def bcs_request(self, command_name: str, param_dict: dict, debugging: bool = False) -> dict:
        if self._conn is None:
            raise RuntimeError("BCSServer not connected")

        response = await asyncio.to_thread(self._conn.bcs_request, command_name, param_dict)

        if debugging:
            api_dt = response.get("API_delta_t", None)
            print(f"API command {command_name} END {api_dt} s.")

        return response
```

All of the high-level BCSz methods (`acquire_data`, `list_motors`, `sc_*` scans, etc.) can continue to call `await self.bcs_request(...)` unchanged. Only the underlying transport and timing move into Rust.

---

## Extensibility

With this scaffolding in place, future enhancements can be added without breaking the Python API:

- Connection pooling in Rust for higher throughput when multiple concurrent requests are issued.
- Typed Rust helpers for heavy endpoints (for example, turning `GetInstrumentAcquired2DBase85` directly into a decoded 2D array) while still exposing a simple function to Python.
- Additional metrics and tracing on the Rust side to profile and debug BCS communication without instrumenting Python.

# BCSz Server rewrite in Rust

The main python functionality of BCSz is to control the beamline
using a zmq socket connection to the BCS server. Here is the relevent
main ptyhon code that will be rewritten in Rust:

## Python Code

<details>
<summary>Relevant modules and status flags</summary>
    <pre><code class="language-python">
import sys
if sys.platform[:3] == "win":
    # zmq.asyncio does not support the default (proactor) event loop on windows.
    # so set the event loop to one zmq supports
    pass  # no asyncio. Do nothing
import zmq
import zmq.utils.z85
import json
import time
from enum import Flag  # for MotorStatus


class MotorStatus(Flag):
    HOME = 1
    FORWARD_LIMIT = 2
    REVERSE_LIMIT = 4
    MOTOR_DIRECTION = 8
    MOTOR_OFF = 16
    MOVE_COMPLETE = 32
    FOLLOWING_ERROR = 64
    NOT_IN_DEAD_BAND = 128
    FORWARD_SW_LIMIT = 256
    REVERSE_SW_LIMIT = 512
    MOTOR_DISABLED = 1024
    RAW_MOTOR_DIRECTION = 2048
    RAW_FORWARD_LIMIT = 4096
    RAW_REVERSE_LIMIT = 8192
    RAW_FORWARD_SW_LIMIT = 16384
    RAW_REVERSE_SW_LIMIT = 32768
    RAW_MOVE_COMPLETE = 65536
    MOVE_LT_THRESHOLD = 131072

    def is_set(self, flag):
        return bool(self._value_ & flag._value_)
    </code></pre>
</details>

<details>
<summary>BCSServer (Python, key public method)</summary>

```python
class BCSServer:
    """..."""
    _zmq_socket: zmq.Socket

    @staticmethod
    async def _get_server_public_key(addr, port):
        clear_socket = _zmq_context.socket(zmq.REQ)
        clear_socket.connect(f'tcp://{addr}:{port}')
        await clear_socket.send('public'.encode())
        server_public = await clear_socket.recv()
        clear_socket.close()
        return server_public
```
</details>

<details>
<summary>connect method (Python, establishing CURVE ZMQ connection)</summary>

```python
async def connect(self, addr='127.0.0.1', port=5577):
    """
    (formerly the Constructor) Supply the zmq address string, addr, to reach this endstation.
    """
    global _zmq_context

    # the first server object will create the global zmq context
    if not _zmq_context:
        if 'zmq.asyncio' in sys.modules:    # Using asyncio? (was the module imported?)
            _zmq_context = zmq.asyncio.Context()
        else:
            _zmq_context = zmq.Context()

    self._zmq_socket = _zmq_context.socket(zmq.REQ)

    (client_public_key, client_secret_key) = zmq.curve_keypair()

    # server_public_key = asyncio.get_running_loop().run_until_complete(self._get_server_public_key(addr, port))
    server_public_key = await self._get_server_public_key(addr, port)

    print(f'Server Public Key {server_public_key}')

    self._zmq_socket.setsockopt(zmq.CURVE_SERVERKEY, server_public_key)
    self._zmq_socket.setsockopt(zmq.CURVE_PUBLICKEY, client_public_key)
    self._zmq_socket.setsockopt(zmq.CURVE_SECRETKEY, client_secret_key)

    self._zmq_socket.connect(f'tcp://{addr}:{port + 1}')
```

</details>
<details>
<summary>bcs_request method (Python, sends a request to the BCS server)</summary>

```python
async def bcs_request(self, command_name, param_dict, debugging=False):
    """
    The method responsible for direct communication to the BCS server

    :param command_name: Name of the API endpoint
    :type command_name: str
    :param param_dict: Parameter dictionary
    :type param_dict: dict
    """
    if debugging:
        print(f"API command {command_name} BEGIN.")

    api_call_start = time.time()
    param_dict['command'] = command_name
    param_dict['_unused'] = '_unused'
    if 'self' in param_dict:
        del param_dict['self']
    await self._zmq_socket.send(json.dumps(param_dict).encode())
    response_dict = json.loads(await self._zmq_socket.recv())
    response_dict['API_delta_t'] = time.time() - api_call_start

    if debugging:
        print(f"API command {command_name} END {response_dict['API_delta_t']} s.")

    return response_dict
```
</details>

## Rust Implementation

The goals of the rust implementation is to construct a rust extension module that can
act as a drop-in replacement that handles all metadata and the `bcs_request()` method.

This leaves the rest of the BCSz client code to be written in Python, and only the
communication with the BCS server to be handled by the rust extension module.

### Crate layout

The `bcs-rs` crate is organized into:

- `z85.rs`: high-performance Z85 decoder (already exported as `decode_z85` / `decode_z85_parallel`).
- `transport.rs`: BCS control client implemented in Rust using ZeroMQ + CURVE.
- `lib.rs`: PyO3 glue exposing both Z85 helpers and the BCS transport API as `bcs_rs._core`.

Only `lib.rs` is visible to Python; `z85` and `transport` remain internal Rust modules.

### Core Rust transport API

The central Rust type is a connection object that owns the ZeroMQ context and REQ socket:

```rust
pub struct BcsConnection {
    ctx: rzmq::Context,
    req: rzmq::Socket,
}
```

It provides a small interface:

```rust
impl BcsConnection {
    pub fn connect(
        addr: &str,
        port: u16,
        recv_timeout: Duration,
        send_timeout: Duration,
    ) -> Result<Self, BcsError>;

    pub fn raw_request(&self, json: &str) -> Result<String, BcsError>;
}
```

- `connect`:
  - Creates a ZeroMQ context.
  - Creates a REQ socket.
  - Generates a client CURVE key pair.
  - Contacts `addr:port` with a plain REQ socket to retrieve the server public key (sending the `"public"` command).
  - Configures `CURVE_SERVERKEY`, `CURVE_PUBLICKEY`, and `CURVE_SECRETKEY`.
  - Sets `RCVTIMEO` and `SNDTIMEO` to the requested timeouts.
  - Connects the secure REQ socket to `tcp://{addr}:{port + 1}`.
- `raw_request`:
  - Sends the provided JSON string on the REQ socket.
  - Blocks until the full response is received.
  - Returns the response body as a JSON string.

The error type `BcsError` wraps underlying ZeroMQ and JSON errors and is converted to `PyErr` in `lib.rs`.

### Motor status bitflags in Rust

The Python `MotorStatus` enum encodes motor controller state as a bitfield. In Rust this will be modeled as a typed flag set so that callers can inspect motor status without re-implementing bit arithmetic in Python.

Planned Rust representation:

```rust
bitflags::bitflags! {
    pub struct MotorStatus: u32 {
        const HOME                 = 1;
        const FORWARD_LIMIT        = 2;
        const REVERSE_LIMIT        = 4;
        const MOTOR_DIRECTION      = 8;
        const MOTOR_OFF            = 16;
        const MOVE_COMPLETE        = 32;
        const FOLLOWING_ERROR      = 64;
        const NOT_IN_DEAD_BAND     = 128;
        const FORWARD_SW_LIMIT     = 256;
        const REVERSE_SW_LIMIT     = 512;
        const MOTOR_DISABLED       = 1024;
        const RAW_MOTOR_DIRECTION  = 2048;
        const RAW_FORWARD_LIMIT    = 4096;
        const RAW_REVERSE_LIMIT    = 8192;
        const RAW_FORWARD_SW_LIMIT = 16384;
        const RAW_REVERSE_SW_LIMIT = 32768;
        const RAW_MOVE_COMPLETE    = 65536;
        const MOVE_LT_THRESHOLD    = 131072;
    }
}
```

This type will be exposed to Python via PyO3 in one of two forms:

- As a `#[pyclass]` with helper methods such as `is_set(name: str) -> bool`, or
- As integer bitfields on response objects with small helper functions that map raw integers to `MotorStatus` instances.

The long-term intent is that any motor-related responses coming back from the BCS server carry a typed `MotorStatus` alongside raw numeric values so that client code can write clear, intention-revealing checks instead of manual bit masking.

### Python-facing PyO3 layer

`lib.rs` exposes the transport API to Python through a PyO3 module:

```rust
#[pymodule]
mod _core {
    use super::*;

    #[pyclass]
    pub struct BcsConnection {
        inner: transport::BcsConnection,
    }

    #[pymethods]
    impl BcsConnection {
        #[new]
        fn new(
            addr: String,
            port: u16,
            recv_timeout_ms: Option<u64>,
            send_timeout_ms: Option<u64>,
        ) -> PyResult<Self> {
            let recv = Duration::from_millis(recv_timeout_ms.unwrap_or(5000));
            let send = Duration::from_millis(send_timeout_ms.unwrap_or(5000));
            let inner = transport::BcsConnection::connect(&addr, port, recv, send)?;
            Ok(Self { inner })
        }

        fn raw_request(&self, json: &str) -> PyResult<String> {
            Ok(self.inner.raw_request(json)?)
        }
    }

    #[pyfunction]
    fn decode_z85(data: &str) -> PyResult<Vec<u8>> { /* existing */ }

    #[pyfunction]
    fn decode_z85_parallel(py: Python<'_>, data: &str) -> PyResult<Vec<u8>> { /* existing */ }
}
```

From Python, this appears as:

```python
from bcs_rs._core import BcsConnection, decode_z85, decode_z85_parallel
```

The existing Z85 functions stay unchanged; only the new `BcsConnection` class is added.

### Future Python BCSz integration

The Python `BCSServer` class in BCSz will use `BcsConnection` as its transport layer, while preserving the existing high-level API.

```python
from bcs_rs._core import BcsConnection

class BCSServer:
    def __init__(self, addr: str = "127.0.0.1", port: int = 5577) -> None:
        self._addr = addr
        self._port = port
        self._conn: BcsConnection | None = None

    async def connect(self, addr: str | None = None, port: int | None = None) -> None:
        if addr is not None:
            self._addr = addr
        if port is not None:
            self._port = port
        self._conn = await asyncio.to_thread(
            BcsConnection,
            self._addr,
            self._port,
            5000,  # recv_timeout_ms
            5000,  # send_timeout_ms
        )

    async def bcs_request(self, command_name: str, param_dict: dict, debugging: bool = False) -> dict:
        if self._conn is None:
            raise RuntimeError("BCSServer not connected")

        api_call_start = time.perf_counter()

        payload = dict(param_dict)
        payload["command"] = command_name
        payload["_unused"] = "_unused"
        payload.pop("self", None)

        json_in = json.dumps(payload)
        json_out = await asyncio.to_thread(self._conn.raw_request, json_in)
        response_dict = json.loads(json_out)
        response_dict["API_delta_t"] = time.perf_counter() - api_call_start
        return response_dict
```

All of the high-level BCSz methods (`acquire_data`, `list_motors`, `sc_*` scans, etc.) can continue to call `await self.bcs_request(...)` unchanged. Only the underlying transport has moved into Rust.

### Extensibility

With this scaffolding in place, future enhancements can be added without breaking the Python API:

- Connection pooling in Rust for higher throughput when multiple concurrent requests are issued.
- Typed Rust helpers for heavy endpoints (for example, turning `GetInstrumentAcquired2DBase85` directly into a decoded 2D array) while still exposing a simple function to Python.
- Additional metrics and tracing on the Rust side to profile and debug BCS communication.

