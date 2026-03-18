use bitflags::bitflags;
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use zmq::{Context, CurveKeyPair, Socket};

use crate::z85;

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct MotorStatus: u32 {
        const HOME = 1;
        const FORWARD_LIMIT = 2;
        const REVERSE_LIMIT = 4;
        const MOTOR_DIRECTION = 8;
        const MOTOR_OFF = 16;
        const MOVE_COMPLETE = 32;
        const FOLLOWING_ERROR = 64;
        const NOT_IN_DEAD_BAND = 128;
        const FORWARD_SW_LIMIT = 256;
        const REVERSE_SW_LIMIT = 512;
        const MOTOR_DISABLED = 1024;
        const RAW_MOTOR_DIRECTION = 2048;
        const RAW_FORWARD_LIMIT = 4096;
        const RAW_REVERSE_LIMIT = 8192;
        const RAW_FORWARD_SW_LIMIT = 16384;
        const RAW_REVERSE_SW_LIMIT = 32768;
        const RAW_MOVE_COMPLETE = 65536;
        const MOVE_LT_THRESHOLD = 131072;
    }
}

#[derive(Debug)]
pub enum BcsError {
    Zmq(zmq::Error),
    Json(serde_json::Error),
    Connect(String),
    ServerKeyDecode(String),
}

impl std::fmt::Display for BcsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BcsError::Zmq(e) => write!(f, "ZMQ: {}", e),
            BcsError::Json(e) => write!(f, "JSON: {}", e),
            BcsError::Connect(s) => write!(f, "connect: {}", s),
            BcsError::ServerKeyDecode(s) => write!(f, "server key: {}", s),
        }
    }
}

impl std::error::Error for BcsError {}

impl From<zmq::Error> for BcsError {
    fn from(e: zmq::Error) -> Self {
        BcsError::Zmq(e)
    }
}

impl From<serde_json::Error> for BcsError {
    fn from(e: serde_json::Error) -> Self {
        BcsError::Json(e)
    }
}

fn duration_to_ms(d: Duration) -> i32 {
    let ms = d.as_millis();
    if ms > i32::MAX as u128 {
        i32::MAX
    } else {
        ms as i32
    }
}

fn fetch_server_public_key(ctx: &Context, addr: &str, port: u16) -> Result<[u8; 32], BcsError> {
    let clear_socket = ctx.socket(zmq::REQ).map_err(BcsError::Zmq)?;
    let endpoint = format!("tcp://{}:{}", addr, port);
    clear_socket.connect(&endpoint).map_err(BcsError::Zmq)?;
    clear_socket.send("public", 0).map_err(BcsError::Zmq)?;
    let raw = clear_socket.recv_bytes(0).map_err(BcsError::Zmq)?;
    drop(clear_socket);
    if raw.len() == 32 {
        let mut out = [0u8; 32];
        out.copy_from_slice(&raw[..32]);
        Ok(out)
    } else if raw.len() == 40 {
        let decoded = z85::decode_scalar(&raw).map_err(|e| BcsError::ServerKeyDecode(e.to_string()))?;
        if decoded.len() != 32 {
            return Err(BcsError::ServerKeyDecode(format!(
                "Z85 decoded to {} bytes, expected 32",
                decoded.len()
            )));
        }
        let mut out = [0u8; 32];
        out.copy_from_slice(&decoded[..32]);
        Ok(out)
    } else {
        Err(BcsError::ServerKeyDecode(format!(
            "server key length {} (expected 32 or 40)",
            raw.len()
        )))
    }
}

pub struct BcsConnection {
    _ctx: Context,
    req: Socket,
}

impl BcsConnection {
    pub fn connect(
        addr: &str,
        port: u16,
        recv_timeout: Duration,
        send_timeout: Duration,
    ) -> Result<Self, BcsError> {
        let ctx = Context::new();
        let server_key = fetch_server_public_key(&ctx, addr, port)?;
        let client_pair = CurveKeyPair::new().map_err(BcsError::Zmq)?;
        let req = ctx.socket(zmq::REQ).map_err(BcsError::Zmq)?;
        req.set_curve_serverkey(&server_key).map_err(BcsError::Zmq)?;
        req.set_curve_publickey(&client_pair.public_key).map_err(BcsError::Zmq)?;
        req.set_curve_secretkey(&client_pair.secret_key).map_err(BcsError::Zmq)?;
        req.set_rcvtimeo(duration_to_ms(recv_timeout)).map_err(BcsError::Zmq)?;
        req.set_sndtimeo(duration_to_ms(send_timeout)).map_err(BcsError::Zmq)?;
        let secure_endpoint = format!("tcp://{}:{}", addr, port + 1);
        req.connect(&secure_endpoint).map_err(|e| {
            BcsError::Connect(format!("{}: {}", secure_endpoint, e))
        })?;
        Ok(BcsConnection { _ctx: ctx, req })
    }

    pub fn send_and_recv(&self, payload: &[u8]) -> Result<Vec<u8>, BcsError> {
        self.req.send(payload, 0).map_err(BcsError::Zmq)?;
        self.req.recv_bytes(0).map_err(BcsError::Zmq)
    }

    pub fn bcs_request(
        &self,
        command_name: &str,
        mut params: BTreeMap<String, Value>,
    ) -> Result<Value, BcsError> {
        params.insert("command".to_string(), Value::String(command_name.to_string()));
        params.insert("_unused".to_string(), Value::String("_unused".to_string()));
        params.remove("self");
        let body = serde_json::to_vec(&params).map_err(BcsError::Json)?;
        let start = Instant::now();
        let response_bytes = self.send_and_recv(&body)?;
        let elapsed_secs = start.elapsed().as_secs_f64();
        let mut response: Value = serde_json::from_slice(&response_bytes).map_err(BcsError::Json)?;
        if let Some(obj) = response.as_object_mut() {
            obj.insert(
                "API_delta_t".to_string(),
                Value::Number(serde_json::Number::from_f64(elapsed_secs).unwrap_or(serde_json::Number::from(0))),
            );
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motor_status_bits() {
        let s = MotorStatus::MOVE_COMPLETE | MotorStatus::HOME;
        assert!(s.contains(MotorStatus::MOVE_COMPLETE));
        assert!(s.contains(MotorStatus::HOME));
        assert!(!s.contains(MotorStatus::MOTOR_OFF));
        assert_eq!(s.bits(), 33);
    }

    #[test]
    fn motor_status_from_bits() {
        let s = MotorStatus::from_bits_truncate(32);
        assert!(s.contains(MotorStatus::MOVE_COMPLETE));
    }
}
