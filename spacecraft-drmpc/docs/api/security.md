# Security and Encryption API Documentation

This document provides comprehensive documentation for the security and encryption systems used in multi-agent spacecraft communications.

## Table of Contents
- [Security Framework](#security-framework)
- [Encryption Manager](#encryption-manager)
- [Key Management](#key-management)
- [Message Authentication](#message-authentication)
- [Communication Security](#communication-security)
- [Code Examples](#code-examples)

---

## Security Framework

The security framework provides military-grade encryption and authentication for spacecraft communications.

### Class Definition
```python
class SecurityManager:
    """
    Central security management for spacecraft communications
    
    Implements AES-256 encryption, RSA key exchange, message authentication,
    and comprehensive security protocols for space applications.
    """
```

### Constructor
```python
def __init__(self, config: Dict)
```

**Configuration Parameters:**
```python
config = {
    # Encryption Settings
    'encryption_algorithm': str,        # 'AES-256-GCM', 'ChaCha20-Poly1305'
    'key_exchange_algorithm': str,      # 'RSA-2048', 'ECDH-P384'
    'authentication_algorithm': str,    # 'HMAC-SHA256', 'HMAC-SHA384'
    
    # Key Management
    'key_rotation_interval': int,       # Key rotation period (seconds)
    'key_derivation_iterations': int,   # PBKDF2 iterations (default: 100000)
    'master_key_length': int,          # Master key length in bits (256/512)
    
    # Security Policies
    'max_message_age': int,            # Maximum message age (seconds)
    'replay_window_size': int,         # Replay protection window
    'failed_auth_lockout': int,        # Lockout time for failed authentication
    
    # Certificate Management
    'certificate_authority': str,       # CA certificate path
    'agent_certificate': str,          # Agent certificate path  
    'private_key_path': str,           # Private key file path
    'certificate_chain': List[str],    # Certificate chain
    
    # Advanced Features
    'perfect_forward_secrecy': bool,   # Enable PFS
    'post_quantum_crypto': bool,       # Enable post-quantum algorithms
    'hardware_security_module': bool,  # Use HSM for key storage
    'audit_logging': bool             # Enable security audit logs
}
```

**Example Configuration:**
```python
config = {
    'encryption_algorithm': 'AES-256-GCM',
    'key_exchange_algorithm': 'RSA-2048',
    'authentication_algorithm': 'HMAC-SHA256',
    'key_rotation_interval': 300,
    'max_message_age': 60,
    'replay_window_size': 1000,
    'perfect_forward_secrecy': True,
    'audit_logging': True
}
security_manager = SecurityManager(config)
```

### Core Security Methods

#### initialize_security_context
```python
def initialize_security_context(self, agent_id: str, 
                               trusted_agents: List[str]) -> SecurityContext
```
**Description:** Initialize security context for spacecraft agent

**Parameters:**
- `agent_id` (str): Unique identifier for the spacecraft
- `trusted_agents` (List[str]): List of trusted agent identifiers

**Returns:**
- `SecurityContext`: Initialized security context object

**SecurityContext Object:**
```python
@dataclass
class SecurityContext:
    agent_id: str
    session_keys: Dict[str, bytes]      # Per-agent session keys
    authentication_tokens: Dict        # Authentication tokens
    encryption_state: Dict             # Encryption state information
    last_key_rotation: float           # Timestamp of last key rotation
    security_level: str                # Current security level
    active_connections: Set[str]       # Active secure connections
```

#### encrypt_message
```python
def encrypt_message(self, message: Dict, recipient: str, 
                   security_level: str = 'standard') -> EncryptedMessage
```
**Description:** Encrypt message for secure transmission

**Parameters:**
- `message` (Dict): Message payload to encrypt
- `recipient` (str): Target recipient agent identifier
- `security_level` (str): Security level ('standard', 'high', 'critical')

**Returns:**
- `EncryptedMessage`: Encrypted message with metadata

**EncryptedMessage Format:**
```python
@dataclass
class EncryptedMessage:
    encrypted_payload: bytes           # AES encrypted message content
    initialization_vector: bytes      # AES IV (96 bits for GCM)
    authentication_tag: bytes         # GCM authentication tag
    sender_id: str                    # Sender identification
    recipient_id: str                 # Recipient identification
    timestamp: float                  # Message creation timestamp
    sequence_number: int              # Message sequence number
    security_level: str               # Applied security level
    key_version: int                  # Key version used for encryption
    message_type: str                 # Message type identifier
```

#### decrypt_message
```python
def decrypt_message(self, encrypted_message: EncryptedMessage, 
                   sender: str) -> Tuple[Dict, bool]
```
**Description:** Decrypt and authenticate received message

**Parameters:**
- `encrypted_message` (EncryptedMessage): Encrypted message to decrypt
- `sender` (str): Sender agent identifier for verification

**Returns:**
- `Tuple[Dict, bool]`: (Decrypted message, Authentication success)

---

## Encryption Manager

Advanced encryption management with multiple cipher support and key derivation.

### Class Definition
```python
class AdvancedEncryptionManager:
    """
    Advanced encryption manager with multiple cipher support
    
    Supports AES-256-GCM, ChaCha20-Poly1305, and post-quantum algorithms
    with hardware acceleration where available.
    """
```

### Encryption Methods

#### generate_session_key
```python
def generate_session_key(self, agent_pair: Tuple[str, str], 
                        key_type: str = 'symmetric') -> bytes
```
**Description:** Generate cryptographically secure session key

**Parameters:**
- `agent_pair` (Tuple[str, str]): Pair of communicating agents
- `key_type` (str): Key type ('symmetric', 'ephemeral', 'long_term')

**Returns:**
- `bytes`: Generated session key (256 bits for AES)

#### derive_key_from_shared_secret
```python
def derive_key_from_shared_secret(self, shared_secret: bytes, 
                                 salt: bytes, info: str) -> bytes
```
**Description:** Derive encryption key using HKDF

**Parameters:**
- `shared_secret` (bytes): Shared secret from key exchange
- `salt` (bytes): Random salt (32 bytes recommended)
- `info` (str): Key derivation context information

**Returns:**
- `bytes`: Derived key material

**Example:**
```python
# ECDH key exchange example
shared_secret = perform_ecdh_exchange(agent_private_key, peer_public_key)
salt = os.urandom(32)
info = f"spacecraft-comm-{agent_id}-{peer_id}"

session_key = encryption_manager.derive_key_from_shared_secret(
    shared_secret, salt, info
)
```

#### encrypt_with_aead
```python
def encrypt_with_aead(self, plaintext: bytes, key: bytes, 
                     associated_data: bytes = b"") -> Tuple[bytes, bytes, bytes]
```
**Description:** Encrypt with Authenticated Encryption with Associated Data

**Parameters:**
- `plaintext` (bytes): Data to encrypt
- `key` (bytes): Encryption key (256 bits)
- `associated_data` (bytes): Additional authenticated data (AAD)

**Returns:**
- `Tuple[bytes, bytes, bytes]`: (Ciphertext, IV, Authentication tag)

#### decrypt_with_aead
```python
def decrypt_with_aead(self, ciphertext: bytes, key: bytes, 
                     iv: bytes, tag: bytes, 
                     associated_data: bytes = b"") -> bytes
```
**Description:** Decrypt and verify AEAD ciphertext

**Parameters:**
- `ciphertext` (bytes): Encrypted data
- `key` (bytes): Decryption key
- `iv` (bytes): Initialization vector
- `tag` (bytes): Authentication tag
- `associated_data` (bytes): Associated data for verification

**Returns:**
- `bytes`: Decrypted plaintext

**Raises:**
- `AuthenticationError`: If authentication tag verification fails

---

## Key Management

Comprehensive key lifecycle management with automatic rotation and secure storage.

### Class Definition
```python
class KeyManager:
    """
    Comprehensive key management system
    
    Handles key generation, distribution, rotation, and secure storage
    with support for hardware security modules (HSM).
    """
```

### Key Management Methods

#### generate_key_pair
```python
def generate_key_pair(self, algorithm: str = 'RSA-2048') -> Tuple[bytes, bytes]
```
**Description:** Generate asymmetric key pair

**Supported Algorithms:**
- `'RSA-2048'`: RSA with 2048-bit keys
- `'RSA-4096'`: RSA with 4096-bit keys
- `'ECDSA-P256'`: ECDSA with P-256 curve
- `'ECDSA-P384'`: ECDSA with P-384 curve
- `'Ed25519'`: EdDSA with Curve25519

**Returns:**
- `Tuple[bytes, bytes]`: (Private key, Public key) in PEM format

#### rotate_session_keys
```python
async def rotate_session_keys(self, affected_agents: List[str]) -> Dict[str, bytes]
```
**Description:** Rotate session keys for specified agents

**Parameters:**
- `affected_agents` (List[str]): Agents requiring key rotation

**Returns:**
- `Dict[str, bytes]`: New session keys indexed by agent ID

**Key Rotation Process:**
```python
# Automatic key rotation example
@asyncio.coroutine
async def automatic_key_rotation():
    while True:
        await asyncio.sleep(key_manager.rotation_interval)
        
        # Check which keys need rotation
        expired_keys = key_manager.check_key_expiration()
        
        if expired_keys:
            # Generate new keys
            new_keys = await key_manager.rotate_session_keys(expired_keys)
            
            # Distribute new keys securely
            await key_manager.distribute_new_keys(new_keys)
            
            # Update local key store
            key_manager.update_key_store(new_keys)
```

#### secure_key_storage
```python
def secure_key_storage(self, key_data: bytes, key_id: str, 
                      protection_level: str = 'software') -> str
```
**Description:** Securely store cryptographic keys

**Protection Levels:**
- `'software'`: Software-based key protection
- `'hardware'`: Hardware Security Module (HSM)
- `'secure_enclave'`: Trusted execution environment

**Parameters:**
- `key_data` (bytes): Key material to store
- `key_id` (str): Unique key identifier
- `protection_level` (str): Storage protection level

**Returns:**
- `str`: Key storage reference/handle

#### retrieve_key
```python
def retrieve_key(self, key_id: str, authentication_token: str) -> bytes
```
**Description:** Retrieve stored key with authentication

**Parameters:**
- `key_id` (str): Key identifier
- `authentication_token` (str): Authentication token for access

**Returns:**
- `bytes`: Retrieved key material

---

## Message Authentication

Message authentication and integrity verification systems.

### Class Definition
```python
class MessageAuthenticator:
    """
    Message authentication and integrity verification
    
    Implements HMAC, digital signatures, and message sequence verification
    for preventing replay attacks and ensuring message integrity.
    """
```

### Authentication Methods

#### generate_message_mac
```python
def generate_message_mac(self, message: bytes, key: bytes, 
                        algorithm: str = 'HMAC-SHA256') -> bytes
```
**Description:** Generate Message Authentication Code

**Supported Algorithms:**
- `'HMAC-SHA256'`: HMAC with SHA-256 (recommended)
- `'HMAC-SHA384'`: HMAC with SHA-384
- `'HMAC-SHA512'`: HMAC with SHA-512
- `'BLAKE2b'`: BLAKE2b keyed hash

**Parameters:**
- `message` (bytes): Message to authenticate
- `key` (bytes): Authentication key
- `algorithm` (str): MAC algorithm

**Returns:**
- `bytes`: Message authentication code

#### verify_message_mac
```python
def verify_message_mac(self, message: bytes, mac: bytes, key: bytes) -> bool
```
**Description:** Verify message authentication code

**Parameters:**
- `message` (bytes): Original message
- `mac` (bytes): Message authentication code
- `key` (bytes): Authentication key

**Returns:**
- `bool`: True if MAC is valid, False otherwise

#### sign_message
```python
def sign_message(self, message: bytes, private_key: bytes, 
                algorithm: str = 'RSA-PSS-SHA256') -> bytes
```
**Description:** Generate digital signature for message

**Signature Algorithms:**
- `'RSA-PSS-SHA256'`: RSA-PSS with SHA-256
- `'ECDSA-SHA256'`: ECDSA with SHA-256
- `'Ed25519'`: EdDSA with Curve25519

**Parameters:**
- `message` (bytes): Message to sign
- `private_key` (bytes): Signer's private key
- `algorithm` (str): Signature algorithm

**Returns:**
- `bytes`: Digital signature

#### verify_signature
```python
def verify_signature(self, message: bytes, signature: bytes, 
                    public_key: bytes, algorithm: str) -> bool
```
**Description:** Verify digital signature

**Parameters:**
- `message` (bytes): Original signed message
- `signature` (bytes): Digital signature
- `public_key` (bytes): Signer's public key
- `algorithm` (str): Signature algorithm used

**Returns:**
- `bool`: True if signature is valid, False otherwise

### Replay Attack Prevention

#### update_sequence_window
```python
def update_sequence_window(self, sender: str, sequence_number: int) -> bool
```
**Description:** Update message sequence window for replay protection

**Parameters:**
- `sender` (str): Message sender identifier
- `sequence_number` (int): Message sequence number

**Returns:**
- `bool`: True if sequence number is valid (not a replay)

#### check_message_freshness
```python
def check_message_freshness(self, timestamp: float, max_age: int = 60) -> bool
```
**Description:** Check if message is within acceptable age limit

**Parameters:**
- `timestamp` (float): Message timestamp
- `max_age` (int): Maximum acceptable message age (seconds)

**Returns:**
- `bool`: True if message is fresh, False if too old

---

## Communication Security

Secure communication protocols and session management.

### Class Definition
```python
class SecureCommunicationProtocol:
    """
    Secure communication protocol implementation
    
    Manages secure channels, session establishment, and encrypted
    message exchange between spacecraft agents.
    """
```

### Session Management

#### establish_secure_session
```python
async def establish_secure_session(self, peer_agent: str, 
                                 security_level: str = 'high') -> SecureSession
```
**Description:** Establish secure communication session with peer

**Security Levels:**
- `'standard'`: AES-256, RSA-2048, basic authentication
- `'high'`: AES-256-GCM, RSA-4096, certificate-based authentication  
- `'critical'`: Post-quantum algorithms, hardware key storage

**Parameters:**
- `peer_agent` (str): Peer agent identifier
- `security_level` (str): Required security level

**Returns:**
- `SecureSession`: Established secure session object

**SecureSession Object:**
```python
@dataclass
class SecureSession:
    session_id: str                    # Unique session identifier
    peer_agent: str                    # Peer agent ID
    session_key: bytes                 # Symmetric session key
    authentication_key: bytes         # Message authentication key
    established_time: float            # Session establishment time
    last_activity: float               # Last communication time
    security_level: str                # Applied security level
    perfect_forward_secrecy: bool      # PFS enabled status
    post_quantum_secure: bool         # Post-quantum security status
```

#### send_secure_message
```python
async def send_secure_message(self, session: SecureSession, 
                            message: Dict, priority: int = 1) -> bool
```
**Description:** Send encrypted message through secure session

**Parameters:**
- `session` (SecureSession): Active secure session
- `message` (Dict): Message payload to send
- `priority` (int): Message priority (1=low, 5=critical)

**Returns:**
- `bool`: True if message sent successfully

#### receive_secure_message
```python
async def receive_secure_message(self, session: SecureSession, 
                               timeout: float = 10.0) -> Optional[Dict]
```
**Description:** Receive and decrypt message from secure session

**Parameters:**
- `session` (SecureSession): Active secure session
- `timeout` (float): Receive timeout in seconds

**Returns:**
- `Optional[Dict]`: Decrypted message or None if timeout

### Certificate Management

#### load_certificate_chain
```python
def load_certificate_chain(self, cert_paths: List[str]) -> CertificateChain
```
**Description:** Load and validate certificate chain

**Parameters:**
- `cert_paths` (List[str]): Paths to certificate files (PEM format)

**Returns:**
- `CertificateChain`: Loaded and validated certificate chain

#### validate_peer_certificate
```python
def validate_peer_certificate(self, peer_cert: bytes, 
                             trusted_cas: List[bytes]) -> CertificateValidation
```
**Description:** Validate peer's certificate against trusted CAs

**Parameters:**
- `peer_cert` (bytes): Peer's certificate in PEM format
- `trusted_cas` (List[bytes]): Trusted Certificate Authority certificates

**Returns:**
- `CertificateValidation`: Certificate validation result

```python
@dataclass
class CertificateValidation:
    is_valid: bool                     # Overall validation result
    trust_chain_valid: bool            # Trust chain verification
    not_expired: bool                  # Certificate not expired
    not_revoked: bool                  # Certificate not revoked
    hostname_match: bool               # Hostname verification
    key_usage_valid: bool              # Key usage constraints
    validation_errors: List[str]       # Validation error messages
```

---

## Code Examples

### Basic Secure Communication
```python
#!/usr/bin/env python3
"""Basic secure communication example"""

import asyncio
from src.security.security_manager import SecurityManager
from src.security.communication_protocol import SecureCommunicationProtocol

async def secure_communication_demo():
    # Initialize security manager
    security_config = {
        'encryption_algorithm': 'AES-256-GCM',
        'key_exchange_algorithm': 'RSA-2048',
        'authentication_algorithm': 'HMAC-SHA256',
        'key_rotation_interval': 300,
        'perfect_forward_secrecy': True
    }
    
    security_manager = SecurityManager(security_config)
    comm_protocol = SecureCommunicationProtocol(security_manager)
    
    # Initialize security context for Agent A
    context_a = security_manager.initialize_security_context(
        'chaser-001', ['target-station', 'observer-001']
    )
    
    # Establish secure session between agents
    session = await comm_protocol.establish_secure_session(
        'target-station', security_level='high'
    )
    
    print(f"Secure session established: {session.session_id}")
    print(f"Security level: {session.security_level}")
    print(f"Post-quantum secure: {session.post_quantum_secure}")
    
    # Send encrypted message
    message = {
        'type': 'docking_request',
        'timestamp': time.time(),
        'position': [10.0, 5.0, 0.0],
        'velocity': [-0.1, -0.05, 0.0],
        'approach_vector': [1.0, 0.0, 0.0]
    }
    
    success = await comm_protocol.send_secure_message(session, message, priority=3)
    if success:
        print("Docking request sent securely")
    
    # Receive response
    response = await comm_protocol.receive_secure_message(session, timeout=5.0)
    if response:
        print(f"Received response: {response['type']}")
        print(f"Docking approved: {response.get('approved', False)}")

if __name__ == "__main__":
    asyncio.run(secure_communication_demo())
```

### Advanced Key Management
```python
#!/usr/bin/env python3
"""Advanced key management example"""

import asyncio
from src.security.key_manager import KeyManager
from src.security.encryption_manager import AdvancedEncryptionManager

class SpacecraftKeyManager:
    def __init__(self, agent_id, hsm_available=False):
        self.agent_id = agent_id
        self.key_manager = KeyManager({
            'key_rotation_interval': 300,
            'hardware_security_module': hsm_available,
            'key_derivation_iterations': 100000
        })
        self.encryption_manager = AdvancedEncryptionManager()
        self.active_sessions = {}
    
    async def initialize_agent_keys(self):
        """Initialize all keys for spacecraft agent"""
        
        # Generate long-term identity key pair
        private_key, public_key = self.key_manager.generate_key_pair('RSA-4096')
        
        # Store keys securely
        protection_level = 'hardware' if self.key_manager.hsm_available else 'software'
        
        private_key_ref = self.key_manager.secure_key_storage(
            private_key, f"{self.agent_id}-identity-private", protection_level
        )
        
        public_key_ref = self.key_manager.secure_key_storage(
            public_key, f"{self.agent_id}-identity-public", protection_level
        )
        
        print(f"Identity keys generated for {self.agent_id}")
        print(f"Private key reference: {private_key_ref}")
        print(f"Public key reference: {public_key_ref}")
        
        return private_key_ref, public_key_ref
    
    async def establish_ephemeral_session(self, peer_agent):
        """Establish session with ephemeral keys for perfect forward secrecy"""
        
        # Generate ephemeral ECDH key pair
        ephemeral_private, ephemeral_public = self.key_manager.generate_key_pair('ECDSA-P384')
        
        # Simulate key exchange (in real implementation, this would use network)
        peer_ephemeral_public = self.simulate_key_exchange(peer_agent, ephemeral_public)
        
        # Compute shared secret using ECDH
        shared_secret = self.compute_ecdh_shared_secret(ephemeral_private, peer_ephemeral_public)
        
        # Derive session keys using HKDF
        salt = os.urandom(32)
        info = f"spacecraft-session-{self.agent_id}-{peer_agent}"
        
        session_key = self.encryption_manager.derive_key_from_shared_secret(
            shared_secret, salt, info
        )
        
        # Store session information
        session_info = {
            'peer': peer_agent,
            'session_key': session_key,
            'established': time.time(),
            'ephemeral_keys': (ephemeral_private, ephemeral_public)
        }
        
        self.active_sessions[peer_agent] = session_info
        print(f"Ephemeral session established with {peer_agent}")
        
        return session_key
    
    async def rotate_all_session_keys(self):
        """Rotate all active session keys"""
        
        print(f"Starting key rotation for {len(self.active_sessions)} sessions")
        
        for peer_agent in self.active_sessions.keys():
            # Generate new session key
            new_session_key = await self.establish_ephemeral_session(peer_agent)
            
            # Securely destroy old key
            self.secure_key_destruction(self.active_sessions[peer_agent]['session_key'])
            
            # Update session information
            self.active_sessions[peer_agent]['session_key'] = new_session_key
            self.active_sessions[peer_agent]['last_rotation'] = time.time()
            
            print(f"Rotated session key for {peer_agent}")
    
    def secure_key_destruction(self, key_material):
        """Securely destroy key material from memory"""
        # Overwrite memory with random data multiple times
        if isinstance(key_material, bytes):
            # This is a simplified example - real implementation would use
            # memory-mapped operations and multiple overwrite passes
            key_array = bytearray(key_material)
            for _ in range(3):
                for i in range(len(key_array)):
                    key_array[i] = os.urandom(1)[0]
            del key_array

# Usage example
async def main():
    # Initialize key managers for multiple spacecraft
    chaser_km = SpacecraftKeyManager('chaser-001', hsm_available=True)
    target_km = SpacecraftKeyManager('target-station', hsm_available=True)
    
    # Initialize agent keys
    await chaser_km.initialize_agent_keys()
    await target_km.initialize_agent_keys()
    
    # Establish ephemeral session
    session_key = await chaser_km.establish_ephemeral_session('target-station')
    
    # Simulate regular key rotation
    await asyncio.sleep(300)  # Wait 5 minutes
    await chaser_km.rotate_all_session_keys()

if __name__ == "__main__":
    asyncio.run(main())
```

### Message Authentication and Integrity
```python
#!/usr/bin/env python3
"""Message authentication and integrity verification"""

import time
import json
from src.security.message_authenticator import MessageAuthenticator
from src.security.security_manager import SecurityManager

class SecureSpacecraftMessaging:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.authenticator = MessageAuthenticator()
        self.sequence_numbers = {}  # Per-peer sequence tracking
        self.message_cache = {}     # Recent message cache for replay detection
    
    def prepare_secure_message(self, message_data, recipient, shared_key):
        """Prepare message with authentication and integrity protection"""
        
        # Add message metadata
        secure_message = {
            'sender': self.agent_id,
            'recipient': recipient,
            'timestamp': time.time(),
            'sequence': self.get_next_sequence_number(recipient),
            'payload': message_data
        }
        
        # Serialize message
        message_bytes = json.dumps(secure_message, sort_keys=True).encode('utf-8')
        
        # Generate MAC for integrity protection
        mac = self.authenticator.generate_message_mac(
            message_bytes, shared_key, 'HMAC-SHA256'
        )
        
        # Create final authenticated message
        authenticated_message = {
            'message': secure_message,
            'mac': mac.hex(),
            'mac_algorithm': 'HMAC-SHA256'
        }
        
        return authenticated_message
    
    def verify_secure_message(self, authenticated_message, sender, shared_key):
        """Verify message authentication and integrity"""
        
        try:
            # Extract message components
            message = authenticated_message['message']
            received_mac = bytes.fromhex(authenticated_message['mac'])
            
            # Verify sender
            if message['sender'] != sender:
                raise SecurityError(f"Sender mismatch: expected {sender}, got {message['sender']}")
            
            # Verify recipient
            if message['recipient'] != self.agent_id:
                raise SecurityError(f"Message not intended for {self.agent_id}")
            
            # Check message freshness (prevent replay attacks)
            if not self.authenticator.check_message_freshness(message['timestamp'], max_age=60):
                raise SecurityError("Message too old - potential replay attack")
            
            # Verify message sequence (prevent replay attacks)
            if not self.authenticator.update_sequence_window(sender, message['sequence']):
                raise SecurityError(f"Invalid sequence number from {sender}")
            
            # Verify MAC
            message_bytes = json.dumps(message, sort_keys=True).encode('utf-8')
            if not self.authenticator.verify_message_mac(message_bytes, received_mac, shared_key):
                raise SecurityError("Message authentication failed")
            
            return message['payload']
            
        except Exception as e:
            print(f"Message verification failed: {e}")
            return None
    
    def get_next_sequence_number(self, recipient):
        """Get next sequence number for recipient"""
        if recipient not in self.sequence_numbers:
            self.sequence_numbers[recipient] = 0
        self.sequence_numbers[recipient] += 1
        return self.sequence_numbers[recipient]

class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass

# Usage example
def secure_messaging_example():
    # Initialize secure messaging for two spacecraft
    chaser = SecureSpacecraftMessaging('chaser-001')
    target = SecureSpacecraftMessaging('target-station')
    
    # Simulate shared key (in practice, this would come from key exchange)
    shared_key = os.urandom(32)  # 256-bit key
    
    # Chaser sends docking request
    docking_request = {
        'type': 'docking_request',
        'approach_vector': [1.0, 0.0, 0.0],
        'velocity': [-0.1, -0.05, 0.0],
        'estimated_contact_time': time.time() + 300
    }
    
    # Prepare secure message
    secure_message = chaser.prepare_secure_message(
        docking_request, 'target-station', shared_key
    )
    
    print("Secure message prepared:")
    print(f"  Sender: {secure_message['message']['sender']}")
    print(f"  Sequence: {secure_message['message']['sequence']}")
    print(f"  MAC: {secure_message['mac'][:16]}...")
    
    # Target verifies and processes message
    verified_payload = target.verify_secure_message(
        secure_message, 'chaser-001', shared_key
    )
    
    if verified_payload:
        print("Message verified successfully!")
        print(f"  Message type: {verified_payload['type']}")
        print(f"  Approach vector: {verified_payload['approach_vector']}")
        
        # Send response
        response = {
            'type': 'docking_response',
            'approved': True,
            'docking_port': 'port-alpha',
            'final_approach_time': verified_payload['estimated_contact_time']
        }
        
        secure_response = target.prepare_secure_message(
            response, 'chaser-001', shared_key
        )
        
        # Chaser verifies response
        verified_response = chaser.verify_secure_message(
            secure_response, 'target-station', shared_key
        )
        
        if verified_response:
            print("Response verified successfully!")
            print(f"  Docking approved: {verified_response['approved']}")
            print(f"  Assigned port: {verified_response['docking_port']}")
        
    else:
        print("Message verification failed!")

if __name__ == "__main__":
    secure_messaging_example()
```

### Certificate-Based Authentication
```python
#!/usr/bin/env python3
"""Certificate-based authentication system"""

import os
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from src.security.certificate_manager import CertificateManager

class SpacecraftCertificateAuthority:
    """Simplified Certificate Authority for spacecraft fleet"""
    
    def __init__(self):
        self.ca_private_key = None
        self.ca_certificate = None
        self.cert_manager = CertificateManager()
        self.issued_certificates = {}
    
    def generate_ca_certificate(self):
        """Generate root CA certificate"""
        
        # Generate CA private key
        self.ca_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        
        # Create CA certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(x509.oid.NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(x509.oid.NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(x509.oid.NameOID.LOCALITY_NAME, "JPL"),
            x509.NameAttribute(x509.oid.NameOID.ORGANIZATION_NAME, "Spacecraft Fleet"),
            x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, "Spacecraft CA"),
        ])
        
        self.ca_certificate = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            self.ca_private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=3650)  # 10 years
        ).add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        ).add_extension(
            x509.KeyUsage(
                key_cert_sign=True,
                crl_sign=True,
                digital_signature=False,
                key_encipherment=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        ).sign(self.ca_private_key, hashes.SHA256())
    
    def issue_spacecraft_certificate(self, agent_id, agent_public_key):
        """Issue certificate for spacecraft agent"""
        
        subject = x509.Name([
            x509.NameAttribute(x509.oid.NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(x509.oid.NameOID.ORGANIZATION_NAME, "Spacecraft Fleet"),
            x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, agent_id),
        ])
        
        certificate = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            self.ca_certificate.issuer
        ).public_key(
            agent_public_key
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)  # 1 year
        ).add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        ).add_extension(
            x509.KeyUsage(
                key_cert_sign=False,
                crl_sign=False,
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=True,
                key_agreement=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        ).add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
            ]),
            critical=True,
        ).sign(self.ca_private_key, hashes.SHA256())
        
        self.issued_certificates[agent_id] = certificate
        return certificate

# Certificate-based authentication usage
def certificate_authentication_example():
    # Create Certificate Authority
    ca = SpacecraftCertificateAuthority()
    ca.generate_ca_certificate()
    
    print("Certificate Authority established")
    
    # Generate key pairs for spacecraft
    chaser_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    
    target_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )
    
    # Issue certificates
    chaser_cert = ca.issue_spacecraft_certificate(
        'chaser-001', chaser_private_key.public_key()
    )
    
    target_cert = ca.issue_spacecraft_certificate(
        'target-station', target_private_key.public_key()
    )
    
    print("Certificates issued for spacecraft")
    print(f"Chaser certificate serial: {chaser_cert.serial_number}")
    print(f"Target certificate serial: {target_cert.serial_number}")
    
    # Verify certificates
    cert_manager = CertificateManager()
    
    chaser_validation = cert_manager.validate_peer_certificate(
        chaser_cert.public_bytes(serialization.Encoding.PEM),
        [ca.ca_certificate.public_bytes(serialization.Encoding.PEM)]
    )
    
    if chaser_validation.is_valid:
        print("Chaser certificate validation successful")
    else:
        print(f"Chaser certificate validation failed: {chaser_validation.validation_errors}")

if __name__ == "__main__":
    certificate_authentication_example()
```

---

## Error Handling and Security Events

### Security Exception Hierarchy
```python
class SecurityException(Exception):
    """Base class for security-related exceptions"""
    pass

class EncryptionError(SecurityException):
    """Encryption/decryption operation failed"""
    pass

class AuthenticationError(SecurityException):
    """Message or entity authentication failed"""
    pass

class KeyManagementError(SecurityException):
    """Key generation, storage, or retrieval failed"""
    pass

class CertificateError(SecurityException):
    """Certificate validation or management failed"""
    pass

class ReplayAttackError(SecurityException):
    """Potential replay attack detected"""
    pass
```

### Security Event Logging
```python
class SecurityAuditLogger:
    """Security event logging and monitoring"""
    
    def log_authentication_failure(self, agent_id, peer_id, reason):
        """Log authentication failure event"""
        event = {
            'timestamp': time.time(),
            'event_type': 'AUTHENTICATION_FAILURE',
            'agent_id': agent_id,
            'peer_id': peer_id,
            'reason': reason,
            'severity': 'HIGH'
        }
        self.write_audit_log(event)
    
    def log_key_rotation(self, agent_id, key_type, success):
        """Log key rotation event"""
        event = {
            'timestamp': time.time(),
            'event_type': 'KEY_ROTATION',
            'agent_id': agent_id,
            'key_type': key_type,
            'success': success,
            'severity': 'MEDIUM'
        }
        self.write_audit_log(event)
    
    def log_security_violation(self, agent_id, violation_type, details):
        """Log security policy violation"""
        event = {
            'timestamp': time.time(),
            'event_type': 'SECURITY_VIOLATION',
            'agent_id': agent_id,
            'violation_type': violation_type,
            'details': details,
            'severity': 'CRITICAL'
        }
        self.write_audit_log(event)
```

---

*For advanced security configurations and enterprise deployment, see the [user manual](../user_manual.md) and [deployment guide](../deployment.md).*