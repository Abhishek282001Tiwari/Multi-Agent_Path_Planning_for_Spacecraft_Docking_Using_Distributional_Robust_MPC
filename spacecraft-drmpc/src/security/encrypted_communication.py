# src/security/encrypted_communication.py
import os
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import base64

from ..fault_tolerance.actuator_fdir import FaultTolerantSpacecraft

class SecureCommunicationSystem:
    """End-to-end encrypted communication for spacecraft"""
    
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.private_key = None
        self.public_key = None
        self.session_keys = {}  # Agent-specific session keys
        self.key_rotation_interval = 3600  # 1 hour
        
        # Initialize cryptographic keys
        self.generate_key_pair()
        
    def generate_key_pair(self):
        """Generate RSA key pair for asymmetric encryption"""
        
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        self.public_key = self.private_key.public_key()
        
    def get_public_key_bytes(self):
        """Export public key for sharing"""
        
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def encrypt_message(self, message: bytes, recipient_public_key) -> bytes:
        """Encrypt message for secure transmission"""
        
        # Generate symmetric session key
        session_key = Fernet.generate_key()
        fernet = Fernet(session_key)
        
        # Encrypt message with session key
        encrypted_message = fernet.encrypt(message)
        
        # Encrypt session key with recipient's public key
        encrypted_session_key = recipient_public_key.encrypt(
            session_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Combine encrypted session key and message
        encrypted_package = encrypted_session_key + b'||' + encrypted_message
        
        return encrypted_package
    
    def decrypt_message(self, encrypted_package: bytes) -> bytes:
        """Decrypt received message"""
        
        # Split package
        encrypted_session_key, encrypted_message = encrypted_package.split(b'||', 1)
        
        # Decrypt session key with private key
        session_key = self.private_key.decrypt(
            encrypted_session_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt message with session key
        fernet = Fernet(session_key)
        decrypted_message = fernet.decrypt(encrypted_message)
        
        return decrypted_message
    
    def create_secure_channel(self, agent_id, public_key):
        """Establish secure communication channel with another agent"""
        
        # Store public key
        self.session_keys[agent_id] = {
            'public_key': serialization.load_pem_public_key(public_key),
            'established_time': time.time()
        }
    
    def rotate_keys(self):
        """Rotate cryptographic keys for enhanced security"""
        
        self.generate_key_pair()
        self.session_keys.clear()
        
    def sign_message(self, message: bytes) -> bytes:
        """Sign message for authentication"""
        
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature
    
    def verify_signature(self, message: bytes, signature: bytes, public_key) -> bool:
        """Verify message signature"""
        
        try:
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except:
            return False

# Secure spacecraft integration
class SecureSpacecraftAgent(FaultTolerantSpacecraft):
    def __init__(self, agent_id):
        super().__init__(agent_id)
        self.security_system = SecureCommunicationSystem(agent_id)
        self.trusted_agents = set()
        
    async def secure_broadcast(self, message: dict):
        """Broadcast encrypted message to all trusted agents"""
        
        # Serialize message
        message_bytes = str(message).encode()
        
        # Sign message
        signature = self.security_system.sign_message(message_bytes)
        
        # Create secure package
        secure_package = {
            'sender': self.agent_id,
            'message': message_bytes,
            'signature': signature,
            'timestamp': time.time()
        }
        
        # Broadcast to all trusted agents
        for agent_id in self.trusted_agents:
            if agent_id != self.agent_id:
                await self.send_secure_message(secure_package, agent_id)
    
    async def send_secure_message(self, package: dict, recipient_id: str):
        """Send encrypted message to specific recipient"""
        
        # Get recipient's public key
        recipient_key = self.security_system.session_keys.get(recipient_id)
        if not recipient_key:
            raise ValueError(f"No secure channel with {recipient_id}")
        
        # Encrypt message
        encrypted_package = self.security_system.encrypt_message(
            package['message'],
            recipient_key['public_key']
        )
        
        # Send via ROS2
        await self.publish_encrypted_message(recipient_id, encrypted_package)