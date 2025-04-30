"""
Code signing module for Saplings tool factory.

This module provides code signing and verification capabilities for
dynamically generated tools.
"""

import base64
import hashlib
import hmac
import logging
import os
from typing import Dict, Optional, Tuple

from saplings.tool_factory.config import SigningLevel, ToolFactoryConfig

logger = logging.getLogger(__name__)


class CodeSigner:
    """
    Code signer for dynamically generated tools.
    
    This class provides functionality for signing code to ensure its integrity
    and authenticity.
    """
    
    def __init__(self, config: Optional[ToolFactoryConfig] = None):
        """
        Initialize the code signer.
        
        Args:
            config: Configuration for the code signer
        """
        self.config = config or ToolFactoryConfig()
        self._private_key = None
        
        # Load the private key if advanced signing is enabled
        if self.config.signing_level == SigningLevel.ADVANCED:
            if not self.config.signing_key_path:
                raise ValueError("Signing key path is required for advanced signing")
            
            try:
                self._load_private_key()
            except Exception as e:
                raise ValueError(f"Failed to load signing key: {e}")
    
    def _load_private_key(self) -> None:
        """
        Load the private key for advanced signing.
        
        Raises:
            ValueError: If the key file doesn't exist or is invalid
        """
        try:
            # Check if cryptography is available
            import cryptography.hazmat.primitives.asymmetric.rsa as rsa
            from cryptography.hazmat.primitives import serialization
        except ImportError:
            raise ImportError(
                "Cryptography package is not installed. "
                "Install it with: pip install cryptography"
            )
        
        # Check if the key file exists
        if not os.path.exists(self.config.signing_key_path):
            raise ValueError(f"Signing key file not found: {self.config.signing_key_path}")
        
        # Load the private key
        try:
            with open(self.config.signing_key_path, "rb") as key_file:
                self._private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,
                )
        except Exception as e:
            raise ValueError(f"Invalid signing key: {e}")
    
    def sign(self, code: str) -> Dict[str, str]:
        """
        Sign code.
        
        Args:
            code: Code to sign
            
        Returns:
            Dict[str, str]: Signature information
        """
        # If signing is disabled, return an empty signature
        if self.config.signing_level == SigningLevel.NONE:
            return {"signature_type": "none"}
        
        # Calculate the code hash
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        
        # Basic signing (hash only)
        if self.config.signing_level == SigningLevel.BASIC:
            return {
                "signature_type": "basic",
                "hash_algorithm": "sha256",
                "code_hash": code_hash,
            }
        
        # Advanced signing (cryptographic signature)
        elif self.config.signing_level == SigningLevel.ADVANCED:
            try:
                # Check if cryptography is available
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import padding
            except ImportError:
                raise ImportError(
                    "Cryptography package is not installed. "
                    "Install it with: pip install cryptography"
                )
            
            # Sign the code hash
            signature = self._private_key.sign(
                code_hash.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            
            # Encode the signature as base64
            signature_b64 = base64.b64encode(signature).decode()
            
            return {
                "signature_type": "advanced",
                "hash_algorithm": "sha256",
                "code_hash": code_hash,
                "signature": signature_b64,
            }
        
        else:
            raise ValueError(f"Unsupported signing level: {self.config.signing_level}")


class SignatureVerifier:
    """
    Signature verifier for dynamically generated tools.
    
    This class provides functionality for verifying code signatures to ensure
    code integrity and authenticity.
    """
    
    def __init__(self, config: Optional[ToolFactoryConfig] = None):
        """
        Initialize the signature verifier.
        
        Args:
            config: Configuration for the signature verifier
        """
        self.config = config or ToolFactoryConfig()
        self._public_key = None
        
        # Load the public key if advanced signing is enabled
        if self.config.signing_level == SigningLevel.ADVANCED:
            if not self.config.signing_key_path:
                raise ValueError("Signing key path is required for advanced signing")
            
            try:
                self._load_public_key()
            except Exception as e:
                raise ValueError(f"Failed to load public key: {e}")
    
    def _load_public_key(self) -> None:
        """
        Load the public key for advanced signature verification.
        
        Raises:
            ValueError: If the key file doesn't exist or is invalid
        """
        try:
            # Check if cryptography is available
            import cryptography.hazmat.primitives.asymmetric.rsa as rsa
            from cryptography.hazmat.primitives import serialization
        except ImportError:
            raise ImportError(
                "Cryptography package is not installed. "
                "Install it with: pip install cryptography"
            )
        
        # Check if the key file exists
        if not os.path.exists(self.config.signing_key_path):
            raise ValueError(f"Signing key file not found: {self.config.signing_key_path}")
        
        # Load the public key
        try:
            with open(self.config.signing_key_path, "rb") as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,
                )
                self._public_key = private_key.public_key()
        except Exception as e:
            raise ValueError(f"Invalid signing key: {e}")
    
    def verify(self, code: str, signature_info: Dict[str, str]) -> bool:
        """
        Verify a code signature.
        
        Args:
            code: Code to verify
            signature_info: Signature information
            
        Returns:
            bool: True if the signature is valid, False otherwise
        """
        # If no signature is provided, return False
        if not signature_info:
            return False
        
        # Get the signature type
        signature_type = signature_info.get("signature_type", "none")
        
        # No signature
        if signature_type == "none":
            # Only accept if signing is disabled
            return self.config.signing_level == SigningLevel.NONE
        
        # Basic signature (hash verification)
        elif signature_type == "basic":
            # Calculate the code hash
            code_hash = hashlib.sha256(code.encode()).hexdigest()
            
            # Compare with the provided hash
            return code_hash == signature_info.get("code_hash", "")
        
        # Advanced signature (cryptographic verification)
        elif signature_type == "advanced":
            try:
                # Check if cryptography is available
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import padding
                from cryptography.exceptions import InvalidSignature
            except ImportError:
                raise ImportError(
                    "Cryptography package is not installed. "
                    "Install it with: pip install cryptography"
                )
            
            # Get the code hash and signature
            code_hash = signature_info.get("code_hash", "")
            signature_b64 = signature_info.get("signature", "")
            
            if not code_hash or not signature_b64:
                return False
            
            try:
                # Decode the signature
                signature = base64.b64decode(signature_b64)
                
                # Verify the signature
                self._public_key.verify(
                    signature,
                    code_hash.encode(),
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH,
                    ),
                    hashes.SHA256(),
                )
                
                return True
            except InvalidSignature:
                return False
            except Exception as e:
                logger.error(f"Error verifying signature: {e}")
                return False
        
        else:
            logger.warning(f"Unsupported signature type: {signature_type}")
            return False


def generate_key_pair(output_dir: str, key_name: str = "signing_key") -> Tuple[str, str]:
    """
    Generate a key pair for code signing.
    
    Args:
        output_dir: Directory to save the keys
        key_name: Base name for the key files
        
    Returns:
        Tuple[str, str]: Paths to the private and public key files
    """
    try:
        # Check if cryptography is available
        import cryptography.hazmat.primitives.asymmetric.rsa as rsa
        from cryptography.hazmat.primitives import serialization
    except ImportError:
        raise ImportError(
            "Cryptography package is not installed. "
            "Install it with: pip install cryptography"
        )
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a key pair
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    # Get the public key
    public_key = private_key.public_key()
    
    # Serialize the private key
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    
    # Serialize the public key
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    
    # Save the keys
    private_key_path = os.path.join(output_dir, f"{key_name}.pem")
    public_key_path = os.path.join(output_dir, f"{key_name}.pub")
    
    with open(private_key_path, "wb") as f:
        f.write(private_pem)
    
    with open(public_key_path, "wb") as f:
        f.write(public_pem)
    
    return private_key_path, public_key_path
