from __future__ import annotations

"""
Code signing module for Saplings tool factory.

This module provides code signing and verification capabilities for
dynamically generated tools.
"""


import base64
import hashlib
import logging
import os

from saplings.tool_factory._internal.factory.config import SigningLevel, ToolFactoryConfig

logger = logging.getLogger(__name__)


class CodeSigner:
    """
    Code signer for dynamically generated tools.

    This class provides functionality for signing code to ensure its integrity
    and authenticity.
    """

    def __init__(self, config: ToolFactoryConfig | None = None) -> None:
        """
        Initialize the code signer.

        Args:
        ----
            config: Configuration for the code signer

        """
        self.config = config or ToolFactoryConfig()
        self._private_key = None

        # Load the private key if advanced signing is enabled
        if self.config.signing_level == SigningLevel.ADVANCED:
            if not self.config.signing_key_path:
                msg = "Signing key path is required for advanced signing"
                raise ValueError(msg)

            try:
                self._load_private_key()
            except Exception as e:
                msg = f"Failed to load signing key: {e}"
                raise ValueError(msg)

    def _load_private_key(self):
        """
        Load the private key for advanced signing.

        Raises
        ------
            ValueError: If the key file doesn't exist or is invalid

        """
        try:
            # Check if cryptography is available
            from cryptography.hazmat.primitives import serialization
        except ImportError:
            msg = "Cryptography package is not installed. Install it with: pip install cryptography"
            raise ImportError(msg)

        # Check if the key file exists
        if not self.config.signing_key_path or not os.path.exists(self.config.signing_key_path):
            msg = f"Signing key file not found: {self.config.signing_key_path}"
            raise ValueError(msg)

        # Load the private key
        try:
            with open(self.config.signing_key_path, "rb") as key_file:
                self._private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,
                )
        except Exception as e:
            msg = f"Invalid signing key: {e}"
            raise ValueError(msg)

    def sign(self, code: str) -> dict[str, str]:
        """
        Sign code.

        Args:
        ----
            code: Code to sign

        Returns:
        -------
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
        if self.config.signing_level == SigningLevel.ADVANCED:
            try:
                # Check if cryptography is available
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import padding
            except ImportError:
                msg = (
                    "Cryptography package is not installed. "
                    "Install it with: pip install cryptography"
                )
                raise ImportError(msg)

            # Sign the code hash
            # Check if the private key is an RSA key
            from cryptography.hazmat.primitives.asymmetric import ec, rsa

            if self._private_key is None:
                msg = "Private key is not loaded"
                raise ValueError(msg)

            if isinstance(self._private_key, rsa.RSAPrivateKey):
                signature = self._private_key.sign(
                    code_hash.encode(),
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH,
                    ),
                    hashes.SHA256(),
                )
            elif isinstance(self._private_key, ec.EllipticCurvePrivateKey):
                # For EC keys
                signature = self._private_key.sign(
                    code_hash.encode(),
                    ec.ECDSA(hashes.SHA256()),
                )
            else:
                msg = f"Unsupported key type: {type(self._private_key)}"
                raise ValueError(msg)

            # Encode the signature as base64
            signature_b64 = base64.b64encode(signature).decode()

            return {
                "signature_type": "advanced",
                "hash_algorithm": "sha256",
                "code_hash": code_hash,
                "signature": signature_b64,
            }

        msg = f"Unsupported signing level: {self.config.signing_level}"
        raise ValueError(msg)


class SignatureVerifier:
    """
    Signature verifier for dynamically generated tools.

    This class provides functionality for verifying code signatures to ensure
    code integrity and authenticity.
    """

    def __init__(self, config: ToolFactoryConfig | None = None) -> None:
        """
        Initialize the signature verifier.

        Args:
        ----
            config: Configuration for the signature verifier

        """
        self.config = config or ToolFactoryConfig()
        self._public_key = None

        # Load the public key if advanced signing is enabled
        if self.config.signing_level == SigningLevel.ADVANCED:
            if not self.config.signing_key_path:
                msg = "Signing key path is required for advanced signing"
                raise ValueError(msg)

            try:
                self._load_public_key()
            except Exception as e:
                msg = f"Failed to load public key: {e}"
                raise ValueError(msg)

    def _load_public_key(self):
        """
        Load the public key for advanced signature verification.

        Raises
        ------
            ValueError: If the key file doesn't exist or is invalid

        """
        try:
            # Check if cryptography is available
            from cryptography.hazmat.primitives import serialization
        except ImportError:
            msg = "Cryptography package is not installed. Install it with: pip install cryptography"
            raise ImportError(msg)

        # Check if the key file exists
        if not self.config.signing_key_path or not os.path.exists(self.config.signing_key_path):
            msg = f"Signing key file not found: {self.config.signing_key_path}"
            raise ValueError(msg)

        # Load the public key
        try:
            with open(self.config.signing_key_path, "rb") as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,
                )
                self._public_key = private_key.public_key()
        except Exception as e:
            msg = f"Invalid signing key: {e}"
            raise ValueError(msg)

    def verify(self, code: str, signature_info: dict[str, str]) -> bool:
        """
        Verify a code signature.

        Args:
        ----
            code: Code to verify
            signature_info: Signature information

        Returns:
        -------
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
        if signature_type == "basic":
            # Calculate the code hash
            code_hash = hashlib.sha256(code.encode()).hexdigest()

            # Compare with the provided hash
            return code_hash == signature_info.get("code_hash", "")

        # Advanced signature (cryptographic verification)
        if signature_type == "advanced":
            try:
                # Check if cryptography is available
                from cryptography.exceptions import InvalidSignature
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import padding
            except ImportError:
                msg = (
                    "Cryptography package is not installed. "
                    "Install it with: pip install cryptography"
                )
                raise ImportError(msg)

            # Get the code hash and signature
            code_hash = signature_info.get("code_hash", "")
            signature_b64 = signature_info.get("signature", "")

            if not code_hash or not signature_b64:
                return False

            try:
                # Decode the signature
                signature = base64.b64decode(signature_b64)

                # Verify the signature
                from cryptography.hazmat.primitives.asymmetric import ec, rsa

                if self._public_key is None:
                    return False

                if isinstance(self._public_key, rsa.RSAPublicKey):
                    self._public_key.verify(
                        signature,
                        code_hash.encode(),
                        padding.PSS(
                            mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH,
                        ),
                        hashes.SHA256(),
                    )
                elif isinstance(self._public_key, ec.EllipticCurvePublicKey):
                    self._public_key.verify(
                        signature,
                        code_hash.encode(),
                        ec.ECDSA(hashes.SHA256()),
                    )
                else:
                    logger.warning(f"Unsupported key type: {type(self._public_key)}")
                    return False

                return True
            except InvalidSignature:
                return False
            except Exception as e:
                logger.exception(f"Error verifying signature: {e}")
                return False

        else:
            logger.warning(f"Unsupported signature type: {signature_type}")
            return False
