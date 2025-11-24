"""
Local Secrets Vault - Encrypted storage for API keys and credentials

Provides secure, local-first secret management for RAGIX agents.

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Set

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = None

logger = logging.getLogger(__name__)


@dataclass
class SecretMetadata:
    """Metadata for a stored secret."""

    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    allowed_agents: Optional[List[str]] = None  # None = all agents


class SecretProvider(Protocol):
    """Protocol for secret storage backends."""

    def set_secret(self, name: str, value: str, metadata: Optional[SecretMetadata] = None):
        """Store a secret."""
        ...

    def get_secret(self, name: str, agent_id: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret value."""
        ...

    def delete_secret(self, name: str):
        """Delete a secret."""
        ...

    def list_secrets(self) -> List[str]:
        """List all secret names."""
        ...

    def has_secret(self, name: str) -> bool:
        """Check if secret exists."""
        ...


class InMemoryVault:
    """
    Simple in-memory vault for testing.

    NOT SECURE - secrets are stored in plain text in memory.
    Use only for development and testing.
    """

    def __init__(self):
        self.secrets: Dict[str, str] = {}
        self.metadata: Dict[str, SecretMetadata] = {}
        logger.warning("Using InMemoryVault - NOT SECURE for production!")

    def set_secret(self, name: str, value: str, metadata: Optional[SecretMetadata] = None):
        """Store a secret in memory."""
        self.secrets[name] = value
        if metadata:
            self.metadata[name] = metadata
        else:
            self.metadata[name] = SecretMetadata(name=name)
        logger.debug(f"Stored secret: {name}")

    def get_secret(self, name: str, agent_id: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret from memory."""
        if name not in self.secrets:
            return None

        # Check access control
        meta = self.metadata.get(name)
        if meta and meta.allowed_agents and agent_id:
            if agent_id not in meta.allowed_agents:
                logger.warning(f"Agent '{agent_id}' denied access to secret '{name}'")
                return None

        return self.secrets.get(name)

    def delete_secret(self, name: str):
        """Delete a secret from memory."""
        if name in self.secrets:
            del self.secrets[name]
            if name in self.metadata:
                del self.metadata[name]
            logger.debug(f"Deleted secret: {name}")

    def list_secrets(self) -> List[str]:
        """List all secret names."""
        return list(self.secrets.keys())

    def has_secret(self, name: str) -> bool:
        """Check if secret exists."""
        return name in self.secrets


class EncryptedFileVault:
    """
    Encrypted file-based vault using Fernet (symmetric encryption).

    Secrets are encrypted with a key derived from a master password.
    Vault file is stored as JSON with encrypted values.
    """

    def __init__(self, vault_path: Path, master_key: Optional[bytes] = None):
        """
        Initialize encrypted vault.

        Args:
            vault_path: Path to vault file
            master_key: Optional master key (32 bytes). If None, must be set later.
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError(
                "cryptography not installed. Install with: pip install cryptography"
            )

        self.vault_path = vault_path
        self._master_key = master_key
        self._fernet: Optional[Fernet] = None
        self.secrets: Dict[str, str] = {}  # Encrypted values
        self.metadata: Dict[str, Dict] = {}

        if master_key:
            self._initialize_fernet(master_key)

        # Load existing vault if it exists
        if self.vault_path.exists():
            self._load()

    def _initialize_fernet(self, master_key: bytes):
        """Initialize Fernet cipher with master key."""
        if len(master_key) != 32:
            raise ValueError("Master key must be exactly 32 bytes")
        self._master_key = master_key
        self._fernet = Fernet(Fernet.generate_key() if not master_key else master_key)

    @staticmethod
    def derive_key_from_password(password: str, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2.

        Args:
            password: Master password
            salt: Salt for key derivation (should be random, 16+ bytes)

        Returns:
            32-byte encryption key
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError("cryptography not installed")

        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(password.encode())

    def set_secret(self, name: str, value: str, metadata: Optional[SecretMetadata] = None):
        """Store an encrypted secret."""
        if not self._fernet:
            raise RuntimeError("Vault not initialized with master key")

        # Encrypt the value
        encrypted = self._fernet.encrypt(value.encode()).decode()
        self.secrets[name] = encrypted

        # Store metadata
        if metadata:
            self.metadata[name] = {
                "name": name,
                "description": metadata.description,
                "created_at": metadata.created_at,
                "updated_at": datetime.now().isoformat(),
                "tags": metadata.tags,
                "allowed_agents": metadata.allowed_agents,
            }
        else:
            self.metadata[name] = {
                "name": name,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

        self._save()
        logger.info(f"Stored encrypted secret: {name}")

    def get_secret(self, name: str, agent_id: Optional[str] = None) -> Optional[str]:
        """Retrieve and decrypt a secret."""
        if not self._fernet:
            raise RuntimeError("Vault not initialized with master key")

        if name not in self.secrets:
            return None

        # Check access control
        meta = self.metadata.get(name, {})
        allowed_agents = meta.get("allowed_agents")
        if allowed_agents and agent_id:
            if agent_id not in allowed_agents:
                logger.warning(f"Agent '{agent_id}' denied access to secret '{name}'")
                return None

        # Decrypt
        try:
            encrypted = self.secrets[name].encode()
            decrypted = self._fernet.decrypt(encrypted).decode()
            return decrypted
        except Exception as e:
            logger.error(f"Failed to decrypt secret '{name}': {e}")
            return None

    def delete_secret(self, name: str):
        """Delete a secret from vault."""
        if name in self.secrets:
            del self.secrets[name]
            if name in self.metadata:
                del self.metadata[name]
            self._save()
            logger.info(f"Deleted secret: {name}")

    def list_secrets(self) -> List[str]:
        """List all secret names."""
        return list(self.secrets.keys())

    def has_secret(self, name: str) -> bool:
        """Check if secret exists."""
        return name in self.secrets

    def get_metadata(self, name: str) -> Optional[Dict]:
        """Get metadata for a secret."""
        return self.metadata.get(name)

    def _save(self):
        """Save vault to encrypted file."""
        data = {"secrets": self.secrets, "metadata": self.metadata}

        # Ensure parent directory exists
        self.vault_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.vault_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved vault to {self.vault_path}")

    def _load(self):
        """Load vault from encrypted file."""
        with open(self.vault_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.secrets = data.get("secrets", {})
        self.metadata = data.get("metadata", {})

        logger.debug(f"Loaded vault from {self.vault_path} ({len(self.secrets)} secrets)")

    def change_master_key(self, old_key: bytes, new_key: bytes):
        """
        Re-encrypt all secrets with a new master key.

        Args:
            old_key: Current master key
            new_key: New master key
        """
        if not self._fernet or self._master_key != old_key:
            raise ValueError("Invalid old master key")

        # Decrypt all secrets with old key
        old_fernet = self._fernet
        decrypted_secrets = {}
        for name, encrypted in self.secrets.items():
            try:
                decrypted = old_fernet.decrypt(encrypted.encode()).decode()
                decrypted_secrets[name] = decrypted
            except Exception as e:
                logger.error(f"Failed to decrypt secret '{name}': {e}")
                raise

        # Re-encrypt with new key
        self._initialize_fernet(new_key)
        for name, value in decrypted_secrets.items():
            encrypted = self._fernet.encrypt(value.encode()).decode()
            self.secrets[name] = encrypted

        self._save()
        logger.info("Master key changed successfully")


@dataclass
class AccessPolicy:
    """Access control policy for secrets."""

    secret_pattern: str  # Glob pattern for secret names
    allowed_agents: List[str]
    allowed_tools: Optional[List[str]] = None  # Tools that can access this secret
    description: str = ""


class VaultManager:
    """
    High-level vault manager with access policies.

    Provides centralized secret management with fine-grained access control.
    """

    def __init__(self, vault: SecretProvider):
        """
        Initialize vault manager.

        Args:
            vault: Secret storage backend
        """
        self.vault = vault
        self.policies: List[AccessPolicy] = []

    def add_policy(self, policy: AccessPolicy):
        """Add an access control policy."""
        self.policies.append(policy)
        logger.debug(f"Added policy: {policy.secret_pattern} -> {policy.allowed_agents}")

    def check_access(self, secret_name: str, agent_id: str, tool_name: Optional[str] = None) -> bool:
        """
        Check if agent/tool can access a secret.

        Args:
            secret_name: Name of secret
            agent_id: Agent requesting access
            tool_name: Optional tool name

        Returns:
            True if access is allowed
        """
        import fnmatch

        for policy in self.policies:
            if fnmatch.fnmatch(secret_name, policy.secret_pattern):
                # Check agent access
                if agent_id not in policy.allowed_agents:
                    return False

                # Check tool access if specified
                if policy.allowed_tools and tool_name:
                    if tool_name not in policy.allowed_tools:
                        return False

                return True

        # No matching policy = deny by default
        return False

    def get_secret_safe(
        self, secret_name: str, agent_id: str, tool_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Get secret with access control check.

        Args:
            secret_name: Name of secret
            agent_id: Agent requesting access
            tool_name: Optional tool name

        Returns:
            Secret value if access is allowed, None otherwise
        """
        if not self.check_access(secret_name, agent_id, tool_name):
            logger.warning(
                f"Access denied: agent='{agent_id}', secret='{secret_name}', tool='{tool_name}'"
            )
            return None

        return self.vault.get_secret(secret_name, agent_id)


def create_vault(
    vault_type: str = "encrypted",
    vault_path: Optional[Path] = None,
    master_key: Optional[bytes] = None,
) -> SecretProvider:
    """
    Factory function to create secret vault.

    Args:
        vault_type: 'memory' or 'encrypted'
        vault_path: Path to vault file (for encrypted vault)
        master_key: Master encryption key (for encrypted vault)

    Returns:
        SecretProvider instance

    Raises:
        ValueError: If vault_type is unknown
    """
    if vault_type == "memory":
        return InMemoryVault()
    elif vault_type == "encrypted":
        if not vault_path:
            raise ValueError("vault_path required for encrypted vault")
        return EncryptedFileVault(vault_path, master_key)
    else:
        raise ValueError(f"Unknown vault type: {vault_type}")
