#!/usr/bin/env python3
"""
RAGIX Vault CLI - Manage encrypted secrets

Usage:
    ragix-vault <command> [options]

Author: Olivier Vitrac, PhD, HDR | olivier.vitrac@adservio.fr | Adservio | 2025-11-24
"""

import argparse
import getpass
import logging
import sys
from pathlib import Path

from ragix_core.secrets_vault import (
    EncryptedFileVault,
    SecretMetadata,
    create_vault,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_vault_path(args) -> Path:
    """Get vault path from args or default."""
    if args.vault_path:
        return Path(args.vault_path)
    return Path.home() / ".ragix" / "vault.json"


def get_master_key(vault_path: Path, creating: bool = False) -> bytes:
    """Prompt for master password and derive key."""
    if creating:
        password = getpass.getpass("Enter master password: ")
        confirm = getpass.getpass("Confirm master password: ")
        if password != confirm:
            logger.error("Passwords do not match")
            sys.exit(1)
    else:
        password = getpass.getpass("Enter master password: ")

    # Use vault path as salt (deterministic for same vault)
    salt = str(vault_path).encode()[:16].ljust(16, b"\x00")
    return EncryptedFileVault.derive_key_from_password(password, salt)


def cmd_init(args):
    """Initialize a new vault."""
    vault_path = get_vault_path(args)

    if vault_path.exists() and not args.force:
        logger.error(f"Vault already exists: {vault_path}")
        logger.error("Use --force to overwrite")
        sys.exit(1)

    master_key = get_master_key(vault_path, creating=True)

    vault = EncryptedFileVault(vault_path, master_key)
    logger.info(f"✅ Vault initialized: {vault_path}")


def cmd_set(args):
    """Set a secret."""
    vault_path = get_vault_path(args)

    if not vault_path.exists():
        logger.error(f"Vault not found: {vault_path}")
        logger.error("Run 'ragix-vault init' first")
        sys.exit(1)

    master_key = get_master_key(vault_path)
    vault = EncryptedFileVault(vault_path, master_key)

    # Get secret value
    if args.value:
        value = args.value
    else:
        value = getpass.getpass(f"Enter value for '{args.name}': ")

    # Create metadata
    metadata = SecretMetadata(
        name=args.name,
        description=args.description or "",
        tags=args.tags or [],
        allowed_agents=args.allowed_agents or None,
    )

    vault.set_secret(args.name, value, metadata)
    logger.info(f"✅ Secret stored: {args.name}")


def cmd_get(args):
    """Get a secret."""
    vault_path = get_vault_path(args)

    if not vault_path.exists():
        logger.error(f"Vault not found: {vault_path}")
        sys.exit(1)

    master_key = get_master_key(vault_path)
    vault = EncryptedFileVault(vault_path, master_key)

    value = vault.get_secret(args.name)
    if value is None:
        logger.error(f"Secret not found: {args.name}")
        sys.exit(1)

    if args.copy:
        # Copy to clipboard (requires pyperclip)
        try:
            import pyperclip

            pyperclip.copy(value)
            logger.info(f"✅ Secret copied to clipboard: {args.name}")
        except ImportError:
            logger.error("pyperclip not installed. Install with: pip install pyperclip")
            sys.exit(1)
    else:
        print(value)


def cmd_list(args):
    """List secrets."""
    vault_path = get_vault_path(args)

    if not vault_path.exists():
        logger.error(f"Vault not found: {vault_path}")
        sys.exit(1)

    master_key = get_master_key(vault_path)
    vault = EncryptedFileVault(vault_path, master_key)

    secrets = vault.list_secrets()

    if not secrets:
        logger.info("No secrets in vault")
        return

    print(f"\nSecrets in {vault_path}:")
    print("-" * 60)

    for name in sorted(secrets):
        meta = vault.get_metadata(name)
        if meta:
            desc = meta.get("description", "")
            tags = meta.get("tags", [])
            created = meta.get("created_at", "")[:10]  # Date only
            print(f"  {name}")
            if desc:
                print(f"    Description: {desc}")
            if tags:
                print(f"    Tags: {', '.join(tags)}")
            print(f"    Created: {created}")
            print()
        else:
            print(f"  {name}")


def cmd_delete(args):
    """Delete a secret."""
    vault_path = get_vault_path(args)

    if not vault_path.exists():
        logger.error(f"Vault not found: {vault_path}")
        sys.exit(1)

    master_key = get_master_key(vault_path)
    vault = EncryptedFileVault(vault_path, master_key)

    if not vault.has_secret(args.name):
        logger.error(f"Secret not found: {args.name}")
        sys.exit(1)

    if not args.yes:
        confirm = input(f"Delete secret '{args.name}'? [y/N] ")
        if confirm.lower() != "y":
            logger.info("Cancelled")
            return

    vault.delete_secret(args.name)
    logger.info(f"✅ Secret deleted: {args.name}")


def cmd_change_password(args):
    """Change vault master password."""
    vault_path = get_vault_path(args)

    if not vault_path.exists():
        logger.error(f"Vault not found: {vault_path}")
        sys.exit(1)

    # Get old key
    print("Current password:")
    old_key = get_master_key(vault_path)

    # Get new key
    print("\nNew password:")
    new_password = getpass.getpass("Enter new master password: ")
    confirm = getpass.getpass("Confirm new master password: ")
    if new_password != confirm:
        logger.error("Passwords do not match")
        sys.exit(1)

    salt = str(vault_path).encode()[:16].ljust(16, b"\x00")
    new_key = EncryptedFileVault.derive_key_from_password(new_password, salt)

    # Change key
    vault = EncryptedFileVault(vault_path, old_key)
    vault.change_master_key(old_key, new_key)

    logger.info("✅ Master password changed successfully")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAGIX Vault - Encrypted secret storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  init              Initialize a new vault
  set NAME          Set a secret value
  get NAME          Get a secret value
  list              List all secrets
  delete NAME       Delete a secret
  change-password   Change vault master password

Examples:
  # Initialize vault
  ragix-vault init

  # Store a secret
  ragix-vault set OPENAI_API_KEY --description "OpenAI API key"

  # Get a secret
  ragix-vault get OPENAI_API_KEY

  # List secrets
  ragix-vault list

  # Delete a secret
  ragix-vault delete OPENAI_API_KEY --yes
""",
    )

    parser.add_argument(
        "--vault-path", type=str, help="Path to vault file (default: ~/.ragix/vault.json)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Init command
    parser_init = subparsers.add_parser("init", help="Initialize a new vault")
    parser_init.add_argument("--force", action="store_true", help="Overwrite existing vault")

    # Set command
    parser_set = subparsers.add_parser("set", help="Set a secret")
    parser_set.add_argument("name", help="Secret name")
    parser_set.add_argument("--value", help="Secret value (prompted if not provided)")
    parser_set.add_argument("--description", "-d", help="Secret description")
    parser_set.add_argument("--tags", nargs="+", help="Tags for secret")
    parser_set.add_argument(
        "--allowed-agents", nargs="+", help="Agent IDs allowed to access this secret"
    )

    # Get command
    parser_get = subparsers.add_parser("get", help="Get a secret")
    parser_get.add_argument("name", help="Secret name")
    parser_get.add_argument(
        "--copy", "-c", action="store_true", help="Copy to clipboard (requires pyperclip)"
    )

    # List command
    parser_list = subparsers.add_parser("list", help="List secrets")

    # Delete command
    parser_delete = subparsers.add_parser("delete", help="Delete a secret")
    parser_delete.add_argument("name", help="Secret name")
    parser_delete.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # Change password command
    parser_change = subparsers.add_parser("change-password", help="Change vault master password")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to command handler
    commands = {
        "init": cmd_init,
        "set": cmd_set,
        "get": cmd_get,
        "list": cmd_list,
        "delete": cmd_delete,
        "change-password": cmd_change_password,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        logger.info("\nCancelled")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
