"""
flexentech_encryption.py
------------------------

Professional, well-documented implementation of a lightweight permutation-based
encryption utility ("FlexenTech") suitable for watermark bitstreams and small
binary payloads.

Features:
- Block-wise permutation encryption/decryption
- Deterministic key generation from integer seed (for reproducibility)
- Secure random key generation (using secrets) for production
- Key save/load
- Utilities to convert bytes <-> bit arrays
- Clear exceptions, type hints, and example usage in __main__

Author: Brahim Ferik
License: MIT
"""

from __future__ import annotations

import json
import math
import logging
import secrets
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class FlexenTechError(Exception):
    """Base exception for FlexenTech module."""


@dataclass
class PermutationKey:
    """
    Represents a permutation key for block-wise encryption.

    Attributes:
        block_size: Number of bits per permutation block.
        permutation: A list/array of indices representing a permutation of range(block_size).
        seed: Optional integer seed used to generate the permutation (for reproducibility).
    """
    block_size: int
    permutation: List[int]
    seed: Optional[int] = None

    def validate(self) -> None:
        """Validate the permutation is correct for the block size."""
        if self.block_size <= 0:
            raise FlexenTechError("block_size must be a positive integer.")
        if len(self.permutation) != self.block_size:
            raise FlexenTechError(
                "Permutation length must equal block_size."
            )
        if sorted(self.permutation) != list(range(self.block_size)):
            raise FlexenTechError("Permutation must be a reordering of 0..block_size-1.")


def generate_permutation_key(
    block_size: int,
    seed: Optional[int] = None,
    secure: bool = False,
) -> PermutationKey:
    """
    Generate a permutation key.

    Args:
        block_size: Number of bits per block to permute.
        seed: Optional integer seed. If provided and secure=False, generation is deterministic.
        secure: If True, use the 'secrets' module for cryptographically secure generation; seed is ignored.

    Returns:
        PermutationKey
    """
    if block_size <= 0:
        raise FlexenTechError("block_size must be > 0")

    if secure:
        # cryptographically secure: use secrets to shuffle indices
        perm = list(range(block_size))
        # Fisher-Yates shuffle using secrets.randbelow
        for i in range(block_size - 1, 0, -1):
            j = secrets.randbelow(i + 1)
            perm[i], perm[j] = perm[j], perm[i]
        logger.debug("Generated secure permutation key (no seed).")
        return PermutationKey(block_size=block_size, permutation=perm, seed=None)

    # deterministic with seed for reproducibility
    if seed is None:
        seed = 42  # default deterministic seed (documented)
        logger.debug("No seed provided; using default seed=42.")

    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(block_size))
    logger.debug("Generated permutation key with seed=%s.", seed)
    return PermutationKey(block_size=block_size, permutation=[int(x) for x in perm], seed=int(seed))


def save_key(key: PermutationKey, path: Path) -> None:
    """Save a PermutationKey to a JSON file."""
    payload = asdict(key)
    # Ensure ints and lists are serializable
    payload["permutation"] = list(payload["permutation"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved permutation key to %s", str(path))


def load_key(path: Path) -> PermutationKey:
    """Load a PermutationKey from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    key = PermutationKey(block_size=int(data["block_size"]), permutation=list(map(int, data["permutation"])), seed=data.get("seed"))
    key.validate()
    logger.info("Loaded permutation key from %s", str(path))
    return key


def _bits_to_numpy(bits: Iterable[int]) -> np.ndarray:
    """Convert an iterable of bits (0/1) to a numpy uint8 array."""
    arr = np.fromiter(bits, dtype=np.uint8)
    if arr.size == 0:
        return arr
    if not np.all(np.isin(arr, [0, 1])):
        raise FlexenTechError("Bits must be 0 or 1.")
    return arr


def bytes_to_bits(data: bytes) -> np.ndarray:
    """
    Convert bytes to a 1D numpy array of bits (MSB-first within each byte).

    Example:
        b'\\x03' -> [0,0,0,0,0,0,1,1]
    """
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits.astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Convert a 1D numpy array of bits (0/1) to bytes.

    Handles lengths not divisible by 8 by padding with zeros on the right (LSB side).
    The caller should track original bit-length if needed.
    """
    if bits.ndim != 1:
        raise FlexenTechError("bits must be a 1D array.")
    # Calculate padding
    pad_len = (-len(bits)) % 8
    if pad_len:
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


def _permute_block(block: np.ndarray, perm: Sequence[int]) -> np.ndarray:
    """
    Apply permutation 'perm' to a 1D block array.

    block: 1D numpy array of length == len(perm)
    perm: sequence of indices representing where each source index should go.
          We implement as permuting positions so that out[i] = block[perm[i]].
    """
    if len(block) != len(perm):
        raise FlexenTechError("Block length does not match permutation length.")
    return block[np.array(perm, dtype=int)]


def _inverse_permutation(perm: Sequence[int]) -> List[int]:
    """Return inverse permutation inv such that inv[perm[i]] = i."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def permutation_encrypt(bits: Iterable[int], key: PermutationKey) -> np.ndarray:
    """
    Encrypt a bit sequence using block-wise permutation.

    Args:
        bits: Iterable of bits (0/1) or numpy array.
        key: PermutationKey with block_size and permutation.

    Returns:
        numpy array of encrypted bits (uint8). Length is padded to multiple of block_size.
    """
    key.validate()
    bit_arr = _bits_to_numpy(bits)
    n = bit_arr.size
    if n == 0:
        return bit_arr  # empty

    bsize = key.block_size
    n_blocks = math.ceil(n / bsize)
    padded_len = n_blocks * bsize
    if padded_len != n:
        pad = np.zeros(padded_len - n, dtype=np.uint8)
        bit_arr = np.concatenate([bit_arr, pad])
        logger.debug("Padded bit array from %d to %d bits.", n, padded_len)

    bit_arr = bit_arr.reshape((n_blocks, bsize))
    perm = np.array(key.permutation, dtype=int)
    # Apply permutation to each block
    encrypted_blocks = bit_arr[:, perm]
    encrypted = encrypted_blocks.reshape(-1)
    logger.debug("Encrypted %d bits using block_size=%d, blocks=%d.", n, bsize, n_blocks)
    return encrypted.astype(np.uint8)


def permutation_decrypt(encrypted_bits: Iterable[int], key: PermutationKey, original_length: Optional[int] = None) -> np.ndarray:
    """
    Decrypt a permutation-encrypted bit sequence.

    Args:
        encrypted_bits: Iterable of bits (0/1) or numpy array.
        key: PermutationKey.
        original_length: Optional original bit length; if provided, final output is truncated to this length.

    Returns:
        numpy array of decrypted bits (uint8).
    """
    key.validate()
    enc = _bits_to_numpy(encrypted_bits)
    if enc.size == 0:
        return enc

    bsize = key.block_size
    if enc.size % bsize != 0:
        raise FlexenTechError("Encrypted bit length must be multiple of block_size.")
    blocks = enc.reshape((-1, bsize))
    inv_perm = _inverse_permutation(key.permutation)
    decrypted_blocks = blocks[:, inv_perm]
    decrypted = decrypted_blocks.reshape(-1).astype(np.uint8)
    if original_length is not None:
        decrypted = decrypted[:original_length]
    logger.debug("Decrypted %d bits with block_size=%d.", decrypted.size, bsize)
    return decrypted


# Convenience functions for byte-oriented payloads


def encrypt_bytes(payload: bytes, key: PermutationKey) -> bytes:
    """
    Encrypt a bytes payload. Returns encrypted bytes. Caller should store original bit length
    if precise recovery is required when message length is not multiple of block_size.
    """
    bits = bytes_to_bits(payload)
    enc_bits = permutation_encrypt(bits, key)
    return bits_to_bytes(enc_bits)


def decrypt_bytes(encrypted_payload: bytes, key: PermutationKey, original_bit_length: Optional[int] = None) -> bytes:
    """
    Decrypt bytes payload previously encrypted with encrypt_bytes.
    If original_bit_length is provided, it will be used to truncate padded bits.
    """
    enc_bits = bytes_to_bits(encrypted_payload)
    dec_bits = permutation_decrypt(enc_bits, key, original_length=original_bit_length)
    return bits_to_bytes(dec_bits)


# Example CLI / quick sanity tests
def _example_usage():
    from textwrap import dedent

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)

    print(dedent("""
    FlexenTech example:
      - create a deterministic key (seeded),
      - encrypt a byte message,
      - decrypt it and verify equality.
    """))

    message = b"Hello FlexenTech!"
    print("Original bytes:", message)
    print("Original bits length:", len(message) * 8)

    key = generate_permutation_key(block_size=16, seed=1234, secure=False)
    save_key(key, Path("example_key.json"))

    enc = encrypt_bytes(message, key)
    print("Encrypted bytes (hex):", enc.hex()[:96] + ("..." if len(enc.hex()) > 96 else ""))

    # For correctness: we must pass original_bit_length because block_size might not divide msg bits
    original_bit_len = len(message) * 8
    dec = decrypt_bytes(enc, key, original_bit_length=original_bit_len)
    print("Decrypted bytes:", dec)
    assert dec.startswith(message), "Decrypted output does not match original (prefix check)."

    # Load key from disk and decrypt again
    loaded_key = load_key(Path("example_key.json"))
    assert loaded_key.block_size == key.block_size
    dec2 = decrypt_bytes(enc, loaded_key, original_bit_length=original_bit_len)
    assert dec2.startswith(message)
    print("Round-trip success.")


if __name__ == "__main__":
    _example_usage()
