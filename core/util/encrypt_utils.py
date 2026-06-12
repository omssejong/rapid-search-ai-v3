import hashlib
import uuid
from Crypto.Cipher import AES

PLAIN_FRAG = "OMS_AI:rapid_search:v3:model_encrypt:v1"
UUID_FRAG = "de508679-0ded-4be9-8c43-86152027fd62"
RANDOM_FRAG = b'\xc7\x81\x12\x34\x56\x78\x9a\xbc\xde\xf0\x11\x22\x33\x44\x55\x66'


def get_aes_key() -> bytes:
    material = (
        uuid.UUID(UUID_FRAG).bytes
        + RANDOM_FRAG
        + PLAIN_FRAG.encode("utf-8")
    )
    return hashlib.sha256(material).digest()

def decrypt_bytes(enc_data: bytes) -> bytes:
    if len(enc_data) < 28:
        raise ValueError("Invalid encrypted data length")

    key = get_aes_key()

    nonce = enc_data[:12]
    tag = enc_data[12:28]
    ciphertext = enc_data[28:]

    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)


def decrypt_file(input_path: str, output_path: str):
    with open(input_path, "rb") as f:
        enc_data = f.read()

    data = decrypt_bytes(enc_data)

    with open(output_path, "wb") as f:
        f.write(data)