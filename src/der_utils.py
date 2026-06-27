"""
DER (Distinguished Encoding Rules) 构建工具。

提供基本的 ASN.1 DER 编码函数，供 spki.py 和 tools/gen_mldsa_certs.py 共用。
消除两处重复的 DER 构建代码。
"""

import datetime


def encode_der_length(length: int) -> bytes:
    """编码 DER 长度字段。"""
    if length < 0x80:
        return bytes([length])
    elif length < 0x100:
        return bytes([0x81, length])
    elif length < 0x10000:
        return bytes([0x82, length >> 8, length & 0xFF])
    else:
        raise ValueError(f"DER length {length} too large")


def build_der_sequence(*items: bytes) -> bytes:
    """构建 DER SEQUENCE。"""
    content = b"".join(items)
    return b"\x30" + encode_der_length(len(content)) + content


def build_der_set(*items: bytes) -> bytes:
    """构建 DER SET。"""
    content = b"".join(items)
    return b"\x31" + encode_der_length(len(content)) + content


def build_der_bitstring(data: bytes, unused_bits: int = 0) -> bytes:
    """构建 DER BIT STRING。"""
    content = bytes([unused_bits]) + data
    return b"\x03" + encode_der_length(len(content)) + content


def build_der_octet_string(data: bytes) -> bytes:
    """构建 DER OCTET STRING。"""
    return b"\x04" + encode_der_length(len(data)) + data


def build_der_oid(oid_str: str) -> bytes:
    """构建 DER OBJECT IDENTIFIER。"""
    parts = [int(x) for x in oid_str.split(".")]
    first_byte = 40 * parts[0] + parts[1]
    encoded = bytes([first_byte])
    for part in parts[2:]:
        if part < 0x80:
            encoded += bytes([part])
        else:
            multi = []
            multi.append(part & 0x7F)
            part >>= 7
            while part > 0:
                multi.append(0x80 | (part & 0x7F))
                part >>= 7
            encoded += bytes(reversed(multi))
    return b"\x06" + encode_der_length(len(encoded)) + encoded


def build_der_null() -> bytes:
    """构建 DER NULL。"""
    return b"\x05\x00"


def build_der_integer(value: int) -> bytes:
    """构建 DER INTEGER（支持负数，二补码编码）。"""
    if value == 0:
        return b"\x02\x01\x00"
    neg = value < 0
    if neg:
        value = -value
    byte_len = (value.bit_length() + 8) // 8
    raw = value.to_bytes(byte_len, "big")
    # 补零防止最高位为 1 被误解为负数
    if raw[0] & 0x80 and not neg:
        raw = b"\x00" + raw
    if neg:
        # 二补码
        int_val = int.from_bytes(raw, "big")
        int_val = (1 << (len(raw) * 8)) - int_val
        raw = int_val.to_bytes(len(raw), "big")
    return b"\x02" + encode_der_length(len(raw)) + raw


def build_der_utc_time(dt: datetime.datetime) -> bytes:
    """构建 DER UTCTime (YYMMDDHHMMSSZ)。"""
    s = dt.strftime("%y%m%d%H%M%SZ")
    return b"\x17" + bytes([len(s)]) + s.encode("ascii")


def build_der_context_specific(tag: int, content: bytes) -> bytes:
    """构建 DER context-specific 标签 [tag]。"""
    return bytes([0xA0 | tag]) + encode_der_length(len(content)) + content


def build_name(common_name: str) -> bytes:
    """构建 X.509 Name (SEQUENCE OF SET OF AttributeTypeAndValue)。

    只设置 CN (commonName)。
    """
    # OID 2.5.4.3 = commonName
    cn_attr = build_der_sequence(
        build_der_oid("2.5.4.3"),
        build_der_octet_string(common_name.encode("utf-8")),
    )
    attr_set = build_der_set(cn_attr)
    return build_der_sequence(attr_set)
