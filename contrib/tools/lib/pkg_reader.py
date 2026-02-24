"""Pure Python reader for macOS .pkg (XAR) installer packages.

A .pkg file is a XAR archive (magic "xar!") containing a Payload that
is gzip-compressed cpio. This reader parses XAR to find the Payload,
gunzips it, and feeds it to CpioReader to enumerate installed files.

Dependencies: stdlib only (struct, zlib, gzip, xml.etree.ElementTree, io).
"""

import gzip
import io
import struct
import zlib
import xml.etree.ElementTree as ET

from contrib.tools.lib.cpio import CpioReader
from contrib.tools.lib.tree_reader import FileInfo, TreeReader

XAR_MAGIC = b"xar!"
XAR_HEADER_SIZE = 28


# ============================================================================
# XAR Parser
# ============================================================================


def parse_xar_header(data: bytes) -> dict:
    """Parse the 28-byte XAR header.

    Layout (big-endian):
      0: uint32  magic ("xar!" = 0x78617221)
      4: uint16  header_size
      6: uint16  version
      8: uint64  toc_compressed_len
     16: uint64  toc_uncompressed_len
     24: uint32  cksum_algo
    """
    if len(data) < XAR_HEADER_SIZE:
        raise ValueError("Data too small for XAR header")

    magic = data[0:4]
    if magic != XAR_MAGIC:
        raise ValueError(f"Not a XAR file: magic {magic!r} (expected {XAR_MAGIC!r})")

    header_size = struct.unpack_from(">H", data, 4)[0]
    version = struct.unpack_from(">H", data, 6)[0]
    toc_compressed_len = struct.unpack_from(">Q", data, 8)[0]
    toc_uncompressed_len = struct.unpack_from(">Q", data, 16)[0]
    cksum_algo = struct.unpack_from(">I", data, 24)[0]

    return {
        "header_size": header_size,
        "version": version,
        "toc_compressed_len": toc_compressed_len,
        "toc_uncompressed_len": toc_uncompressed_len,
        "cksum_algo": cksum_algo,
    }


def parse_xar_toc(data: bytes, header: dict) -> ET.Element:
    """Read and decompress the TOC XML from XAR data."""
    hs = header["header_size"]
    toc_start = hs
    toc_end = toc_start + header["toc_compressed_len"]
    if toc_end > len(data):
        raise ValueError("XAR data truncated: TOC extends past end of data")

    compressed_toc = data[toc_start:toc_end]
    toc_xml = zlib.decompress(compressed_toc)
    return ET.fromstring(toc_xml)


def find_payloads(toc_root: ET.Element) -> list:
    """Walk the TOC XML to find all Payload file entries.

    Returns list of dicts with keys: offset, length, size, encoding, name.
    The offset is relative to the heap start.
    """
    payloads = []
    # XAR TOC has structure: <xar><toc><file>...</file></toc></xar>
    # Files can be nested (directories contain child <file> elements).
    _walk_files(toc_root, payloads)
    return payloads


def _walk_files(element: ET.Element, payloads: list):
    """Recursively walk <file> elements looking for Payload entries."""
    for file_elem in element.iter("file"):
        name_elem = file_elem.find("name")
        if name_elem is None:
            continue
        name = name_elem.text
        if name != "Payload":
            continue

        data_elem = file_elem.find("data")
        if data_elem is None:
            continue

        offset_elem = data_elem.find("offset")
        length_elem = data_elem.find("length")
        size_elem = data_elem.find("size")
        encoding_elem = data_elem.find("encoding")

        offset = int(offset_elem.text) if offset_elem is not None else 0
        length = int(length_elem.text) if length_elem is not None else 0
        size = int(size_elem.text) if size_elem is not None else 0
        encoding = ""
        if encoding_elem is not None:
            encoding = encoding_elem.get("style", "")

        payloads.append({
            "name": name,
            "offset": offset,
            "length": length,
            "size": size,
            "encoding": encoding,
        })


def extract_payload(data: bytes, header: dict, payload_info: dict) -> bytes:
    """Extract and decompress a Payload from XAR data.

    Returns the raw (possibly gzip-compressed) payload bytes from the heap.
    """
    heap_start = header["header_size"] + header["toc_compressed_len"]
    pay_start = heap_start + payload_info["offset"]
    pay_end = pay_start + payload_info["length"]
    if pay_end > len(data):
        raise ValueError("XAR data truncated: Payload extends past end of data")

    raw = data[pay_start:pay_end]

    # If the encoding indicates compression, decompress
    encoding = payload_info["encoding"]
    if "octet-stream" in encoding:
        # application/octet-stream means stored as-is (already compressed
        # at this level, the Payload itself is gzip cpio)
        return raw
    elif "x-gzip" in encoding:
        return gzip.decompress(raw)
    elif "x-bzip2" in encoding:
        import bz2
        return bz2.decompress(raw)
    else:
        # Default: return raw bytes (Payload is typically gzip cpio regardless)
        return raw


# ============================================================================
# PkgReader — TreeReader interface
# ============================================================================


class PkgReader(TreeReader):
    """Reader for macOS .pkg (XAR) installer packages."""

    def __init__(self, pkg_path: str = None, pkg_data: bytes = None):
        self.items = []
        self.index = 0
        if pkg_path is not None:
            with open(pkg_path, "rb") as f:
                pkg_data = f.read()
        if pkg_data is None:
            raise ValueError("Either pkg_path or pkg_data must be provided")
        self._load(pkg_data)

    def _load(self, data: bytes):
        """Parse XAR, find Payloads, gunzip, and read cpio entries."""
        header = parse_xar_header(data)
        toc_root = parse_xar_toc(data, header)
        payloads = find_payloads(toc_root)

        for payload_info in payloads:
            payload_bytes = extract_payload(data, header, payload_info)
            self._read_cpio_from_payload(payload_bytes)

    def _read_cpio_from_payload(self, payload_bytes: bytes):
        """Gunzip the payload and feed to CpioReader."""
        # The Payload is gzip-compressed cpio
        try:
            stream = gzip.GzipFile(fileobj=io.BytesIO(payload_bytes))
            # Wrap in BufferedReader so .tell() works for CpioReader
            buffered = io.BufferedReader(stream)
        except Exception:
            # If it's not gzip, try raw
            buffered = io.BufferedReader(io.BytesIO(payload_bytes))

        cpio = CpioReader(buffered)
        while True:
            info = cpio.next()
            if info is None:
                break
            self.items.append(info)

    def next(self) -> FileInfo:
        if self.index < len(self.items):
            item = self.items[self.index]
            self.index += 1
            return item
        return None

    def is_done(self) -> bool:
        return self.index >= len(self.items)
