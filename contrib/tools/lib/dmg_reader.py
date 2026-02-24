"""Pure Python reader for macOS .dmg (UDIF) disk images.

Parses the UDIF container to decompress into a raw HFS+ image, then
walks the HFS+ catalog B-tree to enumerate all files, directories,
and symlinks.

No external tools required — uses only Python stdlib (struct, plistlib,
zlib, bz2, lzma). Optional: pyliblzfse for LZFSE-compressed DMGs.
"""

import base64
import bz2
import lzma
import plistlib
import struct
import sys
import zlib

from contrib.tools.lib.pkg_reader import PkgReader
from contrib.tools.lib.tree_reader import FileInfo, TreeReader


# ============================================================================
# UDIF (Universal Disk Image Format) Parser
# ============================================================================

KOLY_MAGIC = 0x6B6F6C79  # 'koly'
MISH_MAGIC = 0x6D697368  # 'mish'
SECTOR_SIZE = 512

# Compression types in blkx chunk entries
BLOCK_ZERO_FILL = 0x00000000
BLOCK_RAW = 0x00000001
BLOCK_IGNORE = 0x00000002
BLOCK_ZLIB = 0x80000005
BLOCK_BZ2 = 0x80000006
BLOCK_LZFSE = 0x80000007
BLOCK_LZMA = 0x80000008
BLOCK_ADC = 0x80000004
BLOCK_COMMENT = 0x7FFFFFFE
BLOCK_TERMINATOR = 0xFFFFFFFF


def parse_koly(f) -> dict:
    """Parse the 512-byte koly trailer at end of DMG file.

    Returns dict with XMLOffset, XMLLength, and other fields.
    """
    f.seek(0, 2)
    file_size = f.tell()
    if file_size < 512:
        raise ValueError("File too small to be a DMG")

    f.seek(file_size - 512)
    trailer = f.read(512)

    magic = struct.unpack_from(">I", trailer, 0)[0]
    if magic != KOLY_MAGIC:
        raise ValueError(
            f"Not a DMG file: koly magic not found (got 0x{magic:08x})"
        )

    # koly trailer layout (big-endian):
    #   0: uint32 magic
    #   4: uint32 version
    #   8: uint32 headerSize
    #  12: uint32 flags
    #  ...
    # 216: uint64 XMLOffset
    # 224: uint64 XMLLength
    xml_offset = struct.unpack_from(">Q", trailer, 216)[0]
    xml_length = struct.unpack_from(">Q", trailer, 224)[0]

    return {
        "xml_offset": xml_offset,
        "xml_length": xml_length,
    }


def parse_blkx_chunks(blkx_data: bytes) -> list:
    """Parse a blkx (mish) block descriptor into a list of chunk entries.

    Each blkx starts with a mish header (204 bytes), followed by 40-byte
    chunk entries.
    """
    magic = struct.unpack_from(">I", blkx_data, 0)[0]
    if magic != MISH_MAGIC:
        raise ValueError(f"Expected mish magic, got 0x{magic:08x}")

    # mish header fields (big-endian):
    #  0: uint32 magic
    #  4: uint32 version (== 1)
    #  8: uint64 firstSectorNumber
    # 16: uint64 sectorCount
    # 24: uint64 dataOffset
    # 32: uint32 buffersNeeded
    # 36: uint32 blockDescriptorCount
    # 40-200: reserved / checksums
    # 200: uint32 numChunkEntries
    first_sector = struct.unpack_from(">Q", blkx_data, 8)[0]
    data_offset = struct.unpack_from(">Q", blkx_data, 24)[0]
    num_chunks = struct.unpack_from(">I", blkx_data, 200)[0]

    chunks = []
    offset = 204  # past mish header
    for _ in range(num_chunks):
        if offset + 40 > len(blkx_data):
            break
        # Chunk entry (40 bytes, big-endian):
        #  0: uint32 entryType (compression)
        #  4: uint32 comment
        #  8: uint64 sectorNumber (relative to partition start)
        # 16: uint64 sectorCount
        # 24: uint64 compressedOffset (relative to data fork start)
        # 32: uint64 compressedLength
        entry_type = struct.unpack_from(">I", blkx_data, offset)[0]
        sector_number = struct.unpack_from(">Q", blkx_data, offset + 8)[0]
        sector_count = struct.unpack_from(">Q", blkx_data, offset + 16)[0]
        compressed_offset = struct.unpack_from(">Q", blkx_data, offset + 24)[0]
        compressed_length = struct.unpack_from(">Q", blkx_data, offset + 32)[0]

        chunks.append({
            "type": entry_type,
            "sector_number": sector_number,  # relative to partition start
            "sector_count": sector_count,
            "compressed_offset": data_offset + compressed_offset,
            "compressed_length": compressed_length,
        })
        offset += 40

    return chunks


def _find_hfs_blkx(blkx_list: list) -> dict:
    """Find the blkx entry for the HFS+ partition."""
    # Look for Apple_HFS or Apple_HFSX in the Name field
    for entry in blkx_list:
        name = entry.get("Name", "")
        if "Apple_HFS" in name or "Apple_HFSX" in name:
            return entry

    # Fallback: pick the largest partition (by sector count in mish header)
    best = None
    best_sectors = 0
    for entry in blkx_list:
        blkx_data = entry["Data"]
        if isinstance(blkx_data, str):
            blkx_data = base64.b64decode(blkx_data)
        if len(blkx_data) < 24:
            continue
        magic = struct.unpack_from(">I", blkx_data, 0)[0]
        if magic != MISH_MAGIC:
            continue
        sector_count = struct.unpack_from(">Q", blkx_data, 16)[0]
        if sector_count > best_sectors:
            best_sectors = sector_count
            best = entry

    if best is not None:
        return best
    raise ValueError("No HFS+ partition found in DMG")


def decompress_udif(dmg_path: str) -> bytearray:
    """Parse a UDIF DMG and decompress the HFS+ partition into a raw image."""
    with open(dmg_path, "rb") as f:
        koly = parse_koly(f)

        # Read and parse XML plist
        f.seek(koly["xml_offset"])
        xml_data = f.read(koly["xml_length"])
        plist = plistlib.loads(xml_data)

        # Extract blkx entries from resource-fork
        blkx_list = plist.get("resource-fork", {}).get("blkx", [])
        if not blkx_list:
            raise ValueError("No blkx entries found in DMG plist")

        # Find the HFS+ partition and parse its chunks
        hfs_blkx = _find_hfs_blkx(blkx_list)
        blkx_data = hfs_blkx["Data"]
        if isinstance(blkx_data, str):
            blkx_data = base64.b64decode(blkx_data)
        all_chunks = parse_blkx_chunks(blkx_data)

        # Determine partition size from chunk extents
        max_sector = 0
        for chunk in all_chunks:
            end = chunk["sector_number"] + chunk["sector_count"]
            if end > max_sector:
                max_sector = end

        total_size = max_sector * SECTOR_SIZE
        image = bytearray(total_size)

        # Decompress and place blocks
        for chunk in all_chunks:
            entry_type = chunk["type"]
            dest_offset = chunk["sector_number"] * SECTOR_SIZE
            expected_size = chunk["sector_count"] * SECTOR_SIZE

            if entry_type in (BLOCK_TERMINATOR, BLOCK_COMMENT, BLOCK_IGNORE):
                continue

            if entry_type == BLOCK_ZERO_FILL:
                # Already zero-filled by bytearray
                continue

            # Read compressed data from file
            f.seek(chunk["compressed_offset"])
            compressed = f.read(chunk["compressed_length"])

            if entry_type == BLOCK_RAW:
                data = compressed
            elif entry_type == BLOCK_ZLIB:
                data = zlib.decompress(compressed)
            elif entry_type == BLOCK_BZ2:
                data = bz2.decompress(compressed)
            elif entry_type == BLOCK_LZMA:
                data = lzma.decompress(compressed)
            elif entry_type == BLOCK_LZFSE:
                try:
                    import liblzfse
                    data = liblzfse.decompress(compressed, expected_size)
                except ImportError:
                    raise ImportError(
                        "LZFSE compression requires pyliblzfse: "
                        "pip install pyliblzfse"
                    )
            elif entry_type == BLOCK_ADC:
                print(
                    f"WARNING: ADC compression not supported, skipping block "
                    f"at sector {chunk['sector_number']}",
                    file=sys.stderr,
                )
                continue
            else:
                print(
                    f"WARNING: Unknown block type 0x{entry_type:08x}, skipping",
                    file=sys.stderr,
                )
                continue

            image[dest_offset:dest_offset + len(data)] = data

    return image


# ============================================================================
# HFS+ Catalog B-Tree Walker
# ============================================================================

HFS_PLUS_SIGNATURE = b"H+"
HFSX_SIGNATURE = b"HX"

# Catalog record types
FOLDER_RECORD = 0x0001
FILE_RECORD = 0x0002
FOLDER_THREAD = 0x0003
FILE_THREAD = 0x0004

# HFS+ special CNIDs
ROOT_PARENT_CNID = 1
ROOT_FOLDER_CNID = 2

S_IFLNK = 0o120000
S_IFDIR = 0o040000
S_IFMT = 0o170000


def _read_u16(data, off):
    return struct.unpack_from(">H", data, off)[0]


def _read_u32(data, off):
    return struct.unpack_from(">I", data, off)[0]


def _read_u64(data, off):
    return struct.unpack_from(">Q", data, off)[0]


class HfsPlusReader:
    """Minimal HFS+ filesystem reader — walks the catalog B-tree."""

    def __init__(self, image: bytearray):
        self.image = image
        self._parse_volume_header()
        self._read_catalog()

    def _parse_volume_header(self):
        """Read the HFS+ Volume Header at offset 1024."""
        vh_off = 1024
        sig = self.image[vh_off:vh_off + 2]
        if sig not in (HFS_PLUS_SIGNATURE, HFSX_SIGNATURE):
            raise ValueError(
                f"Not an HFS+ volume: signature {sig!r} "
                f"(expected {HFS_PLUS_SIGNATURE!r} or {HFSX_SIGNATURE!r})"
            )

        # Volume Header layout (big-endian, offset from vh_off):
        #   0: uint16 signature
        #   2: uint16 version
        #  40: uint32 blockSize
        self.block_size = _read_u32(self.image, vh_off + 40)

        # Catalog file fork data starts at vh_off + 272 (catalogFile)
        # HFSPlusForkData: logicalSize(8), clumpSize(4), totalBlocks(4),
        #                   extents[8] (each: startBlock(4), blockCount(4))
        cat_fork_off = vh_off + 272
        self.cat_logical_size = _read_u64(self.image, cat_fork_off)
        # Read up to 8 extent descriptors
        self.cat_extents = []
        ext_off = cat_fork_off + 16  # past logicalSize + clumpSize + totalBlocks
        for _ in range(8):
            start_block = _read_u32(self.image, ext_off)
            block_count = _read_u32(self.image, ext_off + 4)
            if block_count == 0:
                break
            self.cat_extents.append((start_block, block_count))
            ext_off += 8

    def _read_catalog_data(self) -> bytearray:
        """Read the catalog file contents from its extents."""
        data = bytearray()
        for start_block, block_count in self.cat_extents:
            offset = start_block * self.block_size
            length = block_count * self.block_size
            data.extend(self.image[offset:offset + length])
        # Truncate to logical size
        return data[:self.cat_logical_size]

    def _read_catalog(self):
        """Parse the catalog B-tree and collect all file/folder entries."""
        cat_data = self._read_catalog_data()

        # B-tree header node is node 0
        # Node descriptor (14 bytes):
        #   0: uint32 fLink
        #   4: uint32 bLink
        #   8: int8   kind (-1=leaf, 0=index, 1=header, 2=map)
        #   9: uint8  height
        #  10: uint16 numRecords
        #  12: uint16 reserved
        # After node descriptor, the header record (BTHeaderRec) at offset 14:
        #  14: uint16 treeDepth
        #  16: uint32 rootNode
        #  20: uint32 leafRecords
        #  24: uint32 firstLeafNode
        #  28: uint32 lastLeafNode
        #  32: uint16 nodeSize
        node_size = _read_u16(cat_data, 32)
        first_leaf = _read_u32(cat_data, 24)

        # Walk the leaf node chain
        self.entries = {}  # cnid -> (parent_cnid, name, record_type, info)
        node_idx = first_leaf
        while node_idx != 0:
            node_off = node_idx * node_size
            if node_off + node_size > len(cat_data):
                break

            node = cat_data[node_off:node_off + node_size]
            f_link = _read_u32(node, 0)
            kind = struct.unpack_from(">b", node, 8)[0]
            num_records = _read_u16(node, 10)

            if kind != -1:  # not a leaf node
                node_idx = f_link
                continue

            self._parse_leaf_node(node, node_size, num_records)
            node_idx = f_link

    def _parse_leaf_node(self, node: bytes, node_size: int, num_records: int):
        """Parse records from a single leaf node."""
        # Record offsets are stored at the end of the node, growing backwards.
        # offset[i] is at: node_size - 2*(i+1)
        for i in range(num_records):
            rec_off_pos = node_size - 2 * (i + 1)
            rec_off = _read_u16(node, rec_off_pos)

            # Next record offset (for bounds checking)
            if i + 1 < num_records:
                next_off_pos = node_size - 2 * (i + 2)
                next_off = _read_u16(node, next_off_pos)
            else:
                # Free space offset
                free_off_pos = node_size - 2 * (num_records + 1)
                next_off = _read_u16(node, free_off_pos)

            rec_len = next_off - rec_off
            if rec_len < 6 or rec_off + rec_len > node_size:
                continue

            self._parse_catalog_record(node, rec_off, rec_len)

    def _parse_catalog_record(self, node: bytes, rec_off: int, rec_len: int):
        """Parse a single catalog B-tree key+data record."""
        # Catalog key:
        #   0: uint16 keyLength
        #   2: uint32 parentID (CNID)
        #   6: uint16 nameLength (in Unicode chars)
        #   8: name (nameLength * 2 bytes, UTF-16BE)
        key_length = _read_u16(node, rec_off)
        if key_length < 6:
            return

        parent_cnid = _read_u32(node, rec_off + 2)
        name_length = _read_u16(node, rec_off + 6)

        name_bytes_len = name_length * 2
        if rec_off + 8 + name_bytes_len > len(node):
            return
        try:
            name = node[rec_off + 8:rec_off + 8 + name_bytes_len].decode("utf-16-be")
        except UnicodeDecodeError:
            return

        # Data follows the key (aligned to 2 bytes)
        data_off = rec_off + 2 + key_length
        if data_off % 2 != 0:
            data_off += 1

        if data_off + 2 > rec_off + rec_len:
            return

        record_type = _read_u16(node, data_off)

        if record_type == FOLDER_RECORD:
            # HFSPlusCatalogFolder:
            #   0: uint16 recordType
            #   2: uint16 flags
            #   4: uint32 valence
            #   8: uint32 folderID (CNID)
            #  12: uint32 createDate
            #  16: uint32 contentModDate
            #  20: uint32 attributeModDate
            #  24: uint32 accessDate
            #  28: uint32 backupDate
            #  32: HFSPlusBSDInfo (ownerID(4), groupID(4), adminFlags(1), ownerFlags(1), fileMode(2), special(4))
            if data_off + 48 > len(node):
                return
            cnid = _read_u32(node, data_off + 8)
            uid = _read_u32(node, data_off + 32)
            gid = _read_u32(node, data_off + 36)
            # adminFlags(1) + ownerFlags(1) + fileMode(2)
            file_mode = _read_u16(node, data_off + 42)
            self.entries[cnid] = (parent_cnid, name, FOLDER_RECORD, {
                "mode": file_mode,
                "uid": uid,
                "gid": gid,
            })

        elif record_type == FILE_RECORD:
            # HFSPlusCatalogFile:
            #   0: uint16 recordType
            #   2: uint16 flags
            #   4: uint32 reserved
            #   8: uint32 fileID (CNID)
            #  12: uint32 createDate
            #  16: uint32 contentModDate
            #  20: uint32 attributeModDate
            #  24: uint32 accessDate
            #  28: uint32 backupDate
            #  32: HFSPlusBSDInfo (ownerID(4), groupID(4), adminFlags(1)+ownerFlags(1)+fileMode(2), special(4))
            #  48: userInfo (16 bytes)
            #  64: finderInfo (16 bytes)
            #  80: textEncoding (4) + reserved (4)
            #  88: dataFork (HFSPlusForkData): logicalSize(8), clumpSize(4), totalBlocks(4), extents(64)
            if data_off + 96 > len(node):
                return
            cnid = _read_u32(node, data_off + 8)
            uid = _read_u32(node, data_off + 32)
            gid = _read_u32(node, data_off + 36)
            file_mode = _read_u16(node, data_off + 42)
            data_fork_size = _read_u64(node, data_off + 88)
            # Data fork extents start at offset 96 within the record data
            # (88 + 8 for logicalSize). Each extent: startBlock(4), blockCount(4)
            # clumpSize(4) + totalBlocks(4) = 8 bytes, then extents at +104
            extents = []
            ext_base = data_off + 88 + 16  # past logicalSize + clumpSize + totalBlocks
            for ei in range(8):
                eo = ext_base + ei * 8
                if eo + 8 > len(node):
                    break
                sb = _read_u32(node, eo)
                bc = _read_u32(node, eo + 4)
                if bc == 0:
                    break
                extents.append((sb, bc))
            self.entries[cnid] = (parent_cnid, name, FILE_RECORD, {
                "mode": file_mode,
                "uid": uid,
                "gid": gid,
                "size": data_fork_size,
                "extents": extents,
            })

        # Thread records (types 3, 4) are reverse lookups — skip them.

    def build_path(self, cnid: int) -> str:
        """Reconstruct full path for a CNID by walking parent chain."""
        parts = []
        current = cnid
        seen = set()
        while current in self.entries and current not in seen:
            seen.add(current)
            parent_cnid, name, _, _ = self.entries[current]
            parts.append(name)
            if parent_cnid == ROOT_FOLDER_CNID or parent_cnid == ROOT_PARENT_CNID:
                break
            current = parent_cnid
        parts.reverse()
        return "/".join(parts)

    def read_file(self, cnid: int) -> bytes:
        """Read file content for a given CNID using its data fork extents."""
        if cnid not in self.entries:
            raise ValueError(f"CNID {cnid} not found")
        _, _, rec_type, info = self.entries[cnid]
        if rec_type != FILE_RECORD:
            raise ValueError(f"CNID {cnid} is not a file")
        extents = info.get("extents", [])
        logical_size = info.get("size", 0)
        data = bytearray()
        for start_block, block_count in extents:
            offset = start_block * self.block_size
            length = block_count * self.block_size
            data.extend(self.image[offset:offset + length])
        return bytes(data[:logical_size])

    def get_all_files(self) -> list:
        """Return list of FileInfo for all files/dirs/symlinks in the volume."""
        items = []
        for cnid, (parent_cnid, name, rec_type, info) in self.entries.items():
            # Skip the root folder itself
            if cnid == ROOT_FOLDER_CNID:
                continue

            path = self.build_path(cnid)
            if not path:
                continue

            mode_raw = info["mode"]
            uid = info["uid"]
            gid = info["gid"]

            # Construct full mode with file type bits
            file_type_bits = mode_raw & S_IFMT
            is_symlink = file_type_bits == S_IFLNK
            is_dir = rec_type == FOLDER_RECORD

            if is_dir and mode_raw == 0:
                mode_raw = S_IFDIR | 0o755
            elif not is_dir and not is_symlink and mode_raw == 0:
                mode_raw = 0o100644

            if rec_type == FOLDER_RECORD:
                items.append(FileInfo(
                    path=path,
                    size=0,
                    mode=mode_raw if mode_raw != 0 else (S_IFDIR | 0o755),
                    uid=uid,
                    gid=gid,
                    is_dir=True,
                    is_symlink=False,
                ))
            elif rec_type == FILE_RECORD:
                size = info.get("size", 0)
                symlink_target = None

                if is_symlink:
                    size = 0
                    # Symlink target is stored in the data fork — we could
                    # read it but for tree comparison we just note it's a link.

                items.append(FileInfo(
                    path=path,
                    size=0 if is_symlink else size,
                    mode=mode_raw,
                    uid=uid,
                    gid=gid,
                    is_dir=False,
                    is_symlink=is_symlink,
                    symlink_target=symlink_target,
                ))

        items.sort(key=lambda x: x.path)
        return items


# ============================================================================
# DmgReader — TreeReader interface
# ============================================================================


class DmgReader(TreeReader):
    """Reader for macOS .dmg (UDIF) disk images."""

    def __init__(self, dmg_path: str):
        self.items = []
        self.index = 0
        self.done = False
        self._load(dmg_path)

    def _load(self, dmg_path: str):
        """Parse DMG and collect all file entries."""
        # 1. Decompress UDIF to raw HFS+ image
        image = decompress_udif(dmg_path)

        # 2. Walk HFS+ catalog
        hfs = HfsPlusReader(image)
        self.items = hfs.get_all_files()

        # 3. Recurse into .pkg files
        for cnid, (parent_cnid, name, rec_type, info) in hfs.entries.items():
            if rec_type != FILE_RECORD or not name.endswith(".pkg"):
                continue
            pkg_path = hfs.build_path(cnid)
            try:
                pkg_bytes = hfs.read_file(cnid)
                if not pkg_bytes or pkg_bytes[:4] != b"xar!":
                    continue
                pkg = PkgReader(pkg_data=pkg_bytes)
                while True:
                    entry = pkg.next()
                    if entry is None:
                        break
                    entry.path = "@PKG@/" + entry.path
                    self.items.append(entry)
            except Exception as e:
                print(
                    f"WARNING: Failed to read pkg {pkg_path}: {e}",
                    file=sys.stderr,
                )

        self.items.sort(key=lambda x: x.path)

    def next(self) -> FileInfo:
        if self.index < len(self.items):
            item = self.items[self.index]
            self.index += 1
            return item
        self.done = True
        return None

    def is_done(self) -> bool:
        return self.done
