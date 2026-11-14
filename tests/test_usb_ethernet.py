"""Guest acceptance of the USB Ethernet adapter (fm1-live --usb ethernet, CDC-ECM) and of the
dock (--usb disk,ethernet: both behind the virtual hub): modd's firmware, built from its
modd_dev_network defconfig, switches the port to host mode, enumerates what is there, mounts the
disk, gets a lease from the virtual subnet and resolves the weather host. Never silently skipped:
the firmware build is named explicitly.

FM1_MODD_BUILD=/path/to/modd/build FM1_LIVE=target/release/fm1-live \
  python3 -m unittest discover -s tests -p test_usb_ethernet.py

FM1_MODD_BUILD holds modd.fwsc and modd.map (task build -- --defconfig modd_dev_network, with
CONFIG_USB_NET_AUTO=y so the port goes to host mode without a menu visit). The firmware's own
log ring is read out of the RAM dump through the map's symbols: no host-side network parser
stands in for what the guest's lwIP did.
"""
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RAM_BASE = 0x01C00000
LOG_LINE = 48


def contract():
    build = Path(os.environ["FM1_MODD_BUILD"]).resolve()
    fwsc, map_file = build / "modd.fwsc", build / "modd.map"
    for f in (fwsc, map_file):
        if not f.is_file():
            raise FileNotFoundError(f"Build modd first (task build -- --defconfig modd_dev_network): {f}")
    live = Path(os.environ.get("FM1_LIVE", ROOT / "target/release/fm1-live")).resolve()
    if not live.is_file():
        raise FileNotFoundError(f"Build fm1-live first: {live}")
    return fwsc, map_file, live


def symbol(map_text, name):
    """A section's (address, size) from modd.map, or a common symbol's (address, 4)."""
    m = re.search(r"\.bss\." + name + r"\s*\n\s*(0x[0-9a-f]+)\s+(0x[0-9a-f]+)", map_text)
    if m:
        return int(m.group(1), 16), int(m.group(2), 16)
    m = re.search(r"^\s*(0x[0-9a-f]+)\s+" + name + r"\s*$", map_text, re.M)
    if not m:
        raise AssertionError(f"{name} not in modd.map")
    return int(m.group(1), 16), 4


def guest_log(ram, map_text):
    """The firmware's log ring (adapters/modd_log.h), oldest first, by its timestamps."""
    (buf, size), (ts, _) = symbol(map_text, "modd_log_buf"), symbol(map_text, "modd_log_ts")
    rows = []
    for i in range(size // LOG_LINE):
        at = buf - RAM_BASE + i * LOG_LINE
        line = ram[at:at + LOG_LINE].split(b"\0")[0].decode(errors="replace")
        if line:
            rows.append((struct.unpack_from("<I", ram, ts - RAM_BASE + i * 4)[0], i, line))
    rows.sort()
    return [(t, line) for t, _, line in rows]


def empty_fat16(path, mib=32):
    """An empty FAT16 volume: 4 KiB clusters, label FM1EMU (mkfs.fat is not needed)."""
    bps, spc, reserved, nfats, root_entries = 512, 8, 1, 2, 512
    total = mib * 1024 * 1024 // bps
    clusters = (total - reserved - root_entries * 32 // bps) // spc
    spf = (clusters * 2 + bps - 1) // bps
    bs = bytearray(bps)
    bs[0:3] = b"\xeb\x3c\x90"
    bs[3:11] = b"MSWIN4.1"
    struct.pack_into("<HBHBHHBHHHII", bs, 11, bps, spc, reserved, nfats, root_entries, 0, 0xF8, spf, 63, 255, 0, total)
    bs[36], bs[38] = 0x80, 0x29
    struct.pack_into("<I", bs, 39, 0x4D4F4444)
    bs[43:54], bs[54:62], bs[510:512] = b"FM1EMU     ", b"FAT16   ", b"\x55\xaa"
    fat = bytearray(spf * bps)
    fat[0:4] = b"\xf8\xff\xff\xff"
    root = bytearray(root_entries * 32)
    root[0:11], root[11] = b"FM1EMU     ", 0x08
    with path.open("xb") as f:
        f.write(bs)
        f.write(fat * nfats)
        f.write(root)
        f.truncate(mib * 1024 * 1024)


def run_guest(live, fwsc, directory, ms, internet=False, interpreter=False, disk=None):
    ram = directory / "ram.bin"
    env = {k: v for k, v in os.environ.items() if not k.startswith("FM1_")}
    env.update(FM1_DUMPRAM=str(ram), FM1_NET_TRACE="1", FM1_BENCH_NOKEYS="1")
    if interpreter:
        env["FM1_NO_JIT"] = "1"
    port = "disk,ethernet" if disk else "ethernet"
    args = [str(live), "--fwsc", str(fwsc), "--usb", port, "--bench", str(ms)]
    if disk:
        args[-2:-2] = ["--disk-image", str(disk)]
    if internet:
        args[-2:-2] = ["--net", "internet"]
    result = subprocess.run(args, env=env, capture_output=True, timeout=300)
    (directory / "stderr.log").write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode(errors="replace")[-4000:]
    return result.stderr.decode(errors="replace"), ram.read_bytes()


class UsbEthernetGuestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fwsc, cls.map_file, cls.live = contract()
        cls.map_text = cls.map_file.read_text()

    def test_lease_and_lookup_over_the_adapter(self):
        with tempfile.TemporaryDirectory(prefix="fm1-usb-eth-") as temporary:
            directory = Path(temporary)
            for interpreter in (False, True):
                with self.subTest(interpreter=interpreter):
                    trace, ram = run_guest(self.live, self.fwsc, directory, 20000, interpreter=interpreter)
                    # the subnet's side: one lease (a DISCOVER answered, a REQUEST acknowledged),
                    # the weather host's name resolved, its SYN refused for want of --internet
                    self.assertEqual(trace.count("net: dhcp type 3 answered: 10.0.2.15"), 1, trace)
                    self.assertIn('net: dns "api.open-meteo.com" -> 10.0.2.100', trace)
                    self.assertRegex(trace, r'net: tcp syn to Some\(Name\("api.open-meteo.com"\)\):80 \(nat: false\)')
                    # the guest's side: lwIP's interface for the adapter holds the lease (the log
                    # ring is short and other modules write to it, so no line is relied on)
                    # (lwIP's default route is the adapter's netif: next, then ip_addr, netmask, gw)
                    default, _ = symbol(self.map_text, "netif_default")
                    netif = struct.unpack_from("<I", ram, default - RAM_BASE)[0]
                    self.assertNotEqual(netif, 0, "no default route: the link never came up")
                    at = netif - RAM_BASE + 4
                    self.assertEqual(ram[at:at + 12], bytes([10, 0, 2, 15, 255, 255, 255, 0, 10, 0, 2, 2]))
                    log = "\n".join(line for _, line in guest_log(ram, self.map_text))
                    self.assertNotIn("USB host: MSC fail", log)
                    self.assertNotIn("OTG: host enum failed", log)
                    self.assertNotIn("ecm: set ", log, "no failed ECM request")

    def test_dock_hub_with_disk_and_adapter(self):
        """--usb disk,ethernet: both behind the virtual hub, as on the dock. modd walks the hub,
        mounts the disk and brings the adapter up in one host session. 10 s: the lease comes
        once the firmware's first WiFi autoscan (2 to 8 s in the network defconfig) lets the
        adapter's poll run, and the log ring still holds the discovery."""
        with tempfile.TemporaryDirectory(prefix="fm1-usb-dock-") as temporary:
            directory = Path(temporary)
            image = directory / "dock.img"
            empty_fat16(image)
            trace, ram = run_guest(self.live, self.fwsc, directory, 10000, disk=image)
            self.assertEqual(trace.count("net: dhcp type 3 answered: 10.0.2.15"), 1, trace)
            default, _ = symbol(self.map_text, "netif_default")
            netif = struct.unpack_from("<I", ram, default - RAM_BASE)[0]
            self.assertNotEqual(netif, 0, "no default route: the adapter never came up")
            self.assertEqual(ram[netif - RAM_BASE + 4:netif - RAM_BASE + 8], bytes([10, 0, 2, 15]))
            log = "\n".join(line for _, line in guest_log(ram, self.map_text))
            for expected in ("usb: hub detected, running discovery", "hub: #ports=2", "hub: MSC found! addr=2",
                             "hub: ECM found cfg=1", "usb: msc ready", "USB host: sectors 65536", "ecm: attached",
                             "ecm: dhcp bound"):
                self.assertIn(expected, log, log[-3000:])
            self.assertNotIn("hub: MSC init fail", log)
            self.assertNotIn("OTG: host enum failed", log)


if __name__ == "__main__":
    unittest.main()
