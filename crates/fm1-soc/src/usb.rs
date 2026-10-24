//! USB0: the FM-1's full-speed device controller, and a host at the other end of the cable.
//!
//! The controller is a Mentor MUSB core (the classic FS byte map) behind JieLi's register
//! window at 0x11800 (Felucca's hal/fm1_usb.h is the reference, and its src/usb.c, which runs
//! on real FM-1s, the behaviour this follows):
//!   0x11800 CON0   bit 0 on, bit 2 SIE out of reset (USB_NRST), bits 3..5 set at attach,
//!                  bit 12 write = clear SOF pending, bit 13 SOF pending, bits 19.. endpoint
//!                  DMA off (cleared = on)
//!   0x11804 CON1   the SIE's registers, one byte at a time: write (r << 8) | v, or
//!                  (r << 8) | 0x4000 to read; bit 15 reads "done", the low byte the data
//!   0x11808 + 4n   EP_CNT(n), n = 0..3: bytes of the IN packet to send (written last)
//!   0x11818        EP0_ADR: EP0's buffer, both directions
//!   0x1181C + 8(n-1) / 0x11820 + 8(n-1)  EP n's IN (TADR) / OUT (RADR) buffer, n = 1..3
//!   0x11834 / 0x11838 / 0x1183C  EP4's count / IN / OUT buffer
//! and the D+ pull-up in USB_IO_CON0 (0x51000 bit 6): attached when it and CON0 bit 0 are set.
//! The endpoints move their packets straight to and from RAM (the buffers above).
//!
//! The SIE is in the USB clock domain, and every access is a round trip that takes time
//! (`sie_cycles`; an estimate, 1 µs by default, FM1_USB_SIE_NS sets it): firmwares poll the
//! done bit, so their USB code costs CPU here as on the chip. Without a cable the SIE does not
//! answer at all (as Felucca's usb.c notes: powered on without one, the SIE stays silent and
//! its driver gives up), which is what the emulator did before it had this model.
//!
//! The host (`--usb` in fm1-live, `Usb::cable`): every millisecond a frame (SOF, the frame
//! number), and in it the transactions of a full-speed host: after the pull-up, a bus reset,
//! then enumeration (device descriptor, SET_ADDRESS, the configuration and the product
//! string, SET_CONFIGURATION), then a USB-MIDI device's bulk endpoints: one OUT packet a frame
//! from what the computer sends (`midi_send`), one IN poll a frame into what it receives
//! (`midi_take`). The device only answers at its own address (FADDR), so a firmware that gets
//! SET_ADDRESS wrong drops off as it would on a real bus.
//!
//! Not modelled: the USB interrupt. The SIE sets its flags (INTRTX / INTRRX / INTRUSB) but
//! raises no IRQ line, so a driver that polls them (Felucca's usb.c) enumerates, and one that
//! waits for the interrupt (M-VAVE's own firmware) does not.
//!
//! SPDX-License-Identifier: GPL-3.0-only

use std::collections::VecDeque;

pub const USB_LO: u32 = 0x11800;
pub const USB_HI: u32 = 0x1183f;
/// USB_IO_CON0: the pads; bit 6 is the D+ pull-up (attach).
pub const USB_IO_CON0: u32 = 0x51000;

// SIE registers (MUSB FS byte map; usb.c's S_*)
const S_FADDR: u32 = 0;
const S_POWER: u32 = 1;
const S_INTRTX1: u32 = 2;
const S_INTRTX2: u32 = 3;
const S_INTRRX1: u32 = 4;
const S_INTRRX2: u32 = 5;
const S_INTRUSB: u32 = 6;
const S_INTRTX1E: u32 = 7;
const S_INTRTX2E: u32 = 8;
const S_INTRRX1E: u32 = 9;
const S_INTRRX2E: u32 = 10;
const S_INTRUSBE: u32 = 11;
const S_FRAME1: u32 = 12;
const S_FRAME2: u32 = 13;
const S_INDEX: u32 = 14;
const S_TXMAXP: u32 = 16;
const S_CSR0: u32 = 17; // = TXCSR1 for EP1..
const S_TXCSR2: u32 = 18;
const S_RXMAXP: u32 = 19;
const S_RXCSR1: u32 = 20;
const S_RXCSR2: u32 = 21;
const S_COUNT0: u32 = 22; // = RXCOUNT1 for EP1..
const S_RXCOUNT2: u32 = 23;

const NEP: usize = 5;

#[derive(Clone, Copy, Default, Debug)]
struct Ep {
    txmaxp: u8,
    rxmaxp: u8,
    txcsr2: u8,
    rxcsr2: u8,
    /// IN: a packet waits for the host (TxPktRdy).
    tx_rdy: bool,
    /// OUT: a packet waits for the firmware (RxPktRdy), its length.
    rx_rdy: bool,
    rx_count: u16,
    // EP0 only
    /// DataEnd written: the data stage is over (or there is none).
    data_end: bool,
    /// SendStall written: the next IN / OUT of this transfer is stalled.
    stall: bool,
    sent_stall: bool,
    setup_end: bool,
}

/// What the host gets back from a transaction.
#[derive(Debug, PartialEq)]
enum Reply {
    Data(Vec<u8>),
    Ack,
    Nak,
    Stall,
    /// Nobody answered (wrong address, no SIE).
    Silent,
}

/// Where the controller's DMA reads and writes (the SoC's bus).
pub trait Dma {
    fn dma_read8(&mut self, addr: u32) -> u8;
    fn dma_write8(&mut self, addr: u32, v: u8);
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
enum Stage {
    #[default]
    Setup,
    DataIn,
    StatusOut,
    StatusIn,
}

#[derive(Clone, Debug, Default)]
struct Ctl {
    setup: [u8; 8],
    stage: Stage,
    got: Vec<u8>,
    want: usize,
    /// Frames this stage has been NAKed (a stuck device is given up on).
    naks: u32,
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
enum Step {
    #[default]
    DevDesc8,
    SetAddress,
    DevDesc,
    ConfHead,
    Conf,
    Lang,
    Product,
    SetConfig,
}

#[derive(Clone, PartialEq, Debug, Default)]
enum HostState {
    /// No device on the bus (no pull-up).
    #[default]
    Detached,
    /// The pull-up came: the host waits before its bus reset (debounce), until this frame.
    Attached(u64),
    /// Bus reset done: enumeration from this frame on.
    Reset(u64),
    Enum(Step),
    Configured,
    /// Enumeration failed: the reason (the host waits for the device to go and come back).
    Failed(String),
}

/// Something that happened on the bus, for the front end.
#[derive(Clone, Debug, PartialEq)]
pub enum UsbEvent {
    /// Enumerated and configured: the product string, vendor and product ids.
    Configured { product: String, vid: u16, pid: u16 },
    /// The device dropped off the bus (its pull-up went).
    Detached,
    /// Enumeration failed.
    Failed(String),
}

#[derive(Default)]
pub struct Usb {
    /// A cable to a host: without it the SIE never answers and nothing happens on the bus.
    pub cable: bool,
    /// CPU cycles per SIE round trip.
    pub sie_cycles: u64,
    regs: [u32; 16],
    con1: u32,
    sie_done_at: u64,
    sie_data: u8,
    // the SIE
    faddr: u8,
    power: u8,
    intrtx: u16,
    intrrx: u16,
    intrusb: u8,
    intrtxe: u16,
    intrrxe: u16,
    intrusbe: u8,
    frame: u16,
    index: u8,
    ep: [Ep; NEP],
    // the host
    host: HostState,
    frames: u64,
    addr: u8,
    ctl: Option<Ctl>,
    dev_desc: Vec<u8>,
    conf: Vec<u8>,
    lang: u16,
    product: String,
    midi_in_ep: u8,
    midi_out_ep: u8,
    out_q: VecDeque<[u8; 4]>,
    in_q: VecDeque<[u8; 4]>,
    rx_sysex: Vec<u8>,
    pub events: Vec<UsbEvent>,
    /// FM1_USB_TRACE: the host's transactions on stderr.
    pub trace: bool,
    /// SIE accesses so far (CPU cost: times `sie_cycles`).
    pub sie_accesses: u64,
    /// Packets each way on the MIDI endpoints.
    pub out_packets: u64,
    pub in_packets: u64,
}

impl Usb {
    /// The cable in, to a host, with the CPU at `hz`.
    pub fn plug(&mut self, hz: u64) {
        let ns = std::env::var("FM1_USB_SIE_NS").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(1000);
        self.cable = true;
        self.sie_cycles = (hz * ns / 1_000_000_000).max(1);
        self.trace = std::env::var_os("FM1_USB_TRACE").is_some();
    }

    fn con0(&self) -> u32 {
        self.regs[0]
    }

    fn sie_on(&self) -> bool {
        self.cable && self.con0() & 4 != 0
    }

    /// Does a load from `addr` depend on the clock? (CON1's done bit does: a polling loop on
    /// it must run, not be skipped.)
    pub fn pure(&self, addr: u32) -> bool {
        !self.cable || addr != USB_LO + 4
    }

    pub fn read(&mut self, addr: u32, now: u64) -> u32 {
        let i = ((addr - USB_LO) / 4) as usize;
        match i {
            1 => {
                let done = self.sie_on() && now >= self.sie_done_at;
                (self.con1 & 0xff_7f00) | if done { 0x8000 | self.sie_data as u32 } else { 0 }
            }
            _ => self.regs[i],
        }
    }

    pub fn write(&mut self, addr: u32, v: u32, now: u64) {
        let i = ((addr - USB_LO) / 4) as usize;
        match i {
            0 => {
                // bit 13 (SOF pending) is the controller's, bit 12 writes clear it
                let pend = if v & (1 << 12) != 0 { 0 } else { self.regs[0] & (1 << 13) };
                self.regs[0] = (v & !(3 << 12)) | pend;
                if v & 4 == 0 {
                    self.sie_reset();
                }
            }
            1 => {
                self.con1 = v;
                if !self.sie_on() {
                    return;
                }
                self.sie_accesses += 1;
                let r = (v >> 8) & 0x3f;
                if v & 0x4000 != 0 {
                    self.sie_data = self.sie_read(r);
                } else {
                    self.sie_write(r, (v & 0xff) as u8);
                    self.sie_data = 0;
                }
                self.sie_done_at = now + self.sie_cycles;
            }
            _ => self.regs[i] = v,
        }
    }

    fn sie_reset(&mut self) {
        self.faddr = 0;
        self.power = 0;
        self.intrtx = 0;
        self.intrrx = 0;
        self.intrusb = 0;
        self.index = 0;
        self.ep = [Ep::default(); NEP];
    }

    /// Endpoint `n`'s DMA is on (CON0 bit 19 + n cleared).
    fn ep_dma(&self, n: usize) -> bool {
        self.con0() & (1 << (19 + n)) == 0
    }

    fn cnt(&self, n: usize) -> usize {
        (match n {
            0..=3 => self.regs[2 + n],
            _ => self.regs[13],
        } & 0x3ff) as usize
    }

    fn tadr(&self, n: usize) -> u32 {
        match n {
            0 => self.regs[6],
            1..=3 => self.regs[7 + 2 * (n - 1)],
            _ => self.regs[14],
        }
    }

    fn radr(&self, n: usize) -> u32 {
        match n {
            0 => self.regs[6],
            1..=3 => self.regs[8 + 2 * (n - 1)],
            _ => self.regs[15],
        }
    }

    fn sie_read(&mut self, r: u32) -> u8 {
        let e = &self.ep[self.index as usize % NEP];
        match r {
            S_FADDR => self.faddr,
            S_POWER => self.power,
            // the interrupt flags clear when read
            S_INTRTX1 => {
                let v = self.intrtx as u8;
                self.intrtx &= 0xff00;
                v
            },
            S_INTRTX2 => {
                let v = (self.intrtx >> 8) as u8;
                self.intrtx &= 0xff;
                v
            }
            S_INTRRX1 => {
                let v = self.intrrx as u8;
                self.intrrx &= 0xff00;
                v
            },
            S_INTRRX2 => {
                let v = (self.intrrx >> 8) as u8;
                self.intrrx &= 0xff;
                v
            }
            S_INTRUSB => std::mem::take(&mut self.intrusb),
            S_INTRTX1E => self.intrtxe as u8,
            S_INTRTX2E => (self.intrtxe >> 8) as u8,
            S_INTRRX1E => self.intrrxe as u8,
            S_INTRRX2E => (self.intrrxe >> 8) as u8,
            S_INTRUSBE => self.intrusbe,
            S_FRAME1 => self.frame as u8,
            S_FRAME2 => (self.frame >> 8) as u8 & 7,
            S_INDEX => self.index,
            S_TXMAXP => e.txmaxp,
            S_CSR0 if self.index == 0 => {
                e.rx_rdy as u8 | (e.tx_rdy as u8) << 1 | (e.sent_stall as u8) << 2 | (e.setup_end as u8) << 4
            }
            S_CSR0 => e.tx_rdy as u8 | (e.stall as u8) << 4 | (e.sent_stall as u8) << 5,
            S_TXCSR2 => e.txcsr2,
            S_RXMAXP => e.rxmaxp,
            S_RXCSR1 => e.rx_rdy as u8 | (e.stall as u8) << 5 | (e.sent_stall as u8) << 6,
            S_RXCSR2 => e.rxcsr2,
            S_COUNT0 => e.rx_count as u8,
            S_RXCOUNT2 => (e.rx_count >> 8) as u8,
            _ => 0,
        }
    }

    fn sie_write(&mut self, r: u32, v: u8) {
        let n = self.index as usize % NEP;
        match r {
            S_FADDR => self.faddr = v & 0x7f,
            S_POWER => self.power = v,
            S_INTRTX1E => self.intrtxe = (self.intrtxe & 0xff00) | v as u16,
            S_INTRTX2E => self.intrtxe = (self.intrtxe & 0xff) | (v as u16) << 8,
            S_INTRRX1E => self.intrrxe = (self.intrrxe & 0xff00) | v as u16,
            S_INTRRX2E => self.intrrxe = (self.intrrxe & 0xff) | (v as u16) << 8,
            S_INTRUSBE => self.intrusbe = v,
            S_INDEX => self.index = v & 0xf,
            S_TXMAXP => self.ep[n].txmaxp = v,
            S_CSR0 if n == 0 => {
                let e = &mut self.ep[0];
                if v & 0x40 != 0 {
                    e.rx_rdy = false; // ServicedRxPktRdy
                }
                if v & 0x80 != 0 {
                    e.setup_end = false; // ServicedSetupEnd
                }
                if v & 0x20 != 0 {
                    e.stall = true;
                }
                if v & 0x08 != 0 {
                    e.data_end = true;
                }
                if v & 0x02 != 0 {
                    e.tx_rdy = true;
                }
                if v & 0x04 == 0 {
                    e.sent_stall = false; // (written 0)
                }
            }
            S_CSR0 => {
                let e = &mut self.ep[n];
                if v & 0x08 != 0 {
                    e.tx_rdy = false; // FlushFIFO
                } else if v & 1 != 0 {
                    e.tx_rdy = true;
                }
                e.stall = v & 0x10 != 0;
                if v & 0x20 == 0 {
                    e.sent_stall = false;
                }
            }
            S_TXCSR2 => self.ep[n].txcsr2 = v,
            S_RXMAXP => self.ep[n].rxmaxp = v,
            S_RXCSR1 => {
                let e = &mut self.ep[n];
                // FlushFIFO or RxPktRdy written 0: the packet is taken
                if v & 0x10 != 0 || v & 1 == 0 {
                    e.rx_rdy = false;
                }
                e.stall = v & 0x20 != 0;
                if v & 0x40 == 0 {
                    e.sent_stall = false;
                }
            }
            S_RXCSR2 => self.ep[n].rxcsr2 = v,
            _ => {}
        }
    }

    // ------------------------------------------------------------- the device's side ---

    fn answers(&self, addr: u8) -> bool {
        self.sie_on() && self.faddr == addr
    }

    fn bus_reset(&mut self) {
        self.sie_reset();
        self.intrusb |= 4;
    }

    fn dev_setup(&mut self, dma: &mut dyn Dma, addr: u8, pkt: &[u8; 8]) -> Reply {
        if !self.answers(addr) {
            return Reply::Silent;
        }
        let at = self.radr(0);
        for (i, &b) in pkt.iter().enumerate() {
            dma.dma_write8(at.wrapping_add(i as u32), b);
        }
        let e = &mut self.ep[0];
        if e.tx_rdy || e.rx_rdy {
            e.setup_end = true; // the transfer before was cut short
        }
        *e = Ep { setup_end: e.setup_end, ..Default::default() };
        e.rx_rdy = true;
        e.rx_count = 8;
        self.intrtx |= 1;
        Reply::Ack
    }

    fn dev_in(&mut self, dma: &mut dyn Dma, addr: u8, n: usize, status: bool) -> Reply {
        if !self.answers(addr) {
            return Reply::Silent;
        }
        if n == 0 {
            let e = &mut self.ep[0];
            if e.stall {
                e.stall = false;
                e.sent_stall = true;
                self.intrtx |= 1;
                return Reply::Stall;
            }
            if status {
                // the status stage of a transfer without data: once the firmware wrote DataEnd
                if e.data_end && !e.rx_rdy {
                    e.data_end = false;
                    self.intrtx |= 1;
                    return Reply::Data(Vec::new());
                }
                return Reply::Nak;
            }
            if !e.tx_rdy {
                return Reply::Nak;
            }
            let (at, len) = (self.tadr(0), self.cnt(0).min(64));
            let d = (0..len).map(|i| dma.dma_read8(at.wrapping_add(i as u32))).collect();
            self.ep[0].tx_rdy = false;
            self.intrtx |= 1;
            return Reply::Data(d);
        }
        if n >= NEP || !self.ep_dma(n) {
            return Reply::Nak;
        }
        let e = &mut self.ep[n];
        if e.stall {
            e.sent_stall = true;
            return Reply::Stall;
        }
        if !e.tx_rdy {
            return Reply::Nak;
        }
        let (at, len) = (self.tadr(n), self.cnt(n).min(64));
        let d = (0..len).map(|i| dma.dma_read8(at.wrapping_add(i as u32))).collect();
        self.ep[n].tx_rdy = false;
        self.intrtx |= 1 << n;
        Reply::Data(d)
    }

    fn dev_out(&mut self, dma: &mut dyn Dma, addr: u8, n: usize, data: &[u8]) -> Reply {
        if !self.answers(addr) {
            return Reply::Silent;
        }
        if n == 0 {
            // the status stage after IN data (a zero-length packet)
            let e = &mut self.ep[0];
            if e.stall {
                e.stall = false;
                e.sent_stall = true;
                self.intrtx |= 1;
                return Reply::Stall;
            }
            if e.tx_rdy {
                return Reply::Nak;
            }
            e.data_end = false;
            self.intrtx |= 1;
            return Reply::Ack;
        }
        if n >= NEP || !self.ep_dma(n) {
            return Reply::Nak;
        }
        let e = &mut self.ep[n];
        if e.stall {
            e.sent_stall = true;
            return Reply::Stall;
        }
        if e.rx_rdy {
            return Reply::Nak;
        }
        let at = self.radr(n);
        for (i, &b) in data.iter().enumerate() {
            dma.dma_write8(at.wrapping_add(i as u32), b);
        }
        let e = &mut self.ep[n];
        e.rx_rdy = true;
        e.rx_count = data.len() as u16;
        self.intrrx |= 1 << n;
        Reply::Ack
    }

    // --------------------------------------------------------------------- the host ---

    /// Configured, and what it is called.
    pub fn product(&self) -> Option<&str> {
        (self.host == HostState::Configured).then_some(self.product.as_str())
    }

    /// One MIDI message from the computer (a complete one, SysEx included) to the device.
    pub fn midi_send(&mut self, msg: &[u8]) {
        self.out_q.extend(usb_midi_packets(msg));
    }

    /// Bytes waiting to go to the device.
    pub fn midi_pending(&self) -> usize {
        self.out_q.len()
    }

    /// What the device sent the computer: complete MIDI messages.
    pub fn midi_take(&mut self) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while let Some(p) = self.in_q.pop_front() {
            midi_from_packet(&p, &mut self.rx_sysex, &mut out);
        }
        out
    }

    fn log(&self, s: impl FnOnce() -> String) {
        if self.trace {
            eprintln!("usb {:6}: {}", self.frames, s());
        }
    }

    fn fail(&mut self, why: String) {
        self.log(|| format!("enumeration failed: {why}"));
        self.events.push(UsbEvent::Failed(why.clone()));
        self.host = HostState::Failed(why);
        self.ctl = None;
    }

    /// One millisecond on the bus. `pullup`: USB_IO_CON0 bit 6.
    pub fn frame(&mut self, dma: &mut dyn Dma, pullup: bool) {
        self.frames += 1;
        let present = self.cable && pullup && self.con0() & 1 != 0;
        if !present {
            if self.host != HostState::Detached {
                self.log(|| "device gone".into());
                if !matches!(self.host, HostState::Attached(_)) {
                    self.events.push(UsbEvent::Detached);
                }
                self.host = HostState::Detached;
                self.ctl = None;
                self.out_q.clear();
            }
            return;
        }
        // SOF: the frame number, and the pending bit
        self.frame = (self.frame + 1) & 0x7ff;
        self.regs[0] |= 1 << 13;
        match self.host.clone() {
            HostState::Detached => {
                self.log(|| "pull-up: device attached".into());
                self.host = HostState::Attached(self.frames + 100);
            }
            HostState::Attached(at) if self.frames >= at => {
                self.log(|| "bus reset".into());
                self.bus_reset();
                self.addr = 0;
                self.host = HostState::Reset(self.frames + 20);
            }
            HostState::Reset(at) if self.frames >= at => {
                self.host = HostState::Enum(Step::DevDesc8);
                self.start_step(Step::DevDesc8);
            }
            HostState::Enum(step) => self.enumerate(dma, step),
            HostState::Configured => self.midi_frame(dma),
            _ => {}
        }
    }

    fn start_step(&mut self, step: Step) {
        let (req, wv, wi, len): (u16, u16, u16, u16) = match step {
            Step::DevDesc8 => (0x8006, 0x0100, 0, 64),
            Step::SetAddress => (0x0005, 9, 0, 0),
            Step::DevDesc => (0x8006, 0x0100, 0, 18),
            Step::ConfHead => (0x8006, 0x0200, 0, 9),
            Step::Conf => (0x8006, 0x0200, 0, u16::from_le_bytes([self.conf[2], self.conf[3]])),
            Step::Lang => (0x8006, 0x0300, 0, 255),
            Step::Product => (0x8006, 0x0300 | self.dev_desc[15] as u16, self.lang, 255),
            Step::SetConfig => (0x0009, self.conf[5] as u16, 0, 0),
        };
        let s = [(req >> 8) as u8, req as u8, wv as u8, (wv >> 8) as u8, wi as u8, (wi >> 8) as u8, len as u8, (len >> 8) as u8];
        self.ctl = Some(Ctl { setup: s, want: len as usize, ..Default::default() });
    }

    /// Control transfer transactions for this frame; Some(data) when the transfer is done.
    fn control(&mut self, dma: &mut dyn Dma) -> Result<Option<Vec<u8>>, String> {
        let addr = self.addr;
        for _ in 0..8 {
            let Some(mut c) = self.ctl.take() else { return Ok(None) };
            let reply = match c.stage {
                Stage::Setup => self.dev_setup(dma, addr, &c.setup),
                Stage::DataIn => self.dev_in(dma, addr, 0, false),
                Stage::StatusOut => self.dev_out(dma, addr, 0, &[]),
                Stage::StatusIn => self.dev_in(dma, addr, 0, true),
            };
            let req = u16::from_be_bytes([c.setup[0], c.setup[1]]);
            match reply {
                Reply::Nak | Reply::Silent => {
                    c.naks += 1;
                    if c.naks > 5000 {
                        return Err(format!("request {req:04x}: no answer in {:?} for 5 s{}", c.stage,
                                           if reply == Reply::Silent { " (not at its address)" } else { "" }));
                    }
                    self.ctl = Some(c);
                    return Ok(None);
                }
                Reply::Stall => return Err(format!("request {req:04x} stalled in {:?}", c.stage)),
                Reply::Ack => {
                    c.naks = 0;
                    c.stage = match c.stage {
                        Stage::Setup if c.want == 0 => Stage::StatusIn,
                        Stage::Setup => Stage::DataIn,
                        Stage::StatusOut => {
                            self.log(|| format!("{req:04x} done: {} bytes", c.got.len()));
                            return Ok(Some(c.got));
                        }
                        s => s,
                    };
                    self.ctl = Some(c);
                }
                Reply::Data(d) => {
                    c.naks = 0;
                    if c.stage == Stage::StatusIn {
                        self.log(|| format!("{req:04x} done"));
                        return Ok(Some(c.got));
                    }
                    let short = d.len() < 64;
                    c.got.extend_from_slice(&d);
                    if short || c.got.len() >= c.want {
                        c.got.truncate(c.want);
                        c.stage = Stage::StatusOut;
                    }
                    self.ctl = Some(c);
                }
            }
        }
        Ok(None)
    }

    fn enumerate(&mut self, dma: &mut dyn Dma, step: Step) {
        let got = match self.control(dma) {
            Ok(Some(d)) => d,
            Ok(None) => return,
            Err(e) => return self.fail(e),
        };
        let next = match step {
            Step::DevDesc8 => {
                if got.len() < 8 || got[1] != 1 {
                    return self.fail(format!("device descriptor: {got:02x?}"));
                }
                Step::SetAddress
            }
            Step::SetAddress => {
                self.addr = 9;
                Step::DevDesc
            }
            Step::DevDesc => {
                if got.len() < 18 {
                    return self.fail(format!("device descriptor: {} bytes", got.len()));
                }
                self.dev_desc = got;
                Step::ConfHead
            }
            Step::ConfHead => {
                if got.len() < 9 || got[1] != 2 {
                    return self.fail(format!("configuration descriptor: {got:02x?}"));
                }
                self.conf = got;
                Step::Conf
            }
            Step::Conf => {
                // (wTotalLength is the device's word: under 9, or another type, is not one)
                if got.len() < 9 || got[1] != 2 {
                    return self.fail(format!("configuration descriptor: {got:02x?}"));
                }
                self.conf = got;
                if self.dev_desc[15] == 0 {
                    Step::SetConfig
                } else {
                    Step::Lang
                }
            }
            Step::Lang => {
                self.lang = if got.len() >= 4 { u16::from_le_bytes([got[2], got[3]]) } else { 0x0409 };
                Step::Product
            }
            Step::Product => {
                self.product = got.get(2..).unwrap_or(&[]).chunks(2).map(|c| c[0] as char).collect();
                Step::SetConfig
            }
            Step::SetConfig => {
                self.find_midi();
                let vid = u16::from_le_bytes([self.dev_desc[8], self.dev_desc[9]]);
                let pid = u16::from_le_bytes([self.dev_desc[10], self.dev_desc[11]]);
                self.log(|| format!("configured: \"{}\" {vid:04x}:{pid:04x}, MIDI OUT ep {} IN ep {}",
                                    self.product, self.midi_out_ep, self.midi_in_ep & 0xf));
                self.events.push(UsbEvent::Configured { product: self.product.clone(), vid, pid });
                self.host = HostState::Configured;
                self.ctl = None;
                return;
            }
        };
        self.host = HostState::Enum(next);
        self.start_step(next);
    }

    /// The bulk endpoints of the configuration's MIDI streaming interface (audio class,
    /// subclass 3).
    fn find_midi(&mut self) {
        let (mut i, mut midi) = (0usize, false);
        self.midi_in_ep = 0;
        self.midi_out_ep = 0;
        while i + 2 <= self.conf.len() {
            let (len, ty) = (self.conf[i] as usize, self.conf[i + 1]);
            if len < 2 {
                break;
            }
            match ty {
                4 if i + 7 <= self.conf.len() => midi = self.conf[i + 5] == 1 && self.conf[i + 6] == 3,
                5 if midi && i + 4 <= self.conf.len() && self.conf[i + 3] & 3 == 2 => {
                    let ep = self.conf[i + 2];
                    if ep & 0x80 != 0 && self.midi_in_ep == 0 {
                        self.midi_in_ep = ep;
                    } else if ep & 0x80 == 0 && self.midi_out_ep == 0 {
                        self.midi_out_ep = ep;
                    }
                }
                _ => {}
            }
            i += len;
        }
    }

    fn midi_frame(&mut self, dma: &mut dyn Dma) {
        let addr = self.addr;
        if self.midi_out_ep != 0 && !self.out_q.is_empty() {
            let n = self.out_q.len().min(16);
            let data: Vec<u8> = self.out_q.iter().take(n).flatten().copied().collect();
            if self.dev_out(dma, addr, (self.midi_out_ep & 0xf) as usize, &data) == Reply::Ack {
                self.out_q.drain(..n);
                self.out_packets += 1;
                self.log(|| format!("OUT {data:02x?}"));
            }
        }
        if self.midi_in_ep != 0 {
            if let Reply::Data(d) = self.dev_in(dma, addr, (self.midi_in_ep & 0xf) as usize, false) {
                self.in_packets += 1;
                self.log(|| format!("IN {d:02x?}"));
                for p in d.chunks_exact(4) {
                    self.in_q.push_back([p[0], p[1], p[2], p[3]]);
                }
            }
        }
    }
}

/// A complete MIDI message as USB-MIDI event packets (cable 0).
pub fn usb_midi_packets(msg: &[u8]) -> Vec<[u8; 4]> {
    let mut out = Vec::new();
    let Some(&st) = msg.first() else { return out };
    if st == 0xf0 {
        let mut chunks = msg.chunks(3).peekable();
        while let Some(c) = chunks.next() {
            let end = chunks.peek().is_none() && c.last() == Some(&0xf7);
            let cin = if end { 4 + c.len() as u8 } else { 4 };
            out.push([cin, c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)]);
        }
        return out;
    }
    let cin = match st {
        0x80..=0xef => st >> 4,
        0xf1 | 0xf3 => 2,
        0xf2 => 3,
        0xf6 | 0xf7 => 5,
        0xf8..=0xff => 0xf,
        _ => return out,
    };
    out.push([cin, st, *msg.get(1).unwrap_or(&0), *msg.get(2).unwrap_or(&0)]);
    out
}

/// One USB-MIDI event packet into complete messages (`sysex` holds a SysEx under way).
pub fn midi_from_packet(p: &[u8; 4], sysex: &mut Vec<u8>, out: &mut Vec<Vec<u8>>) {
    let n = match p[0] & 0xf {
        0x4 => {
            sysex.extend_from_slice(&p[1..4]);
            return;
        }
        0x5 | 0xf => 1,
        0x2 | 0x6 | 0xc | 0xd => 2,
        0x3 | 0x7 | 0x8..=0xb | 0xe => 3,
        _ => return,
    };
    let cin = p[0] & 0xf;
    if (5..=7).contains(&cin) && (!sysex.is_empty() || p[1] == 0xf0) {
        sysex.extend_from_slice(&p[1..1 + (cin as usize - 4)]);
        out.push(std::mem::take(sysex));
        return;
    }
    out.push(p[1..1 + n].to_vec());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn midi_packets_round_trip() {
        let msgs: Vec<Vec<u8>> = vec![
            vec![0x90, 60, 100],
            vec![0xc0, 4],
            vec![0xf0, 0x00, 0x32, 0x45, 0x00, 0x00, 0x00, 0x40, 0x7f, 0xf7],
            vec![0xf0, 0x22, 0x24, 0x35, 0x7f, 0xf7],
            vec![0xf0, 1, 0xf7],
            vec![0xf8],
        ];
        let mut sx = Vec::new();
        let mut out = Vec::new();
        for m in &msgs {
            for p in usb_midi_packets(m) {
                midi_from_packet(&p, &mut sx, &mut out);
            }
        }
        assert_eq!(out, msgs);
    }

    /// RAM for the endpoints' DMA.
    struct Ram(Vec<u8>);
    impl Dma for Ram {
        fn dma_read8(&mut self, addr: u32) -> u8 {
            self.0[addr as usize & 0xfff]
        }
        fn dma_write8(&mut self, addr: u32, v: u8) {
            self.0[addr as usize & 0xfff] = v;
        }
    }

    const EP0_BUF: u32 = 0x100;
    const EP1_TX: u32 = 0x200;
    const EP1_RX: u32 = 0x300;

    /// A USB-MIDI device's firmware that polls the SIE, as Felucca's usb.c does: EP0's
    /// requests from a descriptor table, then MIDI on EP1 both ways.
    struct Fw {
        t: u64,
        conf: Vec<u8>,
        addr: Option<u8>,
        got: Vec<u8>,
    }
    impl Fw {
        fn new(conf: Vec<u8>) -> Self {
            Fw { t: 0, conf, addr: None, got: Vec::new() }
        }
        fn w(&mut self, u: &mut Usb, r: u32, v: u8) {
            u.write(USB_LO + 4, r << 8 | v as u32, self.t);
            self.t += 10;
        }
        fn r(&mut self, u: &mut Usb, r: u32) -> u8 {
            u.write(USB_LO + 4, r << 8 | 0x4000, self.t);
            self.t += 10;
            let v = u.read(USB_LO + 4, self.t);
            assert!(v & 0x8000 != 0, "the SIE answers");
            v as u8
        }
        fn desc(&self, wv: u16) -> Vec<u8> {
            match wv {
                0x0100 => vec![18, 1, 0x10, 1, 0, 0, 0, 64, 0x34, 0x12, 0x78, 0x56, 0, 1, 0, 2, 0, 1],
                0x0200 => self.conf.clone(),
                0x0300 => vec![4, 3, 0x09, 0x04],
                0x0302 => vec![10, 3, b'T', 0, b'e', 0, b's', 0, b't', 0],
                _ => Vec::new(),
            }
        }
        /// After each frame: what the main loop does.
        fn poll(&mut self, u: &mut Usb, ram: &mut Ram) {
            let tx = self.r(u, S_INTRTX1);
            if tx & 1 != 0 {
                self.w(u, S_INDEX, 0);
                let csr = self.r(u, S_CSR0);
                if csr & 1 == 0 {
                    if let Some(a) = self.addr.take() {
                        self.w(u, S_FADDR, a); // after SET_ADDRESS's status stage
                    }
                } else {
                    let s: Vec<u8> = (0..8).map(|i| ram.dma_read8(EP0_BUF + i)).collect();
                    let (wv, wl) = (u16::from_le_bytes([s[2], s[3]]), u16::from_le_bytes([s[6], s[7]]) as usize);
                    match (s[0], s[1]) {
                        (0x80, 6) => {
                            let d = self.desc(wv);
                            let n = d.len().min(wl);
                            for (i, &b) in d[..n].iter().enumerate() {
                                ram.dma_write8(EP0_BUF + i as u32, b);
                            }
                            u.write(USB_LO + 8, n as u32, self.t);
                            self.w(u, S_CSR0, 0x40 | 0x08 | 0x02);
                        }
                        (0, 5) => {
                            self.addr = Some(s[2]);
                            self.w(u, S_CSR0, 0x40 | 0x08);
                        }
                        _ => self.w(u, S_CSR0, 0x40 | 0x08),
                    }
                }
            }
            if self.r(u, S_INTRRX1) & 2 != 0 {
                self.w(u, S_INDEX, 1);
                let n = self.r(u, S_COUNT0) as u32;
                self.got.extend((0..n).map(|i| ram.dma_read8(EP1_RX + i)));
                self.w(u, S_RXCSR1, 0);
            }
        }
        /// One USB-MIDI packet to the computer on EP1.
        fn send(&mut self, u: &mut Usb, ram: &mut Ram, p: [u8; 4]) {
            for (i, &b) in p.iter().enumerate() {
                ram.dma_write8(EP1_TX + i as u32, b);
            }
            u.write(USB_LO + 0xc, 4, self.t);
            self.w(u, S_INDEX, 1);
            self.w(u, S_CSR0, 1);
        }
    }

    /// A configuration with a MIDI streaming interface, bulk OUT 1 and IN 0x81.
    fn midi_conf(total: u16) -> Vec<u8> {
        let mut c = vec![9, 2, total as u8, (total >> 8) as u8, 2, 1, 0, 0x80, 50];
        c.extend([9, 4, 0, 0, 0, 1, 1, 0, 0]);
        c.extend([9, 4, 1, 0, 2, 1, 3, 0, 0]);
        c.extend([9, 5, 0x01, 2, 64, 0, 0, 0, 0]);
        c.extend([9, 5, 0x81, 2, 64, 0, 0, 0, 0]);
        c
    }

    fn device(conf: Vec<u8>) -> (Usb, Fw, Ram) {
        let mut u = Usb { cable: true, sie_cycles: 1, ..Default::default() };
        u.write(USB_LO, 1 | 4, 0); // on, the SIE out of reset, every endpoint's DMA on
        u.write(USB_LO + 0x18, EP0_BUF, 0);
        u.write(USB_LO + 0x1c, EP1_TX, 0);
        u.write(USB_LO + 0x20, EP1_RX, 0);
        (u, Fw::new(conf), Ram(vec![0; 0x1000]))
    }

    fn run(u: &mut Usb, fw: &mut Fw, ram: &mut Ram, frames: usize) {
        for _ in 0..frames {
            u.frame(ram, true);
            fw.poll(u, ram);
        }
    }

    /// Enumeration step by step (descriptors, SET_ADDRESS at its status stage, the product
    /// string, SET_CONFIGURATION), then MIDI both ways.
    #[test]
    fn a_polling_device_enumerates_and_talks_midi() {
        let (mut u, mut fw, mut ram) = device(midi_conf(45));
        run(&mut u, &mut fw, &mut ram, 200);
        assert_eq!(u.events, [UsbEvent::Configured { product: "Test".into(), vid: 0x1234, pid: 0x5678 }]);
        assert_eq!(u.product(), Some("Test"));
        assert_eq!(u.faddr, 9, "at the address the host gave");

        u.midi_send(&[0x90, 60, 100]);
        run(&mut u, &mut fw, &mut ram, 2);
        assert_eq!(fw.got, [0x09, 0x90, 60, 100]);

        fw.send(&mut u, &mut ram, [0x08, 0x80, 60, 0]);
        run(&mut u, &mut fw, &mut ram, 2);
        assert_eq!(u.midi_take(), [vec![0x80, 60, 0]]);

        // the pull-up goes: off the bus
        u.frame(&mut ram, false);
        assert_eq!(u.events.last(), Some(&UsbEvent::Detached));
        assert_eq!(u.product(), None);
    }

    /// A configuration whose wTotalLength is too short for one (or 0) fails the enumeration
    /// instead of reading past it.
    #[test]
    fn a_short_configuration_fails_the_enumeration() {
        for total in [0u16, 4, 8] {
            let (mut u, mut fw, mut ram) = device(midi_conf(total));
            run(&mut u, &mut fw, &mut ram, 200);
            assert!(matches!(u.events.as_slice(), [UsbEvent::Failed(_)]), "wTotalLength {total}: {:?}", u.events);
            assert_eq!(u.product(), None);
        }
    }
}
