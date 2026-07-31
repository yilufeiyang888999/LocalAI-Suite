// Generate a multi-size Windows .ico file from a PNG, using sharp for image
// decoding/resizing (if available) or pngjs as a fallback. Avoids ImageMagick
// path issues on Windows.
//
// Usage: node scripts/png-to-ico.js <input.png> <output.ico> [sizes...]
//   defaults to: 256,192,128,72,48,32,16
//
// The output .ico embeds each requested size as a PNG payload (ICO format
// allows PNG-compressed images since Vista). This is the simplest path that
// works without native image deps.
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

const input = process.argv[2];
const output = process.argv[3];
const sizes = process.argv.slice(4).map(Number).filter(n => n > 0);

if (!input || !output) {
  console.error('usage: node scripts/png-to-ico.js <input.png> <output.ico> [sizes...]');
  process.exit(1);
}

const targetSizes = sizes.length ? sizes : [256, 192, 128, 72, 48, 32, 16];

const png = fs.readFileSync(input);
if (png.readUInt32BE(0) !== 0x89504e47) {
  console.error('not a PNG file');
  process.exit(1);
}

// Parse PNG chunks
function parseChunks(buf) {
  const chunks = [];
  let pos = 8; // skip signature
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString('ascii', pos + 4, pos + 8);
    const data = buf.subarray(pos + 8, pos + 8 + len);
    chunks.push({ type, data });
    pos += 8 + len + 4; // len + type + data + crc
    if (type === 'IEND') break;
  }
  return chunks;
}

const chunks = parseChunks(png);
const ihdr = chunks.find(c => c.type === 'IHDR');
if (!ihdr) { console.error('no IHDR'); process.exit(1); }
const width = ihdr.data.readUInt32BE(0);
const height = ihdr.data.readUInt32BE(4);
const bitDepth = ihdr.data[8];
const colorType = ihdr.data[9];
console.log(`source: ${width}x${height}, bit_depth=${bitDepth}, color_type=${colorType}`);

// IDAT may be split across multiple chunks; concatenate and inflate
const idat = Buffer.concat(chunks.filter(c => c.type === 'IDAT').map(c => c.data));
const inflated = zlib.inflateSync(idat);

// PNG layout: each row prefixed with filter byte. stride = width * bpp
function bpp(colorType, bitDepth) {
  const samples = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[colorType];
  return Math.max(1, samples * bitDepth / 8);
}

const channels = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[colorType];
const srcStride = width * bpp(colorType, bitDepth);
const srcRows = [];
for (let y = 0; y < height; y++) {
  const filter = inflated[y * (srcStride + 1)];
  srcRows.push({ filter, data: inflated.subarray(y * (srcStride + 1) + 1, y * (srcStride + 1) + 1 + srcStride) });
}

// Un-filter (apply reverse filter)
function paethPredictor(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

const pixels = Buffer.alloc(width * height * channels);
for (let y = 0; y < height; y++) {
  const row = srcRows[y];
  for (let x = 0; x < width * channels; x++) {
    const left = x >= channels ? pixels[y * width * channels + x - channels] : 0;
    const up = y > 0 ? pixels[(y - 1) * width * channels + x] : 0;
    const upLeft = (x >= channels && y > 0) ? pixels[(y - 1) * width * channels + x - channels] : 0;
    let recon;
    switch (row.filter) {
      case 0: recon = row.data[x]; break;
      case 1: recon = (row.data[x] + left) & 0xff; break;
      case 2: recon = (row.data[x] + up) & 0xff; break;
      case 3: recon = (row.data[x] + Math.floor((left + up) / 2)) & 0xff; break;
      case 4: recon = (row.data[x] + paethPredictor(left, up, upLeft)) & 0xff; break;
      default: recon = row.data[x];
    }
    pixels[y * width * channels + x] = recon;
  }
}

// Nearest-neighbor resize. Adds alpha=255 if image lacks alpha channel.
function resize(targetW, targetH) {
  const out = Buffer.alloc(targetW * targetH * 4);
  const hasAlpha = channels === 4;
  for (let y = 0; y < targetH; y++) {
    const sy = Math.floor(y * height / targetH);
    for (let x = 0; x < targetW; x++) {
      const sx = Math.floor(x * width / targetW);
      const si = (sy * width + sx) * channels;
      const di = (y * targetW + x) * 4;
      if (hasAlpha) {
        out[di] = pixels[si];
        out[di + 1] = pixels[si + 1];
        out[di + 2] = pixels[si + 2];
        out[di + 3] = pixels[si + 3];
      } else {
        out[di] = pixels[si];
        out[di + 1] = pixels[si + 1];
        out[di + 2] = pixels[si + 2];
        out[di + 3] = 255;
      }
    }
  }
  return out;
}

// Re-encode as PNG (so we can embed PNG-in-ICO, supported on Vista+)
function encodePng(w, h, rgba) {
  function be32(n) {
    const b = Buffer.alloc(4); b.writeUInt32BE(n, 0); return b;
  }
  // CRC32 table
  const crcTable = (() => {
    const t = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
      let c = i;
      for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
      t[i] = c >>> 0;
    }
    return t;
  })();
  function crc32(buf) {
    let c = 0xffffffff;
    for (let i = 0; i < buf.length; i++) c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }
  function chunk(type, data) {
    const len = be32(data.length);
    const typeBuf = Buffer.from(type, 'ascii');
    const crc = be32(crc32(Buffer.concat([typeBuf, data])));
    return Buffer.concat([len, typeBuf, data, crc]);
  }
  // IHDR: width, height, bit_depth=8, color_type=6 (RGBA), compression=0, filter=0, interlace=0
  const ihdrData = Buffer.alloc(13);
  ihdrData.writeUInt32BE(w, 0);
  ihdrData.writeUInt32BE(h, 4);
  ihdrData[8] = 8;
  ihdrData[9] = 6;
  ihdrData[10] = 0;
  ihdrData[11] = 0;
  ihdrData[12] = 0;
  // Build raw image: per-row filter byte 0 + RGBA bytes
  const raw = Buffer.alloc(h * (w * 4 + 1));
  for (let y = 0; y < h; y++) {
    raw[y * (w * 4 + 1)] = 0;
    rgba.copy(raw, y * (w * 4 + 1) + 1, y * w * 4, (y + 1) * w * 4);
  }
  const compressed = zlib.deflateSync(raw);
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdrData),
    chunk('IDAT', compressed),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

// Build the .ico
const icoImages = targetSizes.map(s => {
  const rgba = resize(s, s);
  const png = encodePng(s, s, rgba);
  return { size: s, png };
});

// ICONDIR (6 bytes): reserved=0, type=1, count
const dir = Buffer.alloc(6);
dir.writeUInt16LE(0, 0);
dir.writeUInt16LE(1, 2);
dir.writeUInt16LE(icoImages.length, 4);

// ICONDIRENTRY (16 bytes each): width, height, color count, reserved, planes,
// bpp, byte size, offset
const entries = [];
const datas = [];
let offset = 6 + 16 * icoImages.length;
for (const img of icoImages) {
  const e = Buffer.alloc(16);
  e[0] = img.size >= 256 ? 0 : img.size; // width (0 means 256)
  e[1] = img.size >= 256 ? 0 : img.size; // height
  e[2] = 0; // color count (palette; 0 for >=8bpp)
  e[3] = 0; // reserved
  e.writeUInt16LE(1, 4); // planes
  e.writeUInt16LE(32, 6); // bpp
  e.writeUInt32LE(img.png.length, 8); // size
  e.writeUInt32LE(offset, 12); // offset
  entries.push(e);
  datas.push(img.png);
  offset += img.png.length;
}

const ico = Buffer.concat([dir, ...entries, ...datas]);
fs.writeFileSync(output, ico);
console.log(`wrote ${output} (${ico.length} bytes, ${icoImages.length} sizes: ${targetSizes.join(', ')})`);
