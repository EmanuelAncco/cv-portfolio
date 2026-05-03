import { removeBackground } from '@imgly/background-removal-node';
import fs from 'node:fs/promises';
import path from 'node:path';

const inputPath = process.argv[2];
const outputPath = process.argv[3];
if (!inputPath || !outputPath) {
  console.error('Usage: node remove-bg.mjs <input> <output>');
  process.exit(1);
}

try {
  const inputBuffer = await fs.readFile(inputPath);
  console.log('Input size:', inputBuffer.length, 'bytes');
  const blob = await removeBackground(inputBuffer, {
    model: 'small',
    output: { quality: 0.9, type: 'foreground' },
    debug: false,
    progress: (key, current, total) => {
      if (current === total) console.log(`[bg-removal] ${key}: done`);
    }
  });
  const outputBuffer = Buffer.from(await blob.arrayBuffer());
  await fs.writeFile(outputPath, outputBuffer);
  console.log('Output written:', outputPath, outputBuffer.length, 'bytes');
} catch (err) {
  console.error('Error:', err.message);
  process.exit(1);
}
