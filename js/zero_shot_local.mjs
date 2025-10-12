import { pipeline, env } from '@xenova/transformers';

// Use local ONNX models from D:\\models
env.localModelPath = 'D:/models';
env.allowLocalModels = true;

// Allocate local zero-shot classifier using our js-intent folder
const clf = await pipeline('zero-shot-classification', 'js-intent');

const text = process.argv[2] || 'Please send the report today.';
const labels = (process.argv.slice(3).length ? process.argv.slice(3) : ['request','status_update','small_talk']);

const out = await clf(text, labels, { multi_label: true });
console.log(JSON.stringify(out, null, 2));
