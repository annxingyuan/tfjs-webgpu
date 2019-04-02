import * as tf from '@tensorflow/tfjs';

import {WebGPUBackend} from './backend_webgpu';
import Shaderc from './shaderc';

export * from '@tensorflow/tfjs';

export const ready = (async () => {
  // @ts-ignore navigator.gpu is required
  const adapter = await navigator.gpu.requestAdapter({});
  const device = await adapter.requestDevice({});
  const shaderc = await Shaderc;

  tf.ENV.registerBackend('webgpu', () => {
    return new WebGPUBackend(device, shaderc);
  }, 3 /*priority*/);

  // If registration succeeded, set the backend.
  if (tf.ENV.findBackend('webgpu') != null) {
    tf.setBackend('webgpu');
  }
})();