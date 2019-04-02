import {KernelBackend, DataMover, Tensor, DataType, util} from '@tensorflow/tfjs-core';

type TensorInfo = {
  shape: number[],
  dtype: DataType,
  values: Float32Array|Int32Array|Uint8Array,
  id: number,
  buffer?: any, // WebGPUBuffer
};

interface DataId {}

declare const GPUBufferUsage: any;
declare const GPUShaderStageBit: any;

export class WebGPUBackend extends KernelBackend {
  device: any;
  queue: any;
  shaderc: any;
  compiler: any;
  compileOpts: any;

  constructor(device: any, shaderc: any) {
    super();
    this.device = device;
    this.queue = device.getQueue();
    this.shaderc = shaderc;
    this.compiler = new shaderc.Compiler();
    this.compileOpts = new shaderc.CompileOptions();
  }

  floatPrecision(): number {
    return 32;
  }

  setDataMover(dataMover: DataMover): void {
    // TODO: tfjs team to implement this.
  }

  private tensorMap = new WeakMap<DataId, TensorInfo>();

  disposeData(dataId: DataId): void {
    // Tensor disposal logic.
  }

  register(dataId: object, shape: number[], dtype: DataType): void {
    if (!this.tensorMap.has(dataId)) {
      const buffer = this.device.createBuffer({
        size: util.sizeFromShape(shape) * util.bytesPerElement(dtype),
        usage: GPUBufferUsage.TRANSFER_SRC | GPUBufferUsage.TRANSFER_DST | GPUBufferUsage.STORAGE,
      });

      this.tensorMap.set(
          dataId, {shape, dtype, values: null, id: -1, buffer});
    }
  }

  write(dataId: object, values: Float32Array|Int32Array|Uint8Array): void {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }

    const info = this.tensorMap.get(dataId);
    info.values = values;
    info.buffer.setSubData(0, values.slice().buffer);
    this.tensorMap.set(dataId, info);
  }

  async getBufferData(info: TensorInfo): Promise<ArrayBuffer> {
    const size = util.sizeFromShape(info.shape) * util.bytesPerElement(info.dtype);
    const staging = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.TRANSFER_DST | GPUBufferUsage.MAP_READ,
    });
    {
      const encoder = this.device.createCommandEncoder({});
      encoder.copyBufferToBuffer(info.buffer, 0, staging, 0, size);
      this.queue.submit([encoder.finish()]);
    }
    const mapped: ArrayBuffer = await staging.mapReadAsync();

    const data = mapped.slice(0);
    info.buffer.unmap();

    return data;
  }

  async read(dataId: object): Promise<Float32Array|Int32Array|Uint8Array> {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }
    const info = this.tensorMap.get(dataId);
    const data = await this.getBufferData(info);

    return new Float32Array(data);
  }

  multiplyPipeline: any;
  getOrCreateMultiplyPipeline(): any {
    if (!this.multiplyPipeline) {
      const source = `#version 450
        layout(std140, set = 0, binding = 0) buffer A {
          float dataA[];
        };
        layout(std140, set = 0, binding = 1) buffer B {
          float dataB[];
        };
        layout(std140, set = 0, binding = 2) buffer Out {
          float dataOut[];
        };

        void main() {
          uint index = gl_GlobalInvocationID.x;
          dataOut[index] = dataA[index] * dataB[index];
        }
      `;
      const result = this.compiler.CompileGlslToSpv(
          source, this.shaderc.shader_kind.compute, "file", "main", this.compileOpts);
      const error = result.GetErrorMessage();
      if (error) {
        console.error(error);
      }
      const code = result.GetBinary().slice(0).buffer;

      const bgl = this.device.createBindGroupLayout({
        bindings: [
          { binding: 0, visibility: GPUShaderStageBit.COMPUTE, type: "storage-buffer" },
          { binding: 1, visibility: GPUShaderStageBit.COMPUTE, type: "storage-buffer" },
          { binding: 2, visibility: GPUShaderStageBit.COMPUTE, type: "storage-buffer" },
        ],
      });
      const layout = this.device.createPipelineLayout({ bindGroupLayouts: [bgl] });

      const module = this.device.createShaderModule({ code });
      const pipeline = this.device.createComputePipeline({
        layout,
        computeStage: { module, entryPoint: "main" }
      });

      this.multiplyPipeline = {
        bgl,
        pipeline,
      };
    }
    return this.multiplyPipeline;
  }

  multiply(a: Tensor, b: Tensor): Tensor {
    const aData = this.tensorMap.get(a.dataId);
    const bData = this.tensorMap.get(b.dataId);

    const output = Tensor.make(a.shape, {}, a.dtype, this);
    const info = this.tensorMap.get(output.dataId);

    const { bgl, pipeline } = this.getOrCreateMultiplyPipeline();

    const size = util.sizeFromShape(output.shape) * util.bytesPerElement(output.dtype);
    const bg = this.device.createBindGroup({
      layout: bgl,
      bindings: [
        { binding: 0, resource: { offset: 0, size, buffer: aData.buffer } },
        { binding: 1, resource: { offset: 0, size, buffer: bData.buffer } },
        { binding: 2, resource: { offset: 0, size, buffer: info.buffer } },
      ]
    });

    const encoder = this.device.createCommandEncoder({});
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatch(util.sizeFromShape(output.shape), 1, 1);
    pass.endPass();
    const cmd = encoder.finish();
    this.queue.submit([cmd]);

    return output;
  }
}